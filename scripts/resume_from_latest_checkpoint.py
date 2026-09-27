#!/usr/bin/env python3
# ============================================================================
# resume_from_latest_checkpoint.py
# ----------------------------------------------------------------------------
# Find the highest-step checkpoint in a directory that the binary can restore
# on the config's grid, inject its path into a config JSON as
# `checkpoint.restart_file`, write the modified config to a
# `<base>_resume.json` sibling file, and print the new path on stdout.
#
# Used by the `make cuda-resume-dendrite` target to drive a checkpoint-based
# resume of a long simulation that was interrupted (Ctrl-C, container kill,
# host suspend, etc.).
#
# Why we need this: SimulationEngine.cu:50 only restarts when
# CheckpointManager::has_restart_file() returns true, which (per
# CheckpointManager.cpp:96-101) only checks the explicit `restart_file`
# field — it does NOT auto-discover by directory. So we have to set that
# field ourselves before invoking the binary.
#
# A checkpoint is accepted only if its 128-byte CheckpointIO::Header has the
# ACCHKPT magic, version 1, num_fields 2, the config's Nx/Ny/Nz, positive
# spacing and the step in its file name; the file is exactly
# 128 + 2*Nx*Ny*Nz*8 bytes; and, when data_crc32 != 0, the CRC-32 of the
# phi||u payload matches it. Anything else (a truncated or foreign file, a
# checkpoint from another grid, corrupted data) is skipped with a warning and
# the next-older checkpoint is tried.
#
# Usage
#   python3 scripts/resume_from_latest_checkpoint.py CONFIG_PATH [CHECKPOINT_DIR]
#
# Stdout  -> path of the new resume-config (consumed by the Makefile)
# Stderr  -> human-readable progress
#
# Paths written to stdout and into `restart_file` are relative to the current
# directory whenever they lie inside it (see portable_path), because the
# Makefile runs this helper on the host from the repository root but runs
# the binary in a container that mounts that directory at /work (-w /work).
#
# Exit codes
#   0 = OK, resume config written
#   1 = no usable checkpoint (cold-start required, caller should use base config)
#   2 = base config not found / invalid, or any other error (never a cold start)
# ============================================================================

from __future__ import annotations

import json
import os
import re
import struct
import sys
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# File names written by CheckpointManager::save (CheckpointManager.cpp:50).
CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.acbin")

# CheckpointIO::Header (src/io/CheckpointIO.hpp), #pragma pack(1), in the
# little-endian byte order of the x86-64 hosts the binary runs on:
#   char magic[8]; int version; int Nx, Ny, Nz; double dx, dy, dz;
#   double dt; double time; int step; int num_fields; uint32_t data_crc32;
#   char reserved[52];
# CheckpointIO::write follows it with phi, then u, each Nx*Ny*Nz doubles.
HEADER = struct.Struct("<8s i 3i 3d d d i i I 52s")
HEADER_BYTES = 128
assert HEADER.size == HEADER_BYTES

# CheckpointIO::read accepts a file whose first 7 magic bytes are "ACCHKPT"
# (strncmp(hdr.magic, "ACCHKPT", 7)); the writer always adds a trailing NUL.
MAGIC_PREFIX = b"ACCHKPT"
SUPPORTED_VERSION = 1
NUM_FIELDS = 2  # phi, u
REAL_BYTES = 8  # Real is double (src/core/Grid.hpp)

# GridParams defaults (src/core/SimulationConfig.hpp) for keys a config omits.
DEFAULT_GRID = {"Nx": 600, "Ny": 600, "Nz": 600}

# The CRC is streamed so memory stays bounded (a 600^3 payload is 3.2 GiB).
CRC_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CheckpointHeader:
    magic: bytes
    version: int
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    dt: float
    time: float
    step: int
    num_fields: int
    data_crc32: int

    @classmethod
    def unpack(cls, raw: bytes) -> "CheckpointHeader":
        *fields, _reserved = HEADER.unpack(raw)
        return cls(*fields)


class InvalidCheckpoint(Exception):
    """The file is not a checkpoint the binary can restore on this grid."""


def portable_path(path: Path) -> Path:
    """Return `path` relative to the cwd if it lies inside it, else absolute.

    `make cuda-resume-dendrite` calls this helper on the host, from the
    repository root, and passes the result to the binary inside
    `docker run -v $PWD:/work -w /work`. A cwd-relative path names the same
    file on both sides; a host-absolute one does not exist in the container.
    """
    absolute = path.resolve()
    try:
        return absolute.relative_to(Path.cwd().resolve())
    except ValueError:
        return absolute


def config_grid(cfg: dict) -> tuple[int, int, int]:
    """Return the (Nx, Ny, Nz) the engine builds from `cfg`.

    SimulationEngine::initialize_from_checkpoint refuses a checkpoint whose
    dimensions differ from these, so they are what a resume must match.
    Raises ValueError if the config does not give integer dimensions.
    """
    grid = cfg.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError(f'"grid" must be an object, got {grid!r}')
    dims = []
    for key, default in DEFAULT_GRID.items():
        value = grid.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"grid.{key} must be a positive integer, got {value!r}")
        dims.append(value)
    return dims[0], dims[1], dims[2]


def read_valid_header(path: Path, step: int,
                      grid: tuple[int, int, int]) -> CheckpointHeader:
    """Return the header of `path` if the binary can restore it on `grid`.

    `step` is the step in the file name. Raises InvalidCheckpoint with the
    reason otherwise. I/O errors propagate: an unreadable newest checkpoint
    must not silently turn into a resume from an older one.
    """
    nx, ny, nz = grid
    payload_bytes = NUM_FIELDS * nx * ny * nz * REAL_BYTES
    expected_bytes = HEADER_BYTES + payload_bytes

    with path.open("rb") as fh:
        raw = fh.read(HEADER_BYTES)
        if len(raw) < HEADER_BYTES:
            raise InvalidCheckpoint(
                f"{len(raw)} B, shorter than the {HEADER_BYTES} B header")
        hdr = CheckpointHeader.unpack(raw)

        if hdr.magic[:len(MAGIC_PREFIX)] != MAGIC_PREFIX:
            raise InvalidCheckpoint(f"bad magic {hdr.magic!r}")
        if hdr.version != SUPPORTED_VERSION:
            raise InvalidCheckpoint(f"unsupported version {hdr.version}")
        if hdr.num_fields != NUM_FIELDS:
            raise InvalidCheckpoint(f"num_fields {hdr.num_fields}, expected {NUM_FIELDS}")
        if (hdr.nx, hdr.ny, hdr.nz) != grid:
            raise InvalidCheckpoint(
                f"grid {hdr.nx}x{hdr.ny}x{hdr.nz} does not match the config's "
                f"{nx}x{ny}x{nz}")
        if hdr.dx <= 0.0 or hdr.dy <= 0.0 or hdr.dz <= 0.0:
            raise InvalidCheckpoint("non-positive grid spacing")
        if hdr.step != step:
            raise InvalidCheckpoint(
                f"header step {hdr.step} does not match the file name")

        size = os.fstat(fh.fileno()).st_size
        if size != expected_bytes:
            raise InvalidCheckpoint(
                f"{size:,} B, expected exactly {expected_bytes:,} B "
                f"(header + phi + u) — truncated or trailing data")

        if hdr.data_crc32 != 0:  # 0 = written before CRCs were stored
            crc = 0
            remaining = payload_bytes
            while remaining:
                chunk = fh.read(min(CRC_CHUNK_BYTES, remaining))
                if not chunk:
                    raise InvalidCheckpoint("payload ended early while reading")
                crc = zlib.crc32(chunk, crc)
                remaining -= len(chunk)
            if crc != hdr.data_crc32:
                raise InvalidCheckpoint(
                    f"CRC32 mismatch (header {hdr.data_crc32:#010x}, "
                    f"data {crc:#010x}) — corrupted")
    return hdr


def find_latest_checkpoint(
        checkpoint_dir: Path,
        grid: tuple[int, int, int]) -> Optional[tuple[Path, CheckpointHeader]]:
    """Return (path, header) of the highest-step valid checkpoint, or None.

    Candidates are tried newest first; each invalid one is reported on stderr
    and the next-older one is tried, so a checkpoint cut short by a kill or a
    full disk, or one left over from a run on another grid, does not sink the
    resume.
    """
    if not checkpoint_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for entry in checkpoint_dir.iterdir():
        m = CHECKPOINT_RE.fullmatch(entry.name)
        if m and entry.is_file():
            candidates.append((int(m.group(1)), entry))

    # Descending step order, so we try the newest first.
    candidates.sort(key=lambda p: -p[0])

    for step, path in candidates:
        try:
            return path, read_valid_header(path, step, grid)
        except InvalidCheckpoint as exc:
            sys.stderr.write(f"  WARN: skipping {path.name}: {exc}\n")
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        sys.stderr.write(
            "usage: resume_from_latest_checkpoint.py CONFIG_PATH [CHECKPOINT_DIR]\n"
        )
        return 2

    config_path = Path(argv[1]).resolve()
    if not config_path.is_file():
        sys.stderr.write(f"ERROR: config not found: {config_path}\n")
        return 2

    try:
        with config_path.open() as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"ERROR: malformed JSON in {config_path}: {exc}\n")
        return 2
    if not isinstance(cfg, dict):
        sys.stderr.write(f"ERROR: {config_path} is not a JSON object\n")
        return 2

    try:
        grid = config_grid(cfg)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {config_path}: {exc}\n")
        return 2

    # Prefer the explicit second argument; otherwise read from the config or
    # default to "./checkpoints" (matches the engine default).
    if len(argv) == 3:
        ckpt_dir = Path(argv[2]).resolve()
    else:
        ckpt_dir = Path(
            cfg.get("checkpoint", {}).get("checkpoint_dir", "./checkpoints")
        ).resolve()

    nx, ny, nz = grid
    expected = HEADER_BYTES + NUM_FIELDS * nx * ny * nz * REAL_BYTES
    sys.stderr.write(f"==> Looking for latest checkpoint in {ckpt_dir}\n")
    sys.stderr.write(f"    grid {nx}x{ny}x{nz}: expecting {expected:,} B per checkpoint\n")
    found = find_latest_checkpoint(ckpt_dir, grid)
    if found is None:
        sys.stderr.write(
            "==> No usable checkpoint found.  Caller should run the BASE config "
            "(cold start) instead of resuming.\n"
        )
        return 1

    latest, hdr = found
    step = hdr.step
    crc_note = "CRC32 verified" if hdr.data_crc32 else "no CRC32 stored"
    sys.stderr.write(f"==> Latest checkpoint: {latest.name} "
                     f"(step={step}, time={hdr.time:.4f}, "
                     f"size={expected / (1024 * 1024):.1f} MB, {crc_note})\n")

    # Inject restart_file into a deep copy and write to <base>_resume.json
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["restart_file"] = str(portable_path(latest))

    resume_path = config_path.with_name(config_path.stem + "_resume.json")
    with resume_path.open("w") as fh:
        json.dump(cfg, fh, indent=4)

    # Sanity: warn if max_steps <= step (resume would do nothing)
    max_steps = cfg.get("time", {}).get("max_steps", 0)
    if max_steps and step >= max_steps:
        sys.stderr.write(
            f"  WARN: checkpoint step {step} >= time.max_steps {max_steps}; "
            "the engine will not advance.  Increase max_steps in the base config.\n"
        )

    sys.stderr.write(f"==> Wrote resume config: {resume_path}\n")
    sys.stderr.write(f"    -> the binary will pick up at step {step + 1}, "
                     f"target step {max_steps}, "
                     f"remaining ~{max_steps - step} steps\n")

    # Stdout = JUST the path, for clean Makefile consumption.
    sys.stdout.write(str(portable_path(resume_path)))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv)
    except Exception:  # noqa: BLE001 - exit status 1 would mean "cold start"
        traceback.print_exc()
        sys.stderr.write("ERROR: resume helper failed; not resuming or cold-starting.\n")
        status = 2
    raise SystemExit(status)
