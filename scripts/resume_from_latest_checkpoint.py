#!/usr/bin/env python3
# ============================================================================
# resume_from_latest_checkpoint.py
# ----------------------------------------------------------------------------
# Find the highest-numbered checkpoint in a directory, inject its path into
# a config JSON as `checkpoint.restart_file`, write the modified config to a
# `<base>_resume.json` sibling file, and print the new path on stdout.
#
# Used by the `make cuda-resume-dendrite` target to drive a checkpoint-based
# resume of a long simulation that was interrupted (Ctrl-C, container kill,
# host suspend, etc.).
#
# Why we need this: SimulationEngine.cu:46 only restarts when
# CheckpointManager::has_restart_file() returns true, which (per
# CheckpointManager.cpp:96-101) only checks the explicit `restart_file`
# field — it does NOT auto-discover by directory. So we have to set that
# field ourselves before invoking the binary.
#
# Usage
#   python3 scripts/resume_from_latest_checkpoint.py CONFIG_PATH [CHECKPOINT_DIR]
#
# Stdout  -> path of the new resume-config (consumed by the Makefile)
# Stderr  -> human-readable progress
# Exit codes
#   0 = OK, resume config written
#   1 = no checkpoint found (cold-start required, caller should use base config)
#   2 = base config not found / invalid JSON
# ============================================================================

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.acbin$")


def expected_checkpoint_bytes(cfg: dict) -> Optional[int]:
    """Return the expected on-disk size in bytes for this config, or None.

    CheckpointIO.cpp writes:  Header (~64B) + phi (Nx*Ny*Nz*8B) + u (Nx*Ny*Nz*8B)
    Anything within 1% of that is considered intact; anything significantly
    smaller is treated as a SIGKILL-during-write truncation.
    """
    g = cfg.get("grid", {})
    nx, ny, nz = g.get("Nx"), g.get("Ny"), g.get("Nz")
    if not (isinstance(nx, int) and isinstance(ny, int) and isinstance(nz, int)):
        return None
    # Header is small (~tens of bytes); use a conservative 256B upper bound
    return 256 + 2 * nx * ny * nz * 8


def find_latest_checkpoint(checkpoint_dir: Path,
                           expected_bytes: Optional[int] = None) -> Optional[Path]:
    """Return the highest-step .acbin that looks intact, or None.

    Iterates in descending step order; returns the first one whose size is
    within tolerance of `expected_bytes`. This way a truncated newest
    checkpoint (from SIGKILL mid-write) doesn't sink the resume — we just
    silently fall back to the previous one.

    Robust against:
      - directory not existing / empty
      - non-checkpoint files in the directory
      - files with malformed names
      - truncated checkpoints from interrupted writes (atomic rename is NOT
        used by CheckpointIO.cpp:11-40; a kill mid-write leaves a partial file)
    """
    if not checkpoint_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for entry in checkpoint_dir.iterdir():
        if not entry.is_file():
            continue
        m = CHECKPOINT_RE.search(entry.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), entry))

    # Descending step order, so we try the newest first.
    candidates.sort(key=lambda p: -p[0])

    tolerance = 0.99  # require >= 99% of expected size to count as intact
    for step, path in candidates:
        size = path.stat().st_size
        if expected_bytes is not None and size < tolerance * expected_bytes:
            sys.stderr.write(
                f"  WARN: skipping likely-truncated checkpoint {path.name} "
                f"(step {step}, {size:,} B vs expected {expected_bytes:,} B). "
                "Probably interrupted mid-write — falling back to next-older.\n"
            )
            continue
        return path
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

    # Prefer the explicit second argument; otherwise read from the config or
    # default to "./checkpoints" (matches the engine default).
    if len(argv) == 3:
        ckpt_dir = Path(argv[2]).resolve()
    else:
        ckpt_dir = Path(
            cfg.get("checkpoint", {}).get("checkpoint_dir", "./checkpoints")
        ).resolve()

    sys.stderr.write(f"==> Looking for latest checkpoint in {ckpt_dir}\n")
    expected = expected_checkpoint_bytes(cfg)
    if expected is not None:
        sys.stderr.write(
            f"    expected checkpoint size for this grid = {expected/1024/1024:.1f} MB\n"
        )
    latest = find_latest_checkpoint(ckpt_dir, expected)
    if latest is None:
        sys.stderr.write(
            "==> No usable checkpoint found.  Caller should run the BASE config "
            "(cold start) instead of resuming.\n"
        )
        return 1

    step = int(CHECKPOINT_RE.search(latest.name).group(1))
    size_mb = latest.stat().st_size / (1024 * 1024)
    sys.stderr.write(f"==> Latest checkpoint: {latest.name} "
                     f"(step={step}, size={size_mb:.1f} MB)\n")

    # Inject restart_file into a deep copy and write to <base>_resume.json
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["restart_file"] = str(latest)

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
    sys.stdout.write(str(resume_path))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
