#!/usr/bin/env python3
# ============================================================================
# Tests for scripts/resume_from_latest_checkpoint.py.
#
# Writes synthetic checkpoints in the on-disk format of src/io/CheckpointIO.hpp
# (packed little-endian 128-byte Header, then phi and u as Nx*Ny*Nz doubles
# each; data_crc32 = CRC-32 of phi||u, in version 1 0 = not stored; version 2,
# which the binary writes, adds header_crc32) into a temp directory,
# runs the helper from the directory holding config/ and checkpoints/, with
# relative arguments, and asserts that:
#
#   1. the highest-step valid checkpoint is chosen, and the printed config
#      path and injected restart_file stay relative to the cwd
#   2. truncated files (short payload, short header) are skipped
#   3. files from another grid are skipped, even when their size matches
#   4. files whose payload does not match the stored CRC32 are skipped
#   5. version-1 files are accepted, with data_crc32 == 0 on header + size
#   6. bad magic / version / num_fields / spacing, a header step that
#      differs from the file name, and trailing bytes are rejected
#   7. with no valid checkpoint the helper requests a cold start: exit 1,
#      nothing on stdout, no resume config written
#   8. the CRC is computed correctly when streamed in small chunks
#   9. a config without usable grid dimensions exits 2, never 1
#  9a. given only the config, as the Makefile's resume targets call it, the
#      helper searches the config's own checkpoint_dir
#  10. checkpoints of another spacing, or with NaN spacing, are skipped
#  11. a config with a key given twice in one object exits 2
#  12. the checkpoint's size is reported in MB of 10^6 bytes
#  13. a version-2 header whose CRC does not match it is skipped, also with a
#      run state in range; version 2 always checks the payload's CRC
#
# Run:  python3 tests/test_resume_checkpoint.py   (stdlib only; exit 0 = pass)
# ============================================================================

from __future__ import annotations

import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "scripts" / "resume_from_latest_checkpoint.py"
sys.path.insert(0, str(ROOT / "scripts"))

import resume_from_latest_checkpoint as resume  # noqa: E402

GRID = (4, 4, 4)
HEADER_FMT = "<8siiiidddddiiII48s"  # CheckpointIO::Header, 128 bytes
assert struct.calcsize(HEADER_FMT) == 128


def payload(grid: tuple[int, int, int], seed: int = 0) -> bytes:
    """phi then u for `grid`, as little-endian doubles."""
    n = grid[0] * grid[1] * grid[2]
    phi = [((i * 7 + seed) % 13) / 13.0 - 0.5 for i in range(n)]
    u = [-0.8 + ((i * 3 + seed) % 5) * 0.01 for i in range(n)]
    return struct.pack(f"<{2 * n}d", *phi, *u)


def write_checkpoint(ckpt_dir: Path, step: int, *,
                     grid: tuple[int, int, int] = GRID,
                     magic: bytes = b"ACCHKPT\0",
                     version: int = 2,
                     num_fields: int = 2,
                     spacing: float = 0.4,
                     header_step: int | None = None,
                     dt: float = 0.005,
                     time: float | None = None,
                     crc: int | None = None,
                     header_crc: int | None = None,
                     flip_payload_byte: bool = False,
                     truncate_to: int | None = None,
                     trailing: bytes = b"") -> Path:
    """Write checkpoint_<step>.acbin; `crc=None` stores the true CRC-32 of the
    payload, `header_crc=None` the true header CRC in version 2 (0 in
    version 1, as CheckpointIO::write of version 1 left the field)."""
    data = payload(grid, seed=step)
    stored_crc = zlib.crc32(data) if crc is None else crc

    def pack(header_crc32: int) -> bytes:
        return struct.pack(HEADER_FMT, magic, version, *grid,
                           spacing, spacing, spacing, dt,
                           step * 0.005 if time is None else time,
                           step if header_step is None else header_step,
                           num_fields, stored_crc, header_crc32, b"\0" * 48)
    if header_crc is None:
        header_crc = zlib.crc32(pack(0)) if version == 2 else 0
    header = pack(header_crc)
    if flip_payload_byte:
        mid = len(data) // 2
        data = data[:mid] + bytes([data[mid] ^ 0x01]) + data[mid + 1:]
    blob = header + data + trailing
    if truncate_to is not None:
        blob = blob[:truncate_to]
    path = ckpt_dir / f"checkpoint_{step}.acbin"
    path.write_bytes(blob)
    return path


class ResumeHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        (self.work / "config").mkdir()
        self.ckpt = self.work / "checkpoints"
        self.ckpt.mkdir()
        self.config = {
            "grid": {"Nx": GRID[0], "Ny": GRID[1], "Nz": GRID[2],
                     "dx": 0.4, "dy": 0.4, "dz": 0.4},
            "time": {"dt": 0.005, "max_steps": 10000},
            "checkpoint": {"frequency": 100, "checkpoint_dir": "./checkpoints",
                           "keep_last": 3},
        }
        self.write_config(self.config)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_config(self, cfg: dict) -> None:
        (self.work / "config" / "run.json").write_text(json.dumps(cfg))

    def run_helper(self, *dir_arg: str) -> subprocess.CompletedProcess:
        args = dir_arg if dir_arg else ("checkpoints",)
        return subprocess.run(
            [sys.executable, str(HELPER), "config/run.json", *args],
            cwd=self.work, capture_output=True, text=True, check=False)

    def assert_resumes_from(self, step: int) -> subprocess.CompletedProcess:
        res = self.run_helper()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout, "config/run_resume.json\n", res.stderr)
        resumed = json.loads((self.work / "config" / "run_resume.json").read_text())
        self.assertEqual(resumed["checkpoint"]["restart_file"],
                         f"checkpoints/checkpoint_{step}.acbin")
        self.assertEqual(resumed["grid"], self.config["grid"])
        return res

    def assert_cold_start(self) -> subprocess.CompletedProcess:
        res = self.run_helper()
        self.assertEqual(res.returncode, 1, res.stderr)
        self.assertEqual(res.stdout, "")
        self.assertFalse((self.work / "config" / "run_resume.json").exists())
        return res

    # 1 ----------------------------------------------------------------------
    def test_picks_highest_valid_step(self) -> None:
        write_checkpoint(self.ckpt, 90)   # numerically lower, lexically higher
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200)
        write_checkpoint(self.ckpt, 300, truncate_to=128 + 100)
        (self.ckpt / "checkpoint_400.acbin.tmp").write_bytes(b"partial")
        res = self.assert_resumes_from(200)
        self.assertIn("skipping checkpoint_300.acbin", res.stderr)

    # 2 ----------------------------------------------------------------------
    def test_skips_truncated(self) -> None:
        write_checkpoint(self.ckpt, 100)
        full = 128 + 2 * GRID[0] * GRID[1] * GRID[2] * 8
        write_checkpoint(self.ckpt, 200, truncate_to=full - 8)  # last double cut
        write_checkpoint(self.ckpt, 300, truncate_to=100)       # inside header
        write_checkpoint(self.ckpt, 400, truncate_to=0)         # empty file
        res = self.assert_resumes_from(100)
        for step in (200, 300, 400):
            self.assertIn(f"skipping checkpoint_{step}.acbin", res.stderr)

    # 3 ----------------------------------------------------------------------
    def test_skips_other_grid(self) -> None:
        write_checkpoint(self.ckpt, 100)
        # Same number of cells (8*4*2 == 4*4*4), so the same byte size.
        write_checkpoint(self.ckpt, 200, grid=(8, 4, 2))
        # Internally consistent checkpoint of a larger grid.
        write_checkpoint(self.ckpt, 300, grid=(5, 5, 5))
        res = self.assert_resumes_from(100)
        self.assertIn("grid 8x4x2 does not match the config's 4x4x4", res.stderr)
        self.assertIn("grid 5x5x5 does not match the config's 4x4x4", res.stderr)

    # 4 ----------------------------------------------------------------------
    def test_skips_bad_crc(self) -> None:
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200, flip_payload_byte=True)
        res = self.assert_resumes_from(100)
        self.assertIn("CRC32 mismatch", res.stderr)

    # 4b ---------------------------------------------------------------------
    # The binary rejects a header whose time or dt is out of range (the CRC
    # covers the fields only), so the helper skips it and resumes from the
    # next-older checkpoint.
    def test_skips_invalid_run_state(self) -> None:
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200, dt=math.nan)
        write_checkpoint(self.ckpt, 300, dt=0.0)
        write_checkpoint(self.ckpt, 400, time=math.inf)
        write_checkpoint(self.ckpt, 500, time=-1.0)
        res = self.assert_resumes_from(100)
        self.assertIn("skipping checkpoint_200.acbin: dt nan is not a finite value > 0",
                      res.stderr)
        self.assertIn("skipping checkpoint_300.acbin: dt 0.0 is not a finite value > 0",
                      res.stderr)
        self.assertIn("skipping checkpoint_400.acbin: time inf is not a finite value >= 0",
                      res.stderr)
        self.assertIn("skipping checkpoint_500.acbin: time -1.0 is not a finite value >= 0",
                      res.stderr)

    # 4c ---------------------------------------------------------------------
    # A config without time.max_steps runs the engine's default, 6000 steps:
    # a checkpoint at step 6000 leaves nothing to do, and the helper says so.
    def test_omitted_max_steps_is_the_engines_default(self) -> None:
        del self.config["time"]["max_steps"]
        self.write_config(self.config)
        write_checkpoint(self.ckpt, 6000)
        res = self.assert_resumes_from(6000)
        self.assertIn("checkpoint step 6000 >= time.max_steps 6000; the engine will not "
                      "advance", res.stderr)
        self.assertIn("target step 6000, remaining ~0 steps", res.stderr)

    # 4d ---------------------------------------------------------------------
    # The defaults the helper fills in for omitted keys are the C++ ones.
    def test_defaults_match_the_engine(self) -> None:
        header = (Path(__file__).resolve().parents[1] / "src" / "core"
                  / "SimulationConfig.hpp").read_text()
        grid = re.search(r"int Nx = (\d+), Ny = (\d+), Nz = (\d+);", header)
        spacing = re.search(r"Real dx = ([\d.]+), dy = ([\d.]+), dz = ([\d.]+);", header)
        steps = re.search(r"int max_steps = (\d+);", header)
        self.assertIsNotNone(grid)
        self.assertIsNotNone(spacing)
        self.assertIsNotNone(steps)
        self.assertEqual(resume.DEFAULT_GRID,
                         dict(zip(("Nx", "Ny", "Nz"), map(int, grid.groups()))))
        self.assertEqual(resume.DEFAULT_SPACING,
                         dict(zip(("dx", "dy", "dz"), map(float, spacing.groups()))))
        self.assertEqual(resume.DEFAULT_MAX_STEPS, int(steps.group(1)))

    # 5 ----------------------------------------------------------------------
    def test_accepts_legacy_file_without_crc(self) -> None:
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200, version=1, crc=0)
        res = self.assert_resumes_from(200)
        self.assertIn("no CRC32 stored (version 1)", res.stderr)
        write_checkpoint(self.ckpt, 300, version=1)
        res = self.assert_resumes_from(300)
        self.assertIn("payload CRC32 verified; version 1 has no header CRC", res.stderr)

    # 6 ----------------------------------------------------------------------
    def test_rejects_bad_headers_and_sizes(self) -> None:
        cases = {
            "bad magic": dict(magic=b"NOTCHKPT"),
            "unsupported version": dict(version=3),
            "num_fields": dict(num_fields=3),
            "non-positive or NaN grid spacing": dict(spacing=0.0),
            "does not match the file name": dict(header_step=999),
            "trailing data": dict(trailing=b"\0" * 8),
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                for f in self.ckpt.iterdir():
                    f.unlink()
                write_checkpoint(self.ckpt, 100, **kwargs)
                res = self.assert_cold_start()
                self.assertIn(reason, res.stderr)

    # 7 ----------------------------------------------------------------------
    def test_no_valid_checkpoint_requests_cold_start(self) -> None:
        self.assert_cold_start()                     # empty directory
        write_checkpoint(self.ckpt, 100, truncate_to=64)
        write_checkpoint(self.ckpt, 200, grid=(5, 5, 5))
        write_checkpoint(self.ckpt, 300, flip_payload_byte=True)
        (self.ckpt / "notes.txt").write_text("not a checkpoint")
        res = self.assert_cold_start()
        self.assertIn("No usable checkpoint found", res.stderr)
        for f in self.ckpt.iterdir():
            f.unlink()
        self.ckpt.rmdir()
        self.assert_cold_start()                     # directory missing

    # 8 ----------------------------------------------------------------------
    def test_crc_is_streamed_in_chunks(self) -> None:
        good = write_checkpoint(self.ckpt, 100)
        bad = write_checkpoint(self.ckpt, 200, flip_payload_byte=True)
        saved = resume.CRC_CHUNK_BYTES
        resume.CRC_CHUNK_BYTES = 7  # not a divisor of the 1024 B payload
        try:
            hdr = resume.read_valid_header(good, 100, GRID)
            self.assertEqual(hdr.data_crc32, zlib.crc32(payload(GRID, seed=100)))
            with self.assertRaises(resume.InvalidCheckpoint):
                resume.read_valid_header(bad, 200, GRID)
        finally:
            resume.CRC_CHUNK_BYTES = saved

    # 9a --------------------------------------------------------------------
    def test_reads_the_checkpoint_dir_from_the_config(self) -> None:
        # The Makefile's resume targets pass only the config (checked by
        # tests/e2e/makefile_resume_call.py): the helper must search the
        # config's own checkpoint_dir, not a shared one (where another run
        # left a higher step of the same grid).
        own = self.ckpt / "run_a"
        own.mkdir()
        cfg = dict(self.config)
        cfg["checkpoint"] = dict(cfg["checkpoint"], checkpoint_dir="./checkpoints/run_a")
        self.write_config(cfg)
        write_checkpoint(own, 200)
        write_checkpoint(self.ckpt, 900)
        res = subprocess.run([sys.executable, str(HELPER), "config/run.json"], cwd=self.work,
                             capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, res.stderr)
        resumed = json.loads((self.work / "config" / "run_resume.json").read_text())
        self.assertEqual(resumed["checkpoint"]["restart_file"],
                         "checkpoints/run_a/checkpoint_200.acbin")

    # 9 ----------------------------------------------------------------------
    def test_config_grid(self) -> None:
        self.assertEqual(resume.config_grid({}), (600, 600, 600))  # engine defaults
        self.assertEqual(resume.config_grid({"grid": {"Nx": 8}}), (8, 600, 600))
        self.assertEqual(resume.config_grid({"grid": {"Nz": 2**31 - 1}}), (600, 600, 2**31 - 1))
        for bad in (0, 2**31, 4294967300, 8.0, "8", True):
            with self.assertRaises(ValueError):
                resume.config_grid({"grid": {"Ny": bad}})
        self.assertEqual(resume.config_spacing({}), (0.4, 0.4, 0.4))  # engine defaults
        self.assertEqual(resume.config_spacing({"grid": {"dy": 0.5}}), (0.4, 0.5, 0.4))
        self.assertEqual(resume.config_spacing({"grid": {"dz": 1}}), (0.4, 0.4, 1.0))
        for bad in (0, -0.4, "0.4", True, float("nan"), float("inf"), 10**400):
            with self.assertRaises(ValueError):
                resume.config_spacing({"grid": {"dx": bad}})
        write_checkpoint(self.ckpt, 100)
        self.write_config({**self.config, "grid": {"Nx": "4", "Ny": 4, "Nz": 4}})
        res = self.run_helper()
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertEqual(res.stdout, "")

    # 10 ---------------------------------------------------------------------
    def test_skips_other_spacing_and_nan_spacing(self) -> None:
        # Retention leaves checkpoints of another spacing alone (a different
        # grid), and the binary refuses them and NaN spacing: the helper must
        # skip both and resume from the newest one the binary restores.
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200, spacing=0.3)
        write_checkpoint(self.ckpt, 300, spacing=float("nan"))
        res = self.assert_resumes_from(100)
        self.assertIn("skipping checkpoint_300.acbin: non-positive or NaN grid spacing",
                      res.stderr)
        self.assertIn("skipping checkpoint_200.acbin: spacing", res.stderr)

    # 11 ---------------------------------------------------------------------
    def test_rejects_a_key_given_twice(self) -> None:
        # json.load keeps the last value of a repeated key, and the resume
        # config written from it would carry that one only. The binary
        # rejects such a base config, so the helper must too (exit 2), at the
        # root and in a section, without writing a resume config.
        write_checkpoint(self.ckpt, 100)
        text = json.dumps(self.config)
        for old, new, key in (
                ('"time": {', '"time": {"scheme": "rk4"}, "time": {', "time"),
                ('"checkpoint_dir": ', '"checkpoint_dir": "./other", "checkpoint_dir": ',
                 "checkpoint_dir")):
            self.assertEqual(text.count(old), 1)
            (self.work / "config" / "run.json").write_text(text.replace(old, new))
            res = self.run_helper()
            self.assertEqual(res.returncode, 2, res.stderr)
            self.assertEqual(res.stdout, "")
            self.assertIn(f"key '{key}' appears twice in one object", res.stderr)
            self.assertFalse((self.work / "config" / "run_resume.json").exists())
        self.write_config(self.config)
        self.assert_resumes_from(100)

    # 12 ---------------------------------------------------------------------
    def test_reports_the_size_in_megabytes(self) -> None:
        # MB = 10^6 bytes, as in the binary's log and the documented sizes: a
        # 64^3 checkpoint, 128 + 2 * 64^3 * 8 = 4194432 bytes, is 4.2 MB (it
        # was reported as 4.0 MB, its size in MiB).
        self.write_config({**self.config,
                           "grid": {**self.config["grid"], "Nx": 64, "Ny": 64, "Nz": 64}})
        write_checkpoint(self.ckpt, 100, grid=(64, 64, 64))
        res = self.run_helper()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("(step=100, time=0.5000, size=4.2 MB, header and payload CRC32 verified)",
                      res.stderr)

    # 13 ---------------------------------------------------------------------
    def test_skips_a_corrupted_header(self) -> None:
        # A version-2 header carries its own CRC: a header changed after it was
        # written is skipped even when its run state is in range (time 1e300,
        # which the run-state checks pass), and the next older checkpoint is
        # resumed. Version 2 always checks the payload's CRC: a data_crc32 of 0
        # does not mean "none", as in version 1.
        write_checkpoint(self.ckpt, 100)
        path = write_checkpoint(self.ckpt, 200)
        raw = bytearray(path.read_bytes())
        struct.pack_into("<d", raw, 8 + 4 + 12 + 24 + 8, 1e300)  # time
        path.write_bytes(bytes(raw))
        res = self.assert_resumes_from(100)
        self.assertIn("skipping checkpoint_200.acbin: header CRC32 mismatch", res.stderr)
        write_checkpoint(self.ckpt, 300, crc=0)  # header CRC over the 0
        res = self.assert_resumes_from(100)
        self.assertIn("skipping checkpoint_300.acbin: CRC32 mismatch (header 0x00000000",
                      res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
