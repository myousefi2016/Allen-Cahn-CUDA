#!/usr/bin/env python3
# ============================================================================
# Tests for scripts/resume_from_latest_checkpoint.py.
#
# Writes synthetic checkpoints in the on-disk format of src/io/CheckpointIO.hpp
# (packed little-endian 128-byte Header, then phi and u as Nx*Ny*Nz doubles
# each; data_crc32 = CRC-32 of phi||u, 0 = not stored) into a temp directory,
# runs the helper the way the Makefile does (from the directory holding
# config/ and checkpoints/, with relative arguments) and asserts that:
#
#   1. the highest-step valid checkpoint is chosen, and the printed config
#      path and injected restart_file stay relative to the cwd
#   2. truncated files (short payload, short header) are skipped
#   3. files from another grid are skipped, even when their size matches
#   4. files whose payload does not match the stored CRC32 are skipped
#   5. legacy files with data_crc32 == 0 are accepted on header + size
#   6. bad magic / version / num_fields / spacing, a header step that
#      differs from the file name, and trailing bytes are rejected
#   7. with no valid checkpoint the helper requests a cold start: exit 1,
#      nothing on stdout, no resume config written
#   8. the CRC is computed correctly when streamed in small chunks
#   9. a config without usable grid dimensions exits 2, never 1
#
# Run:  python3 tests/test_resume_checkpoint.py   (stdlib only; exit 0 = pass)
# ============================================================================

from __future__ import annotations

import json
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
HEADER_FMT = "<8siiiidddddiiI52s"  # CheckpointIO::Header, 128 bytes
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
                     version: int = 1,
                     num_fields: int = 2,
                     spacing: float = 0.4,
                     header_step: int | None = None,
                     crc: int | None = None,
                     flip_payload_byte: bool = False,
                     truncate_to: int | None = None,
                     trailing: bytes = b"") -> Path:
    """Write checkpoint_<step>.acbin; `crc=None` stores the true CRC-32."""
    data = payload(grid, seed=step)
    stored_crc = zlib.crc32(data) if crc is None else crc
    header = struct.pack(HEADER_FMT, magic, version, *grid,
                         spacing, spacing, spacing, 0.005, step * 0.005,
                         step if header_step is None else header_step,
                         num_fields, stored_crc, b"\0" * 52)
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

    def run_helper(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(HELPER), "config/run.json", "checkpoints"],
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

    # 5 ----------------------------------------------------------------------
    def test_accepts_legacy_file_without_crc(self) -> None:
        write_checkpoint(self.ckpt, 100)
        write_checkpoint(self.ckpt, 200, crc=0)
        res = self.assert_resumes_from(200)
        self.assertIn("no CRC32 stored", res.stderr)

    # 6 ----------------------------------------------------------------------
    def test_rejects_bad_headers_and_sizes(self) -> None:
        cases = {
            "bad magic": dict(magic=b"NOTCHKPT"),
            "unsupported version": dict(version=2),
            "num_fields": dict(num_fields=3),
            "non-positive grid spacing": dict(spacing=0.0),
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

    # 9 ----------------------------------------------------------------------
    def test_config_grid(self) -> None:
        self.assertEqual(resume.config_grid({}), (600, 600, 600))  # engine defaults
        self.assertEqual(resume.config_grid({"grid": {"Nx": 8}}), (8, 600, 600))
        write_checkpoint(self.ckpt, 100)
        self.write_config({**self.config, "grid": {"Nx": "4", "Ny": 4, "Nz": 4}})
        res = self.run_helper()
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertEqual(res.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
