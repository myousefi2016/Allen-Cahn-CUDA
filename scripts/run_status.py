#!/usr/bin/env python3
# ============================================================================
# run_status.py — the state of a run, from its configuration
# ----------------------------------------------------------------------------
# Prints, for CONFIG:
#   * time.max_steps, an upper bound: the saturation guard stops a dendrite run
#     once the crystal reaches a wall, usually far earlier;
#   * the checkpoints in checkpoint.checkpoint_dir, in step order, with their
#     sizes (MB of 10^6 bytes);
#   * the one a resume restarts from: the newest the binary can restore on the
#     configuration's grid (the binary's --resume and
#     scripts/resume_from_latest_checkpoint.py choose the same one), with the
#     step and simulated time of its header;
#   * the frames written to output.output_dir in output.format;
#   * the last line of LOG, if it is given and exists.
#
# Used by `make cuda-status-dendrite`. Reads files only.
#
# Usage: python3 scripts/run_status.py CONFIG [LOG]
# Exit status: 0; 2 if CONFIG cannot be read or has no usable grid.
# ============================================================================

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resume_from_latest_checkpoint import (CHECKPOINT_RE, DEFAULT_MAX_STEPS,  # noqa: E402
                                           RepeatedKeyError, config_grid, config_spacing,
                                           find_latest_checkpoint, reject_repeated_keys)

# SimulationConfig's defaults (src/core/SimulationConfig.hpp).
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_OUTPUT_DIR = "./out"
DEFAULT_FORMAT = "vts"
# The frame files of each output.format (src/io/VTKWriter.cpp): output_<step>.vts, or
# output_<step>_phi.raw beside output_<step>_u.raw.
FRAME_RE = {"vts": re.compile(r"output_(\d+)\.vts"), "raw": re.compile(r"output_(\d+)_phi\.raw")}


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        sys.stderr.write("usage: run_status.py CONFIG [LOG]\n")
        return 2
    config_path = Path(argv[1])
    try:
        with config_path.open() as fh:
            cfg = json.load(fh, object_pairs_hook=reject_repeated_keys)
        grid = config_grid(cfg)
        spacing = config_spacing(cfg)
    except (OSError, ValueError, RepeatedKeyError) as exc:
        sys.stderr.write(f"ERROR: {config_path}: {exc}\n")
        return 2

    print(f"==> Run status ({config_path})")
    max_steps = cfg.get("time", {}).get("max_steps", DEFAULT_MAX_STEPS)
    print(f"  max_steps:           {max_steps} (an upper bound: the saturation guard "
          f"stops the run earlier)")

    ckpt_dir = Path(cfg.get("checkpoint", {}).get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    if not ckpt_dir.is_dir():
        print(f"  Checkpoints:         (no {ckpt_dir}/ yet)")
    else:
        files = sorted((int(m.group(1)), p) for p in ckpt_dir.iterdir()
                       if p.is_file() and (m := CHECKPOINT_RE.fullmatch(p.name)))
        print(f"  Checkpoints ({ckpt_dir}):")
        if not files:
            print("    (none yet)")
        for _, p in files:
            print(f"    {p.stat().st_size / 1e6:10.1f} MB  {p.name}")
        found = find_latest_checkpoint(ckpt_dir, grid, spacing)
        if found is None:
            print("  A resume restarts:   at step 0 (no checkpoint of this grid reads back "
                  "intact)")
        else:
            path, header = found
            print(f"  A resume restarts:   from {path.name}: step {header.step}, "
                  f"time {header.time:.6g}")

    out_dir = Path(cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))
    fmt = cfg.get("output", {}).get("format", DEFAULT_FORMAT)
    if out_dir.is_dir() and fmt in FRAME_RE:
        frames = sorted(int(m.group(1)) for p in out_dir.iterdir()
                        if (m := FRAME_RE[fmt].fullmatch(p.name)))
        last = f", the last of step {frames[-1]}" if frames else ""
        print(f"  Frames ({fmt}):        {len(frames)} in {out_dir}{last}")
    elif not out_dir.is_dir():
        print(f"  Frames:              (no {out_dir}/ yet)")

    if len(argv) == 3 and Path(argv[2]).is_file():
        lines = Path(argv[2]).read_text(errors="replace").splitlines()
        print(f"  Last log line:       {lines[-1] if lines else '(empty)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
