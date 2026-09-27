#!/usr/bin/env python3
"""End-to-end check of the simulation binary's graceful shutdown.

For SIGTERM and SIGINT: start the binary on a small grid with an effectively
unbounded max_steps, wait until it has logged step 100, send the signal, and
require
  * exit status 128 + signal (143 / 130), so make, scripts and a Kubernetes
    Job see an interrupted run instead of a completed one;
  * a checkpoint of the step it stopped at, with a valid header (resumable).

Usage: signal_exit.py BINARY WORKDIR
Exit 0 = pass, 1 = fail, 77 = no GPU (skipped; a failure if AC_REQUIRE_GPU=1).
"""

import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

SKIP = 77


def gpu_visible() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "GPU " in out.stdout


def run_case(binary: Path, work: Path, sig: signal.Signals) -> list[str]:
    errors: list[str] = []
    case = work / sig.name
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True)
    cfg = {
        "grid": {"Nx": 48, "Ny": 48, "Nz": 48, "dx": 0.4, "dy": 0.4, "dz": 0.4},
        "time": {"dt": 0.005, "max_steps": 100000000, "scheme": "euler"},
        "output": {"frequency": 1000000000, "output_dir": str(case / "out")},
        "checkpoint": {"frequency": 0, "checkpoint_dir": str(case / "ckpt"), "keep_last": 1},
        "initial": {"seed_radius": 2.0},
    }
    cfg_path = case / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    proc = subprocess.Popen([str(binary), str(cfg_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=case)
    log: list[str] = []
    deadline = time.monotonic() + 120
    reached = False
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line)
        if re.search(r"Step 100/", line):
            reached = True
            break
        if time.monotonic() > deadline:
            break
    if not reached:
        proc.kill()
        proc.wait()
        return [f"{sig.name}: never logged step 100; log tail:\n" + "".join(log[-20:])]

    proc.send_signal(sig)
    try:
        rest, _ = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        rest, _ = proc.communicate()
        errors.append(f"{sig.name}: did not exit within 120 s of the signal")
    log.append(rest or "")
    text = "".join(log)

    expected = 128 + int(sig)
    if proc.returncode != expected:
        errors.append(f"{sig.name}: exit status {proc.returncode}, expected {expected}")
    m = re.search(r"Shutdown requested at step (\d+)", text)
    if not m:
        errors.append(f"{sig.name}: no 'Shutdown requested' in the log")
    else:
        step = int(m.group(1))
        ckpt = case / "ckpt" / f"checkpoint_{step}.acbin"
        if not ckpt.is_file():
            errors.append(f"{sig.name}: {ckpt.name} was not written")
        else:
            raw = ckpt.read_bytes()[:128]
            magic, version, nx, ny, nz = struct.unpack_from("<8si3i", raw)
            (hdr_step,) = struct.unpack_from("<i", raw, 8 + 4 + 12 + 24 + 16)
            if magic[:7] != b"ACCHKPT" or version != 1 or (nx, ny, nz) != (48, 48, 48) \
                    or hdr_step != step:
                errors.append(f"{sig.name}: bad checkpoint header {magic!r} v{version} "
                              f"{nx}x{ny}x{nz} step {hdr_step}")
            else:
                print(f"{sig.name}: exit {proc.returncode}, checkpoint_{step}.acbin valid")
    if errors:
        errors.append("log tail:\n" + "".join(text.splitlines(keepends=True)[-15:]))
    return errors


def main() -> int:
    binary, work = Path(sys.argv[1]), Path(sys.argv[2])
    if not gpu_visible():
        if os.environ.get("AC_REQUIRE_GPU") == "1":
            print("FAIL: no GPU visible to nvidia-smi and AC_REQUIRE_GPU=1")
            return 1
        print("SKIP: no GPU visible to nvidia-smi")
        return SKIP
    errors = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        errors += run_case(binary, work, sig)
    for e in errors:
        print("FAIL:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
