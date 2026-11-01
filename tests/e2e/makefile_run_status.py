#!/usr/bin/env python3
"""`make cuda-status-dendrite` reports the long run's state exactly.

The target runs scripts/run_status.py on LONG_CONFIG and LONG_LOG. Its earlier
shell recipe could never print "(none yet)" (the `||` tested sort's status),
listed checkpoint_144000 before checkpoint_48000 (a text sort), and printed
progress as a percentage of time.max_steps, an upper bound the saturation
guard stops the run far below.

Runs the real target (it only reads files with python3) with LONG_CONFIG a
temporary 4^3 configuration whose checkpoint and output directories are
temporary, and checks, in turn:
  1. no checkpoint or output directory: "(no <dir>/ yet)" for both;
  2. an empty checkpoint directory: "(none yet)", and a resume starts at step 0;
  3. checkpoints of steps 48000, 144000 and 96000: listed in step order, and a
     resume restarts from checkpoint_144000 with the time its header records;
  4. a newer checkpoint_200000 cut short: still listed, skipped by the resume,
     which restarts from checkpoint_144000;
  5. frames output_0.vts and output_800.vts: two, the last of step 800; the
     last line of LONG_LOG; no percentage anywhere.
Checkpoints are written as the binary writes them (tests/test_resume_checkpoint.py).

Needs GNU make and python3; no binary, docker or GPU.

Usage: makefile_run_status.py REPO_ROOT   (exit 0 = pass)
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
from makefile_common import MAKE_STATE  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    sys.path.insert(0, str(root / "tests"))
    import test_resume_checkpoint as writer  # the checkpoint writer of the helper's tests

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt, out, log = tmp / "ckpt", tmp / "out", tmp / "run.log"
        cfg = tmp / "long.json"
        cfg.write_text(json.dumps({
            "grid": {"Nx": 4, "Ny": 4, "Nz": 4, "dx": 0.4, "dy": 0.4, "dz": 0.4},
            "time": {"dt": 0.005, "max_steps": 960000},
            "output": {"frequency": 800, "output_dir": str(out), "format": "vts"},
            "checkpoint": {"frequency": 48000, "checkpoint_dir": str(ckpt), "keep_last": 3}}))
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in MAKE_STATE + ("LONG_CONFIG", "LONG_LOG")}

        def status() -> tuple[str, str]:
            r = subprocess.run([make, "--no-print-directory", "-s", "-C", str(root),
                                "cuda-status-dendrite", f"LONG_CONFIG={cfg}", f"LONG_LOG={log}"],
                               env=env, capture_output=True, text=True, timeout=60)
            return (r.stdout if r.returncode == 0 else f"EXIT {r.returncode}\n{r.stdout}"), r.stderr

        def check(label: str, cond: bool, text: str) -> None:
            nonlocal failures
            if not cond:
                failures += 1
                print(f"FAIL {label}\n     " + text.strip().replace("\n", "\n     "))

        o, _ = status()
        check("1: no checkpoint directory", f"(no {ckpt}/ yet)" in o, o)
        check("1: no output directory", f"(no {out}/ yet)" in o, o)

        ckpt.mkdir()
        o, _ = status()
        check("2: empty checkpoint directory", "(none yet)" in o, o)
        check("2: a resume starts at step 0", "A resume restarts:   at step 0" in o, o)

        for step in (48000, 144000, 96000):
            writer.write_checkpoint(ckpt, step, grid=(4, 4, 4), time=step * 0.005)
        o, _ = status()
        order = [l.split()[-1] for l in o.splitlines() if l.strip().endswith(".acbin")]
        check("3: step order", order == ["checkpoint_48000.acbin", "checkpoint_96000.acbin",
                                         "checkpoint_144000.acbin"], o)
        check("3: the resume's checkpoint and time",
              "from checkpoint_144000.acbin: step 144000, time 720" in o, o)
        check("3: no '(none yet)'", "(none yet)" not in o, o)

        writer.write_checkpoint(ckpt, 200000, grid=(4, 4, 4), truncate_to=200)
        o, err = status()
        check("4: the cut checkpoint is listed", "checkpoint_200000.acbin" in o, o)
        check("4: the resume skips it", "from checkpoint_144000.acbin" in o
              and "skipping checkpoint_200000.acbin" in err, o + err)

        out.mkdir()
        for step in (0, 800):
            (out / f"output_{step}.vts").write_bytes(b"")
        log.write_text("first line\nStep 1600/960000: time=8.0\n")
        o, _ = status()
        check("5: frames", f"2 in {out}, the last of step 800" in o, o)
        check("5: last log line", "Last log line:       Step 1600/960000: time=8.0" in o, o)
        check("5: no percentage", "%" not in o, o)
        check("5: max_steps as an upper bound", "960000 (an upper bound" in o, o)

    print(f"{'ok  ' if not failures else 'FAIL'} cuda-status-dendrite: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
