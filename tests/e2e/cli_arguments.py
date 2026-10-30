#!/usr/bin/env python3
"""The binary takes only its documented command lines and rejects any other.

  allen-cahn-cuda [config.json] | -h | --help | --validate-config config.json
                | --resume config.json

An argument after them used to be ignored: `allen-cahn-cuda config.json
--validate-config`, with the two swapped, started a GPU run that wrote output
and rotated checkpoints. Each such command line must exit 1 with the usage text
and without creating the configuration's output or checkpoint directory; the
documented ones keep working (--help exits 0, --validate-config of a valid file
prints "valid:"). No GPU is needed: every case ends before a run.

Usage: cli_arguments.py BINARY WORKDIR   (exit 0 = pass)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    binary, work = Path(sys.argv[1]), Path(sys.argv[2])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cfg = {
        "grid": {"Nx": 8, "Ny": 8, "Nz": 8},
        "time": {"dt": 0.005, "max_steps": 2},
        "output": {"output_dir": str(work / "out")},
        "checkpoint": {"checkpoint_dir": str(work / "ckpt")},
    }
    cfg_path = work / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    c = str(cfg_path)
    cases = [
        ([c, "--validate-config"], 1, "unexpected argument after"),
        ([c, "extra"], 1, "unexpected argument after"),
        (["--validate-config", c, "extra"], 1, "Usage:"),
        (["--help", "extra"], 1, "Usage:"),
        (["--resume"], 1, "Usage:"),
        (["--resume", c, "extra"], 1, "Usage:"),
        (["--help"], 0, "Usage:"),
        (["--validate-config", c], 0, "valid: "),
    ]
    failures = 0
    for args, status, needle in cases:
        out = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=120,
                             cwd=work)
        text = out.stdout + out.stderr
        made = [d for d in ("out", "ckpt") if (work / d).exists()]
        ok = out.returncode == status and needle in text and not made
        failures += not ok
        shown = " ".join(a if a != c else "config.json" for a in args)
        print(f"{'ok  ' if ok else 'FAIL'} {shown}: exit {out.returncode}"
              + (f" (expected {status})" if out.returncode != status else "")
              + (f", created {made}" if made else "")
              + ("" if needle in text else f", no \"{needle}\" in the output"))
        for d in made:
            shutil.rmtree(work / d)
    print(f"{len(cases)} command lines, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
