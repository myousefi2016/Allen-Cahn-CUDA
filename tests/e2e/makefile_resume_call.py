#!/usr/bin/env python3
"""The Makefile's resume targets run the binary with --resume and the run's config,
and a dry run (`make -n`) runs nothing.

`make cuda-resume-dendrite` and `make cuda-resume-dendrite-large` resume
LONG_CONFIG / LARGE_CONFIG with `allen-cahn-cuda --resume CONFIG`: the binary
restores the newest checkpoint in the config's own checkpoint_dir that it can
restore on the config's grid, or starts at step 0 if there is none
(CheckpointManagerTest, e2e.KubernetesJobResumes). The recipe used to run the
resume helper and, on the same shell line, the cold-start $(MAKE): GNU make
runs a line that references $(MAKE) even under -n, so `make -n` with a usable
checkpoint ran the helper, rewrote config/*_resume.json and started the docker
simulation.

For each target:
  * no recipe line of it or of its prerequisites (as make's data base lists
    them) is one that make runs under -n: none references $(MAKE) or starts
    with '+';
  * `make -n TARGET`, with recording stubs for docker and python3 first on
    PATH (exiting 0, as the helper does when it finds a checkpoint), calls
    neither and leaves config/ as it was;
  * it prints exactly one docker command that runs ./$(NATIVE_BIN), whose
    arguments are `--resume CONFIG`, CONFIG being make's LONG_CONFIG /
    LARGE_CONFIG (config/run_dendrite_long.json, config/run_dendrite_large.json),
    a file of the checkout.

Needs GNU make and python3 on PATH; no binary, docker or GPU.

Usage: makefile_resume_call.py REPO_ROOT   (exit 0 = pass)
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import (data_base, docker_runs, logical_lines, make_values,
                             runs_under_dry_run, scrubbed_env, stub_bin, words)

# Target -> the Makefile variable that names the config it resumes.
TARGETS = {
    "cuda-resume-dendrite": "LONG_CONFIG",
    "cuda-resume-dendrite-large": "LARGE_CONFIG",
}

# Environment values that would replace the Makefile's `?=` defaults of the variables
# these recipes use.
SCRUB = ("LONG_CONFIG", "LARGE_CONFIG", "CUDA_BUILD_DIR", "NATIVE_BIN", "CUDA_DEV_IMAGE",
         "OUT_DIR", "CHECKPOINT_DIR")


def snapshot(directory: Path) -> dict:
    """Name -> (size, mtime in ns) of every file under directory."""
    return {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def closure(target: str, prereqs: dict) -> list[str]:
    """target and every rule it depends on, directly or not."""
    seen, todo = [], [target]
    while todo:
        t = todo.pop()
        if t in seen or t not in prereqs:
            continue
        seen.append(t)
        todo.extend(prereqs[t])
    return seen


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "calls.log"
        env = scrubbed_env(SCRUB, stub_bin(tmp, ["docker", "python3"]), log)
        values = make_values(make, root, env, ["NATIVE_BIN", *TARGETS.values()])
        binary = "./" + values["NATIVE_BIN"]
        prereqs, recipes = data_base(make, root, env)

        for target, variable in TARGETS.items():
            config = values[variable]
            errors = []
            if not (root / config).is_file():
                errors.append(f"{config} ({variable}) does not exist")
            for rule in closure(target, prereqs):
                for line in logical_lines(recipes.get(rule, [])):
                    if runs_under_dry_run(line):
                        errors.append(f"{rule}'s recipe line runs under make -n: {line.strip()}")

            log.write_text("")
            before = snapshot(root / "config")
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target],
                                 env=env, capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                errors.append(f"make -n exited {out.returncode}: {out.stderr.strip()}")
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            if calls:
                errors.append("make -n ran " + "; ".join(" ".join(c) for c in calls))
            if snapshot(root / "config") != before:
                errors.append("make -n changed config/")

            runs = [r for line in logical_lines(out.stdout.splitlines()) if binary in line
                    for r in docker_runs(words(line), binary)]
            if len(runs) != 1:
                errors.append(f"{len(runs)} docker runs of {binary} printed, expected one")
            for run in runs:
                args = run[run.index(binary) + 1:] if binary in run else run
                if args != ["--resume", config]:
                    errors.append(f"the binary runs with {args}, expected ['--resume', '{config}']")

            failures += bool(errors)
            print(f"FAIL {target}:" + "".join(f"\n     {e}" for e in errors) if errors
                  else f"ok   {target}: {binary} --resume {config}; make -n ran nothing")
    print(f"{len(TARGETS)} targets, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
