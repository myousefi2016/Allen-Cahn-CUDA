#!/usr/bin/env python3
"""The Makefile's simulation runs must let Ctrl-C and `docker stop` reach the binary.

The binary writes a checkpoint of the current step and exits 130 / 143 on
SIGINT / SIGTERM (e2e.SignalExitStatus). Under docker that holds only if the
binary is the container's main process: docker passes SIGINT / SIGTERM to that
process alone (its client's signal proxy for Ctrl-C, `docker stop`), and as
PID 1 it receives the signals it has handlers for. The dev image's entrypoint,
inherited from nvidia/cuda (/opt/nvidia/nvidia_entrypoint.sh), execs its
command. Wrapped in `bash -c './binary ...; chown ...'`, bash was PID 1 and
passed neither signal on: Ctrl-C left the run going, `docker stop` ended in
SIGKILL after the grace period, no checkpoint of the current step was written
either way, and a resume started a second run in the same directories.

Checked, for every target that runs the binary in a container:
  * the docker command that runs it has ./$(NATIVE_BIN) as the container's
    command, right after $(CUDA_DEV_IMAGE) (no shell in between), and no
    --entrypoint;
  * it runs as the calling user (--user UID:GID), so the files it writes need
    no chown;
  * $(CUDA_DEV_DOCKERFILE) sets no ENTRYPOINT of its own.
NATIVE_BIN, CUDA_DEV_IMAGE and CUDA_DEV_DOCKERFILE are make's own values.

The targets: those whose recipe runs $(NATIVE_BIN) (outside the double-quoted
strings of its messages) and those that reach one through their prerequisites
or a $(MAKE) call, read from make's data base (`make -pq` of a target that
does not exist, which runs no recipe). The test fails unless they are exactly
TARGETS, so that a new run target is checked too; add it there after making
sure that `make -n` of it runs nothing but sub-makes and the docker stub.

`make -n` prints recipe lines without running them, except a line that
references $(MAKE): it runs that one (cuda-run-vtk, cuda-dendrite-demo and
cuda-all reach the run through such sub-makes, which inherit -n and print).
Printed lines are split into words with shlex, as the shell splits these
unquoted words; a recording docker stub first on PATH gives the exact
arguments of any docker command that runs. Nothing is written to the checkout.

Needs GNU make and python3 on PATH; no binary, docker or GPU.

Usage: makefile_run_signals.py REPO_ROOT   (exit 0 = pass)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import (SEPARATORS, data_base, docker_runs, logical_lines, make_values,
                             scrubbed_env, stub_bin, words)

TARGETS = ["cuda-run", "cuda-run-small", "cuda-run-default", "cuda-run-vtk", "cuda-run-dendrite",
           "cuda-run-dendrite-long", "cuda-run-dendrite-large", "cuda-resume-dendrite",
           "cuda-resume-dendrite-large", "cuda-dendrite-demo", "cuda-all"]

# Environment values that would replace the Makefile's `?=` defaults of the variables
# these recipes use.
SCRUB = ("CONFIG", "CUDA_BUILD_DIR", "NATIVE_BIN", "CUDA_DEV_IMAGE", "CUDA_DEV_DOCKERFILE",
         "LONG_CONFIG", "LARGE_CONFIG", "OUT_DIR", "CHECKPOINT_DIR")


def run_targets(prereqs: dict, recipes: dict) -> list[str]:
    """Targets that run $(NATIVE_BIN), or reach such a target, as make would."""
    runs_binary = re.compile(r"\$[({]NATIVE_BIN[)}]")
    reach = {t for t, lines in recipes.items()
             if any(runs_binary.search(re.sub(r'"(?:[^"\\]|\\.)*"', '""', l))
                    for l in logical_lines(lines))}

    def sub_makes(target: str) -> set:
        called = set()
        for line in logical_lines(recipes.get(target, [])):
            if "$(MAKE)" not in line and "${MAKE}" not in line:
                continue
            w = words(line)
            for i, word in enumerate(w):
                if word.lstrip("@+-") in ("$(MAKE)", "${MAKE}"):
                    for arg in w[i + 1:]:
                        if arg in SEPARATORS:
                            break
                        if "=" not in arg and not arg.startswith("-"):
                            called.add(arg)
        return called

    while True:
        more = {t for t in prereqs if not t.startswith(".") and t not in reach
                and (set(prereqs[t]) & reach or sub_makes(t) & reach)}
        if not more:
            return sorted(reach)
        reach |= more


def check(command: list[str], binary: str, image: str, user: str) -> str:
    """'' if the docker run command runs the binary as the container's command as user."""
    if binary not in command:
        inside = next(word for word in command if binary in word)
        return f"the binary is not the container's command (it runs inside {inside!r})"
    at = command.index(binary)
    if command[at - 1] != image:
        return f"the binary follows {command[at - 1]!r}, not the image {image!r}"
    options = command[2:at - 1]
    if any(o == "--entrypoint" or o.startswith("--entrypoint=") for o in options):
        return f"the run replaces the image's entrypoint ({options})"
    given = None
    for k, o in enumerate(options):
        if o in ("--user", "-u") and k + 1 < len(options):
            given = options[k + 1]
        elif o.startswith("--user="):
            given = o[len("--user="):]
    if given != user:
        return f"it runs as {given or 'the image user (root)'}, not --user {user}"
    return ""


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    user = f"{os.getuid()}:{os.getgid()}"
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "calls.log"
        env = scrubbed_env(SCRUB, stub_bin(tmp, ["docker"]), log)
        values = make_values(make, root, env,
                             ["NATIVE_BIN", "CUDA_DEV_IMAGE", "CUDA_DEV_DOCKERFILE"])
        binary, image = "./" + values["NATIVE_BIN"], values["CUDA_DEV_IMAGE"]
        dockerfile = values["CUDA_DEV_DOCKERFILE"]

        entrypoints = [l for l in (root / dockerfile).read_text().splitlines()
                       if re.match(r"^\s*ENTRYPOINT\b", l, re.IGNORECASE)]
        if entrypoints:
            failures += 1
            print(f"FAIL {dockerfile} sets {entrypoints}: the image's command is then no longer "
                  "exec'd by the nvidia/cuda entrypoint")

        found = run_targets(*data_base(make, root, env))
        if found != sorted(TARGETS):
            failures += 1
            print(f"FAIL the targets that run the binary are {found}, this test checks "
                  f"{sorted(TARGETS)}")

        for target in TARGETS:
            log.write_text("")
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target],
                                 env=env, capture_output=True, text=True, timeout=120)
            printed = [docker_runs(words(l), binary)
                       for l in logical_lines(out.stdout.splitlines()) if binary in l]
            executed = [docker_runs(json.loads(l), binary) for l in log.read_text().splitlines()]
            runs = [r for found_runs in printed + executed for r in found_runs]
            problems = [p for p in (check(r, binary, image, user) for r in runs) if p]
            if out.returncode != 0:
                problems.append(f"make -n exited {out.returncode}: {out.stderr.strip()}")
            if not runs:
                problems.append("no docker run of the binary")
            failures += bool(problems)
            print(f"{'FAIL' if problems else 'ok  '} {target}: {len(runs)} run(s) of {binary}"
                  + ("".join(f"\n     {p}" for p in problems)))
    print(f"{len(TARGETS)} targets, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
