#!/usr/bin/env python3
"""make docker-test and make docker-run run images built from this checkout.

The test and prod images carry the tree they are built from: docker/Dockerfile.test and
docker/Dockerfile.prod copy the sources, tests and configs and compile them in the image.
docker compose builds a service's image only when it is missing, so `docker compose run`
without --build runs an allencahn-cuda:test or :prod built before, from an earlier tree, and
a re-run after an edit reports that tree's results.

Checked, for each target: its recipe is printed by `make -n`, and every line of it is one
this test reads exactly, `echo "TEXT"` (no " $ ` or \\ in TEXT) or a simple command of plain
words (makefile_common.plain_command: no quoting, expansion, globbing, comment or operator
character, so the shell passes each word as it is) run by echo, mkdir or docker. A line of
any other form, such as a fallback after `||`, a branch, a subshell, a command substitution,
`sh -c` or `env`, fails the test: the shell's paths through it are not read here. The recipe
has exactly one docker line, `docker compose [OPTIONS] run [OPTIONS] SERVICE ...` with the
target's service (test, prod) and --build true (--build or --build=BOOL, the last one
counting, as compose reads it). The options are read with the tables of docker compose's
options (their --help, Compose v5.6.1; makefile_common.compose_call), so an option's value,
also after a cluster of short options (-Te VAR=1), is not taken for the service, and an option
the tables do not know fails the test. No line of the target or of its prerequisites runs
under `make -n` ($(MAKE), a '+' prefix), and `make -n`, with a recording docker stub first on
PATH, runs no docker.

Needs GNU make and python3; no docker.

Usage: makefile_compose_build.py REPO_ROOT   (exit 0 = pass)
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import (ECHO_LITERAL, compose_call, data_base, logical_lines,
                             plain_command, runs_under_dry_run, scrubbed_env, stub_bin)

SCRUB = ("HOST_UID", "HOST_GID", "IMAGE_NAME", "DOCKER_REGISTRY", "DOCKER_BUILD_ARGS")
TARGETS = {"docker-test": "test", "docker-run": "prod"}
COMMANDS = ("echo", "mkdir", "docker")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "calls.log"
        env = scrubbed_env(SCRUB, stub_bin(tmp, ["docker"]), log)
        prereqs, recipes = data_base(make, root, env)
        for target, service in TARGETS.items():
            # The target and its prerequisites, none of whose lines may run under -n.
            todo, seen = [target], []
            while todo:
                t = todo.pop()
                if t not in seen:
                    seen.append(t)
                    todo += prereqs.get(t, [])
            dry = [(t, l) for t in seen for l in logical_lines(recipes.get(t, []))
                   if runs_under_dry_run(l)]
            if dry:
                errors.append(f"make {target}: {dry[0][0]}'s line {dry[0][1]!r} runs under "
                              f"make -n ($(MAKE) or '+'), which this test does not run")
                continue
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target],
                                 env=env, capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                errors.append(f"make -n {target} exited {out.returncode}: {out.stderr.strip()}")
                continue
            docker = []
            for line in logical_lines(out.stdout.splitlines()):
                if ECHO_LITERAL.fullmatch(line.strip()):
                    continue
                command = plain_command(line)
                if command is None or command[1][0] not in COMMANDS:
                    errors.append(f"make -n {target}: {line.strip()!r} is not a line this test "
                                  f"reads (echo \"TEXT\", or echo, mkdir or docker with plain "
                                  f"words)")
                elif command[1][0] == "docker":
                    docker.append(command[1])
            if len(docker) != 1:
                if not any(e.startswith(f"make -n {target}:") for e in errors):
                    errors.append(f"make -n {target}: {len(docker)} docker commands, expected "
                                  f"one docker compose run")
                continue
            text = " ".join(docker[0])
            try:
                call = compose_call(docker[0])
            except ValueError as e:
                errors.append(f"make -n {target}: {text}: {e}: not in the tables of docker "
                              f"compose's options (makefile_common.py)")
                continue
            if call is None or call[0] != "run":
                errors.append(f"make -n {target}: {text} is not docker compose run (without "
                              f"options of docker itself before compose)")
                continue
            command, found, operands = call
            if operands[:1] != [service]:
                errors.append(f"make -n {target}: {text} runs {operands[:1]}, expected the "
                              f"{service} service")
            if found.get("--build") is not True:
                errors.append(f"make -n {target}: docker compose run without --build ({text}): "
                              f"an existing allencahn-cuda:{service}, built from another "
                              f"tree, is run")
        if log.exists() and log.read_text().strip():
            errors.append("make -n ran docker: " + "; ".join(
                " ".join(json.loads(l)) for l in log.read_text().splitlines()))

    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   {' and '.join(f'make {t}' for t in TARGETS)} build their images first")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
