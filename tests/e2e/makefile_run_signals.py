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
sure that `make -n` of it runs nothing but sub-makes and the two stubs below.

`make -n` prints recipe lines without running them, except a line that
references $(MAKE): it runs that one. The resume targets' recipe line holds
both the resume run and the cold-start $(MAKE), so it runs, once with a
resume-helper stub that prints a resume config and exits 0 (the resume run,
which a docker stub first on PATH records with its exact arguments) and once
with the stub exiting 1 (the cold start, whose sub-make inherits -n and
prints its run). Printed lines are split into words with shlex, as the shell
splits these unquoted words. Nothing is written to the checkout.

Needs GNU make and python3 on PATH.

Usage: makefile_run_signals.py REPO_ROOT   (exit 0 = pass)
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TARGETS = ["cuda-run", "cuda-run-small", "cuda-run-default", "cuda-run-vtk", "cuda-run-dendrite",
           "cuda-run-dendrite-long", "cuda-run-dendrite-large", "cuda-resume-dendrite",
           "cuda-resume-dendrite-large", "cuda-dendrite-demo", "cuda-all"]
RESUME_TARGETS = ("cuda-resume-dendrite", "cuda-resume-dendrite-large")

HELPER_STUB = """\
import os, sys
print("config/stub_resume.json")
sys.exit(int(os.environ["AC_TEST_HELPER_EXIT"]))
"""

DOCKER_STUB = """\
#!/usr/bin/env python3
import json, os, sys
with open(os.environ["AC_TEST_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
"""

# Inherited make state (a parent make's flags and jobserver, as under `make test`)
# and environment values that would replace the Makefile's `?=` defaults of the
# variables these recipes use.
SCRUB = ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKELEVEL", "MAKEFILES", "CONFIG",
         "CUDA_BUILD_DIR", "NATIVE_BIN", "CUDA_DEV_IMAGE", "CUDA_DEV_DOCKERFILE",
         "LONG_CONFIG", "LARGE_CONFIG", "OUT_DIR", "CHECKPOINT_DIR", "RESUME_HELPER")

SEPARATORS = (";", ";;", "&", "&&", "|", "||")


def words(line: str) -> list[str]:
    """The words of a shell line, with the separators ; & | as words of their own."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    return list(lexer)


def logical_lines(lines: list[str]) -> list[str]:
    """Recipe lines with their backslash-newline continuations joined."""
    joined, acc = [], ""
    for line in lines:
        if line.endswith("\\"):
            acc += line[:-1] + " "
        else:
            joined.append(acc + line)
            acc = ""
    if acc:
        joined.append(acc)
    return joined


def data_base(make: str, root: Path, env: dict) -> tuple[dict, dict]:
    """Prerequisites and recipe lines of every rule in the Makefile's data base."""
    out = subprocess.run([make, "-pq", "--no-print-directory", "-C", str(root),
                          "__ac_no_such_target__"],
                         env=env, capture_output=True, text=True, timeout=60).stdout
    section = out[out.index("\n# Files\n"):out.index("\n# files hash-table stats:")]
    prereqs, recipes, current = {}, {}, None
    for line in section.splitlines():
        if line.startswith("\t"):
            if current is not None:
                recipes[current].append(line[1:])
            continue
        if not line.strip():
            current = None
            continue
        if line.startswith("#"):
            continue
        rule = re.match(r"^([^:\s][^:]*?):(?!:)(.*)$", line)
        if rule is None:
            continue
        rest = rule.group(2)
        # `target: VAR = value` is a target-specific variable, not a rule.
        if re.match(r"^\s*(?:(?:override|export|private)\s+)*[^\s=:]+\s*(?:::=|:=|\?=|\+=|!=|=)",
                    rest):
            continue
        current = rule.group(1).strip()
        prereqs[current] = [w for w in rest.split() if w != "|"]
        recipes[current] = []
    return prereqs, recipes


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


def docker_runs(w: list[str], binary: str) -> list[list[str]]:
    """The `docker run` commands among the words w that mention the binary."""
    runs = []
    for i in range(len(w) - 1):
        if w[i] == "docker" and w[i + 1] == "run":
            end = next((j for j in range(i, len(w)) if w[j] in SEPARATORS), len(w))
            if any(binary in word for word in w[i:end]):
                runs.append(w[i:end])
    return runs


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
        helper = tmp / "resume_helper_stub.py"
        helper.write_text(HELPER_STUB)
        bindir = tmp / "bin"
        bindir.mkdir()
        (bindir / "docker").write_text(DOCKER_STUB)
        (bindir / "docker").chmod(0o755)
        docker_log = tmp / "docker.log"
        env = {k: v for k, v in os.environ.items() if k not in SCRUB}
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
        env["AC_TEST_DOCKER_LOG"] = str(docker_log)

        values = subprocess.run(
            [make, "-s", "--no-print-directory", "-C", str(root),
             "--eval", 'ac-print-%: ; @printf "%s\\n" "$($*)"',
             "ac-print-NATIVE_BIN", "ac-print-CUDA_DEV_IMAGE", "ac-print-CUDA_DEV_DOCKERFILE"],
            env=env, capture_output=True, text=True, timeout=60).stdout.splitlines()
        binary, image, dockerfile = "./" + values[0], values[1], values[2]

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
            branches = (("0", "resume"), ("1", "cold start")) if target in RESUME_TARGETS \
                else ((None, ""),)
            for helper_exit, branch in branches:
                docker_log.write_text("")
                run_env = dict(env)
                if helper_exit is not None:
                    run_env["AC_TEST_HELPER_EXIT"] = helper_exit
                out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target,
                                      f"RESUME_HELPER={helper}"],
                                     env=run_env, capture_output=True, text=True, timeout=120)
                printed = [docker_runs(words(l), binary)
                           for l in logical_lines(out.stdout.splitlines()) if binary in l]
                executed = [docker_runs(["docker"] + json.loads(l), binary)
                            for l in docker_log.read_text().splitlines()]
                runs = [r for found_runs in printed + executed for r in found_runs]
                problems = [p for p in (check(r, binary, image, user) for r in runs) if p]
                if out.returncode != 0:
                    problems.append(f"make -n exited {out.returncode}: {out.stderr.strip()}")
                if not runs:
                    problems.append("no docker run of the binary")
                if helper_exit == "0" and not any(executed):
                    problems.append("the resume run was not executed under make -n")
                name = f"{target} ({branch})" if branch else target
                failures += bool(problems)
                print(f"{'FAIL' if problems else 'ok  '} {name}: {len(runs)} run(s) of {binary}"
                      + ("".join(f"\n     {p}" for p in problems)))
    print(f"{len(TARGETS)} targets, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
