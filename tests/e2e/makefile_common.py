"""What the Makefile tests share: make's data base and values, shell words, stubs.

Imported by makefile_run_signals.py and makefile_resume_call.py (from their
own directory, the first entry of sys.path when they run as scripts).
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Shell separators, as words of their own (words()).
SEPARATORS = (";", ";;", "&", "&&", "|", "||")

# The body of an executable that appends its arguments, as one JSON list per call, to
# the file named by AC_TEST_STUB_LOG (with its own name first) and exits 0. stub_bin()
# gives it this interpreter's absolute path, so that a stub named python3 does not run
# itself.
RECORDING_STUB = """\
import json, os, sys
with open(os.environ["AC_TEST_STUB_LOG"], "a") as log:
    log.write(json.dumps([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\\n")
"""

# Inherited make state (a parent make's flags and jobserver, as under `make test`).
MAKE_STATE = ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKELEVEL", "MAKEFILES")


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
    """Prerequisites and recipe lines of every rule in the Makefile's data base.

    `make -pq` of a target that does not exist prints the data base and runs no
    recipe. Target-specific variable lines (`target: VAR = value`) are skipped.
    """
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
        if re.match(r"^\s*(?:(?:override|export|private)\s+)*[^\s=:]+\s*(?:::=|:=|\?=|\+=|!=|=)",
                    rest):
            continue
        current = rule.group(1).strip()
        prereqs[current] = [w for w in rest.split() if w != "|"]
        recipes[current] = []
    return prereqs, recipes


def runs_under_dry_run(line: str) -> bool:
    """Whether make runs this logical recipe line under -n, -t and -q: a line that
    references $(MAKE) or ${MAKE}, or starts with '+' (after any of @ and -)."""
    return ("$(MAKE)" in line or "${MAKE}" in line
            or line.lstrip().lstrip("@-").startswith("+"))


def make_values(make: str, root: Path, env: dict, names: list[str]) -> dict:
    """The values of the Makefile's variables `names`, as make expands them."""
    out = subprocess.run(
        [make, "-s", "--no-print-directory", "-C", str(root),
         "--eval", 'ac-print-%: ; @printf "%s\\n" "$($*)"'] + [f"ac-print-{n}" for n in names],
        env=env, capture_output=True, text=True, timeout=60, check=True).stdout.splitlines()
    return dict(zip(names, out))


def docker_runs(w: list[str], binary: str) -> list[list[str]]:
    """The `docker run` commands among the words w that mention the binary."""
    runs = []
    for i in range(len(w) - 1):
        if w[i] == "docker" and w[i + 1] == "run":
            end = next((j for j in range(i, len(w)) if w[j] in SEPARATORS), len(w))
            if any(binary in word for word in w[i:end]):
                runs.append(w[i:end])
    return runs


def stub_bin(directory: Path, names: list[str]) -> Path:
    """A directory with a recording stub for each of the commands `names`."""
    bindir = directory / "bin"
    bindir.mkdir()
    for name in names:
        (bindir / name).write_text(f"#!{sys.executable}\n" + RECORDING_STUB)
        (bindir / name).chmod(0o755)
    return bindir


def scrubbed_env(scrub: tuple, bindir: Path, log: Path) -> dict:
    """os.environ without make's inherited state and the variables `scrub`, with the
    stubs of bindir first on PATH, recording into log."""
    env = {k: v for k, v in os.environ.items() if k not in MAKE_STATE + scrub}
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["AC_TEST_STUB_LOG"] = str(log)
    return env
