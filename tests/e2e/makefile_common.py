"""What the Makefile tests share: make's data base and values, shell words, stubs, the
plain recipe lines a test can read exactly, and docker compose's options.

Imported by the makefile_*.py tests (from their own directory, the first entry of
sys.path when they run as scripts).
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

# A plain word: no quoting, expansion, globbing or operator character of the POSIX shell,
# and '#' (which starts a comment at the start of a word) only after its first character, so
# the shell passes it to the command as it is (plain_command).
PLAIN_WORD = re.compile(r"[A-Za-z0-9_./:=@%+,-][A-Za-z0-9_./:=@%+,#-]*")
# The leading words of a simple command that the shell takes as assignments.
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# `echo "TEXT"` with a TEXT the shell passes unchanged (none of " $ ` \).
ECHO_LITERAL = re.compile(r'echo "[^"$`\\]*"')

# The options of docker compose and of its run and up commands (their --help, Compose
# v5.6.1): those that take a value (the next word, unless written --opt=value or, for a
# short option, attached), and the flags (booleans: --flag or --flag=BOOL).
COMPOSE_OPTIONS = {
    "value": {"--ansi", "--env-file", "-f", "--file", "--parallel", "--profile", "--progress",
              "--project-directory", "-p", "--project-name"},
    "flag": {"--all-resources", "--compatibility", "--dry-run"},
}
COMPOSE_COMMAND_OPTIONS = {
    "run": {
        "value": {"--cap-add", "--cap-drop", "--entrypoint", "-e", "--env", "--env-from-file",
                  "-l", "--label", "--name", "-p", "--publish", "--pull", "-u", "--user",
                  "-v", "--volume", "-w", "--workdir"},
        "flag": {"--build", "-d", "--detach", "--dry-run", "-i", "--interactive", "--no-deps",
                 "-T", "--no-tty", "-q", "--quiet", "--quiet-build", "--quiet-pull",
                 "--remove-orphans", "--rm", "-P", "--service-ports", "--use-aliases"},
    },
    "up": {
        "value": {"--attach", "--exit-code-from", "--no-attach", "--pull", "--scale", "-t",
                  "--timeout", "--wait-timeout"},
        "flag": {"--abort-on-container-exit", "--abort-on-container-failure",
                 "--always-recreate-deps", "--attach-dependencies", "--build", "-d",
                 "--detach", "--dry-run", "--force-recreate", "--menu", "--no-build",
                 "--no-color", "--no-deps", "--no-log-prefix", "--no-recreate", "--no-start",
                 "--quiet-build", "--quiet-pull", "--remove-orphans", "-V",
                 "--renew-anon-volumes", "--timestamps", "--wait", "-w", "--watch", "-y",
                 "--yes"},
    },
}
# The options of `docker build` (docker build --help, docker 29.1.3, whose build is buildx's).
DOCKER_BUILD_OPTIONS = {
    "value": {"--add-host", "--allow", "--annotation", "--attest", "--build-arg",
              "--build-context", "--builder", "--cache-from", "--cache-to", "--call",
              "--cgroup-parent", "-f", "--file", "--iidfile", "--label", "--metadata-file",
              "--network", "--no-cache-filter", "-o", "--output", "--platform", "--policy",
              "--progress", "--provenance", "--resource", "--sbom", "--secret", "--shm-size",
              "--ssh", "-t", "--tag", "--target", "--ulimit"},
    "flag": {"--check", "-D", "--debug", "--load", "--no-cache", "--pull", "--push", "-q",
             "--quiet"},
}
# The values Go's strconv.ParseBool takes, as compose reads --flag=BOOL.
BOOL_VALUES = {"1": True, "t": True, "T": True, "TRUE": True, "true": True, "True": True,
               "0": False, "f": False, "F": False, "FALSE": False, "false": False,
               "False": False}


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


def plain_command(line: str):
    """(assignments, words) of a logical recipe line that is one simple command of plain
    words (PLAIN_WORD): the leading NAME=VALUE words as a dict, then the command and its
    arguments. None for any other line: one with an operator, a quote, an expansion, a
    glob or a comment cannot be read without the shell."""
    w = line.split()
    if not w or not all(PLAIN_WORD.fullmatch(x) for x in w):
        return None
    assignments = {}
    while w and ASSIGNMENT.match(w[0]):
        name, value = w.pop(0).split("=", 1)
        assignments[name] = value
    return (assignments, w) if w else None


def options(w: list[str], table: dict, repeated: frozenset = frozenset()) -> tuple[dict, int]:
    """The options at the start of w as {name: value}, a flag's value its boolean (the last
    occurrence counts), a value option's its value (the last, or the list of all for the
    names in `repeated`), and the index of the first word after them. A cluster -Te gives -T
    and -e. Raises ValueError for an option table does not know or a flag's value that is
    not a boolean."""
    found, i = {}, 0

    def value_of(name, value):
        if name in repeated:
            found.setdefault(name, []).append(value)
        else:
            found[name] = value
    while i < len(w) and w[i].startswith("-") and w[i] != "-":
        word = w[i]
        i += 1
        if word == "--":
            break
        if word.startswith("--"):
            name, eq, value = word.partition("=")
            if name in table["flag"]:
                if eq and value not in BOOL_VALUES:
                    raise ValueError(f"{name}={value} (not a boolean)")
                found[name] = BOOL_VALUES[value] if eq else True
            elif name in table["value"]:
                if not eq:
                    if i >= len(w):
                        raise ValueError(f"{name} without its value")
                    value = w[i]
                    i += 1
                value_of(name, value)
            else:
                raise ValueError(name)
            continue
        for k, letter in enumerate(word[1:]):
            name = "-" + letter
            if name in table["flag"]:
                found[name] = True
            elif name in table["value"]:
                if k < len(word) - 2:
                    value_of(name, word[k + 2:])  # the value attached
                else:
                    if i >= len(w):
                        raise ValueError(f"{name} without its value")
                    value_of(name, w[i])
                    i += 1
                break
            else:
                raise ValueError(name)
    return found, i


def compose_call(words: list[str]):
    """(command, its options, its operands) of `docker compose [OPTIONS] COMMAND ...` with
    COMMAND one COMPOSE_COMMAND_OPTIONS lists, else None (also for docker with options of
    its own before compose). Raises ValueError for an option the tables do not know."""
    if words[:2] != ["docker", "compose"]:
        return None
    _, i = options(words[2:], COMPOSE_OPTIONS)
    rest = words[2 + i:]
    if not rest or rest[0] not in COMPOSE_COMMAND_OPTIONS:
        return None
    found, j = options(rest[1:], COMPOSE_COMMAND_OPTIONS[rest[0]])
    return rest[0], found, rest[1 + j:]
