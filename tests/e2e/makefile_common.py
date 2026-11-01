"""What the Makefile tests share: make's data base and values, shell words, stubs, the
plain recipe lines a test can read exactly, the lines make runs (recorded by a shell that runs
nothing), the options of docker compose and docker build the docker tests read, and the files
compose reads.

Imported by the makefile_*.py tests (from their own directory, the first entry of
sys.path when they run as scripts).
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
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

# The body of a shell for make that runs nothing: it appends its arguments, its environment
# and the device and inode of its standard output, as one JSON object per call, to the file
# named by AC_TEST_SHELL_LOG and exits 0 (read_recipes()). make gives a recipe line its own
# standard output and a $(shell ...) a pipe it reads.
RECORDING_SHELL = """\
import json, os, sys
out = os.fstat(1)
with open(os.environ["AC_TEST_SHELL_LOG"], "a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ),
                          "stdout": [out.st_dev, out.st_ino]}) + "\\n")
"""
# How `make -p` prints a setting of SHELL or .SHELLFLAGS: globally (in its Variables section),
# for a target (TARGET: NAME = VALUE, in its Files section) and for a pattern (a commented
# NAME = VALUE under the pattern, in its Pattern-specific Variable Values section).
SHELL_SETTING = r"(SHELL|\.SHELLFLAGS) (?::::=|::=|:=|\?=|\+=|!=|=) ?(.*)"

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

# The options the docker tests read, by each of their spellings: (name, kind), kind "flag" (a
# boolean), "value" (one value, the last given counting) or "values" (repeatable, all kept).
# From the --help of docker compose and of its run and up commands (Compose v5.6.1: --profile a
# stringArray, --rm, --build and -d/--detach booleans) and of docker build (docker 29.1.3,
# whose build is buildx v0.37.1's: -f/--file a string, --build-arg and -t/--tag stringArrays).
# Any other option of these commands fails a test (options()): what it does to the image
# built or run, or to the container (--dry-run builds and runs nothing, -f and
# --project-directory choose another compose file, -u another user), is not read here.
COMPOSE_READ = {"--profile": ("--profile", "values")}
# Per compose command, its options read and whether options may follow its operands: run
# reads options only before its service (the words after it are the container's command),
# up also after its services (Cobra's default).
COMPOSE_COMMAND_READ = {
    "run": ({"--rm": ("--rm", "flag"), "--build": ("--build", "flag")}, False),
    "up": ({"-d": ("--detach", "flag"), "--detach": ("--detach", "flag"),
            "--build": ("--build", "flag")}, True),
}
# docker build reads options after its context too.
DOCKER_BUILD_READ = {"-f": ("--file", "value"), "--file": ("--file", "value"),
                     "--build-arg": ("--build-arg", "values"), "-t": ("--tag", "values"),
                     "--tag": ("--tag", "values")}
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
    stubs of bindir first on PATH, recording into log. Each stub must be what /bin/sh runs
    by its name: one the shell cannot execute (a temporary directory mounted noexec) is
    skipped for the next command of that name on PATH, the real one, whose call no log would
    show; RuntimeError then."""
    env = {k: v for k, v in os.environ.items() if k not in MAKE_STATE + scrub}
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["AC_TEST_STUB_LOG"] = str(log)
    for stub in sorted(bindir.iterdir()):
        subprocess.run(["/bin/sh", "-c", f"{stub.name} ac-stub-probe"], env=env,
                       capture_output=True, timeout=60)
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        if calls != [json.dumps([stub.name, "ac-stub-probe"])]:
            raise RuntimeError(f"/bin/sh does not run the recording stub {stub} for "
                               f"`{stub.name}` (recorded {calls}): a {stub.name} the test "
                               f"meant to stub would run unrecorded")
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


def shell_settings(make: str, root: Path, env: dict) -> tuple[dict, list[str]]:
    """The Makefile's settings of SHELL and .SHELLFLAGS, read from make's data base (`make -pq`
    of a target that does not exist, which runs no recipe): ({name: value} of the global ones,
    a value as the Makefile wrote it, unexpanded; the settings for a target or a pattern, as
    "TARGET: NAME = VALUE")."""
    out = subprocess.run([make, "-pq", "--no-print-directory", "-C", str(root),
                          "__ac_no_such_target__"],
                         env=env, capture_output=True, text=True, timeout=60).stdout
    marks = ["\n# Variables\n", "\n# Pattern-specific Variable Values\n", "\n# Directories\n",
             "\n# Files\n", "\n# files hash-table stats:"]
    at = [out.index(m) for m in marks]
    found, local, pattern = {}, [], None
    for line in out[at[0]:at[1]].splitlines():
        setting = re.fullmatch(SHELL_SETTING, line)
        if setting:
            found[setting.group(1)] = setting.group(2)
    for line in out[at[1]:at[2]].splitlines():
        if line and not line.startswith("#"):
            pattern = line.rstrip(" :")
        elif re.fullmatch("# " + SHELL_SETTING, line):
            local.append(f"{pattern}: {line[2:]}")
    for line in out[at[3]:at[4]].splitlines():
        if re.fullmatch(r"[^#\s][^:]*: " + SHELL_SETTING, line):
            local.append(line)
    return found, local


def read_recipes(make: str, root: Path, targets: list[str], scrub: tuple) -> tuple[dict, list]:
    """The docker commands `make TARGET` runs for each of targets, read exactly: ({target:
    [(assignments, words)] of its docker commands}, for each target read, and the errors).

    make runs each target with this checkout's Makefile, in os.environ without make's inherited
    state and the variables scrub, with a recording docker stub first on PATH (scrubbed_env)
    and with a recording shell (RECORDING_SHELL) as the target's SHELL, which its prerequisites
    inherit: every line make hands the shell for them is recorded with its environment, and
    none is run. That needs make's SHELL and .SHELLFLAGS to be /bin/sh and -c, as the lines
    are read here, and set for no target or pattern (a prerequisite's own SHELL would run its
    lines). A line is read as the shell reads it, without its backslash-newlines: it must be
    `echo "TEXT"` (ECHO_LITERAL) or a simple command of plain words (plain_command) run by echo,
    mkdir or docker, and its environment the one make was given, apart from make's own
    variables (MAKE_STATE), so that the Makefile exports nothing to docker. Errors also: make
    failing, a line of any other form (an operator, a quote, an expansion, several lines in
    one shell as .ONESHELL gives them), a $(shell ...) make runs while it expands the recipe
    or a variable it exports (the recording shell runs it too, but the line holds its
    output, which is not read here), and docker run by make itself (the stub's log)."""
    errors, commands = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub_log = tmp / "calls.log"
        env = scrubbed_env(scrub, stub_bin(tmp, ["docker"]), stub_log)
        shell = tmp / "recording-shell"
        shell.write_text(f"#!{sys.executable}\n" + RECORDING_SHELL)
        shell.chmod(0o755)
        found, local = shell_settings(make, root, env)
        if found != {"SHELL": "/bin/sh", ".SHELLFLAGS": "-c"}:
            errors.append(f"the Makefile's SHELL and .SHELLFLAGS are {found}, not /bin/sh and -c, "
                          f"which run the lines as this test reads them")
        if local:
            errors.append(f"the Makefile sets {'; '.join(local)}: a line of a target read here "
                          f"could run with that shell, unrecorded")
        if not re.fullmatch(r"[A-Za-z0-9_./+,-]+", str(shell)):
            errors.append(f"the recording shell's path {shell} is not one make takes as SHELL as "
                          f"it is (a temporary directory without spaces, '#' or '$')")
        for n, target in enumerate(targets if not errors else []):
            log = tmp / f"shell.{n}.log"
            run_env = dict(env, AC_TEST_SHELL_LOG=str(log))
            with open(tmp / f"stdout.{n}", "w") as stdout:
                out = subprocess.run([make, "--no-print-directory", "-C", str(root), "--eval",
                                      f"{target}: override SHELL := {shell}", target],
                                     env=run_env, stdout=stdout, stderr=subprocess.PIPE,
                                     text=True, timeout=60)
                own = os.fstat(stdout.fileno())
            if out.returncode != 0:
                errors.append(f"make {target} exited {out.returncode}: {out.stderr.strip()}")
                continue
            calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
            docker, unread, shells, changed = [], [], [], set()
            for call in calls:
                changed |= {k for k in set(run_env) | set(call["env"])
                            if k not in MAKE_STATE and run_env.get(k) != call["env"].get(k)}
                line = call["argv"][-1].replace("\\\n", "") if call["argv"] else ""
                if call["stdout"] != [own.st_dev, own.st_ino]:
                    shells.append(line.strip())
                    continue
                if call["argv"][:-1] != ["-c"] or "\n" in line:
                    unread.append(call["argv"])
                    continue
                if ECHO_LITERAL.fullmatch(line.strip()):
                    continue
                command = plain_command(line)
                if command is None or command[1][0] not in ("echo", "mkdir", "docker"):
                    unread.append(line.strip())
                elif command[1][0] == "docker":
                    docker.append(command)
            for u in unread:
                errors.append(f"make {target}: {u!r} is not a line this test reads (echo \"TEXT\", "
                              f"or echo, mkdir or docker with plain words, one line per shell)")
            for c in shells:
                errors.append(f"make {target}: $(shell {c}) runs while make expands the recipe (a "
                              f"line or a variable it exports): the line holds its output, which "
                              f"this test does not read")
            if changed:
                errors.append(f"make {target} runs its recipe with {', '.join(sorted(changed))} "
                              f"changed from make's environment (the Makefile exports or unexports "
                              f"them): docker's environment, which this test does not read")
            if not unread and not shells and not changed:
                commands[target] = docker
        if stub_log.exists() and stub_log.read_text().strip():
            errors.append("make ran docker: " + "; ".join(
                " ".join(json.loads(l)) for l in stub_log.read_text().splitlines()))
    return commands, errors


def options(w: list[str], table: dict, interspersed: bool) -> tuple[dict, list[str]]:
    """The options and the operands of a command's arguments w, read as Cobra (pflag) reads
    them: {name: value} of the options, by the names of table, and the operands. A flag's
    value is its boolean (--flag or -f true, --flag=BOOL or -f=BOOL as BOOL_VALUES reads it), a
    value option's the value (--opt VALUE, --opt=VALUE, -o VALUE, -oVALUE, -o=VALUE), the last
    counting, or the list of them all for kind "values". Short options cluster (-dT, -dtNAME).
    '-' and a word not starting with '-' are operands: the first ends the options unless
    interspersed, which reads options after operands too. '--' ends the options. Raises
    ValueError for an option not in table (one this test does not read, or none of the
    command's) and a flag's value that is not a boolean."""
    found, operands, i = {}, [], 0

    def give(spelling, value):
        name, kind = table[spelling]
        if kind == "flag":
            if value not in BOOL_VALUES:
                raise ValueError(f"{spelling}={value} (not a boolean)")
            found[name] = BOOL_VALUES[value]
        elif kind == "values":
            found.setdefault(name, []).append(value)
        else:
            found[name] = value

    def known(spelling):
        if spelling not in table:
            raise ValueError(f"{spelling}, which this test does not read (it reads "
                             f"{', '.join(sorted(table))})")
        return table[spelling][1]

    while i < len(w):
        word = w[i]
        i += 1
        if word == "--":
            operands += w[i:]
            break
        if word == "-" or not word.startswith("-"):
            operands.append(word)
            if not interspersed:
                operands += w[i:]
                break
            continue
        if word.startswith("--"):
            spelling, eq, value = word.partition("=")
            kind = known(spelling)
            if not eq:
                if kind == "flag":
                    value = "true"
                elif i < len(w):
                    value, i = w[i], i + 1
                else:
                    raise ValueError(f"{spelling} without its value")
            give(spelling, value)
            continue
        letters = word[1:]
        while letters:
            spelling = "-" + letters[0]
            kind = known(spelling)
            if len(letters) > 2 and letters[1] == "=":
                give(spelling, letters[2:])  # -f=VALUE, also a flag's -d=BOOL
                break
            if kind == "flag":
                give(spelling, "true")
                letters = letters[1:]
                continue
            if len(letters) > 1:
                give(spelling, letters[1:])  # the value attached
                break
            if i >= len(w):
                raise ValueError(f"{spelling} without its value")
            give(spelling, w[i])
            i += 1
            break
    return found, operands


def compose_call(words: list[str]):
    """(command, compose's options, the command's options, its operands) of
    `docker compose [OPTIONS] COMMAND [ARGS...]` with COMMAND one COMPOSE_COMMAND_READ lists,
    read with options(): compose's own options come before its command. None for any other
    command, also docker with options of its own before compose. Raises ValueError as
    options() does."""
    if words[:2] != ["docker", "compose"]:
        return None
    top, rest = options(words[2:], COMPOSE_READ, interspersed=False)
    if not rest or rest[0] not in COMPOSE_COMMAND_READ:
        return None
    table, interspersed = COMPOSE_COMMAND_READ[rest[0]]
    found, operands = options(rest[1:], table, interspersed)
    return rest[0], top, found, operands


def docker_build_call(words: list[str]):
    """(options, operands) of `docker build [OPTIONS] PATH`, read with options() and
    DOCKER_BUILD_READ (docker build reads options after its context too); None for any other
    command, also docker with options of its own before build. Raises ValueError as options()
    does."""
    if words[:2] != ["docker", "build"]:
        return None
    return options(words[2:], DOCKER_BUILD_READ, interspersed=True)


# The files compose reads in the directory it runs in, besides docker-compose.yml: compose.yaml
# and compose.yml (read instead of it), the override files (merged into it) and .env (whose
# COMPOSE_FILE names the files to read and whose variables are interpolated). Compose
# v5.6.1, `docker compose config`.
COMPOSE_SIDE_FILES = ("compose.yaml", "compose.yml", "compose.override.yml",
                      "compose.override.yaml", "docker-compose.override.yml",
                      "docker-compose.override.yaml", ".env")


def compose_file_errors(root: Path, compose: dict, services: dict) -> list[str]:
    """Why compose, loaded from root's docker-compose.yml, is not all compose reads of the
    services `services` ({service: the keys of it the caller reads}): a file of
    COMPOSE_SIDE_FILES in root, a top-level key other than services and volumes (include, for
    one, adds other files' services), a service missing, or a key of one of the services not
    in its list (extends, for one, merges another service into it)."""
    errors = [f"{root.name}/{f} exists: compose reads it besides docker-compose.yml, and this "
              f"test does not" for f in COMPOSE_SIDE_FILES if (root / f).exists()]
    extra = sorted(set(compose) - {"services", "volumes"})
    if extra:
        errors.append(f"docker-compose.yml's top-level {', '.join(extra)}: not read by this test")
    for service, keys in services.items():
        definition = (compose.get("services") or {}).get(service)
        if not isinstance(definition, dict):
            errors.append(f"docker-compose.yml has no {service} service")
            continue
        unread = sorted(set(definition) - set(keys))
        if unread:
            errors.append(f"docker-compose.yml's {service} service has {', '.join(unread)}, "
                          f"which this test does not read")
    return errors
