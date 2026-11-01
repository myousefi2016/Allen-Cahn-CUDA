#!/usr/bin/env python3
"""make docker-test and make docker-run run images built from this checkout.

The test and prod images carry the tree they are built from: docker/Dockerfile.test and
docker/Dockerfile.prod copy the sources, tests and configs and compile them in the image.
docker compose builds a service's image only when it is missing, so `docker compose run`
without --build runs an allencahn-cuda:test or :prod built before, from an earlier tree, and
a re-run after an edit reports that tree's results.

Checked, for each target:
  * make runs it with a recording shell as its SHELL (makefile_common.read_recipes), so the
    lines make hands the shell, the recipe's and its prerequisites', are read as the shell
    reads them, with their environment, and none is run. Each must be `echo "TEXT"` (no " $ `
    or \\ in TEXT) or a simple command of plain words (makefile_common.plain_command: no
    quoting, expansion, globbing, comment or operator character, so the shell passes each word
    as it is) run by echo, mkdir or docker. A line of any other form, such as a fallback after
    `||`, a branch, a subshell, a substitution, `sh -c`, `env` or $(MAKE), fails: the shell's
    paths through it are not read here. Every line must run in the environment make was given
    (nothing the Makefile exports reaches docker), make's SHELL and .SHELLFLAGS must be /bin/sh
    and -c and set for no target or pattern, and no docker runs.
  * Exactly one line is docker: `docker compose [--profile P] run [--rm] [--build[=BOOL]]
    SERVICE ...` of the target's service (test, prod) with --build true (the last one counting,
    as compose reads --build=BOOL), assigning nothing on the line but HOST_UID and HOST_GID
    (prod's container user; they build nothing). The options are read as compose reads them
    (makefile_common.compose_call: run's end at its service), and any other option fails, such
    as --dry-run (nothing is built or run), -f, --project-directory and --env-file (another
    compose file), -u or -v; so does docker with options of its own.
  * docker-compose.yml is all compose reads in the checkout (no compose.yaml, compose.yml,
    override file or .env beside it, no include; makefile_common.compose_file_errors), and the
    service is built from this checkout: its build is {context: ., dockerfile:
    docker/Dockerfile.SERVICE}, and its other keys are ones that change neither the image built
    nor the one run (image, container_name, profiles, environment, deploy; prod's user and
    volumes).

Needs GNU make, python3 and PyYAML; no docker.

Usage: makefile_compose_build.py REPO_ROOT   (exit 0 = pass)
"""

import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import compose_call, compose_file_errors, read_recipes

try:
    import yaml  # python3-yaml
except ImportError:
    print("FAIL: PyYAML (python3-yaml) is required to read docker-compose.yml")
    sys.exit(1)

SCRUB = ("HOST_UID", "HOST_GID", "IMAGE_NAME", "DOCKER_REGISTRY", "DOCKER_BUILD_ARGS")
TARGETS = {"docker-test": "test", "docker-run": "prod"}
# The keys of each service this test reads: build, and the keys that change neither the image
# compose builds nor the one it runs.
KEYS = {"test": {"build", "image", "container_name", "profiles", "environment", "deploy"},
        "prod": {"build", "image", "container_name", "profiles", "environment", "deploy", "user",
                 "volumes"}}


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    errors = compose_file_errors(root, compose, KEYS)
    for service in KEYS:
        build = ((compose.get("services") or {}).get(service) or {}).get("build")
        expected = {"context": ".", "dockerfile": f"docker/Dockerfile.{service}"}
        if build != expected:
            errors.append(f"docker-compose.yml's {service} service builds {build!r}, expected "
                          f"{expected} (this checkout, the Dockerfile that copies it)")

    commands, recipe_errors = read_recipes(make, root, list(TARGETS), SCRUB)
    errors += recipe_errors
    for target, service in TARGETS.items():
        if target not in commands:
            continue
        docker = commands[target]
        if len(docker) != 1:
            errors.append(f"make {target}: {len(docker)} docker commands, expected one docker "
                          f"compose run")
            continue
        assignments, w = docker[0]
        text = " ".join(w)
        extra = sorted(set(assignments) - {"HOST_UID", "HOST_GID"})
        if extra:
            errors.append(f"make {target}: {text} runs with {', '.join(extra)} assigned: docker's "
                          f"environment, which this test does not read (COMPOSE_FILE, for one, "
                          f"names another compose file)")
        try:
            call = compose_call(w)
        except ValueError as e:
            errors.append(f"make {target}: {text}: option {e}")
            continue
        if call is None or call[0] != "run":
            errors.append(f"make {target}: {text} is not docker compose run (without options of "
                          f"docker itself before compose)")
            continue
        _, _, found, operands = call
        if operands[:1] != [service]:
            errors.append(f"make {target}: {text} runs {operands[:1]}, expected the {service} "
                          f"service")
        if found.get("--build") is not True:
            errors.append(f"make {target}: docker compose run without --build ({text}): an "
                          f"existing allencahn-cuda:{service}, built from another tree, is run")

    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   {' and '.join(f'make {t}' for t in TARGETS)} build their images first")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
