#!/usr/bin/env python3
"""The dev image is built for the host user, so the bind-mounted checkout is writable.

docker/Dockerfile.dev creates `developer` with the IDs of its build arguments
USER_UID and USER_GID (default 1000). Without them the container user was always
1000, and a checkout bind-mounted at /workspace was not writable, or not readable,
for any other host user.

Checked:
  * docker-compose.yml is all compose reads in the checkout (no compose.yaml, compose.yml,
    override file or .env beside it, no include; makefile_common.compose_file_errors), and its
    dev service builds docker/Dockerfile.dev from this checkout passing USER_UID and USER_GID
    from HOST_UID and HOST_GID (1000 when unset): build is exactly {context: ., dockerfile:
    docker/Dockerfile.dev, args: {USER_UID: ${HOST_UID:-1000}, USER_GID: ${HOST_GID:-1000}}},
    and its other keys are ones this test reads or that change neither the image nor the
    container's user (image, container_name, profiles, stdin_open, tty, volumes, environment,
    deploy; not user, for one, which would run it as another);
  * the dev service mounts its build directory (/workspace/build) and ccache
    (/workspace/.ccache) from named volumes, and every named volume it mounts (in the short
    or the long syntax) has a name with ${COMPOSE_PROJECT_NAME}, ${HOST_UID:-1000} and
    ${HOST_GID:-1000} in it: Docker seeds an empty named volume from the image, owned by
    `developer`, but a volume that holds files keeps their owner, so one a dev image built
    for other IDs filled (1000:1000 by default) is not writable for this one. An anonymous
    volume would be no better: `docker compose up` keeps it when it recreates the container;
  * make runs docker-shell, docker-compose-up and docker-build-dev with a recording shell as
    their SHELL (makefile_common.read_recipes): every line make hands the shell must be
    `echo "TEXT"` (no " $ ` or \\ in TEXT) or a simple command of plain words
    (makefile_common.plain_command) run by echo, mkdir or docker, in the environment make was
    given (nothing the Makefile exports reaches docker), make's SHELL and .SHELLFLAGS /bin/sh
    and -c, and no docker runs. A line of any other form (a fallback after `||`, a branch, a
    subshell, a substitution, `sh -c`, `env`, $(MAKE)) fails: the shell's paths through it are
    not read here. Each target has exactly one docker line:
    - docker-shell's `docker compose [--profile P] run [--rm] [--build[=BOOL]] dev ...` and
      docker-compose-up's `docker compose [--profile P] up [-d] [--build[=BOOL]] [SERVICE...]`,
      which must start dev (named, or of a --profile given), each assigning on its line exactly
      HOST_UID and HOST_GID, the caller's IDs (as `make docker-run` does), with --build true
      (the last one counting, as compose reads --build=BOOL): compose builds a service's image
      only when it is missing, so without it an allencahn-cuda:dev built before for other IDs
      (1000 by default) is reused;
    - docker-build-dev's `docker build [-f FILE] [--build-arg NAME=VALUE]... [-t TAG]... PATH`,
      assigning nothing, with the dev service's dockerfile, its context as PATH, its image among
      the tags and --build-arg USER_UID and USER_GID the caller's IDs (the last of each counting;
      a --build-arg USER_UID without a value takes it from docker's environment).
    The options are read as compose and docker build read them (makefile_common.compose_call,
    docker_build_call: up's and docker build's also after their operands), and any other option
    fails, such as --dry-run (nothing is built), -f (another compose file), -u (another user),
    -v, --no-start, --target or --output; so does docker with options of its own.

Needs GNU make, python3 and PyYAML; no docker.

Usage: makefile_dev_user.py REPO_ROOT   (exit 0 = pass)
"""

import os
import posixpath
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import compose_call, compose_file_errors, docker_build_call, read_recipes

try:
    import yaml  # python3-yaml
except ImportError:
    print("FAIL: PyYAML (python3-yaml) is required to read docker-compose.yml")
    sys.exit(1)

SCRUB = ("HOST_UID", "HOST_GID", "IMAGE_NAME", "DOCKER_REGISTRY", "DOCKER_BUILD_ARGS")
# The dev service's keys this test reads, and those that change neither its image nor its
# container's user.
DEV_KEYS = {"build", "image", "container_name", "profiles", "stdin_open", "tty", "volumes",
            "environment", "deploy"}
DEV_BUILD = {"context": ".", "dockerfile": "docker/Dockerfile.dev",
             "args": {"USER_UID": "${HOST_UID:-1000}", "USER_GID": "${HOST_GID:-1000}"}}


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    uid, gid = str(os.getuid()), str(os.getgid())
    ids = {"HOST_UID": uid, "HOST_GID": gid}

    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    errors = compose_file_errors(root, compose, {"dev": DEV_KEYS})
    services = compose.get("services") or {}
    dev = services.get("dev") or {}
    if dev.get("build") != DEV_BUILD:
        errors.append(f"docker-compose.yml's dev service builds {dev.get('build')!r}, expected "
                      f"{DEV_BUILD}")
    named = compose.get("volumes") or {}
    # Its mounts as (named volume or None, target): SOURCE:TARGET[:MODE] or TARGET (short
    # syntax), or a mapping with type, source and target (long syntax); a source that is a
    # top-level volume is a named volume, a bind's or an anonymous volume's is not. Targets
    # are compared as compose resolves them, cleaned (`/workspace/build/` is
    # /workspace/build, and the later of two mounts there replaces the earlier).
    mounts = []
    for v in dev.get("volumes") or []:
        if isinstance(v, str):
            parts = v.split(":")
            source, target = (parts[0], parts[1]) if len(parts) > 1 else (None, parts[0])
        else:
            source = v.get("source") if v.get("type") == "volume" else None
            target = v.get("target")
        mounts.append((source if source in named else None,
                       posixpath.normpath(target) if target else target))
    for target in ("/workspace/build", "/workspace/.ccache"):
        sources = [s for s, t in mounts if t == target]
        if len(sources) != 1 or sources[0] is None:
            how = ("not at all" if not sources else "more than once" if len(sources) > 1
                   else "from no named volume (a bind, an anonymous volume or a tmpfs)")
            errors.append(f"docker-compose.yml's dev service mounts {target} {how}; expected "
                          f"a named volume per project and per HOST_UID and HOST_GID")
    mounted = [s for s, _ in mounts if s is not None]
    for volume in mounted:
        name = (named[volume] or {}).get("name", "")
        missing = [v for v in ("${COMPOSE_PROJECT_NAME}", "${HOST_UID:-1000}", "${HOST_GID:-1000}")
                   if v not in name]
        if missing:
            errors.append(f"docker-compose.yml's volume {volume} is named {name!r}, without "
                          f"{', '.join(missing)}: a dev image built for other IDs mounts the "
                          f"volume another container user filled")

    commands, recipe_errors = read_recipes(
        make, root, ["docker-shell", "docker-compose-up", "docker-build-dev"], SCRUB)
    errors += recipe_errors
    for target, docker in commands.items():
        if len(docker) != 1:
            errors.append(f"make {target}: {len(docker)} docker commands, expected one")
            continue
        assignments, w = docker[0]
        text = " ".join(w)
        if target == "docker-build-dev":
            if assignments:
                errors.append(f"make {target}: {text} runs with {', '.join(sorted(assignments))} "
                              f"assigned: docker's environment, which this test does not read")
            try:
                call = docker_build_call(w)
            except ValueError as e:
                errors.append(f"make {target}: {text}: option {e}")
                continue
            if call is None:
                errors.append(f"make {target}: {text} is not docker build (without options of "
                              f"docker itself before build)")
                continue
            found, operands = call
            build = dev.get("build") or {}
            if operands != [build.get("context")] or found.get("--file") != build.get("dockerfile"):
                errors.append(f"make {target}: {text} builds {found.get('--file')} in "
                              f"{operands}, expected the dev service's {build.get('dockerfile')} "
                              f"in [{build.get('context')!r}]")
            if dev.get("image") not in found.get("--tag", []):
                errors.append(f"make {target}: {text} tags {found.get('--tag', [])}, not the dev "
                              f"service's image {dev.get('image')}")
            given, from_env = {}, []
            for arg in found.get("--build-arg", []):
                name, eq, value = arg.partition("=")
                if eq:
                    given[name] = value
                elif name in ("USER_UID", "USER_GID"):
                    from_env.append(name)
            if from_env:
                errors.append(f"make {target}: {text}: --build-arg {' and '.join(from_env)} "
                              f"without a value, which docker build takes from its environment")
            if given.get("USER_UID") != uid or given.get("USER_GID") != gid:
                errors.append(f"make {target}: build args {given}, expected USER_UID={uid} "
                              f"USER_GID={gid}")
            continue
        sub = "run" if target == "docker-shell" else "up"
        try:
            call = compose_call(w)
        except ValueError as e:
            errors.append(f"make {target}: {text}: option {e}")
            continue
        if call is None or call[0] != sub:
            errors.append(f"make {target}: {text} is not docker compose {sub} (without options "
                          f"of docker itself before compose)")
            continue
        _, top, found, operands = call
        if sub == "run" and operands[:1] != ["dev"]:
            errors.append(f"make {target}: {text} runs {operands[:1]}, expected the dev service")
        if sub == "up":
            # up starts the services named, else those of the profiles given and those of none.
            profiles = set(top.get("--profile", []))
            started = operands or [s for s, d in services.items()
                                   if not (d or {}).get("profiles")
                                   or profiles & set(d["profiles"])]
            if "dev" not in started:
                errors.append(f"make {target}: {text} starts {sorted(started)}, not the dev "
                              f"service")
        if assignments != ids:
            errors.append(f"make {target}: docker compose runs with {assignments} assigned, "
                          f"expected exactly HOST_UID={uid} HOST_GID={gid}")
        if found.get("--build") is not True:
            errors.append(f"make {target}: docker compose {sub} without --build ({text}): an "
                          f"existing image built for other IDs is reused")

    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   the dev image is built for {uid}:{gid} (compose and the Makefile)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
