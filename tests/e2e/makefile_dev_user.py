#!/usr/bin/env python3
"""The dev image is built for the host user, so the bind-mounted checkout is writable.

docker/Dockerfile.dev creates `developer` with the IDs of its build arguments
USER_UID and USER_GID (default 1000). Without them the container user was always
1000, and a checkout bind-mounted at /workspace was not writable, or not readable,
for any other host user.

Checked:
  * docker-compose.yml's dev service passes USER_UID and USER_GID from HOST_UID and
    HOST_GID (1000 when unset);
  * `make -n docker-shell` and `make -n docker-compose-up` run docker compose with
    HOST_UID and HOST_GID set to the caller's IDs (as `make docker-run` does) and with
    --build: compose builds a service's image only when it is missing, so without it
    an allencahn-cuda:dev built before for other IDs (1000 by default) is reused;
  * `make -n docker-build-dev` passes --build-arg USER_UID and USER_GID with them;
  * the dev service mounts its build directory (/workspace/build) and ccache
    (/workspace/.ccache) from named volumes, and every named volume it mounts (in the short
    or the long syntax) has a name with ${COMPOSE_PROJECT_NAME}, ${HOST_UID:-1000} and
    ${HOST_GID:-1000} in it: Docker seeds an empty named volume from the image, owned by
    `developer`, but a volume that holds files keeps their owner, so one a dev image built
    for other IDs filled (1000:1000 by default) is not writable for this one. An anonymous
    volume would be no better: `docker compose up` keeps it when it recreates the container.
`make -n` prints these recipe lines without running them; a recording docker stub
first on PATH would show a run.

Needs GNU make, python3 and PyYAML; no docker.

Usage: makefile_dev_user.py REPO_ROOT   (exit 0 = pass)
"""

import json
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # importing makefile_common writes nothing to the checkout
from makefile_common import logical_lines, scrubbed_env, stub_bin, words

try:
    import yaml  # python3-yaml
except ImportError:
    print("FAIL: PyYAML (python3-yaml) is required to read docker-compose.yml")
    sys.exit(1)

SCRUB = ("HOST_UID", "HOST_GID", "IMAGE_NAME", "DOCKER_REGISTRY", "DOCKER_BUILD_ARGS")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    uid, gid = str(os.getuid()), str(os.getgid())
    errors = []

    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    dev = compose["services"]["dev"]
    args = (dev.get("build") or {}).get("args") or {}
    expected = {"USER_UID": "${HOST_UID:-1000}", "USER_GID": "${HOST_GID:-1000}"}
    if {k: args.get(k) for k in expected} != expected:
        errors.append(f"docker-compose.yml's dev service passes build args {args}, "
                      f"expected {expected}")
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "calls.log"
        env = scrubbed_env(SCRUB, stub_bin(tmp, ["docker"]), log)
        for target in ("docker-shell", "docker-compose-up", "docker-build-dev"):
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target],
                                 env=env, capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                errors.append(f"make -n {target} exited {out.returncode}: {out.stderr.strip()}")
                continue
            commands = [w for l in logical_lines(out.stdout.splitlines())
                        for w in [words(l)] if "docker" in w]
            if len(commands) != 1:
                errors.append(f"make -n {target}: {len(commands)} docker commands, expected one")
                continue
            w = commands[0]
            if target == "docker-build-dev":
                given = {w[i + 1].split("=", 1)[0]: w[i + 1].split("=", 1)[1]
                         for i in range(len(w) - 1) if w[i] == "--build-arg" and "=" in w[i + 1]}
                if given.get("USER_UID") != uid or given.get("USER_GID") != gid:
                    errors.append(f"make -n {target}: build args {given}, expected USER_UID={uid} "
                                  f"USER_GID={gid}")
            else:
                prefix = w[:w.index("docker")]
                if prefix != [f"HOST_UID={uid}", f"HOST_GID={gid}"]:
                    errors.append(f"make -n {target}: docker compose runs with {prefix}, expected "
                                  f"HOST_UID={uid} HOST_GID={gid}")
                # The options of its subcommand: the words after `run` up to the service,
                # or after `up`.
                sub = "run" if target == "docker-shell" else "up"
                options = []
                if sub in w:
                    for word in w[w.index(sub) + 1:]:
                        if not word.startswith("-"):
                            break
                        options.append(word)
                if "--build" not in options:
                    errors.append(f"make -n {target}: docker compose {sub} without --build "
                                  f"({' '.join(w[w.index('docker'):])}): an existing image "
                                  f"built for other IDs is reused")
        if log.exists() and log.read_text().strip():
            errors.append("make -n ran docker: " + "; ".join(
                " ".join(json.loads(l)) for l in log.read_text().splitlines()))

    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   the dev image is built for {uid}:{gid} (compose and the Makefile)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
