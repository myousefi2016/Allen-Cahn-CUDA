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
    HOST_UID and HOST_GID set to the caller's IDs (as `make docker-run` does);
  * `make -n docker-build-dev` passes --build-arg USER_UID and USER_GID with them.
`make -n` prints these recipe lines without running them; a recording docker stub
first on PATH would show a run.

Needs GNU make, python3 and PyYAML; no docker.

Usage: makefile_dev_user.py REPO_ROOT   (exit 0 = pass)
"""

import json
import os
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

    dev = yaml.safe_load((root / "docker-compose.yml").read_text())["services"]["dev"]
    args = (dev.get("build") or {}).get("args") or {}
    expected = {"USER_UID": "${HOST_UID:-1000}", "USER_GID": "${HOST_GID:-1000}"}
    if {k: args.get(k) for k in expected} != expected:
        errors.append(f"docker-compose.yml's dev service passes build args {args}, "
                      f"expected {expected}")

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
