#!/usr/bin/env python3
"""The Makefile's resume targets must pass the resume helper only the config.

`make cuda-resume-dendrite` and `make cuda-resume-dendrite-large` run
scripts/resume_from_latest_checkpoint.py on LONG_CONFIG / LARGE_CONFIG. Given
only the config, the helper searches the config's own checkpoint_dir
(tests/test_resume_checkpoint.py, 9a); a second argument replaces that
directory. Passing $(CHECKPOINT_DIR) (checkpoints) would make it search
checkpoints/ instead of checkpoints/run_dendrite_long or
checkpoints/run_dendrite_large, where these runs write their checkpoints;
the helper does not look into subdirectories.

For each target, runs `make -n TARGET RESUME_HELPER=<stub>` and requires
  * the stub to be called once, with exactly one argument: the config
    (config/run_dendrite_long.json, config/run_dendrite_large.json);
  * no call of docker.

`make -n` prints recipe lines without running them, except a line that
references $(MAKE): it runs that one. The helper call shares its recipe line
with the cold-start `$(MAKE) cuda-run-dendrite-long|large`, so it runs. The
stub records its arguments and exits 1 ("no usable checkpoint"), so the
recipe takes the cold-start branch, whose sub-make inherits -n and only
prints. A docker stub first on PATH records any call and fails, so a docker
call is reported instead of run. Parsing the Makefile runs `id -u` and
`id -g` (HOST_UID, HOST_GID); its other $(shell) calls, nproc and
git describe, are in deferred variables these recipes do not use. Nothing is
written to the checkout.

Needs GNU make and python3 on PATH (the recipe runs `python3 $(RESUME_HELPER)`);
the nvidia/cuda devel images of the CI jobs and Dockerfiles ship make.

Usage: makefile_resume_call.py REPO_ROOT   (exit 0 = pass)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Target -> the config it must pass (the Makefile's LONG_CONFIG / LARGE_CONFIG).
TARGETS = {
    "cuda-resume-dendrite": "config/run_dendrite_long.json",
    "cuda-resume-dendrite-large": "config/run_dendrite_large.json",
}

HELPER_STUB = """\
import json, os, sys
with open(os.environ["AC_TEST_HELPER_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(1)
"""

DOCKER_STUB = """\
#!/bin/sh
echo "docker $*" >> "$AC_TEST_DOCKER_LOG"
exit 1
"""

# Inherited make state (a parent make's flags and jobserver, as under
# `make test`) and environment values that would replace the Makefile's
# `?=` defaults of the variables this test checks.
SCRUB = ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKELEVEL", "MAKEFILES",
         "LONG_CONFIG", "LARGE_CONFIG", "CHECKPOINT_DIR", "RESUME_HELPER")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = tmp / "resume_helper_stub.py"
        stub.write_text(HELPER_STUB)
        bindir = tmp / "bin"
        bindir.mkdir()
        (bindir / "docker").write_text(DOCKER_STUB)
        (bindir / "docker").chmod(0o755)
        helper_log = tmp / "helper.log"
        docker_log = tmp / "docker.log"
        env = {k: v for k, v in os.environ.items() if k not in SCRUB}
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
        env["AC_TEST_HELPER_LOG"] = str(helper_log)
        env["AC_TEST_DOCKER_LOG"] = str(docker_log)

        for target, config in TARGETS.items():
            helper_log.unlink(missing_ok=True)
            if not (root / config).is_file():
                print(f"FAIL {target}: {config} does not exist")
                failures += 1
                continue
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), target,
                                  f"RESUME_HELPER={stub}"],
                                 env=env, capture_output=True, text=True, timeout=30)
            calls = ([json.loads(line) for line in helper_log.read_text().splitlines()]
                     if helper_log.exists() else [])
            errors = []
            if out.returncode != 0:
                errors.append(f"make -n exited {out.returncode}")
            if not calls:
                errors.append("the resume helper was not run (it must stay on the recipe "
                              "line with the cold-start $(MAKE))")
            elif calls != [[config]]:
                errors.append(f"the resume helper was called with {calls}, expected [['{config}']]")
            if errors:
                failures += 1
                print(f"FAIL {target}: " + "; ".join(errors))
                if out.returncode != 0 or not calls:
                    print("     " + (out.stdout + out.stderr).strip().replace("\n", "\n     "))
            else:
                print(f"ok   {target}: python3 $(RESUME_HELPER) {config}")

        if docker_log.exists():
            failures += 1
            print("FAIL docker was called:\n     "
                  + docker_log.read_text().strip().replace("\n", "\n     "))

    print(f"{len(TARGETS)} resume targets, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
