#!/usr/bin/env python3
"""`make cuda-visualize` must render a run into viz/<name of the run>.

VIZ_OUT defaults to $(VIZ_ROOT)/<last path component of VIZ_IN>. A trailing
slash, a repeated slash or a '.' component in VIZ_IN must not change that
name: with notdir alone, out/run_vtk/ gave viz/, so the frames and
dendrite.mp4 of different runs landed together in viz/.

For each spelling of VIZ_IN, runs `make -n cuda-visualize VIZ_IN=...` and
requires the one `mkdir -p` and the one `--output-dir` of the recipe to name
the expected directory, and no call of docker. `make -n` prints recipe
lines without running them (neither cuda-visualize nor its prerequisite
cuda-image has a line that references $(MAKE), the one kind `make -n` runs);
a docker stub first on PATH records any call and fails. Parsing the Makefile
runs `id -u` and `id -g`. Nothing is written to the checkout.

Needs GNU make on PATH; the nvidia/cuda devel images of the CI jobs and
Dockerfiles ship it.

Usage: makefile_viz_out.py REPO_ROOT   (exit 0 = pass)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# (overrides, expected render directory)
CASES = [
    ((), "viz/run_dendrite"),  # the default VIZ_IN, out/run_dendrite
    *((("VIZ_IN=" + v,), "viz/run_vtk") for v in (
        "out/run_vtk", "out/run_vtk/", "out/run_vtk//", "out/run_vtk/.",
        "./out/run_vtk", "out//run_vtk")),
    (("VIZ_IN=/data/run42/",), "viz/run42"),
    (("VIZ_IN=out/run_vtk/", "VIZ_ROOT=renders"), "renders/run_vtk"),
    (("VIZ_IN=out/run_vtk/", "VIZ_OUT=viz/custom"), "viz/custom"),
]

DOCKER_STUB = """\
#!/bin/sh
echo "docker $*" >> "$AC_TEST_DOCKER_LOG"
exit 1
"""

# Inherited make state (a parent make's flags, as under `make test`) and
# environment values that would replace the Makefile's `?=` defaults.
SCRUB = ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKELEVEL", "MAKEFILES", "OUT_DIR")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bindir = tmp / "bin"
        bindir.mkdir()
        (bindir / "docker").write_text(DOCKER_STUB)
        (bindir / "docker").chmod(0o755)
        docker_log = tmp / "docker.log"
        env = {k: v for k, v in os.environ.items()
               if k not in SCRUB and not k.startswith("VIZ_")}
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
        env["AC_TEST_DOCKER_LOG"] = str(docker_log)

        for overrides, expected in CASES:
            what = " ".join(overrides) or "(defaults)"
            out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root),
                                  "cuda-visualize", *overrides],
                                 env=env, capture_output=True, text=True, timeout=30)
            mkdirs = re.findall(r"mkdir -p (\S+)", out.stdout)
            outputs = re.findall(r"--output-dir (\S+)", out.stdout)
            if out.returncode != 0 or mkdirs != [expected] or outputs != [expected]:
                failures += 1
                print(f"FAIL {what}: mkdir -p {mkdirs}, --output-dir {outputs}, "
                      f"expected {expected} (make -n exit {out.returncode})")
                if out.returncode != 0:
                    print("     " + out.stderr.strip().replace("\n", "\n     "))
            else:
                print(f"ok   {what}: {expected}")

        if docker_log.exists():
            failures += 1
            print("FAIL docker was called:\n     "
                  + docker_log.read_text().strip().replace("\n", "\n     "))

    print(f"{len(CASES)} cases, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
