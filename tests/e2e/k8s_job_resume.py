#!/usr/bin/env python3
"""A Kubernetes retry of the shipped Job continues the interrupted run.

The Job (k8s/base/job.yaml) runs its container with backoffLimit retries and
restartPolicy Never: a pod stopped by SIGTERM (a node drain) writes a
checkpoint of its step and exits 143, and the replacement pod runs the same
container arguments again. Run the binary with exactly those arguments (the
config path replaced by a test copy of the base payload on a small grid and a
local directory), stop it with SIGTERM after step 200, run it again with the
same arguments and stop it at the first step it logs (every 100th). The second
run must restore the first one's checkpoint (log "Restarted from checkpoint:
step=N") and continue past it, and the first one's checkpoint must survive.
Before, the arguments were only the config: the retry started at step 0, and
its shutdown checkpoint at step 101 deleted the first pod's checkpoint_201
("newer than the checkpoint just written"), so the run was recomputed from the
start.

Usage: k8s_job_resume.py BINARY REPO_ROOT WORKDIR
Exit 0 = pass, 1 = fail, 77 = no GPU (skipped; a failure if AC_REQUIRE_GPU=1).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # python3-yaml
except ImportError:
    print("FAIL: PyYAML (python3-yaml) is required to read the k8s manifests")
    sys.exit(1)

SKIP = 77


def gpu_visible() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "GPU " in out.stdout


def run_until(binary: Path, args: list[str], step, cwd: Path) -> tuple[int, str]:
    """Start the binary, SIGTERM it once it logs `step` (None: any step), return
    (status, log)."""
    proc = subprocess.Popen([str(binary), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
    log = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line)
        if re.search(r"Step \d+/" if step is None else rf"Step {step}/", line):
            proc.send_signal(signal.SIGTERM)
            break
    rest, _ = proc.communicate(timeout=300)
    return proc.returncode, "".join(log) + (rest or "")


def main() -> int:
    binary, root, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if not gpu_visible():
        if os.environ.get("AC_REQUIRE_GPU") == "1":
            print("FAIL: no GPU visible to nvidia-smi and AC_REQUIRE_GPU=1")
            return 1
        print("SKIP: no GPU visible to nvidia-smi")
        return SKIP
    job = yaml.safe_load((root / "k8s/base/job.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    args = container.get("args", [])
    config_mount = "/config/simulation.json"
    if config_mount not in args:
        print(f"FAIL: the Job's arguments {args} do not name {config_mount}")
        return 1
    base = yaml.safe_load((root / "k8s/base/configmap.yaml").read_text())
    cfg = json.loads(base["data"]["simulation.json"])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cfg["grid"].update(Nx=32, Ny=32, Nz=32)
    cfg["time"]["max_steps"] = 100000000
    cfg["output"].update(frequency=1000000000, output_dir=str(work / "out"))
    cfg["checkpoint"].update(frequency=0, checkpoint_dir=str(work / "ckpt"))
    cfg_path = work / "simulation.json"
    cfg_path.write_text(json.dumps(cfg))
    run_args = [str(cfg_path) if a == config_mount else a for a in args]

    status, log = run_until(binary, run_args, 200, work)
    m = re.search(r"Shutdown requested at step (\d+)", log)
    if status != 143 or not m:
        print(f"FAIL: first pod: exit {status} (expected 143), no shutdown checkpoint")
        print(log[-2000:])
        return 1
    first = int(m.group(1))
    first_ckpt = work / "ckpt" / f"checkpoint_{first}.acbin"
    print(f"first pod: SIGTERM at step {first}, exit 143, {first_ckpt.name}")

    status, log = run_until(binary, run_args, None, work)
    errors = []
    restarted = re.search(r"Restarted from checkpoint: step=(\d+)", log)
    if not restarted or int(restarted.group(1)) != first:
        errors.append(f"the retry did not restart from step {first} (Job arguments {args})")
    m = re.search(r"Shutdown requested at step (\d+)", log)
    if status != 143 or not m or int(m.group(1)) <= first:
        errors.append(f"the retry ended with exit {status}, not interrupted after step {first}")
    second = int(m.group(1)) if m else None
    if second is not None and not (work / "ckpt" / f"checkpoint_{second}.acbin").is_file():
        errors.append(f"the retry wrote no checkpoint of step {second}")
    if not first_ckpt.is_file():
        errors.append(f"{first_ckpt.name} of the first pod was deleted")
    for e in errors:
        print("FAIL:", e)
    if errors:
        print(log[-2000:])
        return 1
    print(f"retry: restarted from step {first}, SIGTERM at step {second}, exit 143")
    return 0


if __name__ == "__main__":
    sys.exit(main())
