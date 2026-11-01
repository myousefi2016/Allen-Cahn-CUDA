#!/usr/bin/env python3
"""`make k8s-logs` must select the simulation Job's pods in both overlays' namespaces.

A pod carries the labels of its Job's pod template (spec.template.metadata.labels
in k8s/base/job.yaml). kustomize's `labels` entries, in the base and in the
overlays, set no includeTemplates, so they reach only the resources' metadata:
the selector app.kubernetes.io/managed-by=kustomize matched no pod.

`make -n k8s-logs` prints the recipe's kubectl commands without running them (a
recording kubectl stub first on PATH would show a run). Checked:
  * two `kubectl logs` commands, with -n the namespaces of k8s/overlays/dev and
    k8s/overlays/prod, in that order;
  * every pair of each command's -l selector is a label of the pod template in
    k8s/base/job.yaml;
  * no kustomization in k8s/ sets includeTemplates, and no overlay patch (JSON
    6902 or strategic merge) writes the pod template's metadata (which could
    change the pods' labels).

Needs GNU make, python3 and PyYAML; no cluster, kubectl or docker.

Usage: makefile_k8s_logs.py REPO_ROOT   (exit 0 = pass)
"""

import json
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
    print("FAIL: PyYAML (python3-yaml) is required to read the k8s manifests")
    sys.exit(1)

# Environment values that would replace the Makefile's `?=` defaults of the variables
# this recipe uses.
SCRUB = ("KUBECTL", "K8S_NAMESPACE_DEV", "K8S_NAMESPACE_PROD", "K8S_POD_SELECTOR")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    make = shutil.which("make")
    if make is None:
        print("FAIL: GNU make is not on PATH")
        return 1
    errors = []
    job = yaml.safe_load((root / "k8s/base/job.yaml").read_text())
    template_labels = job["spec"]["template"]["metadata"].get("labels") or {}
    overlays = {name: yaml.safe_load((root / f"k8s/overlays/{name}/kustomization.yaml").read_text())
                for name in ("dev", "prod")}
    for path in sorted(root.glob("k8s/**/kustomization.yaml")):
        text = path.read_text()
        if "includeTemplates" in text:
            errors.append(f"{path.relative_to(root)} sets includeTemplates")
    for name, k in overlays.items():
        for patch in k.get("patches") or []:
            body = yaml.safe_load(patch["patch"]) if "patch" in patch else \
                yaml.safe_load((root / f"k8s/overlays/{name}" / patch["path"]).read_text())
            if isinstance(body, list):  # JSON 6902 operations
                for op in body:
                    if str(op.get("path", "")).startswith("/spec/template/metadata"):
                        errors.append(f"overlay {name} patches {op['path']}")
            elif isinstance(body, dict):  # a strategic merge patch
                if ((body.get("spec") or {}).get("template") or {}).get("metadata") is not None:
                    errors.append(f"overlay {name} patches the pod template's metadata")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "calls.log"
        env = scrubbed_env(SCRUB, stub_bin(tmp, ["kubectl"]), log)
        out = subprocess.run([make, "-n", "--no-print-directory", "-C", str(root), "k8s-logs"],
                             env=env, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            errors.append(f"make -n exited {out.returncode}: {out.stderr.strip()}")
        if log.exists() and log.read_text().strip():
            errors.append("make -n ran kubectl: " + "; ".join(
                " ".join(json.loads(l)) for l in log.read_text().splitlines()))
        commands = [w for l in logical_lines(out.stdout.splitlines())
                    for w in [words(l)] if w[:2] == ["kubectl", "logs"]]

    namespaces = [overlays[n]["namespace"] for n in ("dev", "prod")]
    if [c[c.index("-n") + 1] if "-n" in c else None for c in commands] != namespaces:
        errors.append(f"the kubectl logs commands' namespaces are "
                      f"{[c[c.index('-n') + 1] if '-n' in c else None for c in commands]}, "
                      f"the overlays' {namespaces}")
    for c in commands:
        selector = c[c.index("-l") + 1] if "-l" in c else ""
        pairs = dict(p.split("=", 1) for p in selector.split(",") if "=" in p)
        if not pairs:
            errors.append(f"no -l key=value selector in {' '.join(c)}")
        for key, value in pairs.items():
            if template_labels.get(key) != value:
                errors.append(f"{key}={value} is not a label of the Job's pod template "
                              f"{template_labels}")
    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   k8s-logs selects {commands[0][commands[0].index('-l') + 1]} in {namespaces}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
