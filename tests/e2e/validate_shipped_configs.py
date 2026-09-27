#!/usr/bin/env python3
"""Every configuration the repository ships must pass the binary's validator.

Runs `allen-cahn-cuda --validate-config FILE` (no GPU needed) on:
  * config/*.json
  * the simulation.json embedded in k8s/base/configmap.yaml and every JSON
    payload patched into it by k8s/overlays/*/kustomization.yaml
  * the config/run_vtk.json that `make cuda-run-vtk` writes (printf in Makefile)
  * every ```json example in README.md and docs/*.md
and, as a control, requires a copy of config/default.json with one misspelled
key to be REJECTED (so a validator that accepts anything cannot pass).

Usage: validate_shipped_configs.py BINARY REPO_ROOT   (exit 0 = all valid)
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml  # python3-yaml
except ImportError:
    print("FAIL: PyYAML (python3-yaml) is required to read the k8s manifests")
    sys.exit(1)


def k8s_payloads(root: Path):
    base = yaml.safe_load((root / "k8s/base/configmap.yaml").read_text())
    yield "k8s/base/configmap.yaml", base["data"]["simulation.json"]
    for kust in sorted((root / "k8s/overlays").glob("*/kustomization.yaml")):
        doc = yaml.safe_load(kust.read_text())
        for i, patch in enumerate(doc.get("patches", []) or []):
            ops = yaml.safe_load(patch.get("patch", "") or "") or []
            for op in ops if isinstance(ops, list) else []:
                if isinstance(op, dict) and op.get("path") == "/data/simulation.json":
                    yield f"{kust.relative_to(root)} patch {i}", op["value"]


def makefile_run_vtk(root: Path):
    text = (root / "Makefile").read_text()
    m = re.search(r"@printf '%s\\n' \\\n(.*?)> config/run_vtk\.json", text, re.S)
    if not m:
        raise SystemExit("FAIL: could not find the run_vtk.json printf in the Makefile")
    return "\n".join(re.findall(r"'([^']*)'", m.group(1)))


def doc_examples(root: Path):
    for doc in [root / "README.md", *sorted((root / "docs").glob("*.md"))]:
        text = doc.read_text()
        for m in re.finditer(r"```json\n(.*?)```", text, re.S):
            line = text[: m.start()].count("\n") + 1
            yield f"{doc.relative_to(root)}:{line}", m.group(1)


def main() -> int:
    binary, root = Path(sys.argv[1]), Path(sys.argv[2])
    cases = [(str(p.relative_to(root)), p.read_text()) for p in sorted((root / "config").glob("*.json"))
             if not p.name.endswith("_resume.json") and p.name != "run_vtk.json"]
    cases += list(k8s_payloads(root))
    cases.append(("Makefile cuda-run-vtk", makefile_run_vtk(root)))
    cases += list(doc_examples(root))
    if len(cases) < 8:
        print(f"FAIL: only {len(cases)} configurations found; the extraction is broken")
        return 1

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        def validate(name, text):
            path = Path(tmp) / "config.json"
            path.write_text(text)
            return subprocess.run([str(binary), "--validate-config", str(path)],
                                  capture_output=True, text=True, timeout=60)

        for name, text in cases:
            out = validate(name, text)
            ok = out.returncode == 0
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}")
            if not ok:
                print("     " + (out.stdout + out.stderr).strip().replace("\n", "\n     "))

        control = json.loads((root / "config/default.json").read_text())
        control["physics"]["epsilom"] = control["physics"].pop("epsilon")
        out = validate("control", json.dumps(control))
        if out.returncode == 0:
            print("FAIL control: a config with the misspelled key 'epsilom' was accepted")
            failures += 1
        else:
            print("ok   control: misspelled key rejected")

    print(f"{len(cases)} shipped configurations, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
