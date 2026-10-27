#!/usr/bin/env python3
"""Every configuration the repository ships must pass the binary's validator.

Runs `allen-cahn-cuda --validate-config FILE` (no GPU needed) on:
  * config/*.json
  * the simulation.json embedded in k8s/base/configmap.yaml and every JSON
    payload patched into it by k8s/overlays/*/kustomization.yaml
  * the config/run_vtk.json that `make cuda-run-vtk` writes (printf in Makefile)
  * every ```json example in README.md and docs/*.md
and, as controls, requires these copies of shipped configurations to be
REJECTED, each with its own message: one with a misspelled key (rejected by
the parser), and five that parse but break a rule of SimulationConfig::validate()
(epsilon > 1/15 without willmore_beta, willmore_beta with a fully explicit
scheme, a negative willmore_beta, a periodic face opposite a non-periodic one,
a Robin face whose alpha and beta have opposite signs).
So neither a validator that accepts anything nor a loader that parses without
validating can pass.

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

        def mutated(name, change):
            cfg = json.loads((root / "config" / name).read_text())
            change(cfg)
            return json.dumps(cfg)

        def misspell(cfg):
            cfg["physics"]["epsilom"] = cfg["physics"].pop("epsilon")

        def no_beta(cfg):
            del cfg["physics"]["willmore_beta"]

        def heun(cfg):
            cfg["time"]["scheme"] = "heun"

        def negative_beta(cfg):
            cfg["physics"]["epsilon"] = 0.05
            cfg["physics"]["willmore_beta"] = -0.5

        def one_sided_periodic(cfg):
            faces = {f: {"type": "neumann", "flux": 0.0}
                     for f in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")}
            faces["x_lo"] = {"type": "periodic"}
            cfg["boundary"]["phi"] = faces

        def robin_opposite_signs(cfg):
            cfg["boundary"]["u"] = {"type": "robin", "alpha": -1.1, "beta": 0.4, "gamma": 0.0}

        controls = [
            ("misspelled key 'epsilom'", mutated("default.json", misspell), "epsilom"),
            ("eps > 1/15 without willmore_beta", mutated("run_dendrite.json", no_beta),
             "exceeds 1/15"),
            ("willmore_beta with scheme heun", mutated("run_dendrite.json", heun),
             "requires time.scheme"),
            ("negative willmore_beta", mutated("default.json", negative_beta),
             "willmore_beta must be >= 0"),
            ("periodic x_lo opposite a non-periodic x_hi",
             mutated("run_dendrite.json", one_sided_periodic), "periodic BC must be set on both faces"),
            ("Robin u with alpha < 0 < beta", mutated("default.json", robin_opposite_signs),
             "must not have opposite signs"),
        ]
        for what, text, needle in controls:
            out = validate("control", text)
            message = out.stdout + out.stderr
            if out.returncode == 0:
                print(f"FAIL control: {what}: accepted")
                failures += 1
            elif needle not in message:
                print(f"FAIL control: {what}: rejected without \"{needle}\"")
                print("     " + message.strip().replace("\n", "\n     "))
                failures += 1
            else:
                print(f"ok   control: {what}: rejected")

    print(f"{len(cases)} shipped configurations, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
