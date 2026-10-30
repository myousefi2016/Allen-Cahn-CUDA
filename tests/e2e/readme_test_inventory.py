#!/usr/bin/env python3
"""README's test inventory is the one this build registers.

README.md says which tests exist in four places: the summary table (files and ctest
tests of each group), the unit and integration headings (files), one table row per
test file in each group's section, and the project tree's test lines. They went
stale whenever a test was added without them (549b32f's three unit tests left the
table at 356, fbc923f's test_KernelRanges.cpp left the heading at 28 files).
Checked against:
  * the unit and integration test files: the SOURCES of the unit_tests and
    integration_tests targets, which CMake passes (--unit, --integration); every
    .cpp and .cu file of tests/unit and tests/integration is one of them;
  * the ctest tests of each group and the end-to-end scripts: what this build
    registers (ctest --show-only=json-v1). Every test is in one of the three
    groups, and every end-to-end test runs a script of tests/e2e;
  * the project tree's tests/e2e names: the .py files of tests/e2e, each an
    end-to-end script or a module one of them imports.
ctest lists the tests from a temporary directory whose CTestTestfile.cmake includes
the build's. A listing in the build directory writes its Testing/Temporary/LastTest.log:
ctest 3.28 (the CI images') replaced the log of the ctest running this test with its
own, which kept none of the run's entries; ctest 4.2 kept it.

A build that has not built unit_tests or integration_tests (the CI step that builds only
the binary and runs the end-to-end tests) has no inventory to check: gtest_discover_tests
registers the placeholder <target>_NOT_BUILT for such a target, and the test exits 77
(SKIP_RETURN_CODE, reported as Skipped) naming it.

No GPU; Python 3 standard library.

Usage: readme_test_inventory.py CTEST BUILD_DIR REPO_ROOT --unit SOURCE... --integration SOURCE...
(exit 0 = pass, 77 = the test executables are not built)
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GROUPS = {"unit": "Unit tests", "integration": "Integration tests", "e2e": "End-to-end tests"}
# What gtest_discover_tests registers for a target that is not built (CMake's GoogleTest
# module: add_test(<target>_NOT_BUILT ...) when the target's test list does not exist).
NOT_BUILT = ("unit_tests_NOT_BUILT", "integration_tests_NOT_BUILT")
SOURCE_SUFFIXES = (".cpp", ".cu")
SUMMARY_ROW = re.compile(r"^\|\s*(Unit tests|Integration tests|End-to-end tests)\s*\|\s*(\d+)\s*"
                         r"\|\s*(\d+)\s*\|$")
HEADINGS = {
    "unit": re.compile(r"^### Unit tests \(`tests/unit/`, (\d+) files\)$"),
    "integration": re.compile(r"^### Integration tests \(`tests/integration/`, (\d+) files\)$"),
    "e2e": re.compile(r"^### End-to-end tests \(`tests/e2e/`\)()$"),
}
SECTION_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")
TREE_COUNTS = {
    "unit": re.compile(r"^│   ├── unit/\s+# (\d+) unit-test files$"),
    "integration": re.compile(r"^│   ├── integration/\s+# (\d+) integration-test files$"),
}
TREE_E2E = re.compile(r"^│   ├── e2e/\s+# (.*)$")
TREE_CONTINUATION = re.compile(r"^│   │\s+# (.*)$")
PY_NAME = re.compile(r"[A-Za-z0-9_]+\.py")


def parse_args(argv):
    if len(argv) < 5 or argv[4] != "--unit" or "--integration" not in argv[5:]:
        raise SystemExit("usage: readme_test_inventory.py CTEST BUILD_DIR REPO_ROOT "
                         "--unit SOURCE... --integration SOURCE...")
    i = argv.index("--integration", 5)
    return argv[1], Path(argv[2]).resolve(), Path(argv[3]).resolve(), argv[5:i], argv[i + 1:]


def registered_tests(ctest, build):
    """The tests of this build, as ctest lists them (name and command)."""
    with tempfile.TemporaryDirectory() as tmp:
        # A CMake quoted argument: backslash, quote and dollar are escaped.
        path = re.sub(r'([\\"$])', r"\\\1", str(build))
        (Path(tmp) / "CTestTestfile.cmake").write_text(f'subdirs("{path}")\n')
        out = subprocess.run([ctest, "--test-dir", tmp, "--show-only=json-v1"],
                             capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SystemExit(f"FAIL ctest --show-only=json-v1 exited {out.returncode}: "
                         f"{out.stderr.strip()}")
    return [(t["name"], t.get("command")) for t in json.loads(out.stdout)["tests"]]


def built_files(group, sources, root, errors):
    """File names of a target's SOURCES; each must be in tests/<group>, and every test
    source there must be one of them."""
    directory = root / "tests" / group
    names = []
    for s in sources:
        p = Path(s) if Path(s).is_absolute() else root / "tests" / s
        if p.resolve().parent != directory:
            errors.append(f"{group}_tests source {s} is not in tests/{group}/")
        names.append(p.name)
    present = sorted(p.name for p in directory.iterdir() if p.suffix in SOURCE_SUFFIXES)
    for name in sorted(set(present) - set(names)):
        errors.append(f"tests/{group}/{name} is not a source of {group}_tests: it never runs")
    return names


def imported_modules(script):
    tree = ast.parse(script.read_text(), filename=str(script))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def section_rows(lines, heading):
    """The file names of the table rows under the heading, up to the next heading."""
    start = next(i for i, l in enumerate(lines) if heading.match(l))
    rows = []
    for l in lines[start + 1:]:
        if l.startswith("#"):
            break
        m = SECTION_ROW.match(l)
        if m:
            rows.append(m[1])
    return rows


def tree_e2e_names(lines):
    start = [i for i, l in enumerate(lines) if TREE_E2E.match(l)]
    if len(start) != 1:
        return None
    names = PY_NAME.findall(TREE_E2E.match(lines[start[0]])[1])
    for l in lines[start[0] + 1:]:
        m = TREE_CONTINUATION.match(l)
        if not m:
            break
        names += PY_NAME.findall(m[1])
    return names


def compare_names(where, listed, expected, errors):
    duplicates = sorted({n for n in listed if listed.count(n) > 1})
    if duplicates:
        errors.append(f"{where} lists {', '.join(duplicates)} more than once")
    missing, extra = sorted(set(expected) - set(listed)), sorted(set(listed) - set(expected))
    if missing:
        errors.append(f"{where} lacks {', '.join(missing)}")
    if extra:
        errors.append(f"{where} lists {', '.join(extra)}, which the build does not have")


def main() -> int:
    ctest, build, root, unit_sources, integration_sources = parse_args(sys.argv)
    errors = []

    files = {"unit": built_files("unit", unit_sources, root, errors),
             "integration": built_files("integration", integration_sources, root, errors)}
    counts = dict.fromkeys(GROUPS, 0)
    e2e_dir = root / "tests" / "e2e"
    scripts = []
    tests = registered_tests(ctest, build)
    not_built = [name for name, _ in tests if name in NOT_BUILT]
    if not_built:
        print(f"SKIP this build has not built the test executables ({', '.join(not_built)} are "
              f"registered in their place), so their tests are not listed")
        return 77
    for name, command in tests:
        group = name.split(".", 1)[0]
        if group not in GROUPS or "." not in name:
            errors.append(f"ctest test {name} is in none of the groups unit., integration., e2e.")
            continue
        counts[group] += 1
        if group == "e2e":
            script = Path(command[1]).resolve() if command and len(command) > 1 else None
            if script is None or script.parent != e2e_dir or script.suffix != ".py":
                errors.append(f"{name} runs {command}, not a script of tests/e2e/")
            else:
                scripts.append(script.name)
    files["e2e"] = sorted(set(scripts))

    present = sorted(p.name for p in e2e_dir.glob("*.py"))
    imported = set().union(*(imported_modules(e2e_dir / s) for s in files["e2e"]))
    for name in present:
        if name not in files["e2e"] and Path(name).stem not in imported:
            errors.append(f"tests/e2e/{name} is neither run by a test nor imported by one")

    lines = (root / "README.md").read_text().splitlines()
    summary = {}
    for l in lines:
        m = SUMMARY_ROW.match(l)
        if m:
            summary.setdefault(m[1], []).append((int(m[2]), int(m[3])))
    for group, label in GROUPS.items():
        expected = (len(files[group]), counts[group])
        rows = summary.get(label, [])
        if len(rows) != 1:
            errors.append(f"README's summary table has {len(rows)} rows '{label}', expected one")
        elif rows[0] != expected:
            errors.append(f"README's summary table: {label} {rows[0][0]} files, {rows[0][1]} "
                          f"ctest tests; the build has {expected[0]} and {expected[1]}")

        heading = HEADINGS[group]
        found = [m for m in map(heading.match, lines) if m]
        if len(found) != 1:
            errors.append(f"README has {len(found)} headings matching {heading.pattern}, "
                          f"expected one")
            continue
        if found[0][1] and int(found[0][1]) != len(files[group]):
            errors.append(f"README's heading '{found[0][0]}' says {found[0][1]} files; the "
                          f"build has {len(files[group])}")
        compare_names(f"README's {label} section", section_rows(lines, heading), files[group],
                      errors)

    for group, pattern in TREE_COUNTS.items():
        found = [m for m in map(pattern.match, lines) if m]
        if len(found) != 1:
            errors.append(f"README's project tree has {len(found)} lines matching "
                          f"{pattern.pattern}, expected one")
        elif int(found[0][1]) != len(files[group]):
            errors.append(f"README's project tree says {found[0][1]} {group}-test files; the "
                          f"build has {len(files[group])}")
    tree = tree_e2e_names(lines)
    if tree is None:
        errors.append(f"README's project tree has no single line matching {TREE_E2E.pattern}")
    else:
        compare_names("README's project tree (tests/e2e/)", tree, present, errors)

    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print("ok   README's test inventory: " + ", ".join(
            f"{GROUPS[g].lower()} {len(files[g])} files, {counts[g]} ctest tests" for g in GROUPS))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
