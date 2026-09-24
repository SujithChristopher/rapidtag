"""Check that RapidTag's runtime API, type stub, and API reference agree."""

from __future__ import annotations

import ast
from pathlib import Path

import rapidtag


ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "rapidtag.pyi"
API_DOC = ROOT / "docs/python-api.md"


def stub_public_names() -> set[str]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }


def runtime_public_names() -> set[str]:
    module_name = rapidtag.__name__.rsplit(".", 1)[-1]
    return {
        name
        for name in dir(rapidtag)
        if not name.startswith("_") and name != module_name
    }


def main() -> int:
    runtime = runtime_public_names()
    stub = stub_public_names()
    missing_from_stub = sorted(runtime - stub)
    extra_in_stub = sorted(stub - runtime)

    api_text = API_DOC.read_text(encoding="utf-8")
    missing_from_docs = sorted(
        name for name in runtime if f"`{name}" not in api_text
    )

    failures = []
    if missing_from_stub:
        failures.append(f"missing from rapidtag.pyi: {missing_from_stub}")
    if extra_in_stub:
        failures.append(f"not present at runtime: {extra_in_stub}")
    if missing_from_docs:
        failures.append(f"missing from docs/python-api.md: {missing_from_docs}")

    if failures:
        print("Documentation consistency check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Documentation covers all {len(runtime)} public runtime symbols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
