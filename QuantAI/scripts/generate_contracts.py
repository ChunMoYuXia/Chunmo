"""Regenerate OpenAPI and TypeScript, or detect drift without changing tracked files."""

import argparse
import json
import shutil
import subprocess

from export_openapi import ROOT, export_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema = ROOT / "contracts" / "openapi.json"
    if not export_schema(schema, check=args.check):
        print("OpenAPI drift: run scripts/generate_contracts.py")
        return 1
    node = shutil.which("node")
    if node is None:
        raise SystemExit("Node.js is required to generate API types")
    client = ROOT / "apps" / "client"
    cache = ROOT / ".cache" / "contracts"
    cache.mkdir(parents=True, exist_ok=True)
    generated = cache / "schema.d.ts"
    for package, arguments in (
        ("openapi-typescript", [str(schema), "--output", str(generated)]),
        (
            "prettier",
            ["--config", str(client / ".prettierrc.json"), "--write", str(generated)],
        ),
    ):
        directory = client / "node_modules" / package
        metadata = json.loads((directory / "package.json").read_text(encoding="utf-8"))
        binary = (
            metadata["bin"][package]
            if isinstance(metadata["bin"], dict)
            else metadata["bin"]
        )
        subprocess.run(
            [node, str(directory / binary), *arguments], cwd=client, check=True
        )
    target = client / "src" / "lib" / "api" / "schema.d.ts"
    content = generated.read_text(encoding="utf-8")
    if args.check:
        valid = target.exists() and target.read_text(encoding="utf-8") == content
        print(
            "TypeScript contract verified"
            if valid
            else "TypeScript contract drift: regenerate"
        )
        return 0 if valid else 1
    target.write_text(content, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
