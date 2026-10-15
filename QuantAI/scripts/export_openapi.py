"""Export the local application's contract without starting an HTTP server or worker."""

import argparse
import json
from pathlib import Path

from portfolio_client.config import Settings
from portfolio_client.main import create_app
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parents[1]


def render_schema() -> str:
    cache = ROOT / ".cache" / "openapi"
    cache.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        database_url=f"sqlite:///{cache / 'schema.db'}",
        data_dir=cache,
        session_token=SecretStr("openapi-export-only"),
    )
    app = create_app(settings)
    return (
        json.dumps(app.openapi(), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    )


def export_schema(path: Path, *, check: bool) -> bool:
    schema = render_schema()
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == schema
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "contracts" / "openapi.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    valid = export_schema(args.output, check=args.check)
    print(
        "OpenAPI contract verified"
        if valid
        else "OpenAPI drift: regenerate the contract"
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
