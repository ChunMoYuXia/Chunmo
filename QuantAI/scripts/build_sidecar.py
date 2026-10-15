"""Build one owned process plus its runtime directory; never embed personal data."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def publish_runtime(root: Path, relative_destination: str) -> None:
    """Publish only to one of the two build-owned runtime destinations."""
    allowed = {
        "apps/client/src-tauri/resources/backend",
        "dist/stock-portfolio-client/portfolio-api",
    }
    if relative_destination not in allowed:
        raise ValueError("Runtime destination is not build-owned")
    root = root.resolve()
    source = (root / ".cache/sidecar/portfolio-api").resolve()
    destination = root / relative_destination
    if not source.is_relative_to(root) or destination.resolve() != destination:
        raise ValueError(
            "Runtime paths must stay within this workspace without linked targets"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tauri", action="store_true")
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    web = root / "apps/client/build"
    if not (web / "index.html").is_file():
        raise SystemExit("请先运行 npm --prefix apps/client run build")
    cache = root / ".cache/sidecar"
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name",
        "portfolio-api",
        "--distpath",
        str(cache),
        "--workpath",
        str(root / ".cache/pyinstaller-work"),
        "--specpath",
        str(cache),
        "--paths",
        str(root / "apps/api/src"),
        "--add-data",
        f"{web}:web",
        "--collect-all",
        "pypinyin",
        "--collect-submodules",
        "uvicorn",
        "--hidden-import",
        "sqlalchemy.dialects.sqlite",
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "pytest",
        str(root / "scripts/sidecar_entry.py"),
    ]
    subprocess.run(command, cwd=root, check=True)
    artifact = cache / "portfolio-api"
    if args.tauri:
        publish_runtime(root, "apps/client/src-tauri/resources/backend")
    if args.portable:
        publish_runtime(root, "dist/stock-portfolio-client/portfolio-api")
    print(f"Sidecar runtime: {artifact}")


if __name__ == "__main__":
    main()
