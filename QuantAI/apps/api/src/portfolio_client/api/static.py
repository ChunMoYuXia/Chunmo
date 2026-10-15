import base64
import hashlib
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse

SPA_ROUTES = {
    "",
    "data",
    "market",
    "screening",
    "portfolios",
    "backtests",
    "risk",
    "paper",
    "reports",
    "settings",
}


def mount_static_client(app: FastAPI, directory: Path) -> None:
    root = directory.resolve()
    index = root / "index.html"
    if not index.is_file():
        raise ValueError("STATIC_CLIENT_MISSING")
    html = index.read_text(encoding="utf-8")
    hashes = [
        "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"
        for attributes, script in re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.DOTALL)
        if not re.search(r"\bsrc\s*=", attributes)
    ]
    headers = {
        "Content-Security-Policy": "default-src 'self'; script-src 'self' "
        + " ".join(hashes)
        + "; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }

    @app.get("/{path:path}", include_in_schema=False)
    def static_file(path: str) -> FileResponse:
        try:
            target = (root / path).resolve()
            if not target.is_relative_to(root) or any(
                part.startswith(".") for part in Path(path).parts
            ):
                raise HTTPException(404)
            if target.is_file():
                return FileResponse(target, headers=headers)
            if path.rstrip("/") in SPA_ROUTES:
                return FileResponse(index, headers=headers)
        except (OSError, ValueError):
            raise HTTPException(404) from None
        raise HTTPException(404)
