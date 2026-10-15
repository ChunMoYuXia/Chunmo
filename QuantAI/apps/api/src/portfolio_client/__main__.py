import argparse
import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
from pydantic import SecretStr, ValidationError

from portfolio_client.config import Settings
from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.ai.compatible import ProviderSettings
from portfolio_client.infrastructure.files.workspace_lock import workspace_lock
from portfolio_client.main import create_app


def bundled_static_directory() -> Path | None:
    return Path(sys._MEIPASS) / "web" if hasattr(sys, "_MEIPASS") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="组合研究台本地服务")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "runtime")
    parser.add_argument("--static-dir", type=Path, default=bundled_static_directory())
    parser.add_argument("--sidecar", action="store_true")
    parser.add_argument("--parent-stdin", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("PORTFOLIO_SESSION_TOKEN")
    if args.sidecar and not token:
        parser.exit(2, "SIDECAR_TOKEN_REQUIRED: 父进程必须提供本地会话令牌\n")
    token = token or secrets.token_hex(32)
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (
            workspace_lock(data_dir),
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
        ):
            listener.bind((args.host, args.port))
            listener.listen(128)
            port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            ai = (
                ProviderSettings(
                    base_url=os.environ["PORTFOLIO_AI_BASE_URL"],
                    model=os.environ["PORTFOLIO_AI_MODEL"],
                    api_key=SecretStr(os.environ["PORTFOLIO_AI_API_KEY"]),
                )
                if all(
                    os.environ.get(key)
                    for key in (
                        "PORTFOLIO_AI_BASE_URL",
                        "PORTFOLIO_AI_MODEL",
                        "PORTFOLIO_AI_API_KEY",
                    )
                )
                else None
            )
            settings = Settings(
                database_url=f"sqlite:///{(data_dir / 'portfolio.db').as_posix()}",
                data_dir=data_dir,
                session_token=SecretStr(token),
                ai=ai,
                static_dir=args.static_dir,
                allowed_origins=[
                    "http://127.0.0.1:5173",
                    url,
                    "tauri://localhost",
                    "http://tauri.localhost",
                ],
            )
            app = create_app(settings)
            server = uvicorn.Server(
                uvicorn.Config(
                    app, host=args.host, port=port, access_log=False, log_level="warning"
                )
            )
            if args.sidecar or args.port == 0:
                print(json.dumps({"version": 1, "url": url}), flush=True)
            if args.parent_stdin:

                def parent_closed() -> None:
                    try:
                        sys.stdin.buffer.read()
                    finally:
                        server.should_exit = True

                threading.Thread(target=parent_closed, daemon=True).start()
            if args.open_browser:
                if args.static_dir is None:
                    raise ValueError("STATIC_CLIENT_MISSING")

                def open_when_ready() -> None:
                    for _ in range(200):
                        if server.should_exit:
                            return
                        if server.started:
                            webbrowser.open(
                                url + "/#" + urlencode({"session": token, "backend": url})
                            )
                            return
                        time.sleep(0.05)

                threading.Thread(target=open_when_ready, daemon=True).start()
            server.run(sockets=[listener])
    except DomainError as error:
        parser.exit(2, f"{error.code}: {error.message}\n")
    except (ValidationError, ValueError):
        parser.exit(2, "INVALID_LOCAL_CONFIGURATION: 请检查数据目录、静态客户端和AI配置\n")
    except OSError:
        parser.exit(2, "LOCAL_SERVICE_START_FAILED: 无法占用端口或访问数据目录\n")


if __name__ == "__main__":
    main()
