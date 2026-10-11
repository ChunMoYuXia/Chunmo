"""本地训练看板：标准库 HTTP 服务 + 原生 SVG 图表，无需联网。"""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hok1v1_rl.dashboard_data import DashboardData

ASSETS = ROOT / "dashboard"


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, data, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/state", "/api/metrics"):
            query = parse_qs(parsed.query)
            try:
                state = self.data.snapshot(query.get("run", [None])[0], int(query.get("limit", ["1000"])[0]))
                payload = state["metrics"] if parsed.path == "/api/metrics" else state
                self.send_body(200, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json")
            except ValueError as exc:
                self.send_body(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode(), "application/json")
            except OSError:
                self.send_body(503, b'{"error":"Log temporarily unavailable"}', "application/json")
            return
        # 白名单避免把任意本地文件暴露给 HTTP 客户端。
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
        if parsed.path not in files:
            self.send_body(404, b"Not found", "text/plain")
            return
        name, content_type = files[parsed.path]
        self.send_body(200, (ASSETS / name).read_bytes(), content_type)

    def send_body(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="0.0.0.0", help="默认兼容 WSL 的 Windows localhost 转发")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--metrics", help="兼容旧命令：指定 metrics.jsonl")
    selection.add_argument("--run-dir", help="仅查看一个运行目录；不存在时等待首次日志")
    parser.add_argument("--runs-dir", help="运行目录集合；默认自动发现项目 runs 和旧 logs")
    args = parser.parse_args()
    data = DashboardData(ROOT, args.run_dir, args.metrics, args.runs_dir)
    try:
        server = ThreadingHTTPServer((args.host, args.port), partial(DashboardHandler, data=data))
    except OSError as exc:
        parser.exit(1, f"看板启动失败：{exc}。端口已占用时可访问已有页面，或用 --port 8601。\n")
    print(f"训练看板：http://localhost:{args.port} · 每 2 秒刷新 · Ctrl+C 退出", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
