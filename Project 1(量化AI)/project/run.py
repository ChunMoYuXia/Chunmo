"""QuantAI 启动入口。

开发模式：python run.py
打包后：直接运行 exe

自动检测环境，启动 Streamlit 服务并打开浏览器。
"""
import os
import sys
import webbrowser
from pathlib import Path


def main() -> None:
    # 确保项目根目录在 sys.path 中
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    # Streamlit 入口
    app_path = root / "src" / "quantai" / "web" / "app.py"
    if not app_path.exists():
        print(f"[错误] 找不到应用入口：{app_path}")
        sys.exit(1)

    from streamlit.web import cli as stcli

    port = 8501
    sys.argv = [
        "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--global.developmentMode", "false",
    ]

    # 打开浏览器（稍等服务启动）
    try:
        import threading
        def _open_browser():
            import time
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open_browser, daemon=True).start()
    except Exception:
        pass

    print(f"[QuantAI] 启动中…… 浏览器将自动打开 http://localhost:{port}")
    print(f"[QuantAI] 按 Ctrl+C 退出")
    stcli.main()


if __name__ == "__main__":
    main()
