"""低频写入训练进度；看板不需要连接 GPU 或 gamecore。"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time


class LiveStatus:
    def __init__(self, directory, **initial):
        self.path = Path(directory) / "status.json"
        self.data = initial
        self.last_write = 0.0
        self.started = time.monotonic()
        self.warned = False

    def update(self, force=False, **changes):
        self.data.update(changes)
        now = time.monotonic()
        if not force and now - self.last_write < 2:
            return
        self.data.update(heartbeat=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=now - self.started)
        # 原子替换：读方不会读到半份 JSON。
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.path)
            self.last_write = now
        except OSError as exc:
            if not self.warned:
                logging.warning("看板进度写入失败，训练继续：%s", exc)
                self.warned = True
