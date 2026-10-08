"""训练指标 JSONL 存储 —— 实时看板（scripts/dashboard.py）的数据源。

每条记录一行 JSON：``{"step": 3, "train/policy_loss": 0.1, ...}``。
键名统一用 ``组/指标`` 前缀分组，看板按前缀解析出各面板的序列。

设计要点：
  * 纯标准库（json + threading），零第三方依赖
  * 写入加锁，训练主进程写、看板进程读，可安全并发
  * 每步只追加一行，文件体积可控（1e5 步约几十 MB）
"""

import json
import math
import os
import threading
from typing import Dict, List, Optional


def _sanitize(record: Dict) -> Dict:
    """把非有限浮点值（inf / nan）替换为 None。

    Python 的 ``json.dumps`` 会原样输出 Infinity/NaN，这不是合法 JSON，
    浏览器端 ``resp.json()`` 解析失败会导致看板永远拿不到数据
    （实测梯度爆炸时的 grad_norm=inf 让整个看板卡死）。替换为 None 后
    前端按缺失值渲染为 "-"。
    """
    out = {}
    for k, v in record.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


class MetricsStore:
    """追加式 JSONL 指标存储。

    Args:
        path: 指标文件路径（如 logs/metrics.jsonl）
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # 训练脚本可重复运行，每次启动开启新的一段（看板按 step 排序展示）。
        # 只读方（看板）不写文件，避免往训练数据里混入会话标记。
        if not read_only and os.path.exists(path):
            self._append_line({"__session_end__": True})

    def _append_line(self, record: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add(self, step: int, metrics: Dict[str, float]):
        """追加一条指标记录（线程安全）。

        Args:
            step: 全局更新步数
            metrics: 键为 ``组/指标`` 的扁平 dict（值须可 JSON 序列化）
        """
        record = {"step": int(step)}
        record.update(_sanitize(metrics))
        with self._lock:
            self._append_line(record)

    def tail(self, limit: Optional[int] = None) -> List[dict]:
        """读取全部（或最近 limit 条）记录，供看板 API 使用。

        损坏的行（写一半被强杀等）会被跳过，不影响看板整体渲染。
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "__session_end__" not in rec:
                records.append(_sanitize(rec))

        if limit and limit > 0:
            records = records[-limit:]
        return records
