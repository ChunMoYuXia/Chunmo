"""只读看板数据层：增量读日志，训练和评估分别统计。仅使用标准库。"""
from collections import Counter, deque
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import threading


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def read_json(path):
    try:
        value = clean(json.loads(path.read_text(encoding="utf-8")))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class JsonlReader:
    """记住字节位置；未换行的尾行留到下次，文件重写后重新读取。"""
    def __init__(self, path):
        self.path = Path(path)
        self.identity = self.modified = None
        self.reset()

    def reset(self):
        self.offset = 0
        self.rows = deque(maxlen=5000)
        self.groups = Counter()
        self.recent_games = deque(maxlen=100)
        self.recent_train = deque(maxlen=100)

    def refresh(self):
        try:
            with self.path.open("rb") as stream:
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if identity != self.identity or info.st_size < self.offset or (
                    info.st_size == self.offset and info.st_mtime_ns != self.modified
                ):
                    self.reset()
                self.identity, self.modified = identity, info.st_mtime_ns
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    try:
                        row = clean(json.loads(line))
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    if "__session_end__" in row:
                        self.rows.clear()
                        continue
                    self.rows.append(row)
                    if "outcome" not in row or "hero_id" not in row:
                        continue
                    self.recent_games.append(row)
                    mode = row.get("mode", "eval")
                    if mode == "train":
                        self.recent_train.append(row)
                    for kind, key in (("total", "all"), ("hero", str(row["hero_id"])),
                                      ("side", str(row.get("player_id", "?")))):
                        for name, amount in (("games", 1), (row["outcome"], 1),
                                             ("truncated", bool(row.get("truncated")))):
                            self.groups[(mode, kind, key, name)] += amount
        except FileNotFoundError:
            self.reset()
            self.identity = None

    def statistics(self, mode, kind="total"):
        keys = sorted({k[2] for k in self.groups if k[0] == mode and k[1] == kind})
        result = {}
        for key in keys:
            values = {name: self.groups[(mode, kind, key, name)] for name in
                      ("games", "win", "lose", "draw", "unknown", "truncated")}
            values["win_rate"] = values["win"] / values["games"] if values["games"] else None
            result[key] = values
        return result


class DashboardData:
    def __init__(self, project, run_dir=None, metrics=None, runs_dir=None):
        self.project = Path(project).resolve()
        self.pinned = Path(run_dir).resolve() if run_dir else (Path(metrics).resolve().parent if metrics else None)
        self.metrics = Path(metrics).resolve() if metrics else None
        self.runs_dir = Path(runs_dir).resolve() if runs_dir else self.project / "runs"
        self.readers = {}
        self.lock = threading.Lock()

    def runs(self):
        candidates = [self.pinned] if self.pinned else list(self.runs_dir.glob("*"))
        if not self.pinned:
            candidates += [self.project / name for name in ("logs", "logs_p0", "logs_p1")]
        result = []
        for path in candidates:
            if path is None or (path.is_symlink() and not self.pinned):
                continue
            sources = [path / name for name in ("metrics.jsonl", "episodes.jsonl", "manifest.json", "status.json")]
            if self.metrics and path == self.pinned:
                sources.append(self.metrics)
            existing = [p for p in sources if p.is_file()]
            if not existing and not self.pinned:
                continue
            result.append({"id": str(path), "name": path.name,
                           "modified": max((p.stat().st_mtime for p in existing), default=0)})
        return sorted(result, key=lambda r: r["modified"], reverse=True)

    def reader(self, path):
        if str(path) not in self.readers:
            self.readers[str(path)] = JsonlReader(path)
        reader = self.readers[str(path)]
        reader.refresh()
        return reader

    def snapshot(self, run_id=None, limit=1000):
        with self.lock:
            runs = self.runs()
            selected = next((r for r in runs if r["id"] == run_id), None)
            if run_id and selected is None:
                raise ValueError("运行目录不在当前看板范围内")
            selected = selected or (runs[0] if runs else None)
            if selected is None:
                return {"runs": [], "run": None, "metrics": [], "episodes": [], "heroes": {}, "sides": {}}
            directory = Path(selected["id"])
            metrics = self.reader(self.metrics or directory / "metrics.jsonl")
            episodes = self.reader(directory / "episodes.jsonl")
            evaluations = self.reader(directory / "evaluations.jsonl")
            train_recent = list(episodes.recent_train)
            recent = {"games": len(train_recent), "wins": sum(r["outcome"] == "win" for r in train_recent)}
            recent["win_rate"] = recent["wins"] / recent["games"] if recent["games"] else None
            independent = []
            for path in directory.glob("eval-*/summary.json"):
                if path.is_symlink() or path.parent.is_symlink():
                    continue
                summary = read_json(path)
                if summary:
                    independent.append((path.stat().st_mtime, summary))
            return clean({"runs": runs, "run": selected, "manifest": read_json(directory / "manifest.json"),
                          "status": read_json(directory / "status.json"),
                          "server_time": datetime.now(timezone.utc).isoformat(),
                          "metrics": list(metrics.rows)[-max(1, min(limit, 5000)):],
                          "episodes": [self.compact_game(r) for r in list(episodes.recent_games)[-30:]],
                          "train": episodes.statistics("train").get("all", {}), "recent": recent,
                          "heroes": {m: episodes.statistics(m, "hero") for m in ("train", "eval")},
                          "sides": {m: episodes.statistics(m, "side") for m in ("train", "eval")},
                          "evaluation": evaluations.rows[-1] if evaluations.rows else None,
                          "evaluations": list(evaluations.rows)[-200:],
                          "independent_evaluation": max(independent, key=lambda x: x[0])[1] if independent else None})

    @staticmethod
    def compact_game(game):
        result = {key: game.get(key) for key in ("timestamp", "started_at", "ended_at", "sdk_game_id", "sdk_client_id",
                "game_id", "sgame_id",
                "runtime_id", "worker_id", "policy_version", "sampler_device", "control", "initial_frame", "update", "mode", "hero_id", "opponent_id",
                "player_id", "outcome", "episode_len", "reward", "truncated", "termination_reason")}
        # 旧日志只有结束时写入的 timestamp；利用既有耗时估算，不修改原日志。
        duration = game.get("timings", {}).get("trajectory_seconds")
        if not result["started_at"] and game.get("timestamp") and isinstance(duration, (int, float)):
            try:
                if math.isfinite(duration) and duration >= 0:
                    ended = datetime.fromisoformat(game["timestamp"].replace("Z", "+00:00"))
                    result["started_at"] = (ended - timedelta(seconds=duration)).isoformat()
                    result["started_at_estimated"] = True
            except (ValueError, OverflowError):
                pass
        return result
