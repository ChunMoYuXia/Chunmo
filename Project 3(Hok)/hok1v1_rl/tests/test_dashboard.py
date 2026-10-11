"""看板的数据口径、增量读取和局内遥测回归。无需 gamecore 或 GPU。"""
from functools import partial
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

from hok1v1_rl.dashboard_data import DashboardData, JsonlReader
from hok1v1_rl.live_status import LiveStatus
from hok1v1_rl.metrics_store import MetricsStore

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dashboard_server", ROOT / "scripts/dashboard.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


class DashboardTests(unittest.TestCase):
    def test_episode_start_time_legacy_estimate_and_control(self):
        game = {"timestamp": "2026-10-10T08:01:57+00:00", "timings": {"trajectory_seconds": 23},
                "control": {"hero_id": 133, "camp": 2, "verified": True}}
        old = DashboardData.compact_game(game)
        self.assertEqual(old["started_at"], "2026-10-10T08:01:34+00:00")
        self.assertTrue(old["started_at_estimated"])
        self.assertNotIn("started_at", game)  # 不修改历史记录
        game.update(started_at="2026-10-10T16:01:33+08:00", game_id="kaiwu-example")
        new = DashboardData.compact_game(game)
        self.assertEqual(new["started_at"], game["started_at"])
        self.assertNotIn("started_at_estimated", new)
        self.assertEqual(new["control"]["hero_id"], 133)
        self.assertEqual(new["game_id"], "kaiwu-example")

    def test_partial_line_and_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_bytes(b'{"step":1}\n{"step":')
            reader = JsonlReader(path)
            reader.refresh()
            self.assertEqual([r["step"] for r in reader.rows], [1])
            reader.refresh()
            self.assertEqual(len(reader.rows), 1)
            with path.open("ab") as stream:
                stream.write(b'2}\ninvalid\n{"step":3,"loss":NaN}\n')
            reader.refresh()
            self.assertEqual([r["step"] for r in reader.rows], [1, 2, 3])
            self.assertIsNone(reader.rows[-1]["loss"])
            replacement = Path(tmp) / "replacement"
            replacement.write_text('{"step":9}\n')
            replacement.replace(path)
            reader.refresh()
            self.assertEqual(list(reader.rows), [{"step": 9}])

    def test_readonly_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-started"
            data = DashboardData(tmp, run_dir=missing)
            self.assertEqual(data.snapshot()["metrics"], [])
            MetricsStore(str(missing / "metrics.jsonl"), read_only=True).tail()
            self.assertFalse(missing.exists())

    def test_per_game_denominator_and_mode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs/demo"
            run.mkdir(parents=True)
            for step, rate in [(1, 1.0), (2, 0.0)]:
                append(run / "metrics.jsonl", {"step": step, "env/win_rate": rate})
            for index in range(10):
                append(run / "episodes.jsonl", {"update": 2, "mode": "train", "hero_id": 112,
                       "player_id": index % 2, "outcome": "win" if index == 0 else "unknown",
                       "truncated": index == 9, "episode_len": 4, "reward": 0})
            append(run / "episodes.jsonl", {"mode": "eval", "hero_id": 121, "player_id": 0,
                   "outcome": "win", "episode_len": 5, "reward": 50})
            append(run / "evaluations.jsonl", {"games": 1, "win_rate": 1.0, "update": 2})
            data = DashboardData(tmp)
            result = data.snapshot()
            self.assertEqual(result["recent"]["win_rate"], .1)  # 不能平均两批胜率得到 50%
            self.assertEqual(result["train"]["unknown"], 9)
            self.assertEqual(result["train"]["truncated"], 1)
            self.assertEqual(result["heroes"]["eval"]["121"]["games"], 1)
            self.assertNotIn("121", result["heroes"]["train"])
            self.assertNotIn("initial_snapshot", result["episodes"][0])
            self.assertEqual(data.snapshot()["train"]["games"], 10)  # 轮询不重复计数
            other = run.parent / "another-run"
            other.mkdir()
            append(other / "metrics.jsonl", {"step": 99})
            self.assertEqual(data.snapshot(str(other))["metrics"][0]["step"], 99)
            self.assertEqual(data.snapshot(str(other))["train"], {})
            self.assertEqual(data.snapshot(str(run))["train"]["games"], 10)
            formal = run / "eval-test"
            formal.mkdir()
            (formal / "summary.json").write_text(json.dumps({"games": 800, "goal_verified": True}))
            self.assertTrue(data.snapshot(str(run))["independent_evaluation"]["goal_verified"])
            with self.assertRaises(ValueError):
                data.snapshot("/etc")
            json.dumps(result, allow_nan=False)

    def test_live_status_atomic_and_throttled(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = LiveStatus(tmp, phase="sampling")
            status.update(force=True, episode_decisions=1)
            first = json.loads(status.path.read_text())
            self.assertIn("heartbeat", first)
            status.update(episode_decisions=2)
            self.assertEqual(json.loads(status.path.read_text())["episode_decisions"], 1)
            status.update(force=True, phase="finished")
            self.assertEqual(json.loads(status.path.read_text())["episode_decisions"], 2)
            self.assertFalse(status.path.with_suffix(".tmp").exists())
            with patch.object(Path, "write_text", side_effect=OSError("test")):
                status.update(force=True)  # 写看板失败不能使训练退出
            self.assertTrue(status.warned)

    def test_http_assets_api_and_path_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "demo"
            run.mkdir()
            append(run / "metrics.jsonl", {"step": 1, "train/kl": 0.0})
            handler = partial(server_module.DashboardHandler, data=DashboardData(tmp, run_dir=run))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = "http://127.0.0.1:" + str(server.server_port)
            # 本地测试不经过系统 HTTP 代理。
            urlopen = build_opener(ProxyHandler({})).open
            try:
                for asset in ("/", "/style.css", "/app.js"):
                    with urlopen(base + asset) as response:
                        self.assertEqual(response.status, 200)
                        self.assertGreater(len(response.read()), 100)
                with urlopen(base + "/api/state") as response:
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertEqual(json.load(response)["metrics"][0]["step"], 1)
                for url, expected in [("/../config.json", 404), ("/api/state?limit=oops", 400),
                                      ("/api/state?run=/etc", 400)]:
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(base + url)
                    self.assertEqual(caught.exception.code, expected)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
