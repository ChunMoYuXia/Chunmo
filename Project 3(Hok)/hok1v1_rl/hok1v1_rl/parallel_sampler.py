"""独立进程采样，同一批使用同一份策略权重，收齐后才允许 PPO 更新。

主进程安排阵容、合并轨迹、写日志；子进程各自持有一个 native Interface。
进程间只传 CPU numpy 数据，避免共享 CUDA 张量及其生命周期问题。
"""
from collections import deque
import copy
import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import random
import signal
import time
import traceback

import numpy as np
import torch

from .config import Config
from .env_runner import EnvRunner, build_hok_env, bypass_proxy_for_localhost
from .evaluation import summarize_games
from .matchups import TrainingSchedule, evaluation_schedule
from .network import HoK1v1Network
from .rollout import EpisodeCollector, join_trajectories

logger = logging.getLogger(__name__)


def make_runner(cfg, network, device):
    return EnvRunner(build_hok_env(cfg, cfg.env.runtime_id), network, cfg, device)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().numpy().copy(),
            "cuda": [s.cpu().numpy().copy() for s in torch.cuda.get_rng_state_all()]
                    if device.type == "cuda" else []}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8) for s in state["cuda"]])


def sampler_worker(connection, config, index, device_name, saved_rng, runner_factory,
                   inference_mode, fragment_length, burn_in):
    """常驻子进程：同步策略，返回整局或可续接片段；从不更新模型。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C 由主进程统一收尾。
    logging.basicConfig(level=logging.INFO, force=True,
                        format=f"%(asctime)s sampler={index} %(levelname)s | %(message)s")
    runner, collector = None, None
    try:
        cfg, device = Config.from_dict(config), torch.device(device_name)
        torch.set_num_threads(1)
        seed = cfg.train.seed + index + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        network = HoK1v1Network(cfg).to(device).eval()
        runner = runner_factory(cfg, network, device)
        if saved_rng:
            restore_rng(saved_rng, device)
        last_progress = 0.0

        def progress(**changes):
            nonlocal last_progress
            now = time.monotonic()
            if changes.get("episode_decisions") == 0 or now - last_progress >= 2:
                connection.send({"kind": "progress", "changes": changes})
                last_progress = now

        runner.progress_callback = progress
        connection.send({"kind": "ready", "rng": rng_state(device)})
        version = None
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                interrupted = collector.summary(interrupted=True) if collector and collector.active else None
                if interrupted:
                    interrupted.update(worker_id=index, sampler_device=str(device),
                                       sampler_inference=inference_mode)
                runner.close()
                runner = None
                connection.send({"kind": "closed", "interrupted_episode": interrupted,
                                 "rng": rng_state(device)})
                break
            if command["kind"] == "weights":
                network.load_state_dict({name: torch.from_numpy(array) for name, array in command["weights"].items()})
                runner.prepare_inference(inference_mode)
                memory_seconds = collector.refresh_memory() if collector and version != command["version"] else 0.0
                version = command["version"]
                connection.send({"kind": "loaded", "version": version,
                                 "memory_refresh_seconds": memory_seconds})
                continue
            if command["kind"] != "episode" or command["version"] != version:
                raise RuntimeError("采样任务与已加载策略版本不一致")
            evaluation = command["evaluation"]
            training_rng = rng_state(device) if evaluation else None
            try:
                if evaluation:
                    seed = cfg.evaluation.seed + command["job"]
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                if fragment_length and not evaluation:
                    if collector is None or not collector.active:
                        if command["matchup"] is None:
                            raise RuntimeError("新对局缺少阵容")
                        collector = EpisodeCollector(runner, command["matchup"], burn_in=burn_in)
                    elif command["matchup"] is not None:
                        raise RuntimeError("未结束的训练局不能被新阵容覆盖")
                    safety_limit = cfg.env.max_frame_num // cfg.env.predict_frequency + 8
                    trajectory = collector.collect(min(fragment_length, safety_limit-collector.decisions),
                                                    version=version)
                else:
                    if collector and collector.active:
                        raise RuntimeError("评估必须使用独立环境，不能打断训练局")
                    trajectory = runner.collect_trajectory(matchup=command["matchup"],
                        deterministic=evaluation, evaluation=evaluation)
            finally:
                if training_rng is not None:
                    restore_rng(training_rng, device)
            if evaluation:
                # 快评只要逐局统计，不把整局训练张量传回父进程。
                trajectory = {key: trajectory[key] for key in ("episode_stats", "episode_len")}
            else:
                trajectory["lstm_state_init"] = tuple(s.cpu().numpy() for s in trajectory["lstm_state_init"])
            for key in ("episode_stats", "sample_stats"):
                if trajectory.get(key):
                    trajectory[key].update(worker_id=index, policy_version=version,
                                           sampler_device=str(device), sampler_inference=inference_mode)
            connection.send({"kind": "episode", "version": version, "job": command["job"],
                             "trajectory": trajectory, "rng": rng_state(device)})
    except BaseException:
        try:
            connection.send({"kind": "error", "traceback": traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                logger.exception("采样进程清理失败")
        connection.close()


class ParallelSampler:
    """同步 PPO：快进程多接任务，达到下限后收完在途整局或片段。"""
    def __init__(self, cfg, network, workers=2, device="cpu", saved_state=None,
                 runner_factory=make_runner, timeout=180, inference_mode="eager",
                 fragment_length=0, burn_in=64):
        if workers < 2 or cfg.env.zmq_port_begin + 2*workers - 1 > 65535:
            raise ValueError("并行采样至少需要 2 个进程，且每个进程需要两个有效端口")
        if saved_state and (saved_state["workers"] != workers or saved_state["device"] != device):
            raise ValueError("并行续训需保持 samplers 和 sampler-device 一致；更换时请用 --init-from")
        if inference_mode not in ("eager", "jit") or (inference_mode == "jit" and torch.device(device).type != "cpu"):
            raise ValueError("jit 推理仅支持 CPU 采样；其他情况请使用 eager")
        if saved_state and saved_state.get("inference_mode", "eager") != inference_mode:
            raise ValueError("严格续训需保持 sampler-inference 一致；更换时请用 --init-from")
        if fragment_length < 0 or (fragment_length and fragment_length % cfg.network.lstm_time_steps) or burn_in < 0:
            raise ValueError("fragment-length 应为 0 或 LSTM 序列长度的正整数倍，burn-in >= 0")
        if fragment_length and cfg.env.max_decisions:
            raise ValueError("片段模式不使用 max-decisions；片段结束后继续原局")
        if saved_state and (saved_state.get("fragment_length", 0) != fragment_length or
                            (fragment_length and saved_state.get("burn_in", 64) != burn_in)):
            raise ValueError("严格续训需保持 fragment-length 和 burn-in 一致；更换时请用 --init-from")
        self.cfg, self.network, self.device, self.timeout = cfg, network, device, timeout
        self.inference_mode = inference_mode
        self.fragment_length, self.burn_in = fragment_length, burn_in
        self.runner_factory = runner_factory
        self.evaluation_pool = None
        self.collection_samples, self.interrupted_episodes = [], []
        self.collection_metrics = {}
        self.schedule = TrainingSchedule(cfg.env.hero_ids, cfg.env.opponent_hero_ids, cfg.train.seed)
        self.progress_callback = None
        self.workers, self.progress = [], {}
        self.remote_cleanup = runner_factory is make_runner
        self.closed = False
        context = mp.get_context("spawn")  # 不 fork 已经初始化的 CUDA/native 状态。
        try:
            for index in range(workers):
                child_cfg = copy.deepcopy(cfg)
                child_cfg.env.runtime_id = f"{cfg.env.runtime_id}-p{index}"
                child_cfg.env.zmq_port_begin += 2*index
                parent, child = context.Pipe()
                saved_rng = saved_state["rng"][index] if saved_state else None
                process = context.Process(target=sampler_worker, args=(child, child_cfg.to_dict(), index,
                    device, saved_rng, runner_factory, inference_mode, fragment_length, burn_in),
                    name=f"hok-sampler-{index}")
                process.start()
                child.close()
                self.workers.append({"process": process, "pipe": parent, "cfg": child_cfg,
                                     "last_seen": time.monotonic(), "rng": None,
                                     "needs_matchup": True, "episode_decisions": 0})
            pending = set(range(workers))
            while pending:
                index, message = self._receive(pending)
                if message["kind"] != "ready":
                    raise RuntimeError("采样进程没有正常就绪")
                self.workers[index]["rng"] = message["rng"]
                pending.remove(index)
        except BaseException:
            self.close()
            raise

    def state_dict(self):
        return {"workers": len(self.workers), "device": self.device,
                "inference_mode": self.inference_mode,
                "fragment_length": self.fragment_length, "burn_in": self.burn_in,
                "ongoing_games_restored": False,
                "ongoing_games": [{"worker_id": i, "runtime_id": w["cfg"].env.runtime_id,
                                   "episode_decisions": w["episode_decisions"],
                                   "sdk_client_id": self.progress.get(str(i), {}).get("sdk_client_id")}
                                  for i, w in enumerate(self.workers) if not w["needs_matchup"]],
                "rng": [worker["rng"] for worker in self.workers]}

    def _notify(self, index=None, **changes):
        if index is not None:
            self.progress.setdefault(str(index), {}).update(changes)
        if self.progress_callback:
            self.progress_callback(sampler_count=len(self.workers), workers=copy.deepcopy(self.progress))

    def _receive(self, pending):
        """传播子进程异常；定时刷新心跳，进程消失或长时间无响应时拒绝这批数据。"""
        while True:
            pipes = [self.workers[i]["pipe"] for i in pending]
            for pipe in wait(pipes, timeout=1):
                index = next(i for i in pending if self.workers[i]["pipe"] is pipe)
                worker = self.workers[index]
                try:
                    message = pipe.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"采样进程 {index} 的连接已断开") from exc
                worker["last_seen"] = time.monotonic()
                if message["kind"] == "error":
                    raise RuntimeError(f"采样进程 {index} 失败：\n{message['traceback']}")
                if message["kind"] == "progress":
                    self._notify(index, **message["changes"])
                    continue
                return index, message
            self._notify()
            for i in pending:
                worker = self.workers[i]
                if not worker["process"].is_alive():
                    raise RuntimeError(f"采样进程 {i} 已退出，exitcode={worker['process'].exitcode}")
                if time.monotonic() - worker["last_seen"] > self.timeout:
                    raise TimeoutError(f"采样进程 {i} 超过 {self.timeout} 秒没有响应")

    def _load_weights(self, version):
        if self.closed:
            raise RuntimeError("采样池已关闭")
        weights = {name: value.detach().cpu().numpy().copy() for name, value in self.network.state_dict().items()}
        pending = set(range(len(self.workers)))
        self.memory_refresh_seconds = 0.0
        for worker in self.workers:
            worker["pipe"].send({"kind": "weights", "version": version, "weights": weights})
            worker["last_seen"] = time.monotonic()
        while pending:
            index, message = self._receive(pending)
            if message["kind"] != "loaded" or message["version"] != version:
                raise RuntimeError("采样进程未确认当前策略权重")
            self.memory_refresh_seconds += message.get("memory_refresh_seconds", 0.0)
            pending.remove(index)

    def _collect(self, version, min_decisions=None, matchups=None, on_episode=None, on_sample=None):
        sync_started = time.monotonic()
        self._load_weights(version)
        sync_seconds = time.monotonic() - sync_started
        collecting_started = time.monotonic()
        evaluation = matchups is not None
        remaining = deque(enumerate(matchups)) if evaluation else None
        pending, trajectories, in_flight, decisions, job = set(), {}, {}, 0, 0
        dispatched_workers = set()
        dispatched_at, busy_seconds = {}, 0.0
        threshold_at = None

        def dispatch(index):
            nonlocal job
            if evaluation:
                if not remaining:
                    return
                task, matchup = remaining.popleft()
            else:
                if decisions >= min_decisions:
                    return
                if self.fragment_length and self.workers[index]["needs_matchup"] and index in dispatched_workers:
                    # 本批刚终局的进程先让旧引擎自然退出；不要把 10+ 秒的 reset
                    # 塞进最后一个片段。其余仍在对局中的进程可以继续采样。
                    return
                task = job
                matchup = self.schedule.next() if self.workers[index]["needs_matchup"] else None
                job += 1
            worker = self.workers[index]
            worker["pipe"].send({"kind": "episode", "version": version, "job": task,
                                 "matchup": matchup, "evaluation": evaluation})
            worker["last_seen"] = time.monotonic()
            pending.add(index)
            dispatched_workers.add(index)
            in_flight[index] = task
            dispatched_at[index] = time.monotonic()
            self._notify(index, phase="evaluating" if evaluation else "sampling",
                         episode_decisions=0 if matchup else worker["episode_decisions"],
                         policy_version=version, **(matchup.to_dict() if matchup else {}))

        for index in range(len(self.workers)):
            dispatch(index)
        while pending or (not evaluation and decisions < min_decisions):
            if not pending:
                # 所有对局都结束而 batch 仍不足，才一起开始下一轮新局。
                dispatched_workers.clear()
                for index in range(len(self.workers)):
                    dispatch(index)
            index, message = self._receive(pending)
            received_at = time.monotonic()
            if (message["kind"] != "episode" or message["version"] != version or
                    message["job"] != in_flight[index]):
                raise RuntimeError("收到旧策略数据或无效采样结果，拒绝 PPO 更新")
            pending.remove(index)
            busy_seconds += received_at - dispatched_at.pop(index)
            del in_flight[index]
            trajectory = message["trajectory"]
            if not evaluation:
                trajectory["lstm_state_init"] = tuple(torch.from_numpy(s) for s in trajectory["lstm_state_init"])
            trajectories[message["job"]] = trajectory
            decisions += trajectory["episode_len"]
            if not evaluation and threshold_at is None and decisions >= min_decisions:
                threshold_at = received_at
            self.workers[index]["rng"] = message["rng"]
            game = trajectory.get("episode_stats")
            self.workers[index]["needs_matchup"] = game is not None
            sample = trajectory.get("sample_stats") or game
            episode_decisions = sample.get("episode_decisions_end", trajectory["episode_len"])
            self.workers[index]["episode_decisions"] = episode_decisions
            self._notify(index, phase="idle", episode_decisions=episode_decisions)
            if on_sample:
                on_sample(sample)
            if on_episode and game:
                on_episode(trajectory["episode_stats"])
            dispatch(index)
        finished_at = time.monotonic()
        capacity_seconds = (finished_at - collecting_started) * len(self.workers)
        # 这是父进程观察到的等待时间；CPU/GPU 利用率需要由操作系统另行测量。
        idle_seconds = max(0.0, capacity_seconds - busy_seconds)
        self.collection_metrics = {
            "res/weight_sync_seconds": sync_seconds,
            "res/memory_refresh_seconds": self.memory_refresh_seconds,
            "res/batch_drain_seconds": finished_at - threshold_at if threshold_at else 0.0,
            "res/sampler_idle_seconds": idle_seconds,
            "res/sampler_idle_fraction": idle_seconds / max(capacity_seconds, 1e-8),
            "res/batch_overshoot_ratio": decisions / min_decisions if not evaluation else 1.0,
        }
        return [trajectories[key] for key in sorted(trajectories)]

    def collect_batch(self, min_decisions, version, on_episode=None, on_sample=None):
        if min_decisions <= 0:
            raise ValueError("batch 决策数必须为正")
        trajectories = self._collect(version, min_decisions=min_decisions,
                                     on_episode=on_episode, on_sample=on_sample)
        self.collection_samples = [t["sample_stats"] for t in trajectories]
        return join_trajectories(trajectories), [t["episode_stats"] for t in trajectories if t["episode_stats"]]

    def evaluate(self, episodes, version, on_episode=None):
        if self.fragment_length:
            if self.evaluation_pool is None:
                cfg = copy.deepcopy(self.cfg)
                cfg.env.runtime_id += "-eval"
                cfg.env.zmq_port_begin += 2*len(self.workers)
                workers = min(len(self.workers), max(2, episodes))
                self.evaluation_pool = ParallelSampler(cfg, self.network, workers, self.device,
                    runner_factory=self.runner_factory, timeout=self.timeout,
                    inference_mode=self.inference_mode)
            self.evaluation_pool.progress_callback = self.progress_callback
            return self.evaluation_pool.evaluate(episodes, version, on_episode)
        cfg = self.cfg
        matchups = evaluation_schedule(cfg.env.hero_ids, cfg.env.opponent_hero_ids, cfg.evaluation.seed, episodes)
        trajectories = self._collect(version, matchups=matchups, on_episode=on_episode)
        summary = summarize_games([t["episode_stats"] for t in trajectories],
                                  cfg.evaluation.goal_win_rate, cfg.evaluation.final_episodes)
        summary.update(seed=cfg.evaluation.seed, opponent="common_ai", action_selection="argmax",
                       hero_ids=cfg.env.hero_ids, opponent_hero_ids=cfg.env.opponent_hero_ids)
        return summary

    def close(self):
        if self.closed:
            return
        self.closed = True
        for worker in self.workers:
            try:
                worker["pipe"].send({"kind": "close"})
            except (OSError, EOFError):
                pass
        # 所有进程同时收尾，总等待时间不随进程数增加。
        deadline = time.monotonic() + 45
        confirmed_closed = set()
        pending = {i for i, w in enumerate(self.workers) if w["process"].is_alive()}
        while pending and time.monotonic() < deadline:
            pipes = [self.workers[i]["pipe"] for i in pending]
            for pipe in wait(pipes, timeout=.2):
                index = next(i for i in pending if self.workers[i]["pipe"] is pipe)
                try:
                    # 中断时仍可能返回最后一局的大轨迹；读掉它才能让子进程处理 close。
                    message = pipe.recv()
                    if message["kind"] == "closed":
                        if message.get("interrupted_episode"):
                            self.interrupted_episodes.append(message["interrupted_episode"])
                        if message.get("rng"):
                            self.workers[index]["rng"] = message["rng"]
                        confirmed_closed.add(index)
                        pending.remove(index)
                except (EOFError, OSError):
                    pending.remove(index)
            pending = {i for i in pending if self.workers[i]["process"].is_alive()}
        for index, worker in enumerate(self.workers):
            worker["process"].join(max(0, deadline-time.monotonic()))
            if worker["process"].is_alive():
                worker["process"].terminate()
                worker["process"].join(3)
                logger.warning("采样进程未正常退出，已终止：%s", worker["cfg"].env.runtime_id)
            if self.remote_cleanup and index not in confirmed_closed:
                self._stop_remote_game(worker["cfg"])
            worker["pipe"].close()
        if self.evaluation_pool:
            self.evaluation_pool.close()

    @staticmethod
    def _stop_remote_game(cfg):
        # native 接收卡住时由父进程清理对应 runtime，避免遗留对局继续占用端口。
        import requests
        bypass_proxy_for_localhost()
        try:
            response = requests.post(f"http://{cfg.env.gamecore_server_addr}/v2/stopGame",
                                     json={"runtime_id": cfg.env.runtime_id}, timeout=5)
            response.raise_for_status()
        except Exception:
            logger.exception("清理采样进程的 gamecore 失败：%s", cfg.env.runtime_id)
