"""PPO 训练入口。

与原实现的关键差异：
  * ``make_env`` 改用 ``env_runner.build_hok_env``（正确的官方签名 + addrs + 阵容迭代器）
  * 用 ``EnvRunner.collect_trajectory`` 收集**真实轨迹**，不再用 ``torch.randn`` 假 batch
  * 训练循环真正把轨迹喂给 ``PPOLearner.update``
  * 启动时设置全局随机种子（``train.seed``，原实现是死配置）
  * VRAM 降级回调只做"运行时确实可调"的事情（缩小 minibatch）

运行::

    conda activate hok                                  # Python 3.9
    # 先在 Windows 侧启动 gamecore-server，并确保已跑通 scripts/test_env.py
    python scripts/train_ppo.py --config configs/default.yaml --total-steps 1000
"""

import argparse
import logging
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from hok1v1_rl.action_mask import LABEL_SIZE_LIST
from hok1v1_rl.config import load_config
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.logger import UnifiedLogger
from hok1v1_rl.metrics_store import MetricsStore
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.reward import REWARD_COMPONENT_NAMES
from hok1v1_rl.vram_guard import VRamGuard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("train")

#: 训练轨迹里所有需要拼 batch 的键（除 lstm_state_init 外都是 ndarray）
#: ⚠️ ``head_weights`` 是官方样本里的 weight0..5（= sub_action_mask[选中的 button]），
#: learner 用它屏蔽与该 button 无关的动作头。**必须一起转成 tensor 传进 update()**，
#: 否则 head 屏蔽会静默失效（learner 会告警一次）。
#: ⚠️ ``valid_mask`` 是 env_runner 为"尾部补齐成完整 chunk"打的标记
#: （1=真实决策，0=补齐）。**同样必须透传**，否则 learner 会把补齐的 0
#: 算进 Σw 归一化的分母，把真实样本的梯度稀释掉。
_BATCH_KEYS = (
    "features", "legal_actions", "actions", "log_probs",
    "values", "rewards", "dones", "advantages", "returns",
    "head_weights", "valid_mask",
)


def set_global_seed(seed: int):
    """设置全局随机种子，保证实验可复现（原实现声明了 seed 但从未使用）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("全局随机种子已设为 %d", seed)


def make_vram_degrade_callback(learner_holder: dict):
    """构造 VRAM 降级回调。

    ⚠️ 只有"运行时确实可调"的东西才在这里动：
      * ``ppo.minibatch_size`` —— learner 每次 update 时读取，可安全缩小
      * 网络结构（attn_heads / lstm_hidden）与序列长度是**构造期**参数，
        训练中无法收缩，所以不要假装能降它们。

    返回的字符串会写进日志，便于确认降级是否真的生效。
    """

    def _cb(step: int, used_gb: float) -> str:
        cfg = learner_holder.get("cfg")
        if cfg is None:
            return "cfg 未就绪，未做任何收缩"

        old_mb = cfg.ppo.minibatch_size
        new_mb = max(128, old_mb // 2)
        if new_mb == old_mb:
            return f"minibatch_size 已是下限 {old_mb}，未进一步收缩"

        cfg.ppo.minibatch_size = new_mb
        learner = learner_holder.get("learner")
        if learner is not None:
            # learner 每次 update 都重新读 cfg，无需额外同步
            pass
        logger.warning(
            "VRAM 降级：minibatch_size %d -> %d（step=%d, used=%.2fG）",
            old_mb, new_mb, step, used_gb,
        )
        return f"minibatch_size {old_mb} -> {new_mb}"

    return _cb


def collect_batch(runner: EnvRunner, cfg, min_frames: int):
    """收集至少 ``min_frames`` 帧的轨迹，拼成一个 PPO batch。

    每局单独跑完（``collect_trajectory``），再把若干局的 chunk 沿 batch 维拼接。
    只拼接 chunk 长度（T）相同的轨迹；长度不同的轨迹单独成 batch
    （短于一个 chunk 的整局会产生变长 chunk，见 env_runner 的说明）。
    """
    episodes = []
    total_frames = 0
    t_eff = None
    win_cnt = 0

    while total_frames < min_frames:
        traj = runner.collect_trajectory()
        cur_t = traj["features"].shape[1]

        if t_eff is None:
            t_eff = cur_t
        elif cur_t != t_eff:
            # 长度不一致就不混拼，直接返回当前已收集的部分
            logger.debug("chunk 长度 %d != %d，停止本次拼接", cur_t, t_eff)
            if episodes:
                break

        episodes.append(traj)
        total_frames += traj["features"].shape[0] * cur_t
        win_cnt += int(bool(traj["win"]))

    batch = {}
    for k in _BATCH_KEYS:
        batch[k] = torch.as_tensor(
            np.concatenate([e[k] for e in episodes], axis=0), dtype=torch.float32
        )
    # actions 需要 long（gather 的 index）
    batch["actions"] = batch["actions"].long()

    # lstm_state_init: ((1,B,H), (1,B,H))，沿第 1 维（B）拼接
    h = torch.cat([e["lstm_state_init"][0] for e in episodes], dim=1)
    c = torch.cat([e["lstm_state_init"][1] for e in episodes], dim=1)
    batch["lstm_state_init"] = (h, c)

    # 看板统计：本批对局的奖励分量均值与 6 头动作计数（跨局求和）。
    # ⚠️ 分量现在是**归一化后、已乘权重**的贡献，可直接横向比较谁在主导：
    #    tower=-10 就是字面意思"丢了一整座基地"，win/lose=±50。
    comp_sum = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    comp_sum["time"] = 0.0      # 时间惩罚（step 里追加的分量）
    comp_sum["truncate"] = 0.0  # 截断惩罚（达到 max_frame_num 时非零）
    action_total = [np.zeros(s, dtype=np.int64) for s in LABEL_SIZE_LIST]
    #: 终局可见性统计（P3 的哨兵）：确认"含 done=1 的终局步真的进了 batch"。
    #: 历史 bug 是尾部被整段丢弃，那时这些东西根本无从观测。
    n_decisions = 0
    n_padded = 0
    terminal_in_batch_cnt = 0
    for e in episodes:
        es = e["episode_stats"]
        for name in REWARD_COMPONENT_NAMES:
            comp_sum[name] += float(es["reward_components"][name])
        comp_sum["time"] += float(es["reward_components"].get("time", 0.0))
        comp_sum["truncate"] += float(es["reward_components"].get("truncate", 0.0))
        for head_i, size in enumerate(LABEL_SIZE_LIST):
            action_total[head_i] += np.asarray(
                es["action_hist"][f"head_{head_i}"], dtype=np.int64
            )
        n_decisions += int(es.get("episode_len", 0))
        n_padded += int(es.get("chunk_padded_steps", 0))
        if es.get("terminal_in_batch"):
            terminal_in_batch_cnt += 1
        # 每一局都应当有终局步（自然结束 done=1，或截断时手动置 done=1），
        # 所以"应有终局的局数"就等于局数本身。
        episodes_expect_terminal = len(episodes)

    stats = {
        "frames": total_frames,
        "episodes": len(episodes),
        "win_rate": win_cnt / max(1, len(episodes)),
        "avg_episode_len": float(np.mean([e["episode_len"] for e in episodes])),
        # ---- 终局可见性（P3 的哨兵）----
        "decisions": n_decisions,
        "chunk_padded_steps": n_padded,
        "valid_ratio": (
            1.0 - n_padded / max(1, n_decisions + n_padded)
        ),
        "episodes_expect_terminal": episodes_expect_terminal,
        "terminal_in_batch": terminal_in_batch_cnt,
        "reward_components": {
            name: comp_sum[name] / max(1, len(episodes))
            for name in list(comp_sum)   # 含 time（时间惩罚）
        },
        "action_hist": {
            f"head_{head_i}": action_total[head_i].tolist()
            for head_i in range(len(LABEL_SIZE_LIST))
        },
    }
    return batch, stats


def main():
    """训练入口：装配环境/网络/learner，然后跑 PPO 主循环。

    一次 ``update`` 的流程（对应下面 for 循环里的 1)~6)）：
      1) 采样：``collect_batch`` 反复调用 ``collect_trajectory`` 直到凑够
         ``ppo.batch_size`` 个决策，把多个整局的 chunk 沿 batch 维拼起来
      2) 标准化：advantage 做 batch 内 z-score；returns 走 RunningRewardNormalizer
      3) 更新：``learner.update`` 按 ``ppo.ppo_epochs`` × minibatch 做 SGD
      4) 评估：每 ``eval_interval_steps`` 步跑 ``eval_episodes`` 局确定性对局
      5) 落盘：指标写 metrics.jsonl（看板数据源）
      6) 存档：每 ``save_interval_steps`` 步存一次 checkpoint

    ⚠️ ``--total-steps`` 是 **LR/熵退火的基准**，不是"跑够这么多就停"。当前
    默认 ``lr_end == lr``、``entropy_beta_start == end``（都不退火，对齐官方），
    所以传小值做短测也不再把学习率压到 0；但若日后重新打开退火，请把它设成
    **计划步数的 2~3 倍**，否则跑完后段会失去学习能力。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--total-steps", type=int, default=10000,
                        help="PPO 更新总步数（每次 update 消费一个 batch）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    # ---- 并行运行所需的三组隔离参数 -------------------------------------
    # 同一时间只允许一个进程用相同的 runtime_id（gamecore 侧按它区分对局），
    # 且 zmq 端口段不能重叠；log-dir 必须隔离，否则多个实例会往同一个
    # metrics.jsonl / checkpoint 目录里写，把指标混成一条乱曲线。
    parser.add_argument("--runtime-id", default=None,
                        help="覆盖 env.runtime_id（并行实例必须各不相同；不能含下划线）")
    parser.add_argument("--zmq-port", type=int, default=None,
                        help="覆盖 env.zmq_port_begin（并行实例必须错开，每个实例占 2 个连续端口）")
    parser.add_argument("--gamecore-addr", default=None,
                        help="覆盖 env.gamecore_server_addr（跑多个 gamecore-server 时用）")
    parser.add_argument("--log-dir", default=None,
                        help="覆盖 train.log_dir（并行实例必须隔离，存 metrics.jsonl 与 ckpt）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path
    cfg = load_config(str(cfg_path))

    # CLI 覆盖必须在这里做：下面的 UnifiedLogger / MetricsStore / build_hok_env
    # 都会读 cfg，晚于这一步就晚了（尤其 log_dir 决定 metrics.jsonl 落哪）。
    overrides = []
    if args.runtime_id:
        cfg.env.runtime_id = args.runtime_id
        overrides.append("env.runtime_id=%s" % args.runtime_id)
    if args.zmq_port is not None:
        cfg.env.zmq_port_begin = args.zmq_port
        overrides.append("env.zmq_port_begin=%d" % args.zmq_port)
    if args.gamecore_addr:
        cfg.env.gamecore_server_addr = args.gamecore_addr
        overrides.append("env.gamecore_server_addr=%s" % args.gamecore_addr)
    if args.log_dir:
        cfg.train.log_dir = args.log_dir
        cfg.train.tb_log_dir = str(Path(args.log_dir) / "tb")
        overrides.append("train.log_dir=%s" % args.log_dir)

    logger.info("配置已从 %s 加载", cfg_path)
    if overrides:
        logger.info("命令行覆盖：%s", "  ".join(overrides))
    logger.info(
        "num_envs=%d batch_size=%d minibatch_size=%d seq_len=%d amp=%s",
        cfg.env.num_envs, cfg.ppo.batch_size, cfg.ppo.minibatch_size,
        cfg.network.lstm_time_steps, cfg.train.amp,
    )

    if cfg.env.num_envs != 1:
        logger.warning(
            "当前 EnvRunner 只实现单环境；cfg.env.num_envs=%d 将被忽略（按 1 运行）。"
            "多环境需要为每个环境 fork 独立子进程（interface 是进程内单例）。",
            cfg.env.num_envs,
        )

    set_global_seed(cfg.train.seed)

    # ---- 设备 ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("设备：%s", device)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        logger.info("  GPU: %s capability=%s", torch.cuda.get_device_name(0), cap)
        logger.info("  torch 架构列表: %s", torch.cuda.get_arch_list())
        logger.info("  VRAM 总量: %.2f GB",
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
        # 提前暴露 sm_120 不匹配问题（is_available() 为 True 也可能是假象）
        try:
            _t = torch.randn(64, 64, device=device)
            _ = (_t @ _t).sum().item()
            torch.cuda.synchronize()
        except Exception as exc:      # noqa: BLE001
            logger.error("GPU kernel 自检失败，请先修复 torch 版本：%s", exc)
            sys.exit(1)

    # ---- 构建模块 ----
    network = HoK1v1Network(cfg).to(device)
    logger.info("网络参数量：%.2f M", network.count_params() / 1e6)

    unified_logger = UnifiedLogger(
        tb_log_dir=cfg.train.tb_log_dir,
        wandb_project=cfg.train.wandb_project,
        wandb_enabled=cfg.train.wandb_enabled,
        config=cfg.to_dict(),
    )
    metrics_store = MetricsStore(str(Path(cfg.train.log_dir) / "metrics.jsonl"))

    learner_holder = {"cfg": cfg}
    vram_guard = VRamGuard(
        warn_gb=cfg.hardware.vram_warn_gb,
        degrade_gb=cfg.hardware.vram_degrade_gb,
        oom_gb=cfg.hardware.vram_oom_gb,
        degrade_callback=make_vram_degrade_callback(learner_holder),
    )

    learner = PPOLearner(network, cfg, unified_logger, vram_guard)
    learner.set_lr_schedule(total_steps=args.total_steps)
    learner_holder["learner"] = learner

    if args.resume:
        learner.load(args.resume)

    # ---- 构建环境 ----
    try:
        env = build_hok_env(cfg, runtime_id=cfg.env.runtime_id)
    except Exception as exc:      # noqa: BLE001
        logger.error("环境初始化失败：%s: %s", type(exc).__name__, exc)
        logger.error("请先启动 Windows 侧 gamecore-server，并运行 scripts/test_env.py 验证。")
        sys.exit(1)

    # camp_config=None 时 EnvRunner 内部按官方迭代器每局轮换英雄
    runner = EnvRunner(env, network, cfg, device)

    # ---- 训练循环 ----
    logger.info("启动训练循环（total_steps=%d）...", args.total_steps)
    fps_window = []
    step = 0

    try:
        for step in range(args.total_steps):
            t0 = time.time()

            # 1) 采样：收集至少 batch_size 帧
            batch, stats = collect_batch(runner, cfg, cfg.ppo.batch_size)

            # 2) PPO 更新（可能因 VRAM OOM 抛错）
            try:
                result = learner.update(batch, step=step)
            except RuntimeError as exc:
                if "VRamGuard OOM" in str(exc):
                    logger.error("OOM @ step %d：%s", step, exc)
                    logger.error("建议：调小 ppo.minibatch_size 或 batch_size 后重跑。")
                    break
                raise

            # 3) 吞吐与日志
            dt = max(1e-6, time.time() - t0)
            fps_window.append(stats["frames"] / dt)
            if len(fps_window) > 50:
                fps_window.pop(0)

            unified_logger.log_fps(np.mean(fps_window), stats["frames"], step)
            unified_logger.log_metrics({
                "env/win_rate": stats["win_rate"],
                "env/avg_episode_len": stats["avg_episode_len"],
                "env/frames_per_update": stats["frames"],
            }, step)

            # 4) 周期性评估：确定性动作，不打入训练 batch
            eval_metrics = {}
            if (
                step > 0
                and cfg.train.eval_interval_steps > 0
                and step % cfg.train.eval_interval_steps == 0
            ):
                eval_wins, eval_lens = [], []
                for _ in range(cfg.train.eval_episodes):
                    try:
                        et = runner.collect_trajectory(deterministic=True)
                    except Exception as exc:      # noqa: BLE001
                        logger.warning("评估对局失败（跳过本次评估）：%s", exc)
                        break
                    eval_wins.append(int(et["win"]))
                    eval_lens.append(int(et["episode_len"]))
                if eval_wins:
                    eval_metrics = {
                        "eval/win_rate": sum(eval_wins) / len(eval_wins),
                        "eval/avg_episode_len": float(np.mean(eval_lens)),
                    }
                    unified_logger.log_metrics(eval_metrics, step)

            # 5) 写入指标存储（看板数据源，一步一行）
            vram = vram_guard.check(step=step)
            record = {
                "train/policy_loss": result.policy_loss,
                "train/value_loss": result.value_loss,
                "train/entropy": result.entropy,
                "train/kl": result.kl,
                "train/grad_norm": result.grad_norm,
                "train/lr": result.lr,
                "train/clip_fraction": result.clip_fraction,
                "train/explained_variance": result.explained_variance,
                "train/entropy_beta": result.entropy_beta,
                "env/win_rate": stats["win_rate"],
                "env/avg_episode_len": stats["avg_episode_len"],
                "env/frames_per_update": stats["frames"],
                "res/fps": float(np.mean(fps_window)),
                "res/vram_warn_gb": vram_guard.warn_gb,
                "res/vram_degrade_gb": vram_guard.degrade_gb,
                "res/vram_oom_gb": vram_guard.oom_gb,
                **{f"res/{k}": v for k, v in vram.to_dict().items()},
                **eval_metrics,
            }
            for name in REWARD_COMPONENT_NAMES:
                record[f"env/reward/{name}"] = stats["reward_components"][name]
            if "time" in stats["reward_components"]:
                record["env/reward/time"] = stats["reward_components"]["time"]
            # ⚠️ truncate 不在 REWARD_COMPONENT_NAMES 里（它不是 env 给的 8 分量之一，
            # 而是达到 max_frame_num 时我们追加的惩罚）。旧代码因此**永远不记录它**，
            # 导致"拖局被截断扣 -50"这件事在日志里完全不可见。
            if "truncate" in stats["reward_components"]:
                record["env/reward/truncate"] = stats["reward_components"]["truncate"]

            # ---- 终局可见性 / 有效性（P3 的哨兵）----
            record["env/chunk_padded_steps"] = stats["chunk_padded_steps"]
            record["env/valid_ratio"] = stats["valid_ratio"]
            record["env/terminal_in_batch"] = (
                stats["terminal_in_batch"] / max(1, stats["episodes_expect_terminal"])
            )
            record["train/valid_ratio"] = result.valid_ratio
            # 哨兵指标：采样侧与训练侧是不是同一个前向（正常 ≈1e-5，>0.05 就要停）。
            # 它同时进 TB 与 metrics.jsonl，这样看板和日志都能盯它。
            record["train/old_new_logp_gap"] = result.old_new_logp_gap

            for head_i, hist in stats["action_hist"].items():
                for a, cnt in enumerate(hist):
                    record[f"env/act_dist/{head_i}/{a}"] = float(cnt)
            metrics_store.add(step, record)

            # 6) 周期性 checkpoint
            if (
                step > 0
                and cfg.train.save_interval_steps > 0
                and step % cfg.train.save_interval_steps == 0
            ):
                learner.save(str(Path(cfg.train.log_dir) / f"ckpt_step_{step}.pt"))

            if step % cfg.train.log_interval_steps == 0:
                logger.info(
                    "step=%d frames=%d eps=%d win=%.2f | ploss=%.4f vloss=%.4f "
                    "kl=%.4f ent=%.4f beta=%.4f gnorm=%.2f lr=%.2e mb=%d fps=%.1f",
                    step, stats["frames"], stats["episodes"], stats["win_rate"],
                    result.policy_loss, result.value_loss, result.kl,
                    result.entropy, result.entropy_beta, result.grad_norm,
                    result.lr, result.num_minibatches, float(np.mean(fps_window)),
                )

    except KeyboardInterrupt:
        logger.info("收到中断，保存当前 checkpoint 后退出")
    finally:
        # ---- 先把未处理的异常落盘 ----
        # close_game 可能长时间阻塞（见 env_runner.close 的看门狗说明），
        # 若不先记录，traceback 会被压在大量日志之后甚至永远不打印
        exc_info = sys.exc_info()
        if exc_info[0] is not None and exc_info[0] is not KeyboardInterrupt:
            logger.error("训练异常退出：%s: %s", exc_info[0].__name__, exc_info[1])
            logger.error("traceback:\n%s", traceback.format_exc())

        # ---- 保存最终模型 ----
        ckpt_path = Path(cfg.train.log_dir) / "final_ckpt.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        learner.save(str(ckpt_path))
        runner.close()
        unified_logger.close()
        logger.info("训练结束（共 %d 次 update）。checkpoint: %s", step + 1, ckpt_path)


if __name__ == "__main__":
    main()
