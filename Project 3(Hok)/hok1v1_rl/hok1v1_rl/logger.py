"""统一 logger —— 同时封装 TensorBoard 与 WandB。

记录：
  - 训练指标：policy_loss, value_loss（按 head）, entropy, KL, grad_norm, LR
  - 资源指标：vram_used_gb, vram_peak_gb, fps, env_steps_per_sec
  - 评估指标：win_rate, elo_rating, avg_reward（Phase 4+）
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class UnifiedLogger:
    """以 TensorBoard 为主、可选同步到 WandB 的 logger。

    Args:
        tb_log_dir: TensorBoard 日志目录
        wandb_project: WandB 项目名（None 禁用）
        wandb_enabled: 显式开关
        config: dict，写入 WandB 的配置信息
    """

    def __init__(
        self,
        tb_log_dir: str,
        wandb_project: str = "hok1v1_rl",
        wandb_enabled: bool = False,
        config: Optional[Dict] = None,
    ):
        import os
        os.makedirs(tb_log_dir, exist_ok=True)
        self.tb_log_dir = tb_log_dir
        self._tb = None
        self._wandb = None

        # 优先初始化 TensorBoard：本地调试必需，且无网络依赖
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=tb_log_dir)
        except Exception as e:
            logger.warning(f"TensorBoard 初始化失败：{e}")

        # WandB 可选：需网络与登录，单机调试阶段通常关闭
        if wandb_enabled:
            try:
                import wandb
                self._wandb = wandb.init(
                    project=wandb_project, config=config or {}, reinit=True
                )
            except Exception as e:
                logger.warning(f"WandB 初始化失败：{e}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """在指定 step 记录 dict 形式的指标。

        Args:
            metrics: name -> scalar value 的 dict
            step: 全局步数（env 帧数或 PPO 更新数）
        """
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, float(v), step)
                except Exception as e:
                    logger.debug(f"TB 记录 {k} 失败：{e}")
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"WandB 记录失败：{e}")

    def log_fps(self, fps: float, env_steps: int, step: int):
        """记录吞吐指标。"""
        self.log_metrics({
            "fps": fps,
            "env_steps_per_sec": fps,
            "total_env_steps": env_steps,
        }, step)

    def log_vram(self, status_dict: Dict[str, float], step: int):
        """记录 VRAM 状态。"""
        self.log_metrics(status_dict, step)

    def log_train_step(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl: float,
        grad_norm: float,
        lr: float,
        step: int,
        extra: Optional[Dict] = None,
    ):
        """记录单步 PPO 训练指标。

        所有指标加 train/ 前缀，便于在 TensorBoard 中分组显示。
        """
        metrics = {
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/entropy": entropy,
            "train/kl": kl,
            "train/grad_norm": grad_norm,
            "train/lr": lr,
        }
        if extra:
            for k, v in extra.items():
                metrics[f"train/{k}"] = float(v)
        self.log_metrics(metrics, step)

    def alert(self, msg: str, level: str = "warning"):
        """发送告警：WandB 可用时同步推送；否则仅本地日志。"""
        if self._wandb is not None:
            try:
                import wandb
                wandb.alert(text=msg, level=level)
            except Exception:
                pass
        getattr(logger, level, logger.warning)(msg)

    def close(self):
        """关闭所有 logger 句柄。"""
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
