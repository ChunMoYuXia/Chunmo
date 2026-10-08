"""VRam Guard —— 显存监控与优雅降级。

────────────────────────────────────────────────────────────────────────────
原实现的问题（C9）
────────────────────────────────────────────────────────────────────────────
1. **降级是空操作**：``train_ppo.vram_degrade_callback`` 只打了一条日志
   就返回 ``"logged_only_skeleton"``，注释里的三级降级
   （num_envs → seq_len → attn_heads）一行都没实现，
   但 config 里却明确承诺了这个行为。
2. **降级在物理上做不到**：``attn_heads`` / ``lstm_time_steps`` 都是
   网络构造期参数，训练中出现显存压力时**无法**通过改配置来收缩，
   只有 ``num_envs``（采样侧）与 ``batch/minibatch``（learner 侧）是可调的。
3. **低估显存**：``torch.cuda.memory_allocated()`` 只统计"已分配给张量"的部分，
   不含 PyTorch 缓存分配器 **reserved** 的部分，而 reserved 同样占用物理显存。
   只看 allocated 会在真正 OOM 前毫无察觉。这里改为同时报告并取较大者。
4. **检查时机太晚**：原实现只在 ``learner.update()`` 末尾检查，
   等于"出事之后才报警"。本实现提供 ``check``（可放在 update 前）与
   ``check_after_step`` 两个入口。

本模块现在的定位：
  * 如实报告 allocated / reserved / peak
  * 超阈值时调用**调用方提供的**回调，由调用方决定能做什么
  * 回调返回的字符串会被记录，便于排查"降级到底做没做"

⚠️ 注意 ``oom_gb`` 不要贴近物理上限：RTX 5070 标称 12 GB（实测 12227 MiB），
缓存分配器的碎片会让实际可用显存低于标称值。11.5 GB 已相当激进，
建议先按 10.5 / 11.0 观察。
"""

from dataclasses import dataclass
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VRamStatus:
    """一次显存探测的快照。"""
    used_gb: float          # allocated：真正分配给张量的显存
    reserved_gb: float      # reserved：缓存分配器保留的显存（>= used）
    peak_gb: float          # 历史峰值（allocated）
    threshold_warn: float
    threshold_degrade: float
    threshold_oom: float
    degraded: bool = False
    degrade_action: Optional[str] = None

    def to_dict(self):
        """转成可直接喂给 logger 的扁平 dict。"""
        return {
            "vram_used_gb": self.used_gb,
            "vram_reserved_gb": self.reserved_gb,
            "vram_peak_gb": self.peak_gb,
            "vram_degraded": int(self.degraded),
        }


class VRamGuard:
    """监控 CUDA 显存并在阈值突破时触发降级回调。

    Args:
        warn_gb, degrade_gb, oom_gb: 三级阈值，单位 GiB
        degrade_callback: ``callable(step, used_gb) -> str``，
            触发降级时调用；返回值会记录到日志，便于确认降级是否真的做了事。
            回调内部**应当**只做那些运行时确实可调的事情
            （例如缩小 learner 的 minibatch、暂停部分环境），
            不要试图去改网络结构。
        oom_is_fatal: True 时超过 oom_gb 直接抛 RuntimeError；
            False 时只记录并触发 degrade 回调（便于先观察再收紧）。
    """

    def __init__(
        self,
        warn_gb: float = 9.0,
        degrade_gb: float = 10.5,
        oom_gb: float = 11.5,
        degrade_callback: Optional[Callable[[int, float], str]] = None,
        oom_is_fatal: bool = True,
    ):
        self.warn_gb = warn_gb
        self.degrade_gb = degrade_gb
        self.oom_gb = oom_gb
        self.degrade_callback = degrade_callback
        self.oom_is_fatal = oom_is_fatal
        self.degraded = False       # 一旦降级不再恢复，避免抖动
        self.degrade_step = -1

    # -- 探测 -------------------------------------------------------------

    @staticmethod
    def _probe() -> Optional[tuple]:
        """返回 (allocated_gb, reserved_gb, peak_gb)；CUDA 不可用则 None。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return allocated, reserved, peak
        except Exception as exc:      # noqa: BLE001
            logger.warning("VRamGuard: CUDA 探测失败：%s", exc)
            return None

    def check(self, step: int = 0) -> VRamStatus:
        """探测显存并在超阈值时降级。

        返回 VRamStatus。若 ``oom_is_fatal`` 且超过 oom 阈值，抛 RuntimeError。

        三级响应：
          warn    仅记录日志
          degrade 触发回调（只触发一次）
          oom     抛错（或仅记录），由上层决定收缩 batch 后重试
        """
        probed = self._probe()
        if probed is None:
            return VRamStatus(0.0, 0.0, 0.0,
                              self.warn_gb, self.degrade_gb, self.oom_gb)
        allocated, reserved, peak = probed

        # 用 allocated 与 reserved 的较大者作为"实际占用"：
        # 缓存分配器保留但暂未使用的显存同样不能被别的进程使用。
        effective = max(allocated, reserved)

        status = VRamStatus(
            used_gb=allocated,
            reserved_gb=reserved,
            peak_gb=peak,
            threshold_warn=self.warn_gb,
            threshold_degrade=self.degrade_gb,
            threshold_oom=self.oom_gb,
            degraded=self.degraded,
        )

        # ---- 硬 OOM ----
        if effective > self.oom_gb:
            msg = (
                f"VRamGuard OOM：allocated={allocated:.2f}G reserved={reserved:.2f}G "
                f"> oom_gb={self.oom_gb}G。请减小 minibatch_size / 序列长度后重试。"
            )
            if self.oom_is_fatal:
                raise RuntimeError(msg)
            logger.error(msg)

        # ---- 降级（只触发一次）----
        if effective > self.degrade_gb and not self.degraded:
            self.degraded = True
            self.degrade_step = step
            status.degrade_action = self._do_degrade(step, effective)
            logger.warning(
                "VRamGuard DEGRADE @ step=%d allocated=%.2fG reserved=%.2fG -> %s",
                step, allocated, reserved, status.degrade_action,
            )

        # ---- 警告 ----
        if effective > self.warn_gb and effective <= self.degrade_gb:
            logger.info(
                "VRamGuard WARN @ step=%d allocated=%.2fG reserved=%.2fG",
                step, allocated, reserved,
            )

        return status

    def _do_degrade(self, step: int, used_gb: float) -> str:
        """调用外部降级回调；未注册回调时明确返回 no-op 标记。"""
        if self.degrade_callback is not None:
            return self.degrade_callback(step, used_gb)
        return "no_callback_registered（未注册降级回调，本次未做任何收缩）"

    def reset_peak(self):
        """重置峰值统计，便于按阶段观察。"""
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:      # noqa: BLE001
            pass
