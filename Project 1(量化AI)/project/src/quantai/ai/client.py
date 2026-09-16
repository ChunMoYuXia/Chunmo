"""AI 调用与降级：提示词 → 多家大模型；任何异常降级为模板兜底。

设计目标：
- 统一封装多家 LLM 供应商（DeepSeek / OpenAI 兼容协议等）的调用接口，
  使上层业务（gen_report）无需感知具体供应商差异。
- 采用「AI 优先 + 模板兜底」的双保险策略：LLM 输出更具分析师语感，
  但若网络、钥匙、配额等任何环节失败，立即回退到确定性的模板报告，
  保证研报功能在离线/故障场景下依然可用。

钥匙三层优先级：环境变量 > keys.env > 报错提醒。
"""
import os

import requests

from ..config import (
    load_keys_into_env,      # 把 keys.env 中的钥匙注入 os.environ（幂等，不覆盖已设环境变量）
    load_prompt_template,    # 加载外部可编辑的提示词模板文本
    load_providers,          # 读取供应商注册表：{name: (url, model, key_env_name)}
    section_titles,          # 从模板正文中解析出【】包裹的段落标题列表
)
from ..logger import get_logger

# 独立 logger 名 "ai"，便于在日志中按模块过滤 AI 调用相关记录
log = get_logger("ai")


def ask_llm(prompt: str, provider: str = "deepseek", timeout: int = 30) -> str:
    """送信拆信：POST + 钥匙 + 题面 → 取 choices[0].message.content。

    参数:
        prompt: 已组装好的完整提示词（含股票代码、指标快照、风格要求）。
        provider: 供应商名，必须在 config 的 providers 注册表中，默认 deepseek。
        timeout: 单次 HTTP 请求超时秒数，默认 30s；研报生成通常 <15s，
                 设 30s 兼顾长尾请求与及时失败。

    返回:
        模型返回的纯文本内容（choices[0].message.content）。

    说明:
        本函数是「底层薄封装」，故意不捕获异常——所有异常（网络、4xx/5xx、
        钥匙缺失、KeyError）都会向上抛出，由 gen_report 统一捕获并降级。
        这样关注点分离：ask_llm 只负责「成功调用」，gen_report 负责「失败兜底」。
    """
    # 钥匙注入放在调用最前：确保 keys.env 中的钥匙在读取 key_env 之前已进入 os.environ
    load_keys_into_env()
    # FAIL_LLM 是演练/测试开关：设置该环境变量可模拟 AI 通道整体故障，
    # 便于在不真实消耗配额的情况下验证降级链路是否畅通
    if os.environ.get("FAIL_LLM"):
        raise ConnectionError("演练：AI 通道被掐断")
    # 供应商注册表由外部配置驱动，新增供应商只需改配置，无需改此处代码
    providers = load_providers()
    if provider not in providers:
        raise KeyError(f"未注册的模型供应商：{provider}")
    url, model, key_env = providers[provider]
    # 钥匙缺失是最常见的用户错误，给出明确指引而非裸 KeyError
    if key_env not in os.environ:
        raise ConnectionError(f"没找到 {key_env}：请设环境变量，或在设置页填写钥匙")
    # 所有供应商均遵循 OpenAI 兼容的 chat completions 协议，因此请求体结构统一
    headers = {
        "Authorization": f"Bearer {os.environ[key_env]}",
        "Content-Type": "application/json",
    }
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    # raise_for_status 会把 4xx/5xx 转为 HTTPError，交由上层统一降级
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def gen_report(prompt: str, df, code: str, provider: str = "deepseek") -> str:
    """主入口：先试 AI，任何异常都降级到模板兜底。

    参数:
        prompt: 组装好的提示词文本。
        df: 指标数据表（用于降级时生成模板报告）。
        code: 股票代码（降级报告中展示用）。
        provider: 优先尝试的 LLM 供应商。

    返回:
        最终研报文本。优先使用 AI 生成结果；失败则返回 template_report 的兜底文本。

    设计选型:
        为什么不在 ask_llm 内部重试？因为研报是「读多写少」的在线交互场景，
        用户等待耐心有限。与其在 ask_llm 里做指数退避重试（可能累计数秒），
        不如快速失败一次后立即给出可读的模板报告，体验更可控。
        若后续需要重试，可在此处加 retry 装饰器，不影响兜底链路。
    """
    try:
        return ask_llm(prompt, provider)
    except Exception as exc:
        # 记录失败原因便于排查，但不中断流程——降级是核心价值
        log.warning("[AI] 失败：%s，已降级为模板兜底", exc)
        return template_report(df, code)


def _rating_from_facts(f: dict) -> tuple[str, str]:
    """基于技术指标给出五档评级 + 理由。

    参数:
        f: snapshot_facts 产出的事实字典。

    返回:
        (评级, 理由) 二元组。评级为「买入/增持/中性/减持/卖出」五档之一。

    算法思路:
        采用「多因子加权打分」的简易模型：
        - 均线排列：多头 +2 / 空头 -2（权重最高，因趋势是长周期主导）
        - 收盘价相对 MA20：之上 +1 / 之下 -1
        - RSI：<30 超卖 +1（反弹预期）/ >70 超买 -1（回调风险）
        - MACD：金叉或多头 +1 / 死叉或空头 -1
        - 5 日动量：>5% +1 / <-5% -1
        分数区间 [-7, +5]，按阈值映射到五档评级。
        这是纯技术面的经验模型，不考虑基本面，仅作降级兜底用，故保持简单可解释。
    """
    score = 0
    # 趋势——均线排列是最核心的趋势信号，权重最高（±2）
    if f["ma_arrange"] == "多头排列":
        score += 2
    elif f["ma_arrange"] == "空头排列":
        score -= 2
    # 收盘价相对 MA20 的位置反映中期趋势
    if f["up20"] == "之上":
        score += 1
    else:
        score -= 1
    # RSI：超卖视为潜在反弹机会（+1），超买视为回调风险（-1）
    if f["rsi"] < 30:
        score += 1
    elif f["rsi"] > 70:
        score -= 1
    # MACD 金叉/多头：短期动能向上
    if f["macd_cross"] in ("金叉", "多头"):
        score += 1
    else:
        score -= 1
    # 5 日动量：短期涨跌幅度反映近期资金态度
    if f["pct5"] > 5:
        score += 1
    elif f["pct5"] < -5:
        score -= 1

    # 阈值切分：>=4 买入、>=2 增持、>=0 中性、>=-2 减持、其余卖出
    if score >= 4:
        return "买入", "技术面强势：均线多头排列、MACD 多头、动量向上，可积极配置"
    elif score >= 2:
        return "增持", "技术面偏强，趋势向上，可逢低布局"
    elif score >= 0:
        return "中性", "技术面信号混杂，趋势不明，建议观望"
    elif score >= -2:
        return "减持", "技术面偏弱，均线空头或 MACD 死叉，建议减仓"
    else:
        return "卖出", "技术面弱势明显，多重看空信号共振，建议离场"


def template_report(df, code: str) -> str:
    """模板兜底：6 段结构，跟随提示词模板的【】段落。

    参数:
        df: 指标数据表。
        code: 股票代码（当前仅用于占位，兜底文本中不直接输出 code，
              但保留参数以便未来在段尾加注代码信息）。

    返回:
        以「【段落标题】\\n内容」形式拼接的 6 段研报文本，
        末尾追加免责声明。

    设计要点:
        - 段落顺序严格跟随提示词模板的【】标题（由 section_titles 解析），
          保证兜底报告与 AI 报告在结构上完全一致，前端渲染无需分支处理。
        - 字典 lines 的 key 即段落标题；若模板中出现未覆盖的新段落，
          用通用占位文案兜底，避免 KeyError 导致降级链路本身崩溃。
        - 评级与仓位建议由 _rating_from_facts 产出，保证兜底报告
          也具备「结论性」，而非纯数据罗列。
    """
    from .prompts import snapshot_facts
    f = snapshot_facts(df)
    rating, reason = _rating_from_facts(f)
    # 仓位建议与评级联动：看多加仓、看空减仓，给用户一个可执行的参考区间
    position = "建议仓位 6-8 成" if rating in ("买入", "增持") else (
        "建议仓位 3-5 成" if rating == "中性" else "建议仓位 0-2 成")

    # 6 段事实文本，key 与模板中的【】标题一一对应
    lines = {
        "行情概览": f"收盘价 {f['close']:.2f}，当日涨跌 {f['pct1']:+.2f}%，成交量 {f['volume']:.0f}，量比 {f['vol_ratio']:.2f}。",
        "趋势研判": f"MA5={f['ma5']:.2f}、MA10={f['ma10']:.2f}、MA20={f['ma20']:.2f}，均线{f['ma_arrange']}；收盘价位于 MA20 {f['up20']}；布林带位置：{f['boll_pos']}。",
        "量价分析": f"近 5 日涨跌 {f['pct5']:+.2f}%，量比 {f['vol_ratio']:.2f}，{'价涨量增，多头积极' if f['pct1'] > 0 and f['vol_ratio'] > 1 else '价量配合一般'}。",
        "技术指标": f"RSI14={f['rsi']:.1f}（{f['rsi_zone']}）；MACD {f['macd_cross']}，柱值 {f['macd']:.2f}；KDJ {f['kdj_cross']}。",
        "关键价位": f"支撑位 {f['lo']:.2f}（近 20 日低）/ {f['boll_lower']:.2f}（BOLL 下轨）；压力位 {f['hi']:.2f}（近 20 日高）/ {f['boll_upper']:.2f}（BOLL 上轨）。",
        "投资评级": f"评级：{rating}。理由：{reason}。{position}。",
    }
    parts = []
    # 按模板中的段落顺序拼接，确保与 AI 输出结构一致
    for title in section_titles(load_prompt_template()):
        parts.append(f"【{title}】\n{lines.get(title, '本段由模板兜底生成，仅供参考。')}")
    # 末尾统一追加免责声明，提示兜底报告的局限性
    return "\n\n".join(parts) + "\n\n（本段文字为模板兜底生成，仅供参考，不构成投资建议）"
