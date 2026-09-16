"""分组对比：两组股票归一化收益率曲线 + 对比表 + AI 对比研报。

本模块服务于「行业对比 / 风格对比」场景：把两组股票放到同一时间轴上比较
区间表现。核心在于把绝对价格转换为「起点对齐的累计收益率」，
这样不同价位的股票才具有可比性。
"""
import pandas as pd

from ..ai.client import ask_llm
from ..config import load_compare_template
from ..data.cache import fetch_cached
from ..data.indicators import add_all_indicators


def _normalize(series):
    """价格序列 → 归一化收益率：起点为 0。

    计算方法：series / series.iloc[0] - 1，
    即以首个有效收盘价为基准，后续每个时点表示相对起点的涨跌幅（小数）。

    选择「除以首点」而非 pct_change 的原因：
    pct_change 给出的是单日收益率，序列会在 0 附近震荡，
    无法直观看出区间累计收益；除以首点后得到累计收益曲线，
    跨组对比时走势一目了然。
    """
    return series / series.iloc[0] - 1


def compare_data(groups, start, end, refresh=False):
    """groups: {组名: {代码: 名称}} → (curves, means, table)。

    参数：
        groups:  分组映射，如 {"白酒": {"600519": "贵州茅台", "000858": "五粮液"}, ...}。
        start:   起始日期，透传给 fetch_cached。
        end:     截止日期。
        refresh: 是否跳过缓存强制重拉。

    返回：
        tuple[dict, dict, pd.DataFrame]:
            - curves: {组名: {代码: Series}}，每个 Series 是归一化收益率。
            - means:  {组名: Series}，组内所有个股在同一时间轴上的均值收益率。
            - table:  每行一只股票，含组名/代码/名称/近20日涨跌/RSI14/收盘相对MA20。

    说明：
        - 取数与指标计算复用单股流水线的底层函数（fetch_cached + add_all_indicators），
          保证对比场景与单股场景的指标口径一致。
        - 统计表中的「近20日涨跌」用 close.iloc[-21] 作基准：
          因为 iloc[-1] 是当日，往前推 20 个交易日即 iloc[-21]，
          这样得到的正好是最近 20 个交易日的区间涨跌幅。
        - 组均值用 pd.DataFrame(cs).mean(axis=1) 计算，前提是各股时间索引对齐；
          fetch_cached 已保证同区间同频率，因此可直接横向求均值。
    """
    curves, rows = {}, []
    for gname, stocks in groups.items():
        curves[gname] = {}
        for code, name in stocks.items():
            # 取数 + 指标：与单股研报保持一致口径
            df = add_all_indicators(fetch_cached(code, start, end, refresh=refresh))
            curves[gname][code] = _normalize(df["close"])
            last = df.iloc[-1]
            rows.append({
                "组": gname, "代码": code, "名称": name,
                # 近 20 日涨跌：iloc[-21] 是 20 个交易日前的收盘
                "近20日涨跌%": round((last["close"] / df["close"].iloc[-21] - 1) * 100, 2),
                "RSI14": round(last["rsi14"], 1),
                "收盘vsMA20": "上" if last["close"] >= last["ma20"] else "下",
            })
    # 组均值：把同组个股 Series 拼成 DataFrame 后按行求均值
    means = {g: pd.DataFrame(cs).mean(axis=1) for g, cs in curves.items()}
    return curves, means, pd.DataFrame(rows)


def build_compare_stats(curves) -> str:
    """组均值期末值 + 每组最强/最弱个股 → 给模型的统计文字。

    参数：
        curves: compare_data 返回的 curves 结构。

    返回：
        str: 多行 Markdown 风格文本，每行描述一个组的期末均值及最强/最弱个股。

    设计理由：
        LLM 直接阅读原始曲线数据（大量浮点数）既浪费 token 又容易抓不住重点，
        因此先做「摘要压缩」：期末均值代表整体胜负，最强/最弱代表组内分化，
        这些统计量足以支撑模型产出有数据支撑的对比结论。
    """
    lines = []
    for gname, cs in curves.items():
        # 期末值：每只股票归一化收益率的最后一个时点
        ends = {code: s.iloc[-1] for code, s in cs.items()}
        mean_end = sum(ends.values()) / len(ends)
        best = max(ends, key=ends.get)
        worst = min(ends, key=ends.get)
        lines.append(
            f"- {gname}：组均值期末 {mean_end:+.2%}；最强 {best}（{ends[best]:+.2%}）；"
            f"最弱 {worst}（{ends[worst]:+.2%}）"
        )
    return "\n".join(lines)


def gen_compare_report(groups, curves, start, end, provider="deepseek"):
    """AI 对比研报；失败抛异常，由调用方兜底。

    参数：
        groups:   分组映射（仅用于模板里展示组名）。
        curves:   归一化收益率曲线，用于 build_compare_stats 生成统计摘要。
        start/end: 区间日期，填入模板。
        provider: LLM 提供商标识。

    返回：
        str: AI 生成的对比研报文本。

    异常策略：
        本函数不捕获异常，直接向上抛出。设计上把「是否重试 / 降级」的决策
        交给调用方（如 batch 层或 CLI），保持本函数职责单一。
    """
    template = load_compare_template()
    # 模板中 {stats} 占位符由统计摘要填充，避免把整份数据塞给模型
    prompt = template.format(start=start, end=end, stats=build_compare_stats(curves))
    return ask_llm(prompt, provider)
