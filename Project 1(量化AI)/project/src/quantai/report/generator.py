"""单股全流程：下载 → 清洗 → 指标 → 图 → 股评。

本模块是研报生成的最小编排单元，按顺序串联数据层、指标层、图表层与 AI 层。
之所以把这四步合并为一个函数，是为了让上层（CLI / 批量 / Web 接口）
只需关心「输入代码与区间」即可拿到「图 + 文本 + 原始数据」三件套，
避免调用方重复书写流水线样板代码。
"""
from ..ai.client import gen_report
from ..ai.prompts import build_prompt
from ..data.cache import fetch_cached
from ..data.indicators import add_all_indicators
from .chart import make_chart


def run_one(code: str, start: str, end: str, style: str = "要点式",
            provider: str = "deepseek", refresh: bool = False):
    """单股全流程。返回 (fig, report, df)。

    参数：
        code:     股票代码（如 "600519"），透传给数据层。
        start:    起始日期字符串（格式与 fetch_cached 约定一致，如 "2024-01-01"）。
        end:      截止日期字符串。
        style:    AI 股评的写作风格，对应 prompts 层的风格分支，默认"要点式"。
        provider: 选用的 LLM 提供商标识，透传给 ai.client。
        refresh:  是否强制跳过缓存重新下载数据，默认 False 以优先命中本地缓存。

    返回：
        tuple[go.Figure, str, pd.DataFrame]:
            - fig: Plotly 图表对象（主图 + 已开启的副图）。
            - report: AI 生成的股评文本。
            - df: 已附加全部技术指标的行情 DataFrame，便于上层再加工（如导出统计表）。

    设计说明：
        - 流水线顺序不可调换：图表与股评都依赖 add_all_indicators 产出的列；
          build_prompt 也基于 df 的最新指标字段，因此指标计算必须在前。
        - 本函数不做异常兜底：单只失败时由 batch 层统一捕获并记录，
          保持单股接口的「异常即失败」语义清晰。
    """
    # 1. 取数：优先命中缓存；refresh=True 时强制重新拉取原始行情
    df = fetch_cached(code, start, end, refresh=refresh)
    # 2. 指标：在原始 OHLCV 基础上追加 MA/BOLL/RSI/MACD/KDJ 等列
    df = add_all_indicators(df)
    # 3. 图表：根据全局 settings 决定副图组合，生成交互式 Plotly 图
    fig = make_chart(df)
    # 4. 股评：基于指标数据构造提示词，调用 LLM 生成文字分析
    prompt = build_prompt(df, code, style)
    report = gen_report(prompt, df, code, provider)
    return fig, report, df
