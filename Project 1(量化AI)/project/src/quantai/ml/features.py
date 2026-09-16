"""特征与标签工程：行情 + 指标 → 特征矩阵 X 和标签 y。

防泄露铁律：每行特征只用当天及之前的数据，标签用未来一日涨跌；
切分必须按时间顺序。

设计思路：
- 所有特征均为"截至当天"可计算的量（pct_change、rolling、与均线比值等），
  绝不引用未来数据，避免训练时信息泄露。
- 标签 y 使用次日收盘价相对当日的涨跌，是典型的二分类监督信号。
- 返回的 DataFrame 按时间自然排序，配合下游 TimeSeriesSplit 做时序切分。
"""
import pandas as pd

from ..data.cache import fetch_cached
from ..data.indicators import add_all_indicators

# 最终送入模型的特征列顺序；顺序即含义见 make_dataset 中的构造注释
FEATURE_COLUMNS = [
    "pct1", "pct5", "close_ma5_gap", "close_ma20_gap",
    "vol_ratio", "body", "rng", "rsi14", "dif_dea", "macd",
]


def make_dataset(code: str, start: str, end: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """构造单只股票的 (特征矩阵 X, 标签 y, 日期序列 dates)。

    参数:
        code: 股票代码，用于从缓存拉取日线行情。
        start: 起始日期字符串（含），如 "2020-01-01"。
        end: 结束日期字符串（含），如 "2024-12-31"。

    返回:
        X: DataFrame，列为 FEATURE_COLUMNS，行对应每个有效交易日。
        y: Series，1 表示下一交易日上涨，0 表示下跌/持平。
        dates: Series，X 每行对应的交易日期，供结果对齐与展示。

    说明:
        任何一行只要有任一特征或标签为 NaN（如滚动窗口不足、最后一行无未来标签）
        即被 dropna 剔除，保证训练数据完整。
    """
    df = fetch_cached(code, start, end)
    # 先添加 MA/RSI/MACD 等通用指标，再在此基础上派生本模块专用特征
    df = add_all_indicators(df)

    # 动量类：单日涨跌幅，捕捉最近一日方向与强度
    df["pct1"] = df["close"].pct_change()
    # 动量类：5 日累计涨跌幅，反映短期趋势
    df["pct5"] = df["close"].pct_change(5)
    # 位置类：收盘价相对 5 日均线的偏离度，衡量短期超买超卖
    df["close_ma5_gap"] = df["close"] / df["ma5"] - 1
    # 位置类：收盘价相对 20 日均线的偏离度，衡量中期位置
    df["close_ma20_gap"] = df["close"] / df["ma20"] - 1
    # 量能类：当日成交量相对 5 日均量的倍数，识别放量/缩量
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(5).mean()
    # K线形态：实体占开盘价比例，阳线为正、阴线为负
    df["body"] = (df["close"] - df["open"]) / df["open"]
    # 波动类：当日振幅（最高-最低）/收盘，刻画日内波动强度
    df["rng"] = (df["high"] - df["low"]) / df["close"]
    # MACD 状态：DIF 与 DEA 的差，金叉/死叉的量化表达（add_all_indicators 已生成 dif/dea/macd）
    df["dif_dea"] = df["dif"] - df["dea"]

    # 标签：次日收盘是否高于当日收盘；shift(-1) 引用未来值，仅用于标签不用于特征
    df["y"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # 丢弃因滚动窗口不足或末行无未来标签而产生的 NaN 行
    data = df.dropna(subset=FEATURE_COLUMNS + ["y"]).copy()
    return data[FEATURE_COLUMNS], data["y"], data["date"]
