"""机器学习主入口：特征 → 赛跑 → 冠军预测 → AI 风控建议。

本模块串联 features / models 两层的能力，并对接大模型生成面向风控的文字建议。
设计上保持纯函数式编排，不持有状态，便于测试与复用。
"""
from .features import make_dataset
from .models import predict_latest, run_race
from ..ai.client import ask_llm


def build_advice(info: dict, provider: str = "deepseek") -> str:
    """基于赛跑结果与冠军预测，调用大模型生成风控导向的文字建议。

    参数:
        info: 由 analyze_stock 组装的字典，包含 code/date/ranking/scores/best/prob_up。
        provider: 大模型提供商标识，透传给 ask_llm，默认 deepseek。

    返回:
        str: 大模型返回的建议文本；若调用失败则返回内置的规则版降级文案，
             保证上层始终能拿到非空字符串。

    降级策略:
        大模型接口不稳定（网络/限流/异常）时，捕获所有异常并回退到规则版文案，
        只展示概率与基础风控提示，避免整个分析流程因 AI 环节失败而中断。
    """
    # 把各模型的 LogLoss/AUC 格式化为列表，作为大模型判断模型置信度的输入
    ranking_lines = "\n".join(
        f"- {name}：LogLoss {s['log_loss']:.4f} / AUC {s['auc']:.3f}"
        for name, s in info["ranking"]
    )
    prompt = (
        "你是一名风控导向的量化研究员。\n"
        f"股票 {info['code']}（截至 {info['date'].date()}）{len(info['ranking'])} 个模型时序交叉验证结果：\n"
        f"{ranking_lines}\n"
        f"冠军模型 {info['best']} 预测下一交易日上涨概率 {info['prob_up']:.1%}。\n"
        "请按【预测解读】【风险提示】【操作纪律】三段给出风控导向的建议，总字数 150 字以内；"
        "强调概率不是承诺、注意仓位与止损纪律；不要编造数据中没有的信息。"
    )
    try:
        return ask_llm(prompt, provider)
    except Exception as exc:
        # 大模型调用失败时降级：给出最朴素的概率提示与风控纪律，不抛出异常
        return f"（AI 建议生成失败：{exc}）\n规则版：上涨概率 {info['prob_up']:.1%}，仅作参考，注意仓位与止损。"


def analyze_stock(code: str, start: str, end: str, provider: str = "deepseek",
                  model_names: list[str] | None = None) -> dict:
    """对单只股票执行完整的机器学习分析流程。

    参数:
        code: 股票代码。
        start: 数据起始日期。
        end: 数据结束日期。
        provider: 大模型提供商标识，透传给 build_advice。
        model_names: 参赛模型名列表，None 表示使用全部可用模型。

    返回:
        dict: 包含 code、date（最新交易日）、ranking（模型排名）、scores（各模型得分）、
        best（冠军模型名）、prob_up（上涨概率）、models_used（实际参赛模型列表）、
        advice（AI 风控建议文本）。

    流程:
        1. make_dataset 拉取行情并构造特征/标签。
        2. run_race 做多模型时序交叉验证，得到排名。
        3. predict_latest 用冠军模型在全量数据上重训并预测最新一日。
        4. build_advice 把结果交给大模型生成风控建议。
    """
    X, y, dates = make_dataset(code, start, end)
    scores, ranking = run_race(X, y, names=model_names)
    # ranking 首位即冠军模型（LogLoss 最低）
    best = ranking[0][0]
    prob_up = predict_latest(X, y, best)
    info = {
        "code": code, "date": dates.iloc[-1], "ranking": ranking,
        "scores": scores, "best": best, "prob_up": prob_up,
        "models_used": [n for n, _ in ranking],
    }
    info["advice"] = build_advice(info, provider)
    return info
