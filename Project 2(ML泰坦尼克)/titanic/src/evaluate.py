"""评估与排行榜：把并行训练结果整理成排行榜，选出最优模型。

主排序指标用 accuracy（Kaggle 泰坦尼克竞赛的官方评分指标）；
并列时按 roc_auc 降序（概率排序能力更稳健）。
"""
import pandas as pd


def make_leaderboard(results: list[dict]) -> pd.DataFrame:
    """把并行训练结果整理成排行榜 DataFrame。

    参数:
        results: parallel.run_parallel 返回的结果列表。

    返回:
        DataFrame，列 = [算法, 准确率, 精确率, 召回率, F1, AUC, 耗时s, 状态]，
        按 accuracy 降序、roc_auc 降序排列。

    说明:
        训练失败（error 非 None）的算法仍列出，状态列标注错误信息，
        方便排查是哪个算法因依赖或参数问题挂掉。
    """
    rows = []
    for r in results:
        rows.append({
            "算法": r["name"],
            "准确率": round(r["accuracy"], 4),
            "精确率": round(r["precision"], 4),
            "召回率": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
            "AUC": round(r["roc_auc"], 4),
            "耗时s": round(r["fit_time"], 2),
            "状态": "错误: " + r["error"] if r["error"] else "OK",
        })
    df = pd.DataFrame(rows)
    # 主键 accuracy 降序，次键 roc_auc 降序
    df = df.sort_values(["准确率", "AUC"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "排名", range(1, len(df) + 1))
    return df


def select_best(results: list[dict]) -> str:
    """从结果列表中选出最优算法名（accuracy 最高，并列看 roc_auc）。

    参数:
        results: parallel.run_parallel 返回的结果列表。

    返回:
        最优算法名字符串。若全部失败则返回 None。
    """
    # 过滤掉训练失败的算法
    ok = [r for r in results if r["error"] is None]
    if not ok:
        return None
    best = max(ok, key=lambda r: (r["accuracy"], r["roc_auc"]))
    return best["name"]


def print_leaderboard(leaderboard: pd.DataFrame, best_name: str) -> None:
    """打印排行榜与冠军模型，供控制台运行时查看。"""
    print("\n" + "=" * 70)
    print("多算法排行榜（分层 5 折交叉验证）")
    print("=" * 70)
    # 用 to_string 保证完整列宽打印，不被截断
    print(leaderboard.to_string(index=False))
    print("-" * 70)
    print(f"冠军算法: {best_name}  (按准确率择优)")
    print("=" * 70)
