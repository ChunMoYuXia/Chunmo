"""快速测试脚本：验证数据、指标、ML、导出全流程。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quantai.ml.predictor import analyze_stock
from quantai.report.export import export_html

# 测试 ML 预测
print("=== ML 预测测试 ===")
r = analyze_stock("600519", "2024-01-01", "2025-06-30", "deepseek")
print("冠军模型:", r["best"])
print("上涨概率:", round(r["prob_up"] * 100, 1), "%")
print("排行榜:")
for n, s in r["ranking"]:
    print(f"  {n}: LogLoss={s['log_loss']:.4f}  AUC={s['auc']:.3f}")

# 测试导出
print("\n=== 导出测试 ===")
from quantai.report.generator import run_one
fig, report, df = run_one("600519", "2025-01-01", "2025-06-30", "要点式", "deepseek")
html_path = export_html(fig, report, "600519", "贵州茅台")
print("HTML 导出:", html_path)
print("全流程测试通过！")
