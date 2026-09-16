"""机器学习层：特征工程与多模型时序预测。

模块组成：
- features.py：行情 + 技术指标 → 特征矩阵 X 与标签 y，严格防数据泄露。
- models.py：多模型（逻辑回归 / XGBoost / LightGBM / MLP / LSTM）赛跑，
  使用 TimeSeriesSplit 时序交叉验证，缺失依赖自动降级。
- predictor.py：串联特征 → 赛跑 → 冠军模型预测 → 大模型风控建议。
"""
