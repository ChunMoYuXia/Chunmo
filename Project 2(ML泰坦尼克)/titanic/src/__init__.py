"""泰坦尼克号生存预测 - 模块化机器学习包。

子模块职责：
- config:     全局配置（路径、随机种子、交叉验证折数）
- data_loader: 原始 CSV 读取与拼接
- features:   特征工程（Title/家庭规模/票价分箱/缺失值填充/编码）
- models:     多算法定义（LR/SVM/KNN/DT/RF/GBDT/XGB/LGB）
- parallel:   joblib 多进程并行训练 + 交叉验证
- evaluate:   评估指标、排行榜、最优模型筛选
- submit:     生成 Kaggle 提交文件 submission.csv
"""
