"""多模型赛跑：逻辑回归 / XGBoost / LightGBM / MLP / LSTM × 时序交叉验证。

支持用户选择参赛模型（单选或多选）。切分用 TimeSeriesSplit。
XGBoost / LightGBM / torch 缺失时自动跳过对应模型。

设计要点：
- 工厂模式：get_models 统一实例化并返回 {模型名: 模型对象} 字典，
  调用方无需关心各模型的构造差异与依赖缺失问题。
- 降级策略：可选依赖（xgboost/lightgbm/torch）缺失时仅跳过对应模型，
  至少保留逻辑回归作为兜底，保证流程不中断。
- 评估指标：LogLoss 衡量概率校准质量（越低越好），AUC 衡量排序能力
  （越高越好）；主排名用 LogLoss，因为风控更依赖概率可信度而非仅排序。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..logger import get_logger

log = get_logger("ml")

# 全部支持的模型名称，用于 UI 展示与参数校验
ALL_MODEL_NAMES = ["逻辑回归", "XGBoost", "LightGBM", "MLP", "LSTM"]

# 概率裁剪边界：log_loss 在 p=0 或 p=1 时趋向无穷，裁剪到 [eps, 1-eps]
# 保证数值稳定又不损失有效精度（1e-7 远小于常规概率分辨率）
_PROB_EPS = 1e-7

# 统一随机种子，满足 NFR-SEC-03 可复现性需求；所有含随机性的模型共用
RANDOM_STATE = 42


def get_models(names: list[str] | None = None) -> dict:
    """获取指定模型字典，工厂方法。

    参数:
        names: 期望使用的模型名列表；None 或空列表表示返回全部可用模型。

    返回:
        dict: {模型名: 模型实例}。对于 LSTM，值为字符串 "LSTM" 占位符，
        由 run_race / predict_latest 走 _train_lstm / _lstm_cv 特殊训练路径。

    降级逻辑:
        - xgboost / lightgbm / torch 任一可选依赖缺失，仅跳过对应模型并记录 warning。
        - 若用户所选模型全部不可用，则降级为仅含逻辑回归的字典，确保流程不崩。
        - 逻辑回归与 MLP 依赖 sklearn，属于必选依赖，必定可用。
    """
    available = {}

    # 逻辑回归（必有）：兜底基线模型；套 StandardScaler 消除量纲差异，
    # 否则大量级特征（rsi14 0~100、macd ±90）会压掉小量级特征（pct1 ±0.07）的贡献。
    available["逻辑回归"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # MLP（sklearn 内置，必有）：小规模前馈网络，捕捉非线性关系；
    # 同样套 StandardScaler（神经网络对输入尺度极敏感，不缩放会收敛极慢）；
    # early_stopping 防止过拟合，random_state 固定保证可复现
    available["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, random_state=RANDOM_STATE,
        )),
    ])

    # XGBoost（可选）：树模型，擅长表格数据；缺失则跳过
    # 正则化调优：默认 n_estimators=100/max_depth=3 在数百行小数据上严重过拟合
    # （实测 LogLoss 0.86 远劣于 0.69 随机基线），故下调树数与深度、加采样与 L2 正则
    try:
        from xgboost import XGBClassifier
        available["XGBoost"] = XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,   # 行/列采样降方差
            reg_lambda=1.0,                         # L2 正则抑制过拟合
            eval_metric="logloss", verbosity=0,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        log.warning("xgboost 加载失败，跳过：%s", exc)

    # LightGBM（可选）：与 XGBoost 互补的梯度提升树；缺失则跳过
    # 同样下调复杂度：浅树 + 采样 + L2，匹配小样本场景
    try:
        from lightgbm import LGBMClassifier
        available["LightGBM"] = LGBMClassifier(
            n_estimators=50, num_leaves=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,    # bagging + 特征采样
            reg_lambda=1.0,                          # L2 正则
            verbose=-1, random_state=RANDOM_STATE,
        )
    except Exception as exc:
        log.warning("lightgbm 加载失败，跳过：%s", exc)

    # LSTM（需要 torch，可选）：用字符串占位而非真实模型，
    # 因为 LSTM 需要把二维 X 重排为三维序列，fit/predict 接口与 sklearn 不同
    try:
        import torch  # noqa: F401
        available["LSTM"] = "LSTM"  # 标记，训练时特殊处理
    except ImportError:
        log.warning("torch 未安装，跳过 LSTM 模型")

    # 按用户选择过滤；若所选全部不可用，降级到逻辑回归保证可运行
    if names:
        selected = {n: available[n] for n in names if n in available}
        if not selected:
            log.warning("所选模型均不可用，降级为逻辑回归")
            selected = {"逻辑回归": available["逻辑回归"]}
        return selected
    return available


def _build_sequences(X_arr, y_arr, seq_len):
    """把二维 (N, F) 滑窗重排为三维 (样本, seq_len, 特征)。

    参数:
        X_arr: ndarray，形状 (N, F)。
        y_arr: ndarray，长度 N（可为 None，仅构造 X 序列时）。
        seq_len: 滑动窗口长度。

    返回:
        (X_seq, y_seq)：X_seq 形状 (N-seq_len, seq_len, F)，
        y_seq 长度 N-seq_len（y_arr 为 None 时返回 None）。

    说明:
        第 i 个样本取 X[i-seq_len:i] 作为输入、y[i] 作为标签，
        因此前 seq_len 行无法构成完整序列被丢弃。LSTM 训练与测试均复用本函数，
        消除原先 _train_lstm / _lstm_cv 里重复的序列构造代码。
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_arr)):
        X_seq.append(X_arr[i - seq_len:i])
        if y_arr is not None:
            y_seq.append(y_arr[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32) if y_arr is not None else None
    return X_seq, y_seq


def _make_lstm_net(input_size, hidden, layers, device):
    """构造 LSTM 网络实例（torch 延迟导入，缺失时不影响其余模型）。

    结构：堆叠 LSTM → 取最后时刻隐藏态 → 全连接 → Sigmoid 输出二分类概率。
    """
    import torch
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden, layers):
            super().__init__()
            # batch_first=True 表示输入形状为 (batch, seq_len, features)
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            # 取最后一个时间步的隐藏态 out[:, -1, :] 作为序列的聚合表示
            out, _ = self.lstm(x)
            return self.sig(self.fc(out[:, -1, :]))

    return LSTMNet(input_size, hidden, layers).to(device)


def _train_lstm(X, y, hidden=64, layers=2, epochs=50):
    """在全量数据上训练 LSTM，并返回每个样本的上涨概率（predict_proba 风格）。

    参数:
        X: 二维特征矩阵（DataFrame 或 ndarray），形状 (N, F)。
        y: 一维标签，长度 N。
        hidden: LSTM 隐藏层维度，默认 64。
        layers: LSTM 堆叠层数，默认 2。
        epochs: 训练轮数，默认 50。

    返回:
        np.ndarray: 一维概率数组，长度为 N - seq_len，对应每个可用序列样本的上涨概率。

    序列构造:
        把二维 (N, F) 重排为三维 (样本数, seq_len, 特征数)。
        - seq_len = min(10, N//2)：取最近 10 个时间步作为一条序列，
          数据不足时折半避免序列过长导致样本过少。
        - 第 i 个样本取 X[i-seq_len : i] 作为输入，y[i] 作为标签，
          因此前 seq_len 行无法构成序列被丢弃。
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 统一转为 ndarray，兼容 DataFrame 与 ndarray 输入
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    # 序列化为 (样本, 时间步, 特征)，复用共享滑窗函数
    seq_len = min(10, len(X_arr) // 2)
    X_seq, y_seq = _build_sequences(X_arr, y_arr, seq_len)

    model = _make_lstm_net(X_arr.shape[1], hidden, layers, device)
    # 二分类交叉熵，输入已是 Sigmoid 概率故用 BCELoss 而非 BCEWithLogitsLoss
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        proba = model(X_t).cpu().numpy().flatten()
    return proba


def run_race(X, y, names: list[str] | None = None, n_splits: int = 5) -> tuple[dict, list]:
    """多模型时序赛跑：对所选模型做 TimeSeriesSplit 交叉验证，输出各模型得分与排名。

    参数:
        X: 特征矩阵，需按时间升序排列（来自 make_dataset）。
        y: 标签序列。
        names: 参赛模型名列表，None 表示全部可用模型。
        n_splits: 时序交叉验证折数，默认 5。

    返回:
        scores: {模型名: {"log_loss": 均值, "auc": 均值}}。
        ranking: 按 (log_loss 升序, auc 降序) 排序后的 [(模型名, 得分), ...]，
                 首位为冠军模型。

    时序交叉验证思路:
        金融数据具有强时序相关性与潜在非平稳性，随机切分会把未来信息泄露给训练集。
        TimeSeriesSplit 保证每折训练集为历史、测试集为其后的未来片段，
        模拟真实"用历史预测未来"的场景，避免乐观偏差。

    指标选择理由:
        - LogLoss：直接惩罚概率与真实标签的偏差，对风控中"概率是否可信"最敏感，
          因此作为主排序键（越小越好）。
        - AUC：衡量模型把正类排在负类前面的能力，不依赖阈值，作为辅助参考。
    """
    if len(X) < 100:
        # 数据过少时时序切分后每折样本不足，评估结果不稳定，直接报错提示
        raise ValueError(f"数据仅 {len(X)} 行（不足 100 行），时序赛跑没有意义")
    models = get_models(names)
    scores = {}
    tss = TimeSeriesSplit(n_splits=n_splits)

    for name, model in models.items():
        ll_list, auc_list = [], []
        for tr_idx, te_idx in tss.split(X):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            if name == "LSTM":
                # LSTM 需要序列，用训练集训练，测试集验证
                proba = _lstm_cv(X_tr, y_tr, X_te)
            else:
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_te)[:, 1]
            # 裁剪到 [eps, 1-eps]：防止 Sigmoid 输出精确 0/1 导致 log_loss 为 inf
            proba = np.clip(proba, _PROB_EPS, 1 - _PROB_EPS)
            ll_list.append(log_loss(y_te, proba))
            auc_list.append(roc_auc_score(y_te, proba))
        # 跨折取均值作为最终得分，降低单折偶然波动影响
        scores[name] = {
            "log_loss": float(np.mean(ll_list)),
            "auc": float(np.mean(auc_list)),
        }
    # 主排序键 log_loss 升序（越低越好），次排序键 auc 降序（越高越好）
    ranking = sorted(scores.items(), key=lambda kv: (kv[1]["log_loss"], -kv[1]["auc"]))
    return scores, ranking


def _lstm_cv(X_tr, y_tr, X_te):
    """LSTM 交叉验证单折：用训练集训练，返回测试集概率。

    参数:
        X_tr, y_tr: 训练集特征与标签。
        X_te: 测试集特征（需为训练集之后的时间片段）。

    返回:
        np.ndarray: 测试集每个样本的上涨概率数组。

    测试序列构造:
        LSTM 预测需要 seq_len 长度的历史窗口，而测试集前 seq_len 行没有足够历史。
        因此把训练集末尾 seq_len 行拼接到测试集之前，再滑动构造测试序列，
        保证测试集每个样本都有完整的历史窗口。
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_arr = X_tr.values if hasattr(X_tr, "values") else np.asarray(X_tr)
    y_tr_arr = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
    X_te_arr = X_te.values if hasattr(X_te, "values") else np.asarray(X_te)

    seq_len = min(10, len(X_tr_arr) // 2)
    # 训练序列，复用共享滑窗函数
    X_seq, y_seq = _build_sequences(X_tr_arr, y_tr_arr, seq_len)

    model = _make_lstm_net(X_tr_arr.shape[1], 64, 2, device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    # 测试集：需要构造序列（用训练集末尾 + 测试集开头补全）
    model.eval()
    with torch.no_grad():
        # 拼接训练集末尾用于构造测试序列，y=None 表示只构造 X 序列
        full = np.vstack([X_tr_arr[-seq_len:], X_te_arr])
        X_te_seq, _ = _build_sequences(full, None, seq_len)
        X_te_t = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        proba = model(X_te_t).cpu().numpy().flatten()
    return proba


def predict_latest(X, y, best_name: str) -> float:
    """用冠军模型在全部数据上重训，预测最新一交易日的上涨概率。

    参数:
        X: 全量特征矩阵。
        y: 全量标签。
        best_name: 冠军模型名称。

    返回:
        float: 下一交易日上涨概率，范围 [0, 1]；若 LSTM 无可用样本则返回 0.5（中性）。

    说明:
        与赛跑阶段不同，这里用全部数据训练以利用最多信息，再取最后一行特征做预测。
        对 LSTM，取训练输出概率数组的最后一个元素作为最新预测。
    """
    models = get_models([best_name])
    model = models[best_name]
    if best_name == "LSTM":
        proba = _train_lstm(X, y)
        # 若序列样本为空（数据极少），返回 0.5 表示无倾向，避免异常
        return float(proba[-1]) if len(proba) > 0 else 0.5
    model.fit(X, y)
    return float(model.predict_proba(X.iloc[[-1]])[:, 1][0])
