"""用户自定义设置中心：指标参数、图表显示、研报样式、缓存策略。

所有设置存于 config/settings.json，首次运行自举生成默认值。
"""
import json
from copy import deepcopy

from .paths import CONFIG_DIR, ensure_dirs

SETTINGS_FILE = CONFIG_DIR / "settings.json"

DEFAULT_SETTINGS = {
    # ── 技术指标参数 ──
    "indicators": {
        "ma_windows": [5, 10, 20],      # 均线窗口
        "rsi_window": 14,               # RSI 周期
        "macd_fast": 12,                # MACD 快线
        "macd_slow": 26,                # MACD 慢线
        "macd_signal": 9,               # MACD 信号线
        "boll_window": 20,              # 布林带窗口
        "boll_std": 2.0,                # 布林带标准差倍数
        "kdj_n": 9,                     # KDJ N
        "kdj_m1": 3,                    # KDJ M1
        "kdj_m2": 3,                    # KDJ M2
    },
    # ── 图表显示 ──
    "chart": {
        "show_volume": True,            # 成交量副图
        "show_rsi": True,               # RSI 副图
        "show_macd": True,              # MACD 副图
        "show_kdj": False,              # KDJ 副图（默认关，避免太挤）
        "show_boll": True,              # 布林带
        "show_ma": True,                # 均线
        "height": 900,                  # 图表高度
    },
    # ── 研报样式 ──
    "report": {
        "max_words": 300,               # 股评字数上限
        "style": "要点式",              # 要点式 / 成段
        "include_rating": True,         # 是否输出投资评级
    },
    # ── 数据缓存 ──
    "cache": {
        "enabled": True,                # 是否启用缓存
        "ttl_days": 7,                  # 缓存有效期（天）
    },
    # ── 数据源 ──
    "data": {
        "primary": "baostock",          # 主数据源 baostock / akshare
    },
    # ── 机器学习 ──
    "ml": {
        "default_models": ["逻辑回归", "XGBoost", "LightGBM"],  # 默认参赛模型
        "n_splits": 5,                  # 时序交叉验证折数
        "lstm_hidden": 64,              # LSTM 隐藏层维度
        "lstm_layers": 2,               # LSTM 层数
        "lstm_epochs": 50,              # LSTM 训练轮数
    },
}


def _bootstrap() -> None:
    ensure_dirs()
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(
            json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_settings() -> dict:
    """读取设置；缺失字段自动用默认值补全（向前兼容）。"""
    _bootstrap()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    # 深度合并：默认值兜底
    merged = deepcopy(DEFAULT_SETTINGS)
    for section, values in data.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def save_settings(settings: dict) -> None:
    """写回设置。"""
    ensure_dirs()
    SETTINGS_FILE.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def reset_settings() -> None:
    """恢复出厂设置。"""
    save_settings(deepcopy(DEFAULT_SETTINGS))
