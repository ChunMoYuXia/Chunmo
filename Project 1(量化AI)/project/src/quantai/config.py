"""配置中心：模型注册表、提示词模板、API 钥匙、自选池、自定义组。

出厂设置存在代码常量里；首次运行自动生成配置文件；之后一律读文件。
所有路径基于 paths.py 解析，保证任何工作目录下都能正常读写。

设计选型说明：
    - 采用「代码内置默认值 + 文件覆盖」模式，而非纯环境变量。
      好处是用户可在网页端直接编辑模板/厂商列表，无需重启进程；
      坏处是 keys.env 属敏感信息，必须排除在版本控制之外。
    - providers.json 使用 tuple(URL, model, env_var_name) 三元组保存，
      解析时还原为 tuple，避免 list 被误改后顺序错乱。
"""
import json
import os
import re

from .paths import (
    COMPARE_FILE,
    CONFIG_DIR,
    KEY_FILE,
    KEY_FILE_EXAMPLE,
    PROMPT_FILE,
    PROVIDERS_FILE,
    ensure_dirs,
)

CUSTOM_GROUPS_FILE = CONFIG_DIR / "custom_groups.json"

# ── 出厂设置 ──────────────────────────────────────────────

# 模型提供商注册表：厂商名 → (API 端点 URL, 默认模型 ID, 对应的环境变量名)
# 选择三元组而非 dict，是为了保持字段顺序稳定，序列化/反序列化时不易错位。
DEFAULT_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4/chat/completions", "glm-4-flash", "ZHIPU_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-plus", "DASHSCOPE_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1/chat/completions", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
}

# 自选池：组名 → {代码: 名称}，扩充到 10 组覆盖主要行业
STOCK_POOL = {
    "白酒组": {"600519": "贵州茅台", "000858": "五粮液", "000568": "泸州老窖", "600809": "山西汾酒"},
    "新能源组": {"300750": "宁德时代", "002594": "比亚迪", "601012": "隆基绿能", "300014": "亿纬锂能"},
    "金融组": {"601318": "中国平安", "600036": "招商银行", "601688": "华泰证券", "601601": "中国太保"},
    "科技组": {"002415": "海康威视", "000725": "京东方A", "600703": "三安光电", "002230": "科大讯飞"},
    "医药组": {"600276": "恒瑞医药", "603259": "药明康德", "300760": "迈瑞医疗", "600436": "片仔癀"},
    "消费组": {"603288": "海天味业", "600887": "伊利股份", "000333": "美的集团", "000651": "格力电器"},
    "半导体组": {"688981": "中芯国际", "603501": "韦尔股份", "002371": "北方华创", "603986": "兆易创新"},
    "军工组": {"600760": "中航沈飞", "600893": "航发动力", "600150": "中国船舶", "002179": "中航光电"},
    "地产组": {"600048": "保利发展", "000002": "万科A", "001979": "招商蛇口", "600383": "金地集团"},
    "券商组": {"600030": "中信证券", "300059": "东方财富", "601688": "华泰证券", "600837": "海通证券"},
}

# 6 段结构化研报模板（对标高级金融分析师）
DEFAULT_PROMPT_TEMPLATE = (
    "你是一名资深 A 股高级金融分析师，面向专业投资者。\n"
    "以下是股票 {code} 截至 {date} 的技术指标数据：\n"
    "{facts}\n"
    "请只基于上述数据，按以下六段输出专业研报：\n"
    "【行情概览】收盘价、当日涨跌幅、成交量变化、量比\n"
    "【趋势研判】均线排列（MA5/10/20）、价格相对均线位置、布林带位置\n"
    "【量价分析】放量/缩量判断、价量配合关系\n"
    "【技术指标】RSI 超买超卖、MACD 金叉死叉、KDJ 信号\n"
    "【关键价位】支撑位（近20日低/BOLL下轨）、压力位（近20日高/BOLL上轨）\n"
    "【投资评级】给出 买入/增持/中性/减持/卖出 五档评级，说明理由并给出仓位建议（如 建议仓位 X 成）\n"
    "{style_req}；不要编造数据中没有的信息，不确定就明说。"
)

DEFAULT_COMPARE_TEMPLATE = (
    "你是一名资深 A 股策略分析师，面向专业投资者。\n"
    "以下是两组股票在 {start} 至 {end} 的归一化收益率对比（起点对齐为 0，代表相对涨幅）：\n"
    "{stats}\n"
    "请比较两组强弱，按【整体强弱】【组内分化】【结论与配置建议】三段输出，总字数 300 字以内；"
    "结论部分需给出配置建议（建议超配/标配/低配哪组）；不要编造数据中没有的信息，不确定就明说。"
)

# 匹配提示词模板中的【章节标题】占位符，用于动态生成研报段落结构。
# 非贪婪匹配 .+? 确保遇到下一个 】 时立即终止，避免跨章节误匹配。
SECTION_RE = re.compile(r"【(.+?)】")


def _bootstrap() -> None:
    """首次运行：把出厂配置写入文件。

    设计意图：
        - 采用「存在则跳过」策略（if not exists），保证用户已修改的配置
          不会被出厂默认值覆盖。
        - 幂等设计：可在任意 load_* 入口调用，多次执行不会产生副作用。
        - 写入时使用 ensure_ascii=False 保留中文可读性，indent=2 便于人工编辑。
    """
    ensure_dirs()
    if not PROVIDERS_FILE.exists():
        PROVIDERS_FILE.write_text(
            json.dumps(DEFAULT_PROVIDERS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not PROMPT_FILE.exists():
        PROMPT_FILE.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    if not COMPARE_FILE.exists():
        COMPARE_FILE.write_text(DEFAULT_COMPARE_TEMPLATE, encoding="utf-8")
    if not KEY_FILE_EXAMPLE.exists():
        # 仅生成 .example 模板，不直接生成 keys.env，避免误提交空文件到版本库。
        KEY_FILE_EXAMPLE.write_text(
            "# 复制本文件为 keys.env 并填入钥匙。已设置的环境变量优先于本文件。\n"
            "DEEPSEEK_API_KEY=\n"
            "# ZHIPU_API_KEY=\n"
            "# DASHSCOPE_API_KEY=\n"
            "# MOONSHOT_API_KEY=\n",
            encoding="utf-8",
        )


# ── 读写接口 ──────────────────────────────────────────────

def load_providers() -> dict:
    """读取模型提供商配置。

    Returns:
        dict: 厂商名 → (URL, model, env_var_name) 三元组。
              JSON 反序列化后 list 会被还原为 tuple，保证调用方取值稳定。

    说明：每次读取都调用 _bootstrap()，确保首次运行时文件已生成。
    """
    _bootstrap()
    with open(PROVIDERS_FILE, encoding="utf-8") as fh:
        return {k: tuple(v) for k, v in json.load(fh).items()}


def save_providers(providers: dict) -> None:
    """保存模型提供商配置。

    Args:
        providers: 厂商名 → (URL, model, env_var_name) 的字典。
    """
    ensure_dirs()
    with open(PROVIDERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(providers, fh, ensure_ascii=False, indent=2)


def load_prompt_template() -> str:
    """读取研报提示词模板。

    Returns:
        str: 完整的提示词模板文本，含 {code}/{date}/{facts}/{style_req} 占位符。
    """
    _bootstrap()
    return PROMPT_FILE.read_text(encoding="utf-8")


def save_prompt_template(text: str) -> None:
    """保存研报提示词模板。

    Args:
        text: 用户自定义的提示词模板全文。
    """
    ensure_dirs()
    PROMPT_FILE.write_text(text, encoding="utf-8")


def reset_prompt_template() -> None:
    """恢复出厂研报提示词模板。

    直接覆盖写入 DEFAULT_PROMPT_TEMPLATE，用于用户改坏模板时一键还原。
    """
    save_prompt_template(DEFAULT_PROMPT_TEMPLATE)


def load_compare_template() -> str:
    """读取分组对比提示词模板。

    Returns:
        str: 对比模板文本，含 {start}/{end}/{stats} 占位符。
    """
    _bootstrap()
    return COMPARE_FILE.read_text(encoding="utf-8")


def section_titles(template: str) -> list[str]:
    """从提示词模板中提取所有【章节标题】。

    Args:
        template: 提示词模板全文。

    Returns:
        list[str]: 按出现顺序排列的章节标题列表（不含方括号）。
                   用于网页端展示模板结构，便于用户理解输出段落。
    """
    return SECTION_RE.findall(template)


def save_keys(updates: dict[str, str]) -> None:
    """增量保存 API 钥匙到 keys.env。

    Args:
        updates: 环境变量名 → 密钥值 的字典。

    设计要点：
        - 增量合并：先读取现有 keys.env，再用 updates 覆盖/追加，
          避免一次保存只写一个 key 导致其他 key 丢失。
        - 空值跳过：updates 中 value 为空字符串时不写入，
          允许用户通过传空来表示「不修改该字段」。
        - 未使用 ConfigParser：因为 keys.env 是简单的 KEY=VALUE 格式，
          手动解析更轻量，且能保留注释行的写入顺序。
    """
    current: dict[str, str] = {}
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            # 跳过空行和注释行，仅解析 KEY=VALUE 形式
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                current[k.strip()] = v.strip()
    for k, v in updates.items():
        if v:
            current[k] = v
    ensure_dirs()
    with open(KEY_FILE, "w", encoding="utf-8") as fh:
        for k, v in current.items():
            fh.write(f"{k}={v}\n")


def load_keys_into_env() -> None:
    """把 keys.env 中的钥匙加载到 os.environ。

    容错策略：
        - 文件不存在时直接返回，不抛异常。
        - 使用 os.environ.setdefault 而非直接赋值，保证「已设置的环境变量
          优先级高于文件」，方便 CI/容器环境通过 env 注入密钥。
        - 同样跳过空行和注释行。
    """
    if not KEY_FILE.exists():
        return
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


# ── 自定义股票组 ──────────────────────────────────────────

def load_custom_groups() -> dict:
    """读取用户保存的自定义组，返回 {组名: {代码: 名称}}。

    容错策略：
        - 文件不存在时返回空字典，而非抛异常。
        - JSON 解析失败或 IO 错误时降级为空字典，避免单个损坏文件
          导致整个应用无法启动。

    Returns:
        dict: 自定义股票组字典，格式同 STOCK_POOL。
    """
    if not CUSTOM_GROUPS_FILE.exists():
        return {}
    try:
        return json.loads(CUSTOM_GROUPS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_custom_groups(groups: dict) -> None:
    """保存自定义股票组。

    Args:
        groups: {组名: {代码: 名称}} 字典，会完全覆盖已有文件内容。
    """
    ensure_dirs()
    CUSTOM_GROUPS_FILE.write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def all_groups() -> dict:
    """内置组 + 自定义组合并（自定义组可覆盖同名内置组）。

    合并策略：
        - 先用 dict(STOCK_POOL) 浅拷贝内置组，避免修改原始常量。
        - 再用 update 合并自定义组，同 key 时自定义组优先，
          允许用户用自定义组替换内置组（如修改白酒组成分）。

    Returns:
        dict: 合并后的全部股票组字典。
    """
    merged = dict(STOCK_POOL)
    merged.update(load_custom_groups())
    return merged
