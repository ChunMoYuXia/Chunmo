"""路径管理：统一解析项目根目录、配置目录、数据目录、导出目录。

支持两种运行模式：
  - 开发模式：从源码运行，根目录 = project/
  - 打包模式：PyInstaller 打包后，根目录 = exe 所在目录

所有路径都基于此文件解析，避免相对路径在不同工作目录下出错。

核心思路：
    - 不依赖 os.getcwd()，而是以「本文件位置」为锚点推导根目录。
      这样无论用户在哪个目录启动程序（如 cd 到子目录运行），
      都能正确找到 config/data 等目录。
    - 使用 pathlib.Path 而非 os.path，API 更面向对象，跨平台更安全。
"""
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    """项目根目录：开发模式取 project/，打包模式取 exe 同级目录。

    判定逻辑：
        - sys.frozen 仅在 PyInstaller 等打包环境下为 True，
          此时 sys.executable 指向生成的 exe 文件。
        - 开发模式下 __file__ = .../src/quantai/paths.py，
          parents[2] 即上溯 3 级到 project/ 目录。

    Returns:
        Path: 项目根目录的绝对路径。
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后，exe 所在目录
        return Path(sys.executable).resolve().parent
    # 开发模式：src/quantai/paths.py -> 上溯 3 级到 project/
    return Path(__file__).resolve().parents[2]


# 以下常量在模块加载时即计算，全局共享
ROOT_DIR = _project_root()
# 配置文件目录（providers.json / prompt_template.txt / keys.env 等）
CONFIG_DIR = ROOT_DIR / "config"
# 运行时数据目录（股票历史数据缓存等）
DATA_DIR = ROOT_DIR / "data"
# 导出文件目录（研报 PDF / 图表 PNG 等）
EXPORT_DIR = ROOT_DIR / "exports"
# 日志目录（quantai.log / audit.jsonl）
LOG_DIR = ROOT_DIR / "logs"

# 配置文件路径
PROVIDERS_FILE = CONFIG_DIR / "providers.json"
PROMPT_FILE = CONFIG_DIR / "prompt_template.txt"
COMPARE_FILE = CONFIG_DIR / "compare_template.txt"
KEY_FILE = CONFIG_DIR / "keys.env"
KEY_FILE_EXAMPLE = CONFIG_DIR / "keys.env.example"


def ensure_dirs() -> None:
    """确保运行时需要的目录存在。

    使用 mkdir(parents=True, exist_ok=True)：
        - parents=True：递归创建缺失的父目录，避免逐级判断。
        - exist_ok=True：目录已存在时不报错，保证幂等。
    应在任何文件读写操作前调用，作为统一的目录守卫。
    """
    for d in (CONFIG_DIR, DATA_DIR, EXPORT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
