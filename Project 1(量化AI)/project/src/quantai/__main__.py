"""python -m quantai 入口。

设计说明：
    - 本文件不直接实现业务逻辑，而是把项目根目录加入 sys.path 后
      委托执行 project/run.py。这样做的好处是：
        1. 保持 run.py 作为唯一的应用启动入口，避免两套启动逻辑分叉。
        2. python -m quantai 与直接 python run.py 行为一致。
    - parents[3] 的推导：本文件位于 src/quantai/__main__.py，
      上溯 3 级到 project/（即 run.py 所在目录）。
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    # 将项目根目录插入 sys.path 最前面，确保 import 优先找到本地模块
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    # 以 __main__ 名义运行 run.py，使其内部的 if __name__ == "__main__" 生效
    runpy.run_path(str(root / "run.py"), run_name="__main__")
