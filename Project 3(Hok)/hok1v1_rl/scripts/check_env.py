"""环境验证脚本 —— 只检查"环境是否可用"，不检查业务代码逻辑。

在开始训练前跑一次；任何一项 FAIL 都会给出具体的修复命令。

运行::

    conda activate hok
    python scripts/check_env.py                 # 默认还测 gamecore 连通性
    python scripts/check_env.py --no-gamecore   # 只查本地依赖，不连 gamecore

检查项（对应审查报告的 A 组）：
  A1  依赖完整性：numpy / pyyaml / tensorboard / loguru / requests / h5py / pyzmq
  A2  Python 版本与 hok 原生扩展 ABI 是否匹配
  A3  gamecore-server 是否可达（TCP 23432）与 zmq 端口是否被占用
  A4  torch 是否带当前 GPU 架构的 kernel（RTX 5070 = Blackwell sm_120）
"""

import argparse
import importlib
import os
import socket
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# 结果收集：(名称, 状态, 说明)
RESULTS = []

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    icon = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}[status]
    print(f"{icon} {name}" + (f"  — {detail}" if detail else ""))


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# A2. Python 版本与 ABI
# ---------------------------------------------------------------------------

def check_python():
    section("A2  Python 版本与 hok 原生扩展 ABI")

    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info[2]}"
    # hok_env 的 setup.py: python_requires=">=3.6, <3.10"
    if (3, 6) <= (major, minor) < (3, 10):
        record("Python 版本", OK, f"{ver}（hok_env 要求 >=3.6,<3.10）")
    else:
        record("Python 版本", FAIL,
               f"{ver} 不受支持；hok_env 要求 >=3.6,<3.10，"
               f"且原生扩展只有 cpython-36m/37m/38/39 四个 ABI。请用 Python 3.9")

    # ABI：EXT_SUFFIX 是否能在 lib/ 里找到对应的 interface .so
    try:
        import sysconfig
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    except Exception:      # noqa: BLE001
        suffix = ""

    found_lib = None
    for cand in _candidate_hok_lib_dirs():
        if (cand / f"interface{suffix}").exists():
            found_lib = cand
            break

    if found_lib is not None:
        record("hok 原生扩展 ABI", OK, f"interface{suffix} 存在（{found_lib}）")
    elif suffix:
        avail = []
        for cand in _candidate_hok_lib_dirs():
            avail += [p.name for p in cand.glob("interface*.so")]
        if avail:
            record("hok 原生扩展 ABI", FAIL,
                   f"当前解释器需要 interface{suffix}，但只有 {sorted(set(avail))}")
        else:
            record("hok 原生扩展 ABI", WARN,
                   f"未找到 interface*.so（当前需要 {suffix}）；"
                   f"若 hok_env 是 pip 安装的，可能位于 site-packages 内")


def _candidate_hok_lib_dirs():
    """返回可能的 hok/hok1v1/lib 目录（hok_env 源码树 + 已安装位置）。"""
    dirs = []
    env_hint = os.getenv("HOK_ENV_ROOT")
    bases = [str(_HERE.parent.parent / "hok_env" / "hok_env")]
    if env_hint:
        bases.insert(0, env_hint)
    for base in bases:
        d = Path(base) / "hok" / "hok1v1" / "lib"
        if d.is_dir():
            dirs.append(d)
    try:
        import hok.hok1v1.lib as lib
        d = Path(lib.__file__).parent
        if d.is_dir():
            dirs.append(d)
    except Exception:      # noqa: BLE001
        pass
    return dirs


# ---------------------------------------------------------------------------
# A1. 依赖完整性
# ---------------------------------------------------------------------------

#: (import 名, pip 名, 是否关键)
_REQUIRED = [
    ("numpy", "numpy", True),
    ("yaml", "pyyaml", True),
    ("torch", "torch", True),
    ("loguru", "loguru", True),        # hok.common.log 依赖，缺了 env1v1 直接 import 失败
    ("requests", "requests", True),    # hok_env 官方依赖
    ("tensorboard", "tensorboard", False),
    ("h5py", "h5py", False),           # 官方 actor 的 h5 采样需要
    ("zmq", "pyzmq", False),           # 官方 actor / gamecore_client 可能需要
    ("wandb", "wandb", False),
]


def check_dependencies():
    section("A1  依赖完整性")
    missing_critical = []
    for mod, pkg, critical in _REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            record(f"import {mod}", OK, f"{ver}")
        except Exception as exc:      # noqa: BLE001
            if critical:
                missing_critical.append(pkg)
                record(f"import {mod}", FAIL, f"{type(exc).__name__}: {exc}")
            else:
                record(f"import {mod}", WARN, f"未安装（可选）：{type(exc).__name__}")

    if missing_critical:
        print()
        print("  修复：pip install " + " ".join(missing_critical))
        print("        另需 hok 本体：cd ~/hok/hok_env/hok_env && pip install -e .")


# ---------------------------------------------------------------------------
# hok 包本身
# ---------------------------------------------------------------------------

def check_hok_package():
    section("A1b  hok 包（hok_env）")
    try:
        import hok
        record("import hok", OK, getattr(hok, "__file__", "?"))
    except Exception as exc:      # noqa: BLE001
        record("import hok", FAIL, f"{type(exc).__name__}: {exc}")
        print("  修复：cd ~/hok/hok_env/hok_env && pip install -e .")
        return

    for mod in ("hok.hok1v1.env1v1", "hok.hok1v1.hero_config",
                "hok.common.camp", "hok.common.gamecore_client",
                "hok.hok1v1.lib.interface"):
        try:
            importlib.import_module(mod)
            record(f"import {mod}", OK)
        except Exception as exc:      # noqa: BLE001
            record(f"import {mod}", FAIL, f"{type(exc).__name__}: {exc}")

    # 明确验证审查报告中指出的"不存在的符号"，避免再次误用
    try:
        from hok.hok1v1.hero_config import get_default_hero_config  # noqa: F401
        record("hero_config.get_default_hero_config", OK)
    except Exception as exc:      # noqa: BLE001
        record("hero_config.get_default_hero_config", FAIL, str(exc))

    try:
        from hok.common.camp import (      # noqa: F401
            HERO_DICT, GameMode, camp_iterator_1v1_roundrobin_camp_heroes,
        )
        record("camp: HERO_DICT/GameMode/camp_iterator_1v1_*", OK,
               f"{len(HERO_DICT)} 个英雄")
    except Exception as exc:      # noqa: BLE001
        record("camp 公开符号", FAIL, str(exc))

    try:
        from hok.hok1v1.env1v1 import HoK1v1
        import inspect
        sig = str(inspect.signature(HoK1v1.__init__))
        record("HoK1v1 构造签名", OK, sig)
    except Exception as exc:      # noqa: BLE001
        record("HoK1v1 构造签名", WARN, f"无法内省：{exc}")


def check_workdir_config():
    """检查 CWD 下是否有 config.json —— 这是 native 层的硬性要求。

    实测：native `interface` 在 ``env.reset()` 时会以**相对路径**读取 ``config.json``
    （即奖励权重 + log_level 那个文件）。若 CWD 下没有该文件，会在
    ``sgame_ai_src/common/resource_helper.cpp:203`` 触发
    ``Assertion 'm_doc.IsObject()' failed`` 并 **core dump**，且报错信息
    完全看不出是缺文件。

    官方文档让 test_env.py 在 ``hok_env/hok/hok1v1/unit_test/`` 下运行，
    正是因为那个目录里有一个 config.json。本项目在根目录放了一份。
    """
    section("A3b  工作目录 config.json（native 层按 CWD 相对路径读取）")

    here = Path.cwd() / "config.json"
    if here.exists():
        try:
            import json
            with open(here, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                record("CWD/config.json", OK,
                       f"存在且是 JSON 对象（{len(obj)} 个键）: {sorted(obj)[:4]}...")
            else:
                record("CWD/config.json", FAIL,
                       f"存在但不是 JSON 对象（是 {type(obj).__name__}）")
        except Exception as exc:      # noqa: BLE001
            record("CWD/config.json", FAIL, f"解析失败：{exc}")
    else:
        record("CWD/config.json", FAIL,
               f"当前工作目录 {Path.cwd()} 下没有 config.json")
        print()
        print("  请在项目根目录从 config.example.json 生成本地 config.json 后重试。")


# ---------------------------------------------------------------------------
# A4. torch / GPU 架构
# ---------------------------------------------------------------------------

def check_torch_gpu():
    section("A4  torch 与 GPU 架构（RTX 5070 = Blackwell sm_120）")
    try:
        import torch
    except Exception as exc:      # noqa: BLE001
        record("import torch", FAIL, str(exc))
        return

    record("torch 版本", OK, f"{torch.__version__} (cuda {torch.version.cuda})")

    if not torch.cuda.is_available():
        record("CUDA 可用性", WARN,
               "torch.cuda.is_available() == False（将回退 CPU，训练会极慢）")
        return

    # 注意：is_available() 为 True 并**不代表** kernel 能用
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_arch_list()
    record("GPU", OK, f"{name} capability={cap}")
    record("torch 编译架构", OK, str(arch))

    sm = f"sm_{cap[0]}{cap[1]}"
    if sm in arch:
        record("架构匹配", OK, f"{sm} 在 arch_list 中")
    else:
        record("架构匹配", FAIL,
               f"{sm} 不在 arch_list {arch} 中 —— 任何 kernel 都会失败")

    # 真正跑一次 kernel
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        _ = (a @ b).sum().item()
        torch.cuda.synchronize()
        record("GPU 真实 kernel", OK, "matmul 成功执行")
    except Exception as exc:      # noqa: BLE001
        record("GPU 真实 kernel", FAIL, f"{type(exc).__name__}: {exc}")
        print()
        print("  修复（RTX 5070 需要 CUDA 12.8+ 的轮子，且 cu128 不含 cp38）：")
        print("    conda create -n hok python=3.9     # 3.9 才能同时满足 .so 与 cu128")
        print("    conda activate hok")
        print("    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision")

    # 显存
    try:
        props = torch.cuda.get_device_properties(0)
        record("显存", OK, f"{props.total_memory / 1e9:.2f} GB")
    except Exception:      # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# A3. gamecore 连通性
# ---------------------------------------------------------------------------

def _tcp_open(host, port, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:      # noqa: BLE001
        return False


def check_gamecore(cfg=None):
    section("A3  gamecore 连通性与 zmq 端口")

    addr = "127.0.0.1:23432"
    payload = None
    if cfg is not None:
        addr = cfg.env.gamecore_server_addr

    host, _, port_s = addr.partition(":")
    port = int(port_s or 23432)

    if _tcp_open(host, port):
        record(f"gamecore {addr}", OK, "TCP 可连接")
    else:
        record(f"gamecore {addr}", FAIL, "连接被拒绝（服务端未启动？）")
        print()
        print("  请按授权环境的说明启动本地 gamecore 服务，并核对地址与端口。")

    # zmq 端口：player i -> zmq_port_begin + i
    begin = cfg.env.zmq_port_begin if cfg is not None else 35150
    n = cfg.env.player_num if cfg is not None else 2
    busy = []
    for i in range(n):
        p = begin + i
        if _tcp_open("127.0.0.1", p, timeout=1.0):
            busy.append(p)
    if busy:
        record("zmq 端口", WARN, f"以下端口已被占用：{busy}（残留进程？）")
    else:
        record("zmq 端口", OK, f"{begin}..{begin + n - 1} 空闲")

    # WSL -> Windows 主机地址（gamecore 与 actor 不同机时才需要）
    win_host = None
    try:
        import subprocess
        out = subprocess.run(["ip", "route", "show", "default"],
                             capture_output=True, text=True, timeout=5).stdout
        parts = out.split()
        if "via" in parts:
            win_host = parts[parts.index("via") + 1]
    except Exception:      # noqa: BLE001
        pass
    if win_host:
        record("WSL 默认网关（Windows 主机）", OK, win_host)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gamecore", action="store_true",
                        help="跳过 gamecore 连通性检查")
    parser.add_argument("--config", default=None,
                        help="YAML 配置路径（用于读取 gamecore 地址与 zmq 端口）")
    args = parser.parse_args()

    print("HoK 1v1 RL 环境验证")
    print(f"解释器: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")

    cfg = None
    if args.config:
        try:
            from hok1v1_rl.config import load_config
            cfg = load_config(args.config)
            print(f"配置: {args.config}")
        except Exception as exc:      # noqa: BLE001
            print(f"（配置加载失败，使用默认值：{exc}）")
    if cfg is None:
        try:
            from hok1v1_rl.config import load_config
            default_yaml = _HERE.parent / "configs" / "default.yaml"
            if default_yaml.exists():
                cfg = load_config(str(default_yaml))
        except Exception:      # noqa: BLE001
            pass

    check_python()
    check_dependencies()
    check_hok_package()
    check_workdir_config()
    check_torch_gpu()
    if not args.no_gamecore:
        check_gamecore(cfg)

    # ---- 汇总 ----
    section("汇总")
    n_fail = sum(1 for _, s, _ in RESULTS if s == FAIL)
    n_warn = sum(1 for _, s, _ in RESULTS if s == WARN)
    n_ok = sum(1 for _, s, _ in RESULTS if s == OK)
    print(f"OK={n_ok}  WARN={n_warn}  FAIL={n_fail}")
    if n_fail:
        print()
        print("未通过项：")
        seen = set()
        for name, s, detail in RESULTS:
            if s == FAIL and name not in seen:
                seen.add(name)
                print(f"  - {name}: {detail}")
        print()
        print("环境尚未就绪，请先修复上面的 FAIL 项。")
        sys.exit(1)

    print()
    print("环境检查通过。下一步：python scripts/_smoke_test.py（离线）")
    print("                       python scripts/check_gamecore_run.py（需 gamecore）")


if __name__ == "__main__":
    main()
