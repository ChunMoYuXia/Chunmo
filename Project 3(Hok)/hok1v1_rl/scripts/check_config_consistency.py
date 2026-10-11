"""检查 YAML 与 dataclass 的全部配置字段。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hok1v1_rl.config import Config, load_config


def main():
    defaults, actual = Config().to_dict(), load_config(ROOT / "configs/default.yaml").to_dict()
    differences = [f"{section}.{name}: {value!r} != {actual[section][name]!r}"
                   for section, fields in defaults.items() for name, value in fields.items()
                   if value != actual[section][name]]
    if differences:
        raise SystemExit("\n".join(differences))
    print(f"全部 {sum(map(len, defaults.values()))} 个配置字段一致")

if __name__ == "__main__":
    main()
