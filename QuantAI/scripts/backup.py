import argparse
from pathlib import Path

from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.files.backup import create_backup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在客户端关闭后备份数据库与原始来源文件"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = create_backup(args.data_dir, args.output)
    except DomainError as error:
        parser.exit(2, f"{error.code}: {error.message}\n")
    print(f"备份完成：{receipt.directory}；来源文件 {receipt.source_files} 个")


if __name__ == "__main__":
    main()
