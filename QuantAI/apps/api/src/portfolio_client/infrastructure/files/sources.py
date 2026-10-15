import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4

from portfolio_client.domain.errors import DomainError


class LocalSourceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, content_hash: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise DomainError("INVALID_SOURCE_HASH", "源文件摘要无效")
        path = self.root / f"{content_hash}.bin"
        if path.resolve().parent != self.root:
            raise DomainError("INVALID_SOURCE_PATH", "源文件路径越界")
        return path

    def put(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        target = self._path(digest)
        if target.exists():
            self.read(digest)
            return digest
        temporary = self.root / f"{digest}.{uuid4().hex}.tmp"
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def read(self, content_hash: str) -> bytes:
        path = self._path(content_hash)
        if not path.is_file():
            raise DomainError("SOURCE_NOT_FOUND", "原始资料缺失，请从备份恢复")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != content_hash:
            raise DomainError("SOURCE_INTEGRITY_ERROR", "原始资料摘要不匹配，拒绝作为证据返回")
        return content
