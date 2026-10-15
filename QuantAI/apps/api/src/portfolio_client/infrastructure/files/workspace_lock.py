import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from portfolio_client.domain.errors import DomainError


@contextmanager
def workspace_lock(data_dir: Path) -> Iterator[None]:
    """Only the service owning this OS lock may recover persisted running jobs."""
    with (data_dir / "service.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DomainError(
                "DATA_DIRECTORY_IN_USE", "该数据目录已有本地服务，请关闭原客户端后重试"
            ) from None
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
