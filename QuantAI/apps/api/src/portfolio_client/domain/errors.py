from typing import Any


class DomainError(Exception):
    """A stable business error, mapped to transport status at the API boundary."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
