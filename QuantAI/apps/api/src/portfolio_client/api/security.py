import secrets

from fastapi import Request

from portfolio_client.config import Settings
from portfolio_client.domain.errors import DomainError


def access_failure(request: Request, settings: Settings) -> DomainError | None:
    """Local-only binding still needs authentication against hostile browser origins."""
    if request.url.hostname not in settings.allowed_hosts:
        return DomainError("HOST_REJECTED", "请求主机不受信任")
    origin = request.headers.get("Origin")
    if origin is not None and origin not in settings.allowed_origins:
        return DomainError("ORIGIN_REJECTED", "请求来源不受信任")
    if request.url.path == "/health":
        return None
    if not request.url.path.startswith("/v1/"):
        return None
    if request.method == "OPTIONS":
        return None
    authorization = request.headers.get("Authorization", "")
    header_token = authorization[7:] if authorization.startswith("Bearer ") else ""
    cookie_token = request.cookies.get("portfolio_session", "")
    if (
        not header_token
        and cookie_token
        and request.method not in {"GET", "HEAD", "OPTIONS"}
        and origin is None
    ):
        return DomainError("ORIGIN_REJECTED", "浏览器写操作需要可信来源")
    token = header_token or cookie_token
    expected = settings.session_token.get_secret_value()
    if not token or not secrets.compare_digest(token.encode(), expected.encode()):
        return DomainError("UNAUTHENTICATED", "需要有效的本地会话")
    return None
