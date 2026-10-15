import logging
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from portfolio_client.domain.errors import DomainError

STATUS_CODES = {
    "INTERNAL_ERROR": 500,
    "JOB_CAPACITY": 429,
    "JOB_NOT_READY": 409,
    "UNAUTHENTICATED": 401,
    "ORIGIN_REJECTED": 403,
    "HOST_REJECTED": 403,
    "IDEMPOTENCY_CONFLICT": 409,
    "ACCOUNT_PAUSED": 409,
    "INVALID_JOB_TRANSITION": 409,
    "IMPORT_INVALID": 422,
    "CALENDAR_REQUIRED": 422,
    "INVALID_CALENDAR": 422,
    "INVALID_MAPPING": 422,
    "INVALID_AS_OF": 422,
    "FILE_TOO_LARGE": 413,
}


def error_response(error: DomainError, trace_id: str) -> JSONResponse:
    status = STATUS_CODES.get(error.code, 404 if error.code.endswith("_NOT_FOUND") else 400)
    return JSONResponse(
        status_code=status,
        headers={"X-Trace-Id": trace_id, "Cache-Control": "no-store"},
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "trace_id": trace_id,
            }
        },
    )


async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
    code = "ROUTE_NOT_FOUND" if error.status_code == 404 else "HTTP_ERROR"
    message = "接口不存在" if error.status_code == 404 else "请求方法或HTTP参数不受支持"
    response = error_response(DomainError(code, message), request.state.trace_id)
    response.status_code = error.status_code
    return response


async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", uuid4().hex)
    logging.getLogger(__name__).error(
        "request failure trace=%s type=%s", trace_id, type(error).__name__
    )
    return error_response(
        DomainError("INTERNAL_ERROR", "本地服务发生异常，请根据trace_id检查诊断"), trace_id
    )


async def domain_error_handler(request: Request, error: DomainError) -> JSONResponse:
    return error_response(error, request.state.trace_id)


async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    # Raw input may contain personal data or credentials; return locations and reasons only.
    issues: list[dict[str, Any]] = [
        {"location": list(issue["loc"]), "message": issue["msg"], "type": issue["type"]}
        for issue in error.errors()
    ]
    response = error_response(
        DomainError("VALIDATION_ERROR", "输入参数不合法", {"issues": issues}),
        request.state.trace_id,
    )
    response.status_code = 422
    return response
