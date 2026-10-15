import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from portfolio_client.api.errors import (
    domain_error_handler,
    error_response,
    http_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from portfolio_client.api.routes.ai import create_ai_router
from portfolio_client.api.routes.jobs import create_jobs_router
from portfolio_client.api.routes.market import create_market_router
from portfolio_client.api.routes.paper import create_paper_router
from portfolio_client.api.routes.portfolios import create_portfolio_router
from portfolio_client.api.routes.quotes import create_quotes_router
from portfolio_client.api.routes.reports import create_reports_router
from portfolio_client.api.routes.research import create_risk_router
from portfolio_client.api.routes.screening import create_screening_router
from portfolio_client.api.schemas import Envelope, ErrorEnvelope, LocalConfig, Metadata
from portfolio_client.api.security import access_failure
from portfolio_client.api.static import mount_static_client
from portfolio_client.application.ai import AIService
from portfolio_client.application.datasets import DatasetService
from portfolio_client.application.jobs import JobService
from portfolio_client.application.market import MarketService
from portfolio_client.application.paper import PaperTradingService
from portfolio_client.application.portfolios import PortfolioService
from portfolio_client.application.reports import ReportService
from portfolio_client.application.research import ResearchService
from portfolio_client.application.risk import RiskService
from portfolio_client.application.screening import ScreeningService
from portfolio_client.config import Settings
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.ai_models import AIProvider
from portfolio_client.infrastructure.ai.compatible import CompatibleProvider
from portfolio_client.infrastructure.db.ai_runs import SQLiteAIRuns
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.jobs import SQLiteJobs
from portfolio_client.infrastructure.db.market_queries import SQLiteMarketData
from portfolio_client.infrastructure.db.paper import SQLitePaperRepository
from portfolio_client.infrastructure.db.portfolios import SQLitePortfolios
from portfolio_client.infrastructure.db.repositories import SQLiteDatasetRepository
from portfolio_client.infrastructure.db.screening_presets import SQLiteScreeningPresets
from portfolio_client.infrastructure.files.sources import LocalSourceStore
from portfolio_client.infrastructure.imports.csv import CsvImporter
from portfolio_client.infrastructure.imports.synthetic import generate_dataset
from portfolio_client.infrastructure.jobs.runner import JobRunner


def create_app(settings: Settings, *, ai_provider: AIProvider | None = None) -> FastAPI:
    database = Database(settings.database_url)
    datasets = DatasetService(
        SQLiteDatasetRepository(database),
        CsvImporter(),
        generate_dataset,
        LocalSourceStore(settings.data_dir / "sources"),
    )
    market = MarketService(datasets.repository, SQLiteMarketData(database))
    research = ResearchService(datasets.repository, market.market)
    jobs = JobService(SQLiteJobs(database), research)
    runner = JobRunner(jobs, research)
    reports = ReportService(jobs.repository)
    ai = AIService(
        SQLiteAIRuns(database),
        ai_provider or (CompatibleProvider(settings.ai) if settings.ai else None),
        settings.ai.model if settings.ai else None,
        reports,
        market,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            ai.repository.recover()
            runner.start()
            yield
        finally:
            await ai.close()
            runner.stop()
            database.close()

    app = FastAPI(
        title="组合研究台",
        version="0.1.0",
        docs_url=None,
        openapi_url=None,
        lifespan=lifespan,
        responses={
            status: {"model": ErrorEnvelope}
            for status in (400, 401, 403, 404, 409, 413, 422, 429, 500)
        },
    )
    app.state.settings = settings
    app.add_exception_handler(DomainError, domain_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.middleware("http")
    async def local_boundary(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get("X-Trace-Id", "")
        trace_id = supplied if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", supplied) else uuid4().hex
        request.state.trace_id = trace_id
        failure = access_failure(request, settings)
        if failure is None and request.url.path.startswith("/v1/"):
            try:
                declared_size = int(request.headers.get("Content-Length", "0"))
                if declared_size < 0:
                    raise ValueError("negative length")
                if declared_size > 52 * 1024 * 1024:
                    failure = DomainError("FILE_TOO_LARGE", "单次上传总大小不得超过52MB")
            except ValueError:
                failure = DomainError("INVALID_REQUEST_LENGTH", "请求长度无效")
        try:
            response = error_response(failure, trace_id) if failure else await call_next(request)
        except Exception as error:
            # Catch before ServerErrorMiddleware re-raises to Uvicorn's raw traceback logger.
            response = await unexpected_error_handler(request, error)
        response.headers["X-Trace-Id"] = trace_id
        response.headers["Cache-Control"] = "no-store"
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Trace-Id"],
        expose_headers=["X-Trace-Id"],
        allow_credentials=True,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/config", response_model=Envelope[LocalConfig])
    def local_config(request: Request) -> Envelope[LocalConfig]:
        return Envelope(data=LocalConfig(), meta=Metadata(trace_id=request.state.trace_id))

    @app.post("/v1/session", response_model=Envelope[dict[str, bool]])
    def browser_session(request: Request, response: Response) -> Envelope[dict[str, bool]]:
        response.set_cookie(
            "portfolio_session",
            settings.session_token.get_secret_value(),
            httponly=True,
            samesite="strict",
            secure=False,  # Loopback HTTP only; the server cannot bind a public address.
            path="/",
        )
        return Envelope(data={"ready": True}, meta=Metadata(trace_id=request.state.trace_id))

    app.include_router(create_market_router(datasets))
    app.include_router(create_quotes_router(market))
    app.include_router(create_jobs_router(jobs))
    app.include_router(create_reports_router(reports))
    app.include_router(create_ai_router(ai))
    portfolios = SQLitePortfolios(database)
    app.include_router(create_portfolio_router(PortfolioService(market, portfolios)))
    app.include_router(create_risk_router(RiskService(market, portfolios)))
    app.include_router(
        create_paper_router(
            PaperTradingService(SQLitePaperRepository(database), datasets.repository, market.market)
        )
    )
    app.include_router(
        create_screening_router(ScreeningService(market, SQLiteScreeningPresets(database)))
    )
    # Finalize lazy route/dependency schemas before concurrent requests can build them.
    if settings.static_dir is not None:
        mount_static_client(app, settings.static_dir)
    # The schema stays internal: HTTP docs and /openapi.json are disabled.
    app.openapi()
    return app
