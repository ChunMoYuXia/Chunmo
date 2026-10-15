import logging
from threading import Event, Thread

from portfolio_client.application.jobs import JobService
from portfolio_client.application.research import ResearchService
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.backtest_models import Progress

logger = logging.getLogger(__name__)


class JobRunner:
    """One worker, a bounded durable queue, and cooperative cancellation at each session."""

    def __init__(self, jobs: JobService, research: ResearchService) -> None:
        self.jobs, self.research = jobs, research
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.jobs.recover()
        self._stop.clear()
        self._thread = Thread(target=self._run, name="research-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("研究任务未在关闭期限内停止")

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(0.25)

    def run_once(self) -> bool:
        job = self.jobs.claim()
        if job is None:
            return False
        try:

            def update(progress: Progress) -> None:
                if self._stop.is_set():
                    raise DomainError("JOB_INTERRUPTED", "应用关闭，研究任务已中断")
                self.jobs.progress(job.id, progress)

            run = self.research.run(job.payload, update)
            if self._stop.is_set():
                raise DomainError("JOB_INTERRUPTED", "应用关闭，研究任务已中断")
            self.jobs.complete(job.id, run)
        except DomainError as error:
            if error.code != "JOB_CANCELLED":
                self.jobs.fail(job.id, error.code, error.message)
        except Exception as error:
            logger.error("research failure trace=%s type=%s", job.trace_id, type(error).__name__)
            self.jobs.fail(job.id, "JOB_FAILED", "研究任务异常终止，请根据trace_id检查本地诊断")
        return True
