from celery import Celery
from same_api.settings.config import get_settings


settings = get_settings()


celery_app = Celery(
    "same_api",
    broker=settings.run.celery_broker_url,
    backend=settings.run.celery_result_backend,
)

# Basic sensible defaults
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_max_tasks_per_child=50,
    # Увеличиваем таймауты для больших файлов
    task_soft_time_limit=3600,  # 1 час мягкий таймаут
    task_time_limit=7200,       # 2 часа жесткий таймаут
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)


def get_celery_app() -> Celery:
    return celery_app

# Ensure tasks module is imported so Celery can register tasks when worker starts
# noqa: F401
try:
    import same_api.api.configuration.tasks.catalog_tasks  # type: ignore
    import same_api.api.configuration.tasks.search_tasks  # type: ignore
except Exception:  # pragma: no cover
    # In API runtime this import is not required; only worker needs it
    pass


