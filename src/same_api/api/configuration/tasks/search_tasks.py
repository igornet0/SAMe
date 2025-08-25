import time
from typing import Dict, Any, List

from .celery_app import get_celery_app
from same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig


celery_app = get_celery_app()


@celery_app.task(name="search.run_analogs")
def run_analogs(
    queries: List[str],
    method: str = "hybrid",
    similarity_threshold: float = 0.6,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Выполнить поиск аналогов в фоне.

    Предполагается, что модели уже сохранены ранее (после загрузки каталога),
    поэтому здесь мы загружаем модели и выполняем поиск.
    """
    start = time.time()

    # Конфигурация движка для поиска
    config = AnalogSearchConfig(
        search_method=method,
        similarity_threshold=similarity_threshold,
        max_results_per_query=max_results,
    )
    engine = AnalogSearchEngine(config)

    # Загружаем модели, сохраненные ранее в стандартной директории
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        try:
            loop.run_until_complete(engine.load_models())
        except Exception as load_err:
            return {
                "status": "error",
                "message": f"Failed to load models for search: {load_err}",
            }

        # Выполняем поиск
        try:
            results = loop.run_until_complete(engine.search_analogs_async(queries, method))
        except Exception as search_err:
            return {
                "status": "error",
                "message": f"Search execution failed: {search_err}",
            }

        duration = time.time() - start
        return {
            "status": "success",
            "results": results,
            "statistics": engine.get_statistics(),
            "processing_time": duration,
        }
    finally:
        try:
            loop.close()
        except Exception:
            pass



