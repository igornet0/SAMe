from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.security import (HTTPBearer, )
from fastapi.responses import FileResponse
from datetime import timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
from pathlib import Path
from datetime import datetime

from same_api.api.configuration.schemas import (InitializeRequest,
                                                SearchRequest,
                                                SearchResponse)

from same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig
from celery.result import AsyncResult
from same_api.api.configuration.tasks.celery_app import get_celery_app
from same_api.api.configuration.tasks.catalog_tasks import process_catalog_upload
from same_api.api.configuration.tasks.catalog_tasks import _compute_dataset_statistics
from same_api.api.configuration.tasks.search_tasks import run_analogs

from src.same.excel_processor import AdvancedExcelProcessor

import logging

http_bearer = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/search", tags=["Search"])

logger = logging.getLogger("app_fastapi.search")

# Глобальный экземпляр поискового движка
search_engine: Optional[AnalogSearchEngine] = None

# API endpoints
@router.post("/initialize")
async def initialize_search_engine(request: InitializeRequest):
    """Инициализация поискового движка"""
    global search_engine

    try:
        # # Создаем конфигурацию
        config = AnalogSearchConfig(
            search_method=request.search_method,
            similarity_threshold=request.similarity_threshold
        )

        # Создаем новый экземпляр движка, если его нет
        if search_engine is None:
            globals()['search_engine'] = AnalogSearchEngine(config)

        # Инициализация по указанному пути к каталогу (csv/xlsx)
        if request.catalog_file_path:
            await globals()['search_engine'].initialize(request.catalog_file_path)

        return {
            "status": "success",
            "message": "Search engine initialized successfully",
            "statistics": globals()['search_engine'].get_statistics()
        }

    except Exception as e:
        logger.error(f"Error initializing search engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-catalog")
async def upload_catalog(file: UploadFile = File(...)):
    """Загрузка каталога: ставим задачу в Celery и возвращаем task_id"""
    try:
        if not file.filename.lower().endswith((".xlsx", ".csv")):
            raise HTTPException(status_code=400, detail="Поддерживаются только файлы .xlsx и .csv")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл пустой")

        # Отправляем задачу в Celery (фоновая обработка)
        task = process_catalog_upload.delay(contents, file.filename)

        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Файл принят. Обработка запущена в фоне"
        }
    except Exception as e:
        logger.error(f"Error uploading catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload-status/{task_id}")
async def upload_status(task_id: str):
    """Статус фоновой обработки каталога"""
    try:
        app = get_celery_app()
        result = AsyncResult(task_id, app=app)
        response: Dict[str, Any] = {
            "task_id": task_id,
            "state": result.state,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else False,
        }
        if result.failed():
            response["error"] = str(result.result)
        if result.successful():
            # В результатах храним статистику и сообщение
            res = result.result or {}
            response.update(res)
        return response
    except Exception as e:
        logger.error(f"Error fetching upload status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel-upload/{task_id}")
async def cancel_upload(task_id: str):
    """Отмена фоновой обработки файла (Celery revoke)."""
    try:
        app = get_celery_app()
        result = AsyncResult(task_id, app=app)
        if result.ready():
            return {"status": "already_finished", "state": result.state}
        # revoke с terminate=True попытается завершить выполнение
        result.revoke(terminate=True)
        return {"status": "canceled", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error canceling upload task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/activate-models/{task_id}")
async def activate_models(task_id: str):
    """Активировать обученные модели после успешной фоновой обработки"""
    global search_engine
    try:
        app = get_celery_app()
        result = AsyncResult(task_id, app=app)
        if not result.successful():
            raise HTTPException(status_code=400, detail=f"Задача не завершена успешно: {result.state}")

        # Загружаем сохраненные модели в процесс API
        if search_engine is None:
            search_engine = AnalogSearchEngine()
        await search_engine.load_models()

        return {
            "status": "success",
            "message": "Модели активированы",
            "statistics": search_engine.get_statistics()
        }
    except Exception as e:
        logger.error(f"Error activating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-analogs", response_model=SearchResponse)
async def search_analogs(request: SearchRequest):
    """Поиск аналогов"""
    global search_engine

    if search_engine is None or not search_engine.is_ready:
        raise HTTPException(status_code=400, detail="Search engine is not initialized")

    try:
        import time
        start_time = time.time()

        # Выполняем поиск
        results = await search_engine.search_analogs_async(
            request.queries,
            request.method
        )

        processing_time = time.time() - start_time

        return SearchResponse(
            results=results,
            statistics=search_engine.get_statistics(),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error searching analogs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Поиск через Celery (RabbitMQ) ===
@router.post("/enqueue-search")
async def enqueue_search(request: SearchRequest):
    """Поставить задачу поиска в очередь Celery и вернуть task_id."""
    try:
        task = run_analogs.delay(
            queries=request.queries,
            method=request.method or "hybrid",
            similarity_threshold=request.similarity_threshold or 0.6,
            max_results=request.max_results or 10,
        )
        return {"status": "queued", "task_id": task.id}
    except Exception as e:
        logger.error(f"Error enqueueing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-status/{task_id}")
async def search_status(task_id: str):
    """Проверить статус задачи поиска в Celery."""
    try:
        app = get_celery_app()
        result = AsyncResult(task_id, app=app)
        response: Dict[str, Any] = {
            "task_id": task_id,
            "state": result.state,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else False,
        }
        if result.failed():
            response["error"] = str(result.result)
        if result.successful():
            payload = result.result or {}
            response.update(payload)
        return response
    except Exception as e:
        logger.error(f"Error fetching search status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-search/{task_id}")
async def cancel_search(task_id: str):
    """Отменить задачу поиска в Celery (если еще выполняется)."""
    try:
        app = get_celery_app()
        result = AsyncResult(task_id, app=app)
        if result.ready():
            return {"status": "already_finished", "state": result.state}
        result.revoke(terminate=True)
        return {"status": "canceled", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error canceling search task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-single/{query}")
async def search_single_analog(query: str, method: str = "fuzzy", max_results: int = 10):
    """Поиск аналогов для одного запроса"""
    global search_engine

    if search_engine is None or not search_engine.is_ready:
        raise HTTPException(status_code=400, detail="Search engine is not initialized")

    try:
        results = await search_engine.search_analogs_async(
            [query],
            method
        )

        return {
            "query": query,
            "results": results.get(query, [])[:max_results],
            "method": method
        }

    except Exception as e:
        logger.error(f"Error searching single analog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export-results")
async def export_results(results: Dict[str, List[Dict[str, Any]]], format: str = "excel"):
    """Экспорт результатов поиска"""
    global search_engine

    if search_engine is None:
        raise HTTPException(status_code=400, detail="Search engine is not initialized")

    try:
        # Экспортируем результаты
        filepath = await search_engine.export_results(results, export_format=format)

        # Возвращаем файл
        return FileResponse(
            path=filepath,
            filename=f"analog_search_results.{format}",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics():
    """Получение статистики системы"""
    global search_engine

    if search_engine is None:
        return {"status": "not_initialized"}

    return search_engine.get_statistics()

@router.post("/save-models")
async def save_models():
    """Сохранение обученных моделей"""
    global search_engine

    if search_engine is None or not search_engine.is_ready:
        raise HTTPException(status_code=400, detail="Search engine is not initialized")

    try:
        await search_engine.save_models()
        return {"status": "success", "message": "Models saved successfully"}

    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-models")
async def load_models():
    """Загрузка сохраненных моделей"""
    global search_engine

    try:
        if search_engine is None:
            search_engine = AnalogSearchEngine()

        await search_engine.load_models()
        return {"status": "success", "message": "Models loaded successfully"}

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Новый эндпоинт: продвинутая обработка каталога с возвратом расширенных полей ===
@router.post("/process-catalog-advanced")
async def process_catalog_advanced(file: UploadFile = File(...), max_rows: Optional[int] = None):
    """Загрузка файла каталога и продвинутая обработка как в main_proccesed.py.

    Возвращает превью строк, список колонок и статистику по колонкам.
    """
    try:
        if AdvancedExcelProcessor is None:
            raise HTTPException(status_code=500, detail="AdvancedExcelProcessor is not available")

        if not file.filename.lower().endswith((".xlsx", ".csv")):
            raise HTTPException(status_code=400, detail="Поддерживаются только файлы .xlsx и .csv")

        # Готовим пути
        uploads_dir = Path("src/data/input/uploads")
        outputs_dir = Path("src/data/output/processed")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_ext = ".xlsx" if file.filename.lower().endswith(".xlsx") else ".csv"
        input_path = uploads_dir / f"uploaded_{timestamp}{input_ext}"
        output_path = outputs_dir / f"processed_{timestamp}.csv"

        # Сохраняем загруженный файл
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл пустой")

        with open(input_path, "wb") as f:
            f.write(contents)

        # Запускаем процессор
        processor = AdvancedExcelProcessor()
        success = processor.process_excel_file(str(input_path), str(output_path), max_rows=max_rows)
        if not success or not output_path.exists():
            raise HTTPException(status_code=500, detail="Не удалось обработать файл")

        # Читаем результат и считаем статистику
        df_result = pd.read_csv(output_path)
        stats = _compute_dataset_statistics(df_result)

        # Формируем сокращённое превью для фронта
        preview_limit = 100
        preview_rows = df_result.head(preview_limit).to_dict(orient="records")

        # Возвращаем ответ
        return {
            "status": "success",
            "message": f"Файл обработан: {len(df_result)} строк",
            "output_csv_path": str(output_path),
            "columns": list(map(str, df_result.columns)),
            "rows_preview": preview_rows,
            "preview_limit": preview_limit,
            "statistics": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_catalog_advanced: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-processed")
async def download_processed(path: str):
    """Скачивание подготовленного CSV из безопасной директории output."""
    try:
        outputs_dir = Path("src/data/output/processed").resolve()
        requested = Path(path).resolve()
        # Безопасность: разрешаем только внутри целевой директории
        if outputs_dir not in requested.parents and requested != outputs_dir:
            raise HTTPException(status_code=400, detail="Недопустимый путь")
        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail="Файл не найден")
        return FileResponse(str(requested), filename=requested.name, media_type="text/csv")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_processed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Старые endpoints для совместимости
@router.get("/")
async def read_root(request: Request):
    return {"message": "SAMe Analog Search API", "status": "active"}

@router.get("/health")
async def health_check():
    """Проверка состояния API"""
    global search_engine

    return {
        "status": "healthy",
        "search_engine_ready": search_engine is not None and search_engine.is_ready if search_engine else False,
        "timestamp": pd.Timestamp.now().isoformat()
    }