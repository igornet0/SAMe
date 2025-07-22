from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Depends, status, Body, Request, UploadFile, File
from fastapi.security import (HTTPBearer,
                              OAuth2PasswordRequestForm)
from fastapi.responses import FileResponse
from datetime import timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import io

from same import settings
from same.api.configuration import (Server, )
from same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig

import logging

http_bearer = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/search", tags=["Search"])

logger = logging.getLogger("app_fastapi.search")

# Глобальный экземпляр поискового движка
search_engine: Optional[AnalogSearchEngine] = None

# Pydantic модели для API
class SearchRequest(BaseModel):
    queries: List[str]
    method: Optional[str] = "hybrid"
    similarity_threshold: Optional[float] = 0.6
    max_results: Optional[int] = 10

class SearchResponse(BaseModel):
    results: Dict[str, List[Dict[str, Any]]]
    statistics: Dict[str, Any]
    processing_time: float

class InitializeRequest(BaseModel):
    catalog_file_path: Optional[str] = None
    search_method: Optional[str] = "hybrid"
    similarity_threshold: Optional[float] = 0.6

# API endpoints
@router.post("/initialize")
async def initialize_search_engine(request: InitializeRequest):
    """Инициализация поискового движка"""
    global search_engine

    try:
        # Создаем конфигурацию
        config = AnalogSearchConfig(
            search_method=request.search_method,
            similarity_threshold=request.similarity_threshold
        )

        # Создаем новый экземпляр движка
        search_engine = AnalogSearchEngine(config)

        # Инициализируем с данными
        if request.catalog_file_path:
            await search_engine.initialize(request.catalog_file_path)
        else:
            # Используем тестовые данные если файл не указан
            test_data = pd.DataFrame({
                'name': [
                    'Болт М10х50 ГОСТ 7798-70',
                    'Гайка М10 ГОСТ 5915-70',
                    'Шайба 10 ГОСТ 11371-78',
                    'Винт М8х30 DIN 912',
                    'Труба стальная 57х3.5 ГОСТ 8732-78'
                ],
                'id': range(1, 6)
            })
            await search_engine.initialize(test_data)

        return {
            "status": "success",
            "message": "Search engine initialized successfully",
            "statistics": search_engine.get_statistics()
        }

    except Exception as e:
        logger.error(f"Error initializing search engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-catalog")
async def upload_catalog(file: UploadFile = File(...)):
    """Загрузка каталога из файла"""
    global search_engine

    try:
        # Проверяем формат файла
        if not file.filename.endswith(('.xlsx', '.csv')):
            raise HTTPException(status_code=400, detail="Поддерживаются только файлы .xlsx и .csv")

        # Читаем файл
        contents = await file.read()

        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Создаем движок если не существует
        if search_engine is None:
            search_engine = AnalogSearchEngine()

        # Инициализируем с загруженными данными
        await search_engine.initialize(df)

        return {
            "status": "success",
            "message": f"Catalog uploaded successfully. {len(df)} items loaded.",
            "statistics": search_engine.get_statistics()
        }

    except Exception as e:
        logger.error(f"Error uploading catalog: {e}")
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
        results = await search_engine.search_analogs(
            queries=request.queries,
            method=request.method
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

@router.get("/search-single/{query}")
async def search_single_analog(query: str, method: str = "fuzzy", max_results: int = 10):
    """Поиск аналогов для одного запроса"""
    global search_engine

    if search_engine is None or not search_engine.is_ready:
        raise HTTPException(status_code=400, detail="Search engine is not initialized")

    try:
        results = await search_engine.search_analogs(
            queries=[query],
            method=method
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