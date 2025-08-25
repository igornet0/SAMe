"""
Модуль семантического поиска с использованием BERT/Transformers и FAISS
"""

import logging
import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
import numpy as np
from functools import wraps
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import torch
import hashlib
from functools import lru_cache
from difflib import SequenceMatcher

from ..models import get_model_manager

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Декоратор для повторных попыток при ошибках

    Args:
        max_retries: Максимальное количество попыток
        delay: Начальная задержка между попытками
        backoff: Коэффициент увеличения задержки
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                     f"Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                     f"Retrying in {current_delay:.1f}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            raise last_exception

        # Возвращаем соответствующую обертку в зависимости от типа функции
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


@dataclass
class SemanticSearchConfig:
    """Конфигурация семантического поиска"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    
    # FAISS параметры
    index_type: str = "flat"  # flat, ivf, hnsw
    nlist: int = 100  # для IVF индекса
    nprobe: int = 10  # для поиска в IVF
    
    # Пороги схожести
    similarity_threshold: float = 0.5
    top_k_results: int = 10
    max_results: int = 10  # Alias for top_k_results for notebook compatibility
    
    # Оптимизация
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_gpu: bool = False

    # Категориальная фильтрация
    enable_category_filtering: bool = True
    category_similarity_threshold: float = 0.7
    max_category_candidates: int = 1000

    # Улучшенное скоринг
    enable_enhanced_scoring: bool = True
    numeric_token_weight: float = 0.3  # Снижаем вес числовых токенов
    semantic_weight: float = 0.6
    lexical_weight: float = 0.3
    key_term_weight: float = 0.1

    # Кэширование
    enable_cache: bool = True
    cache_size: int = 1000

    # Обработка ошибок и fallback режимы
    enable_fallback: bool = True
    fallback_model_names: List[str] = None  # Список fallback моделей
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    graceful_degradation: bool = True  # Продолжать работу при частичных ошибках


class SemanticSearchEngine:
    """Движок семантического поиска"""
    
    def __init__(self, config: SemanticSearchConfig = None):
        self.config = config or SemanticSearchConfig()
        self.model_manager = get_model_manager()
        self._model = None
        self.index = None
        self.documents = []
        self.document_ids = []
        self.embeddings = None
        self.is_fitted = False
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="semantic_search")

        # Категориальная информация для фильтрации
        self.document_categories = {}  # document_id -> category
        self.category_index = {}  # category -> set of document_ids
        self.document_metadata = {}  # document_id -> metadata dict

        # Кэш результатов поиска для оптимизации производительности
        self._search_cache = {}
        self._cache_max_size = 1000  # Максимальный размер кэша

        # Кэш эмбеддингов для повторно используемых текстов
        self._embeddings_cache = {}
        self._embeddings_cache_max_size = 5000  # Максимальный размер кэша эмбеддингов

        logger.info("SemanticSearchEngine initialized")

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    async def _ensure_model_loaded(self):
        """Ленивая загрузка модели с fallback режимами"""
        if self._initialized:
            return

        # Список моделей для попытки загрузки
        models_to_try = [self.config.model_name]

        # Добавляем fallback модели если включен fallback режим
        if self.config.enable_fallback and self.config.fallback_model_names:
            models_to_try.extend(self.config.fallback_model_names)

        # Стандартные fallback модели
        default_fallbacks = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]

        for fallback in default_fallbacks:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_exception = None

        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_name}")

                # Пытаемся загрузить через model_manager
                if hasattr(self.model_manager, 'get_sentence_transformer'):
                    self._model = await self.model_manager.get_sentence_transformer()
                else:
                    # Прямая загрузка как fallback
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(model_name)

                self._initialized = True
                logger.info(f"Successfully loaded model: {model_name}")
                return

            except Exception as e:
                last_exception = e
                logger.warning(f"Failed to load model {model_name}: {e}")
                continue

        # Если все модели не удалось загрузить
        if self.config.graceful_degradation:
            logger.error("All models failed to load, creating dummy model for graceful degradation")
            self._model = self._create_dummy_model()
            self._initialized = True
        else:
            raise RuntimeError(f"Failed to load any semantic model. Last error: {last_exception}")

    @property
    def model(self):
        """Получение модели (для обратной совместимости)"""
        if self._model is None:
            # Синхронная инициализация для обратной совместимости
            try:
                # Пытаемся получить модель синхронно через model_manager
                self._model = self.model_manager.get_sentence_transformer_sync()
            except Exception as e:
                logger.warning(f"Failed to load model synchronously: {e}")
                return None
        return self._model
    
    async def fit_async(self, documents: List[str], document_ids: List[Any] = None, metadata: List[Dict[str, Any]] = None):
        """
        Асинхронное обучение поискового движка на корпусе документов

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
            metadata: Список метаданных для каждого документа, включая категории
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        await self._ensure_model_loaded()

        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))

        # Обрабатываем метаданные
        self._process_metadata(metadata)

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # Выполняем синхронную генерацию эмбеддингов в ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        self.embeddings = await loop.run_in_executor(self._executor, self._generate_embeddings_sync, documents)

        # Создание FAISS индекса
        self._build_index()

        self.is_fitted = True
        logger.info("Semantic search engine fitted successfully")

    # def fit(self, documents: List[str], document_ids: List[Any] = None):
    #     """
    #     Синхронное обучение (для обратной совместимости)
    #     """
    #     loop = asyncio.get_event_loop()
    #     if loop.is_running():
    #         logger.warning("Active event loop detected, using synchronous fit fallback")
    #         return self._fit_sync(documents, document_ids)
    #     else:
    #         return loop.run_until_complete(self.fit_async(documents, document_ids))

    def fit(self, documents: List[str], document_ids: List[Any] = None, metadata: List[Dict[str, Any]] = None):
        """
        Синхронное обучение (для обратной совместимости)

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
            metadata: Список метаданных для каждого документа, включая категории
        """
        # Всегда используем синхронную версию для избежания проблем с event loop
        return self._fit_sync(documents, document_ids, metadata)

    def _fit_sync(self, documents: List[str], document_ids: List[Any] = None, metadata: List[Dict[str, Any]] = None):
        """
        Синхронное обучение поискового движка на корпусе документов

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
            metadata: Список метаданных для каждого документа, включая категории
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Синхронная загрузка модели
        if self._model is None:
            self._model = self.model_manager.get_sentence_transformer_sync()
            self._initialized = True
            logger.info(f"SemanticSearchEngine model loaded: {self.config.model_name}")

        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))

        # Обработка метаданных и категорий
        self._process_metadata(metadata)

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # Генерация эмбеддингов синхронно
        self.embeddings = self._generate_embeddings(documents)

        # Создание FAISS индекса
        self._build_index()

        self.is_fitted = True
        logger.info("Semantic search engine fitted successfully")

    def _process_metadata(self, metadata: List[Dict[str, Any]] = None):
        """Обработка метаданных и построение категориального индекса"""
        if not metadata:
            return

        # Очищаем существующие индексы
        self.document_categories.clear()
        self.category_index.clear()
        self.document_metadata.clear()

        for i, doc_metadata in enumerate(metadata):
            if i >= len(self.document_ids):
                break

            doc_id = self.document_ids[i]
            self.document_metadata[doc_id] = doc_metadata

            # Извлекаем категорию
            category = doc_metadata.get('category', 'unknown')
            self.document_categories[doc_id] = category

            # Обновляем категориальный индекс
            if category not in self.category_index:
                self.category_index[category] = set()
            self.category_index[category].add(doc_id)

        logger.info(f"Processed metadata for {len(self.document_metadata)} documents, "
                   f"found {len(self.category_index)} categories")

    async def _generate_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """Асинхронная генерация эмбеддингов для текстов"""
        await self._ensure_model_loaded()

        # Выполняем в executor для неблокирующей работы
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._generate_embeddings_sync, texts
        )

        return embeddings.astype(np.float32)

    def _generate_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """Синхронная генерация эмбеддингов с оптимизацией памяти и кэшированием"""
        if not texts:
            return np.array([])

        # Используем кэширование для всех случаев
        return self._generate_embeddings_with_cache(texts)

    def _generate_embeddings_streaming(self, texts: List[str]) -> np.ndarray:
        """Потоковая генерация эмбеддингов для больших датасетов"""
        logger.info(f"Using streaming processing for {len(texts)} texts")

        # Определяем размер чанка на основе доступной памяти
        chunk_size = min(self.config.batch_size * 10, 500)  # Максимум 500 текстов за раз
        embeddings_list = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            logger.debug(f"Processing chunk {i//chunk_size + 1}/{(len(texts) + chunk_size - 1)//chunk_size}")

            try:
                # Генерируем эмбеддинги для чанка
                chunk_embeddings = self._model.encode(
                    chunk,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,  # Отключаем для чанков
                    normalize_embeddings=self.config.normalize_embeddings
                )

                embeddings_list.append(chunk_embeddings)

                # Принудительная очистка памяти после каждого чанка
                if i % (chunk_size * 5) == 0:  # Каждые 5 чанков
                    import gc
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                # Создаем пустые эмбеддинги для этого чанка
                embedding_dim = getattr(self._model, 'get_sentence_embedding_dimension', lambda: 384)()
                chunk_embeddings = np.zeros((len(chunk), embedding_dim), dtype=np.float32)
                embeddings_list.append(chunk_embeddings)

        # Объединяем все эмбеддинги
        if embeddings_list:
            result = np.vstack(embeddings_list)
            logger.info(f"Generated embeddings shape: {result.shape}")
            return result
        else:
            # Fallback если ничего не получилось
            embedding_dim = getattr(self._model, 'get_sentence_embedding_dimension', lambda: 384)()
            return np.zeros((len(texts), embedding_dim), dtype=np.float32)

    def _get_text_hash(self, text: str) -> str:
        """Получение хэша текста для кэширования"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга из кэша"""
        text_hash = self._get_text_hash(text)
        return self._embeddings_cache.get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Сохранение эмбеддинга в кэш с управлением размером"""
        text_hash = self._get_text_hash(text)

        # Управление размером кэша
        if len(self._embeddings_cache) >= self._embeddings_cache_max_size:
            # Удаляем случайный элемент (простая стратегия)
            oldest_key = next(iter(self._embeddings_cache))
            del self._embeddings_cache[oldest_key]

        self._embeddings_cache[text_hash] = embedding.copy()

    def _generate_embeddings_with_cache(self, texts: List[str]) -> np.ndarray:
        """Генерация эмбеддингов с использованием кэша"""
        if not texts:
            return np.array([])

        embeddings_list = []
        texts_to_process = []
        indices_to_process = []

        # Проверяем кэш для каждого текста
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings_list.append((i, cached_embedding))
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)

        # Генерируем эмбеддинги для некэшированных текстов
        if texts_to_process:
            if len(texts_to_process) > 1000:
                new_embeddings = self._generate_embeddings_streaming(texts_to_process)
            else:
                new_embeddings = self._model.encode(
                    texts_to_process,
                    batch_size=self.config.batch_size,
                    show_progress_bar=len(texts_to_process) > 100,
                    normalize_embeddings=self.config.normalize_embeddings
                )

            # Кэшируем новые эмбеддинги
            for text, embedding in zip(texts_to_process, new_embeddings):
                self._cache_embedding(text, embedding)

            # Добавляем новые эмбеддинги в список
            for i, embedding in zip(indices_to_process, new_embeddings):
                embeddings_list.append((i, embedding))

        # Сортируем по исходным индексам и возвращаем
        embeddings_list.sort(key=lambda x: x[0])
        result = np.array([embedding for _, embedding in embeddings_list])

        logger.debug(f"Generated embeddings with cache: {len(texts)} texts, "
                    f"{len(texts_to_process)} processed, "
                    f"{len(texts) - len(texts_to_process)} from cache")

        return result

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Генерация эмбеддингов для текстов (синхронная версия)"""
        # Для синхронного использования используем синхронный метод
        if self._model is None:
            # Синхронная загрузка модели если необходимо
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Если есть активный loop, используем синхронную загрузку
                self._model = self.model_manager.get_sentence_transformer_sync()
            except RuntimeError:
                # Если нет активного loop, используем синхронную версию
                self._model = self.model_manager.get_sentence_transformer_sync()

        return self._generate_embeddings_sync(texts)
    
    def _build_index(self):
        """Построение оптимизированного FAISS индекса"""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available for index building")

        dimension = self.embeddings.shape[1]
        num_vectors = self.embeddings.shape[0]

        logger.info(f"Building index for {num_vectors} vectors of dimension {dimension}")

        # Выбираем оптимальный тип индекса на основе размера данных
        if num_vectors < 1000:
            # Для малых датасетов используем точный поиск
            index_type = "flat"
        elif num_vectors < 10000:
            # Для средних датасетов используем IVF
            index_type = "ivf"
        else:
            # Для больших датасетов используем HNSW
            index_type = "hnsw"

        # Переопределяем тип индекса если задан в конфигурации
        if hasattr(self.config, 'index_type') and self.config.index_type:
            index_type = self.config.index_type

        if index_type == "flat":
            # Простой плоский индекс (точный поиск)
            if self.config.normalize_embeddings:
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product для нормализованных векторов
            else:
                self.index = faiss.IndexFlatL2(dimension)  # L2 расстояние

        elif index_type == "ivf":
            # IVF индекс (приближенный поиск) с оптимальными параметрами
            nlist = min(int(np.sqrt(num_vectors)), 1000)  # Оптимальное количество кластеров
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Обучение индекса
            self.index.train(self.embeddings)
            # Оптимальное значение nprobe (количество кластеров для поиска)
            self.index.nprobe = min(max(nlist // 10, 1), 50)

        elif index_type == "hnsw":
            # HNSW индекс (быстрый приближенный поиск) с оптимальными параметрами
            M = 32  # Количество соединений
            self.index = faiss.IndexHNSWFlat(dimension, M)

            # Оптимизируем параметры на основе размера данных
            if num_vectors < 50000:
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
            else:
                self.index.hnsw.efConstruction = 400
                self.index.hnsw.efSearch = 200

        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Добавление векторов в индекс пакетами для больших датасетов
        batch_size = 10000
        if num_vectors > batch_size:
            logger.info(f"Adding vectors in batches of {batch_size}")
            for i in range(0, num_vectors, batch_size):
                end_idx = min(i + batch_size, num_vectors)
                batch = self.embeddings[i:end_idx]
                self.index.add(batch)
                logger.debug(f"Added batch {i//batch_size + 1}/{(num_vectors + batch_size - 1)//batch_size}")
        else:
            self.index.add(self.embeddings)

        logger.info(f"Built {index_type} index with {self.index.ntotal} vectors")

    def _get_cache_key(self, query: str, top_k: int, category_filter: str = None) -> str:
        """Генерация ключа кэша для запроса"""
        cache_string = f"{query}:{top_k}:{category_filter or 'all'}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _manage_cache_size(self):
        """Управление размером кэша"""
        if len(self._search_cache) > self._cache_max_size:
            # Удаляем 20% старых записей (простая стратегия FIFO)
            keys_to_remove = list(self._search_cache.keys())[:int(self._cache_max_size * 0.2)]
            for key in keys_to_remove:
                del self._search_cache[key]

    def search(self, query: str, top_k: int = None, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Семантический поиск с улучшенным скорингом и категориальной фильтрацией

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            category_filter: Фильтр по категории (опционально)

        Returns:
            Список результатов поиска
        """
        if not self.is_fitted:
            raise ValueError("Search engine is not fitted. Call fit() first.")

        if not query or not isinstance(query, str):
            return []

        top_k = top_k or self.config.top_k_results

        # Проверяем кэш
        cache_key = self._get_cache_key(query, top_k, category_filter)
        if self.config.enable_cache and cache_key in self._search_cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._search_cache[cache_key]

        # Определяем кандидатов для поиска
        candidate_indices = self._get_search_candidates(query, category_filter)

        # Генерация эмбеддинга запроса
        query_embedding = self._generate_embeddings([query])

        # Поиск в индексе с увеличенным количеством кандидатов для фильтрации
        search_k = min(self.config.max_category_candidates, len(self.documents))
        scores, indices = self.index.search(query_embedding, search_k)

        # Формирование результатов с улучшенным скорингом и ранней остановкой
        results = []
        min_confidence_threshold = 0.1  # Минимальный порог уверенности для ранней остановки
        consecutive_low_scores = 0
        max_consecutive_low = 10  # Максимальное количество подряд идущих низких скоров

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS возвращает -1 для отсутствующих результатов
                continue

            doc_id = self.document_ids[idx]

            # Применяем категориальную фильтрацию
            if candidate_indices and doc_id not in candidate_indices:
                continue

            # Базовое семантическое сходство
            if self.config.normalize_embeddings and self.config.index_type == "flat":
                semantic_similarity = float(score)
            else:
                semantic_similarity = 1.0 / (1.0 + float(score))

            # Ранняя остановка при очень низкой уверенности
            if semantic_similarity < min_confidence_threshold:
                consecutive_low_scores += 1
                if consecutive_low_scores >= max_consecutive_low:
                    logger.debug(f"Early stopping at position {i} due to low confidence scores")
                    break
            else:
                consecutive_low_scores = 0  # Сбрасываем счетчик при хорошем скоре

            if semantic_similarity >= self.config.similarity_threshold:
                # Вычисляем улучшенный скор
                enhanced_score = self._calculate_enhanced_score(
                    query, self.documents[idx], semantic_similarity, doc_id
                )

                results.append({
                    'document_id': doc_id,
                    'document': self.documents[idx],
                    'similarity_score': enhanced_score,
                    'semantic_score': semantic_similarity,
                    'raw_score': float(score),
                    'rank': i + 1,
                    'index': int(idx),
                    'category': self.document_categories.get(doc_id, 'unknown')
                })

                # Ранняя остановка если уже набрали достаточно хороших результатов
                if len(results) >= top_k * 2:  # Собираем в 2 раза больше для лучшей сортировки
                    logger.debug(f"Early stopping at position {i} - collected enough results")
                    break

        # Сортируем по улучшенному скору
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Ограничиваем количество результатов
        results = results[:top_k]

        # Обновляем ранги
        for i, result in enumerate(results):
            result['rank'] = i + 1

        # Сохраняем результат в кэш
        if self.config.enable_cache:
            self._search_cache[cache_key] = results
            self._manage_cache_size()

        return results

    def _get_search_candidates(self, query: str, category_filter: str = None) -> Set[Any]:
        """Определение кандидатов для поиска на основе категориальной фильтрации"""
        if not self.config.enable_category_filtering or not category_filter:
            return None

        # Прямое совпадение категории
        if category_filter in self.category_index:
            return self.category_index[category_filter]

        # Поиск похожих категорий
        similar_categories = self._find_similar_categories(category_filter)
        candidates = set()
        for cat in similar_categories:
            if cat in self.category_index:
                candidates.update(self.category_index[cat])

        return candidates if candidates else None

    def _find_similar_categories(self, target_category: str) -> List[str]:
        """Поиск похожих категорий для расширения поиска"""
        similar = []
        target_lower = target_category.lower()

        for category in self.category_index.keys():
            category_lower = category.lower()

            # Проверяем вхождение подстроки
            if target_lower in category_lower or category_lower in target_lower:
                similar.append(category)
            # Проверяем схожесть по Левенштейну
            elif self._levenshtein_similarity(target_lower, category_lower) > self.config.category_similarity_threshold:
                similar.append(category)

        return similar

    def _calculate_enhanced_score(self, query: str, document: str, semantic_score: float, doc_id: Any) -> float:
        """Вычисление улучшенного скора с учетом различных факторов"""
        if not self.config.enable_enhanced_scoring:
            return semantic_score

        # Базовый семантический скор
        final_score = semantic_score * self.config.semantic_weight

        # Лексическое сходство (Левенштейн)
        lexical_score = self._levenshtein_similarity(query.lower(), document.lower())
        final_score += lexical_score * self.config.lexical_weight

        # Скор ключевых терминов (без чисел)
        key_term_score = self._calculate_key_term_score(query, document)
        final_score += key_term_score * self.config.key_term_weight

        # Штраф за числовые токены
        numeric_penalty = self._calculate_numeric_penalty(query, document)
        final_score *= (1.0 - numeric_penalty * (1.0 - self.config.numeric_token_weight))

        return min(1.0, final_score)  # Ограничиваем максимальным значением 1.0

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Вычисление сходства по Левенштейну"""
        return SequenceMatcher(None, s1, s2).ratio()

    def _calculate_key_term_score(self, query: str, document: str) -> float:
        """Вычисление скора на основе ключевых терминов (исключая числа)"""
        # Извлекаем нечисловые токены
        query_terms = set(re.findall(r'\b[а-яёa-z]+\b', query.lower()))
        doc_terms = set(re.findall(r'\b[а-яёa-z]+\b', document.lower()))

        if not query_terms:
            return 0.0

        # Пересечение терминов
        common_terms = query_terms.intersection(doc_terms)
        return len(common_terms) / len(query_terms)

    def _calculate_numeric_penalty(self, query: str, document: str) -> float:
        """Вычисление штрафа за совпадение только по числам"""
        # Извлекаем числа
        query_numbers = set(re.findall(r'\b\d+\b', query))
        doc_numbers = set(re.findall(r'\b\d+\b', document))

        if not query_numbers or not doc_numbers:
            return 0.0

        # Если есть совпадающие числа, но мало текстовых совпадений
        common_numbers = query_numbers.intersection(doc_numbers)
        if common_numbers:
            text_similarity = self._calculate_key_term_score(query, document)
            # Штраф тем больше, чем меньше текстовых совпадений
            return (1.0 - text_similarity) * 0.5  # Максимальный штраф 50%

        return 0.0

    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """Пакетный поиск"""
        if not self.is_fitted:
            raise ValueError("Search engine is not fitted. Call fit() first.")
        
        top_k = top_k or self.config.top_k_results
        
        # Генерация эмбеддингов для всех запросов
        query_embeddings = self._generate_embeddings(queries)
        
        # Пакетный поиск
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Формирование результатов для каждого запроса
        all_results = []
        for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            
            for i, (score, idx) in enumerate(zip(query_scores, query_indices)):
                if idx == -1:
                    continue
                
                if self.config.normalize_embeddings and self.config.index_type == "flat":
                    similarity = float(score)
                else:
                    similarity = 1.0 / (1.0 + float(score))
                
                if similarity >= self.config.similarity_threshold:
                    query_results.append({
                        'document_id': self.document_ids[idx],
                        'document': self.documents[idx],
                        'similarity_score': similarity,
                        'raw_score': float(score),
                        'rank': i + 1,
                        'index': int(idx)
                    })
            
            all_results.append(query_results)
        
        return all_results
    
    def get_similar_documents(self, document_id: Any, top_k: int = None) -> List[Dict[str, Any]]:
        """Поиск документов, похожих на заданный"""
        if document_id not in self.document_ids:
            raise ValueError(f"Document ID {document_id} not found")
        
        # Находим индекс документа
        doc_index = self.document_ids.index(document_id)
        document_text = self.documents[doc_index]
        
        # Ищем похожие документы
        results = self.search(document_text, top_k)
        
        # Исключаем сам документ из результатов
        results = [r for r in results if r['document_id'] != document_id]
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга для текста"""
        return self._generate_embeddings([text])[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Вычисление семантического сходства между двумя текстами"""
        embeddings = self._generate_embeddings([text1, text2])
        
        if self.config.normalize_embeddings:
            # Косинусное сходство для нормализованных векторов
            similarity = np.dot(embeddings[0], embeddings[1])
        else:
            # Косинусное сходство для ненормализованных векторов
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
        
        return float(similarity)
    
    def save_model(self, filepath: str):
        """Сохранение модели"""
        # Конвертируем конфигурацию в словарь для сериализации
        config_dict = {
            'model_name': self.config.model_name,
            'embedding_dim': self.config.embedding_dim,
            'index_type': self.config.index_type,
            'nlist': self.config.nlist,
            'nprobe': self.config.nprobe,
            'similarity_threshold': self.config.similarity_threshold,
            'top_k_results': self.config.top_k_results,
            'max_results': self.config.max_results,
            'batch_size': self.config.batch_size,
            'normalize_embeddings': self.config.normalize_embeddings,
            'use_gpu': self.config.use_gpu,
            'enable_category_filtering': self.config.enable_category_filtering,
            'category_similarity_threshold': self.config.category_similarity_threshold,
            'max_category_candidates': self.config.max_category_candidates,
            'enable_enhanced_scoring': self.config.enable_enhanced_scoring,
            'numeric_token_weight': self.config.numeric_token_weight,
            'semantic_weight': self.config.semantic_weight,
            'lexical_weight': self.config.lexical_weight,
            'key_term_weight': self.config.key_term_weight,
            'enable_cache': self.config.enable_cache,
            'cache_size': self.config.cache_size,
            'enable_fallback': self.config.enable_fallback,
            'fallback_model_names': self.config.fallback_model_names,
            'max_retries': self.config.max_retries,
            'retry_delay': self.config.retry_delay,
            'retry_backoff': self.config.retry_backoff,
            'graceful_degradation': self.config.graceful_degradation
        }
        
        model_data = {
            'config': config_dict,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'embeddings': self.embeddings,
            'is_fitted': self.is_fitted
        }
        
        # Сохраняем FAISS индекс отдельно
        index_path = str(Path(filepath).with_suffix('.faiss'))
        faiss.write_index(self.index, index_path)
        
        # Сохраняем остальные данные
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath} and {index_path}")
    
    def load_model(self, filepath: str):
        """Загрузка модели"""
        # Загружаем данные
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Восстанавливаем конфигурацию из словаря
        if isinstance(model_data['config'], dict):
            config_dict = model_data['config']
            self.config = SemanticSearchConfig(
                model_name=config_dict['model_name'],
                embedding_dim=config_dict['embedding_dim'],
                index_type=config_dict['index_type'],
                nlist=config_dict['nlist'],
                nprobe=config_dict['nprobe'],
                similarity_threshold=config_dict['similarity_threshold'],
                top_k_results=config_dict['top_k_results'],
                max_results=config_dict['max_results'],
                batch_size=config_dict['batch_size'],
                normalize_embeddings=config_dict['normalize_embeddings'],
                use_gpu=config_dict['use_gpu'],
                enable_category_filtering=config_dict['enable_category_filtering'],
                category_similarity_threshold=config_dict['category_similarity_threshold'],
                max_category_candidates=config_dict['max_category_candidates'],
                enable_enhanced_scoring=config_dict['enable_enhanced_scoring'],
                numeric_token_weight=config_dict['numeric_token_weight'],
                semantic_weight=config_dict['semantic_weight'],
                lexical_weight=config_dict['lexical_weight'],
                key_term_weight=config_dict['key_term_weight'],
                enable_cache=config_dict['enable_cache'],
                cache_size=config_dict['cache_size'],
                enable_fallback=config_dict['enable_fallback'],
                fallback_model_names=config_dict['fallback_model_names'],
                max_retries=config_dict['max_retries'],
                retry_delay=config_dict['retry_delay'],
                retry_backoff=config_dict['retry_backoff'],
                graceful_degradation=config_dict['graceful_degradation']
            )
        else:
            # Обратная совместимость со старым форматом
            self.config = model_data['config']
        
        self.documents = model_data['documents']
        self.document_ids = model_data['document_ids']
        self.embeddings = model_data['embeddings']
        self.is_fitted = model_data['is_fitted']
        
        # Загружаем FAISS индекс
        index_path = str(Path(filepath).with_suffix('.faiss'))
        self.index = faiss.read_index(index_path)

        # Перезагружаем модель (асинхронно)
        try:
            # Пытаемся загрузить модель синхронно для обратной совместимости
            if self._model is None:
                self._model = self.model_manager.get_sentence_transformer_sync()
                self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to load model synchronously: {e}")
            # Модель будет загружена при первом использовании

        logger.info(f"Model loaded from {filepath} and {index_path}")

    def _create_dummy_model(self):
        """
        Создание dummy модели для graceful degradation

        Returns:
            Объект с минимальным интерфейсом для работы
        """
        class DummyModel:
            def __init__(self):
                self.embedding_dim = 384  # Стандартная размерность

            def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
                """Создает случайные эмбеддинги для graceful degradation"""
                if isinstance(texts, str):
                    texts = [texts]

                import numpy as np
                # Создаем детерминированные "эмбеддинги" на основе хэша текста
                embeddings = []
                for text in texts:
                    # Простой хэш для создания воспроизводимых эмбеддингов
                    hash_val = hash(text) % (2**32)
                    np.random.seed(hash_val)
                    embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

                    if normalize_embeddings:
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                    embeddings.append(embedding)

                result = np.array(embeddings)
                logger.warning(f"Using dummy embeddings for {len(texts)} texts (graceful degradation mode)")
                return result

            def get_sentence_embedding_dimension(self):
                return self.embedding_dim

        return DummyModel()

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики поискового движка"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1],
            'index_type': self.config.index_type,
            'index_size': self.index.ntotal,
            'model_name': self.config.model_name,
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'normalize_embeddings': self.config.normalize_embeddings,
                'batch_size': self.config.batch_size
            }
        }

    def clear_cache(self):
        """Очистка всех кэшей для освобождения памяти"""
        self._search_cache.clear()
        self._embeddings_cache.clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики использования кэшей"""
        return {
            'search_cache_size': len(self._search_cache),
            'search_cache_max_size': self._cache_max_size,
            'embeddings_cache_size': len(self._embeddings_cache),
            'embeddings_cache_max_size': self._embeddings_cache_max_size,
            'memory_usage_mb': self._estimate_cache_memory_usage()
        }

    def _estimate_cache_memory_usage(self) -> float:
        """Оценка использования памяти кэшами в МБ"""
        try:
            import sys
            search_cache_size = sum(sys.getsizeof(v) for v in self._search_cache.values())
            embeddings_cache_size = sum(
                embedding.nbytes if hasattr(embedding, 'nbytes') else sys.getsizeof(embedding)
                for embedding in self._embeddings_cache.values()
            )
            total_bytes = search_cache_size + embeddings_cache_size
            return total_bytes / (1024 * 1024)  # Конвертируем в МБ
        except Exception:
            return 0.0

    def __del__(self):
        """Корректное закрытие ThreadPoolExecutor при удалении объекта"""
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass  # Игнорируем ошибки при закрытии
