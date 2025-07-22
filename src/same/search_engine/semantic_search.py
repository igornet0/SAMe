"""
Модуль семантического поиска с использованием BERT/Transformers и FAISS
"""

import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import torch
import hashlib
from functools import lru_cache

from ..models import get_model_manager

logger = logging.getLogger(__name__)


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

        # Кэш результатов поиска для оптимизации производительности
        self._search_cache = {}
        self._cache_max_size = 1000  # Максимальный размер кэша

        logger.info("SemanticSearchEngine initialized")

    async def _ensure_model_loaded(self):
        """Ленивая загрузка модели"""
        if self._initialized:
            return

        self._model = await self.model_manager.get_sentence_transformer()
        self._initialized = True
        logger.info(f"SemanticSearchEngine model loaded: {self.config.model_name}")

    @property
    def model(self):
        """Получение модели (для обратной совместимости)"""
        if self._model is None:
            # Синхронная инициализация для обратной совместимости
            try:
                loop = asyncio.get_running_loop()
                task = asyncio.create_task(self._ensure_model_loaded())
                # Не можем ждать в синхронном контексте, возвращаем None
                logger.warning("Model not loaded yet, use async methods")
                return None
            except RuntimeError:
                asyncio.run(self._ensure_model_loaded())
        return self._model
    
    async def fit_async(self, documents: List[str], document_ids: List[Any] = None):
        """
        Асинхронное обучение поискового движка на корпусе документов

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        await self._ensure_model_loaded()

        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # Генерация эмбеддингов
        self.embeddings = await self._generate_embeddings_async(documents)

        # Создание FAISS индекса
        self._build_index()

        self.is_fitted = True
        logger.info("Semantic search engine fitted successfully")

    def fit(self, documents: List[str], document_ids: List[Any] = None):
        """
        Синхронное обучение (для обратной совместимости)
        """
        try:
            loop = asyncio.get_running_loop()
            # Если есть активный loop (например, в Jupyter), используем синхронную версию
            logger.warning("Active event loop detected, using synchronous fit")
            return self._fit_sync(documents, document_ids)
        except RuntimeError:
            # Если нет активного loop, можем использовать async
            return asyncio.run(self.fit_async(documents, document_ids))

    def _fit_sync(self, documents: List[str], document_ids: List[Any] = None):
        """
        Синхронное обучение поискового движка на корпусе документов

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
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

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # Генерация эмбеддингов синхронно
        self.embeddings = self._generate_embeddings(documents)

        # Создание FAISS индекса
        self._build_index()

        self.is_fitted = True
        logger.info("Semantic search engine fitted successfully")

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
        """Синхронная генерация эмбеддингов"""
        return self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.normalize_embeddings
        )

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
                # Если нет активного loop, можем использовать async
                self._model = asyncio.run(self.model_manager.get_sentence_transformer())

        return self._generate_embeddings_sync(texts)
    
    def _build_index(self):
        """Построение FAISS индекса"""
        dimension = self.embeddings.shape[1]
        
        if self.config.index_type == "flat":
            # Простой плоский индекс (точный поиск)
            if self.config.normalize_embeddings:
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product для нормализованных векторов
            else:
                self.index = faiss.IndexFlatL2(dimension)  # L2 расстояние
        
        elif self.config.index_type == "ivf":
            # IVF индекс (приближенный поиск)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            
            # Обучение индекса
            self.index.train(self.embeddings)
            self.index.nprobe = self.config.nprobe
        
        elif self.config.index_type == "hnsw":
            # HNSW индекс (быстрый приближенный поиск)
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Добавление векторов в индекс
        self.index.add(self.embeddings)
        
        logger.info(f"Built {self.config.index_type} index with {self.index.ntotal} vectors")

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Генерация ключа кэша для запроса"""
        return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()

    def _manage_cache_size(self):
        """Управление размером кэша"""
        if len(self._search_cache) > self._cache_max_size:
            # Удаляем 20% старых записей (простая стратегия FIFO)
            keys_to_remove = list(self._search_cache.keys())[:int(self._cache_max_size * 0.2)]
            for key in keys_to_remove:
                del self._search_cache[key]

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Семантический поиск
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список результатов поиска
        """
        if not self.is_fitted:
            raise ValueError("Search engine is not fitted. Call fit() first.")
        
        if not query or not isinstance(query, str):
            return []
        
        top_k = top_k or self.config.top_k_results

        # Проверяем кэш
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self._search_cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._search_cache[cache_key]

        # Генерация эмбеддинга запроса
        query_embedding = self._generate_embeddings([query])
        
        # Поиск в индексе
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Формирование результатов
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS возвращает -1 для отсутствующих результатов
                continue
            
            # Преобразование скора в зависимости от типа индекса
            if self.config.normalize_embeddings and self.config.index_type == "flat":
                similarity = float(score)  # Inner product уже дает косинусное сходство
            else:
                # Для L2 расстояния преобразуем в сходство
                similarity = 1.0 / (1.0 + float(score))
            
            if similarity >= self.config.similarity_threshold:
                results.append({
                    'document_id': self.document_ids[idx],
                    'document': self.documents[idx],
                    'similarity_score': similarity,
                    'raw_score': float(score),
                    'rank': i + 1,
                    'index': int(idx)
                })

        # Сохраняем результат в кэш
        self._search_cache[cache_key] = results
        self._manage_cache_size()

        return results
    
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
        model_data = {
            'config': self.config,
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
