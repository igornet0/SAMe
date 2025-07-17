"""
Модуль индексации для эффективного поиска и хранения данных
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Конфигурация индексатора"""
    # Типы индексов
    enable_text_index: bool = True
    enable_embedding_index: bool = True
    enable_parameter_index: bool = True
    enable_category_index: bool = True
    
    # Настройки хранения
    storage_backend: str = "sqlite"  # sqlite, memory, file
    index_dir: Path = Path("data/indexes")
    
    # Настройки производительности
    batch_size: int = 1000
    enable_parallel_indexing: bool = True
    max_workers: int = 4
    
    # Настройки кэширования
    enable_cache: bool = True
    cache_size: int = 10000
    
    # Настройки обновления
    enable_incremental_updates: bool = True
    version_control: bool = True


class SearchIndexer:
    """Класс для индексации и управления поисковыми индексами"""
    
    def __init__(self, config: IndexConfig = None):
        self.config = config or IndexConfig()
        
        # Инициализация хранилищ
        self.text_index: Dict[str, Set[int]] = {}
        self.embedding_index: Optional[np.ndarray] = None
        self.parameter_index: Dict[str, Dict[str, Set[int]]] = {}
        self.category_index: Dict[str, Set[int]] = {}
        self.document_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Кэш
        self.search_cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()
        
        # Статистика
        self.stats = {
            'total_documents': 0,
            'index_size': 0,
            'last_update': None,
            'version': 1
        }
        
        # Инициализация хранилища
        self._init_storage()
        
        logger.info("SearchIndexer initialized")
    
    def _init_storage(self):
        """Инициализация системы хранения"""
        if self.config.storage_backend == "sqlite":
            self._init_sqlite_storage()
        elif self.config.storage_backend == "file":
            self.config.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage backend '{self.config.storage_backend}' initialized")
    
    def _init_sqlite_storage(self):
        """Инициализация SQLite хранилища"""
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config.index_dir / "search_index.db"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Таблица документов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица текстового индекса
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_index (
                    term TEXT,
                    document_id INTEGER,
                    frequency INTEGER,
                    PRIMARY KEY (term, document_id),
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Таблица параметров
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_index (
                    parameter_name TEXT,
                    parameter_value TEXT,
                    document_id INTEGER,
                    PRIMARY KEY (parameter_name, parameter_value, document_id),
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Таблица категорий
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category_index (
                    category TEXT,
                    document_id INTEGER,
                    PRIMARY KEY (category, document_id),
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Индексы для производительности
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_term ON text_index (term)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_param_name ON parameter_index (parameter_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON category_index (category)")
            
            conn.commit()
    
    def index_documents(self, 
                       documents: List[str], 
                       document_ids: List[Any] = None,
                       metadata: List[Dict[str, Any]] = None,
                       embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Индексация документов
        
        Args:
            documents: Список текстов документов
            document_ids: Список ID документов
            metadata: Метаданные документов
            embeddings: Векторные представления документов
            
        Returns:
            Статистика индексации
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        document_ids = document_ids or list(range(len(documents)))
        metadata = metadata or [{}] * len(documents)
        
        logger.info(f"Indexing {len(documents)} documents")
        
        start_time = datetime.now()
        
        # Пакетная индексация
        if self.config.enable_parallel_indexing and len(documents) > self.config.batch_size:
            results = self._parallel_index_documents(documents, document_ids, metadata, embeddings)
        else:
            results = self._sequential_index_documents(documents, document_ids, metadata, embeddings)
        
        # Обновление статистики
        self.stats.update({
            'total_documents': len(documents),
            'last_update': datetime.now(),
            'indexing_time': (datetime.now() - start_time).total_seconds()
        })
        
        logger.info(f"Indexing completed in {self.stats['indexing_time']:.2f} seconds")
        
        return results
    
    def _parallel_index_documents(self, 
                                 documents: List[str], 
                                 document_ids: List[Any],
                                 metadata: List[Dict[str, Any]],
                                 embeddings: Optional[np.ndarray]) -> Dict[str, Any]:
        """Параллельная индексация документов"""
        batch_size = self.config.batch_size
        batches = [
            (documents[i:i + batch_size], 
             document_ids[i:i + batch_size], 
             metadata[i:i + batch_size],
             embeddings[i:i + batch_size] if embeddings is not None else None)
            for i in range(0, len(documents), batch_size)
        ]
        
        results = {'indexed_documents': 0, 'errors': []}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._index_batch, batch_docs, batch_ids, batch_meta, batch_emb)
                for batch_docs, batch_ids, batch_meta, batch_emb in batches
            ]
            
            for future in futures:
                try:
                    batch_result = future.result()
                    results['indexed_documents'] += batch_result['indexed_documents']
                    results['errors'].extend(batch_result['errors'])
                except Exception as e:
                    logger.error(f"Error in parallel indexing: {e}")
                    results['errors'].append(str(e))
        
        return results
    
    def _sequential_index_documents(self, 
                                   documents: List[str], 
                                   document_ids: List[Any],
                                   metadata: List[Dict[str, Any]],
                                   embeddings: Optional[np.ndarray]) -> Dict[str, Any]:
        """Последовательная индексация документов"""
        return self._index_batch(documents, document_ids, metadata, embeddings)
    
    def _index_batch(self, 
                    documents: List[str], 
                    document_ids: List[Any],
                    metadata: List[Dict[str, Any]],
                    embeddings: Optional[np.ndarray]) -> Dict[str, Any]:
        """Индексация пакета документов"""
        results = {'indexed_documents': 0, 'errors': []}
        
        for i, (doc, doc_id, meta) in enumerate(zip(documents, document_ids, metadata)):
            try:
                # Вычисляем хэш документа для проверки изменений
                doc_hash = hashlib.md5(doc.encode()).hexdigest()
                
                # Проверяем, нужно ли обновлять документ
                if self.config.enable_incremental_updates and self._is_document_unchanged(doc_id, doc_hash):
                    continue
                
                # Индексируем документ
                self._index_single_document(
                    doc, doc_id, meta, doc_hash,
                    embeddings[i] if embeddings is not None else None
                )
                
                results['indexed_documents'] += 1
                
            except Exception as e:
                logger.error(f"Error indexing document {doc_id}: {e}")
                results['errors'].append(f"Document {doc_id}: {str(e)}")
        
        return results
    
    def _index_single_document(self, 
                              document: str, 
                              document_id: Any, 
                              metadata: Dict[str, Any],
                              doc_hash: str,
                              embedding: Optional[np.ndarray]):
        """Индексация одного документа"""
        # Сохраняем метаданные документа
        self.document_metadata[document_id] = {
            'content': document,
            'metadata': metadata,
            'hash': doc_hash,
            'indexed_at': datetime.now().isoformat()
        }
        
        # Текстовый индекс
        if self.config.enable_text_index:
            self._index_text(document, document_id)
        
        # Индекс эмбеддингов
        if self.config.enable_embedding_index and embedding is not None:
            self._index_embedding(embedding, document_id)
        
        # Индекс параметров
        if self.config.enable_parameter_index and 'parameters' in metadata:
            self._index_parameters(metadata['parameters'], document_id)
        
        # Индекс категорий
        if self.config.enable_category_index and 'category' in metadata:
            self._index_category(metadata['category'], document_id)
        
        # Сохраняем в базу данных
        if self.config.storage_backend == "sqlite":
            self._save_to_sqlite(document, document_id, metadata, doc_hash)
    
    def _index_text(self, text: str, document_id: Any):
        """Индексация текста"""
        # Простая токенизация (можно улучшить)
        tokens = text.lower().split()
        
        for token in tokens:
            if len(token) >= 2:  # Игнорируем слишком короткие токены
                if token not in self.text_index:
                    self.text_index[token] = set()
                self.text_index[token].add(document_id)
    
    def _index_embedding(self, embedding: np.ndarray, document_id: Any):
        """Индексация векторного представления"""
        if self.embedding_index is None:
            self.embedding_index = embedding.reshape(1, -1)
        else:
            self.embedding_index = np.vstack([self.embedding_index, embedding.reshape(1, -1)])
    
    def _index_parameters(self, parameters: Dict[str, Any], document_id: Any):
        """Индексация параметров"""
        for param_name, param_value in parameters.items():
            if param_name not in self.parameter_index:
                self.parameter_index[param_name] = {}
            
            param_value_str = str(param_value)
            if param_value_str not in self.parameter_index[param_name]:
                self.parameter_index[param_name][param_value_str] = set()
            
            self.parameter_index[param_name][param_value_str].add(document_id)
    
    def _index_category(self, category: str, document_id: Any):
        """Индексация категории"""
        if category not in self.category_index:
            self.category_index[category] = set()
        self.category_index[category].add(document_id)
    
    def _save_to_sqlite(self, document: str, document_id: Any, metadata: Dict[str, Any], doc_hash: str):
        """Сохранение в SQLite"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Сохраняем документ
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, content, metadata, hash, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (document_id, document, json.dumps(metadata), doc_hash))
            
            conn.commit()
    
    def _is_document_unchanged(self, document_id: Any, doc_hash: str) -> bool:
        """Проверка, изменился ли документ"""
        if self.config.storage_backend == "sqlite":
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT hash FROM documents WHERE id = ?", (document_id,))
                result = cursor.fetchone()
                return result is not None and result[0] == doc_hash
        
        return False
    
    def search_text(self, query: str) -> Set[int]:
        """Поиск по текстовому индексу"""
        query_tokens = query.lower().split()
        
        if not query_tokens:
            return set()
        
        # Пересечение результатов для всех токенов
        result_sets = []
        for token in query_tokens:
            if token in self.text_index:
                result_sets.append(self.text_index[token])
        
        if not result_sets:
            return set()
        
        # Пересечение всех множеств
        result = result_sets[0]
        for result_set in result_sets[1:]:
            result = result.intersection(result_set)
        
        return result
    
    def search_parameters(self, parameter_filters: Dict[str, Any]) -> Set[int]:
        """Поиск по параметрам"""
        result_sets = []
        
        for param_name, param_value in parameter_filters.items():
            param_value_str = str(param_value)
            
            if (param_name in self.parameter_index and 
                param_value_str in self.parameter_index[param_name]):
                result_sets.append(self.parameter_index[param_name][param_value_str])
        
        if not result_sets:
            return set()
        
        # Пересечение всех множеств
        result = result_sets[0]
        for result_set in result_sets[1:]:
            result = result.intersection(result_set)
        
        return result
    
    def search_category(self, category: str) -> Set[int]:
        """Поиск по категории"""
        return self.category_index.get(category, set())
    
    def get_document_metadata(self, document_id: Any) -> Optional[Dict[str, Any]]:
        """Получение метаданных документа"""
        return self.document_metadata.get(document_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики индекса"""
        return {
            **self.stats,
            'text_index_size': len(self.text_index),
            'parameter_index_size': len(self.parameter_index),
            'category_index_size': len(self.category_index),
            'embedding_index_shape': self.embedding_index.shape if self.embedding_index is not None else None,
            'cache_size': len(self.search_cache),
            'storage_backend': self.config.storage_backend
        }
    
    def save_index(self, filepath: str):
        """Сохранение индекса"""
        index_data = {
            'config': self.config,
            'text_index': {k: list(v) for k, v in self.text_index.items()},
            'parameter_index': {
                k: {kk: list(vv) for kk, vv in v.items()} 
                for k, v in self.parameter_index.items()
            },
            'category_index': {k: list(v) for k, v in self.category_index.items()},
            'document_metadata': self.document_metadata,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Сохраняем эмбеддинги отдельно
        if self.embedding_index is not None:
            embedding_path = str(Path(filepath).with_suffix('.npy'))
            np.save(embedding_path, self.embedding_index)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Загрузка индекса"""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.config = index_data['config']
        self.text_index = {k: set(v) for k, v in index_data['text_index'].items()}
        self.parameter_index = {
            k: {kk: set(vv) for kk, vv in v.items()} 
            for k, v in index_data['parameter_index'].items()
        }
        self.category_index = {k: set(v) for k, v in index_data['category_index'].items()}
        self.document_metadata = index_data['document_metadata']
        self.stats = index_data['stats']
        
        # Загружаем эмбеддинги
        embedding_path = str(Path(filepath).with_suffix('.npy'))
        if Path(embedding_path).exists():
            self.embedding_index = np.load(embedding_path)
        
        logger.info(f"Index loaded from {filepath}")
    
    def clear_cache(self):
        """Очистка кэша"""
        with self.cache_lock:
            self.search_cache.clear()
        logger.info("Search cache cleared")
