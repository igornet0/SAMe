"""
Модуль семантического поиска с использованием BERT/Transformers и FAISS
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import torch

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
    
    # Оптимизация
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_gpu: bool = False


class SemanticSearchEngine:
    """Движок семантического поиска"""
    
    def __init__(self, config: SemanticSearchConfig = None):
        self.config = config or SemanticSearchConfig()
        self.model = None
        self.index = None
        self.documents = []
        self.document_ids = []
        self.embeddings = None
        self.is_fitted = False
        
        self._load_model()
        logger.info("SemanticSearchEngine initialized")
    
    def _load_model(self):
        """Загрузка модели sentence-transformers"""
        try:
            device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.config.model_name, device=device)
            logger.info(f"Loaded model {self.config.model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def fit(self, documents: List[str], document_ids: List[Any] = None):
        """
        Обучение поискового движка на корпусе документов
        
        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))
        
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        # Генерация эмбеддингов
        self.embeddings = self._generate_embeddings(documents)
        
        # Создание FAISS индекса
        self._build_index()
        
        self.is_fitted = True
        logger.info("Semantic search engine fitted successfully")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Генерация эмбеддингов для текстов"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        return embeddings.astype(np.float32)
    
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
        
        # Перезагружаем модель
        self._load_model()
        
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
