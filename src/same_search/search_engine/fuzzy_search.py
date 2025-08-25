"""
Модуль нечеткого поиска с использованием TF-IDF и расстояний
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FuzzySearchConfig:
    """Конфигурация нечеткого поиска"""
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_min_df: int = 1  # Changed from 2 to 1 to handle small datasets
    tfidf_max_df: float = 0.95

    # Пороги схожести
    cosine_threshold: float = 0.3
    fuzzy_threshold: int = 60
    levenshtein_threshold: int = 70
    similarity_threshold: float = 0.3  # Alias for cosine_threshold for notebook compatibility

    # Веса для комбинированного скора
    cosine_weight: float = 0.4
    fuzzy_weight: float = 0.3
    levenshtein_weight: float = 0.3

    # Параметры поиска
    max_candidates: int = 100
    top_k_results: int = 10
    max_results: int = 10  # Alias for top_k_results for notebook compatibility
    use_stemming: bool = False  # For notebook compatibility


class FuzzySearchEngine:
    """Движок нечеткого поиска"""
    
    def __init__(self, config: FuzzySearchConfig = None):
        self.config = config or FuzzySearchConfig()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.document_ids = []
        self.is_fitted = False
        
        logger.info("FuzzySearchEngine initialized")
    
    def fit(self, documents: List[str], document_ids: List[Any] = None):
        """
        Обучение поискового движка на корпусе документов
        
        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
        """
        if not documents:
            # Разрешаем пустые документы для тестирования
            self.documents = []
            self.document_ids = []
            self.vectorizer = None
            self.tfidf_matrix = None
            self.is_fitted = True  # Устанавливаем флаг для пустых документов
            logger.warning("Initialized with empty documents list")
            return
        
        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))
        
        logger.info(f"Fitting TF-IDF vectorizer on {len(documents)} documents")

        # Динамическая настройка min_df для малых наборов данных
        min_df = self.config.tfidf_min_df
        if len(documents) < 10:
            min_df = 1  # Для очень малых наборов данных используем min_df=1
            logger.info(f"Small dataset detected ({len(documents)} docs), using min_df=1")

        # Инициализация TF-IDF векторизатора
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            min_df=min_df,
            max_df=self.config.tfidf_max_df,
            lowercase=True,
            stop_words=None,  # Стоп-слова уже удалены на этапе предобработки
            token_pattern=r'\b\w+\b'
        )
        
        # Построение TF-IDF матрицы
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.is_fitted = True
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию из конфига)
            
        Returns:
            Список результатов поиска
        """
        if not self.is_fitted:
            raise ValueError("Search engine is not fitted. Call fit() first.")

        if not query or not isinstance(query, str):
            return []

        # Если нет документов, возвращаем пустой список
        if not self.documents:
            return []
        
        top_k = top_k or self.config.top_k_results
        
        # Этап 1: TF-IDF поиск
        tfidf_results = self._tfidf_search(query, self.config.max_candidates)
        
        # Этап 2: Нечеткий поиск для уточнения
        fuzzy_results = self._fuzzy_search(query, tfidf_results)
        
        # Этап 3: Комбинирование скоров
        combined_results = self._combine_scores(fuzzy_results)
        
        # Сортировка и отбор топ-K
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Добавляем ранги к результатам и поле content для совместимости
        final_results = combined_results[:top_k]
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
            # Добавляем поле content для совместимости с интерфейсом
            if 'content' not in result and 'document' in result:
                result['content'] = result['document']
            # Добавляем поле score для совместимости
            if 'score' not in result and 'combined_score' in result:
                result['score'] = result['combined_score']

        return final_results
    
    def _tfidf_search(self, query: str, max_candidates: int) -> List[Dict[str, Any]]:
        """TF-IDF поиск"""
        # Векторизация запроса
        query_vector = self.vectorizer.transform([query])
        
        # Вычисление косинусного сходства
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Отбор кандидатов выше порога
        candidates = []
        for idx, score in enumerate(cosine_scores):
            if score >= self.config.cosine_threshold:
                candidates.append({
                    'document_id': self.document_ids[idx],
                    'document': self.documents[idx],
                    'cosine_score': float(score),
                    'index': idx
                })
        
        # Сортировка по косинусному сходству
        candidates.sort(key=lambda x: x['cosine_score'], reverse=True)
        
        return candidates[:max_candidates]
    
    def _fuzzy_search(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Нечеткий поиск по кандидатам"""
        results = []
        
        for candidate in candidates:
            document = candidate['document']
            
            # Различные метрики нечеткого сравнения
            fuzzy_ratio = fuzz.ratio(query, document)
            fuzzy_partial = fuzz.partial_ratio(query, document)
            fuzzy_token_sort = fuzz.token_sort_ratio(query, document)
            fuzzy_token_set = fuzz.token_set_ratio(query, document)
            
            # Расстояние Левенштейна (нормализованное)
            levenshtein_score = fuzz.ratio(query, document)
            
            # Лучший нечеткий скор
            best_fuzzy_score = max(fuzzy_ratio, fuzzy_partial, fuzzy_token_sort, fuzzy_token_set)
            
            # Добавляем метрики к кандидату
            candidate.update({
                'fuzzy_ratio': fuzzy_ratio,
                'fuzzy_partial': fuzzy_partial,
                'fuzzy_token_sort': fuzzy_token_sort,
                'fuzzy_token_set': fuzzy_token_set,
                'best_fuzzy_score': best_fuzzy_score,
                'levenshtein_score': levenshtein_score
            })
            
            # Фильтрация по порогам
            if (best_fuzzy_score >= self.config.fuzzy_threshold or 
                levenshtein_score >= self.config.levenshtein_threshold):
                results.append(candidate)
        
        return results
    
    def _combine_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Комбинирование различных скоров"""
        for result in results:
            # Нормализация скоров (приведение к диапазону 0-1)
            cosine_norm = result['cosine_score']  # Уже в диапазоне 0-1
            fuzzy_norm = result['best_fuzzy_score'] / 100.0  # Приведение к 0-1
            levenshtein_norm = result['levenshtein_score'] / 100.0  # Приведение к 0-1
            
            # Взвешенная комбинация
            combined_score = (
                self.config.cosine_weight * cosine_norm +
                self.config.fuzzy_weight * fuzzy_norm +
                self.config.levenshtein_weight * levenshtein_norm
            )
            
            result['combined_score'] = combined_score
            result['cosine_norm'] = cosine_norm
            result['fuzzy_norm'] = fuzzy_norm
            result['levenshtein_norm'] = levenshtein_norm
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """Пакетный поиск"""
        results = []
        for query in queries:
            try:
                query_results = self.search(query, top_k)
                results.append(query_results)
            except Exception as e:
                logger.error(f"Error searching for query '{query}': {e}")
                results.append([])
        
        return results
    
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
    
    def save_model(self, filepath: str):
        """Сохранение модели"""
        # Конвертируем конфигурацию в словарь для сериализации
        config_dict = {
            'tfidf_max_features': self.config.tfidf_max_features,
            'tfidf_ngram_range': self.config.tfidf_ngram_range,
            'tfidf_min_df': self.config.tfidf_min_df,
            'tfidf_max_df': self.config.tfidf_max_df,
            'cosine_threshold': self.config.cosine_threshold,
            'fuzzy_threshold': self.config.fuzzy_threshold,
            'levenshtein_threshold': self.config.levenshtein_threshold,
            'similarity_threshold': self.config.similarity_threshold,
            'cosine_weight': self.config.cosine_weight,
            'fuzzy_weight': self.config.fuzzy_weight,
            'levenshtein_weight': self.config.levenshtein_weight,
            'max_candidates': self.config.max_candidates,
            'top_k_results': self.config.top_k_results,
            'max_results': self.config.max_results,
            'use_stemming': self.config.use_stemming
        }
        
        model_data = {
            'config': config_dict,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Восстанавливаем конфигурацию из словаря
        if isinstance(model_data['config'], dict):
            config_dict = model_data['config']
            self.config = FuzzySearchConfig(
                tfidf_max_features=config_dict['tfidf_max_features'],
                tfidf_ngram_range=config_dict['tfidf_ngram_range'],
                tfidf_min_df=config_dict['tfidf_min_df'],
                tfidf_max_df=config_dict['tfidf_max_df'],
                cosine_threshold=config_dict['cosine_threshold'],
                fuzzy_threshold=config_dict['fuzzy_threshold'],
                levenshtein_threshold=config_dict['levenshtein_threshold'],
                similarity_threshold=config_dict['similarity_threshold'],
                cosine_weight=config_dict['cosine_weight'],
                fuzzy_weight=config_dict['fuzzy_weight'],
                levenshtein_weight=config_dict['levenshtein_weight'],
                max_candidates=config_dict['max_candidates'],
                top_k_results=config_dict['top_k_results'],
                max_results=config_dict['max_results'],
                use_stemming=config_dict['use_stemming']
            )
        else:
            # Обратная совместимость со старым форматом
            self.config = model_data['config']
        
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.documents = model_data['documents']
        self.document_ids = model_data['document_ids']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики поискового движка"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'tfidf_matrix_shape': self.tfidf_matrix.shape,
            'tfidf_matrix_density': self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]),
            'config': {
                'max_features': self.config.tfidf_max_features,
                'ngram_range': self.config.tfidf_ngram_range,
                'cosine_threshold': self.config.cosine_threshold,
                'fuzzy_threshold': self.config.fuzzy_threshold
            }
        }
