"""
Система поиска по токенам для проекта SAMe

Этот модуль предоставляет различные методы поиска на основе векторизации токенов:
- Поиск по ID токенов
- Семантический поиск по эмбеддингам
- Гибридный поиск (комбинация методов)
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска"""
    code: str
    raw_name: str
    tokens: str
    token_vectors: str
    parameters: str
    score: float
    method: str
    matched_tokens: List[str] = None
    similarity_details: Dict[str, Any] = None


@dataclass
class SearchConfig:
    """Конфигурация системы поиска"""
    # Поиск по ID токенов
    token_id_weight: float = 0.6
    min_token_matches: int = 1
    
    # Семантический поиск
    semantic_weight: float = 0.4
    similarity_threshold: float = 0.3
    
    # Общие настройки
    max_results: int = 100
    score_threshold: float = 0.1
    
    # Бустинг для технических терминов
    technical_boost: float = 1.5
    technical_patterns: List[str] = None
    
    def __post_init__(self):
        if self.technical_patterns is None:
            self.technical_patterns = [
                r'ГОСТ\s*\d+[-]\d+',
                r'IP\d+',
                r'М\d+х\d+',
                r'\d+х\d+х\d+',
                r'[А-Я]+\d+[А-Я]*',
                r'\d+[WВвт]+',
                r'\d+[VВв]',
                r'\d+[Aa]'
            ]


class TokenSearchEngine:
    """Система поиска по токенам"""
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        
        # Данные для поиска
        self.data_df = None
        self.token_vectors_matrix = None
        self.embeddings_matrix = None
        
        # Векторизатор для обработки запросов
        self.vectorizer = None
        self.tokenizer = None
        
        # Индексы для быстрого поиска
        self.token_id_index = {}  # ID токена -> список индексов записей
        self.code_to_index = {}   # Код записи -> индекс в DataFrame
        
        logger.info("TokenSearchEngine initialized")
    
    def load_data(self, csv_file: str, vectorizer=None, tokenizer=None) -> bool:
        """
        Загрузка данных для поиска
        
        Args:
            csv_file: Путь к CSV файлу с обработанными данными
            vectorizer: Обученный векторизатор токенов
            tokenizer: Токенизатор для обработки запросов
            
        Returns:
            True если загрузка прошла успешно
        """
        try:
            logger.info(f"Loading search data from: {csv_file}")
            
            # Загружаем данные
            self.data_df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(self.data_df)} records")
            
            # Сохраняем векторизатор и токенизатор
            self.vectorizer = vectorizer
            self.tokenizer = tokenizer
            
            # Строим индексы
            self._build_token_id_index()
            self._build_code_index()
            
            # Строим матрицу эмбеддингов если доступна
            if vectorizer and hasattr(vectorizer, 'token_embeddings'):
                self._build_embeddings_matrix()
            
            logger.info("Search engine data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading search data: {e}")
            return False
    
    def _build_token_id_index(self):
        """Построение индекса ID токенов"""
        logger.info("Building token ID index...")
        
        self.token_id_index = {}
        
        for idx, row in self.data_df.iterrows():
            token_vectors = row.get('token_vectors', '')
            
            if pd.notna(token_vectors) and token_vectors:
                # Извлекаем ID токенов из строки "IDs: [1, 2, 3, ...]"
                token_ids = self._parse_token_ids(token_vectors)
                
                for token_id in token_ids:
                    if token_id not in self.token_id_index:
                        self.token_id_index[token_id] = []
                    self.token_id_index[token_id].append(idx)
        
        logger.info(f"Built token ID index with {len(self.token_id_index)} unique tokens")
    
    def _build_code_index(self):
        """Построение индекса кодов записей"""
        self.code_to_index = {}
        
        for idx, row in self.data_df.iterrows():
            code = row.get('Код', '')
            if code:
                self.code_to_index[code] = idx
    
    def _build_embeddings_matrix(self):
        """Построение матрицы эмбеддингов для семантического поиска"""
        logger.info("Building embeddings matrix...")
        
        embeddings_list = []
        
        for idx, row in self.data_df.iterrows():
            # Получаем токены записи
            tokens_str = row.get('tokenizer', '')
            if pd.notna(tokens_str) and tokens_str:
                tokens = tokens_str.split()
                
                # Получаем эмбеддинги токенов
                record_embeddings = []
                for token in tokens:
                    token_info = self.vectorizer.get_token_info(token)
                    if token_info.get('found', False) and 'embedding' in token_info:
                        record_embeddings.append(token_info['embedding'])
                
                # Агрегируем эмбеддинги (среднее)
                if record_embeddings:
                    avg_embedding = np.mean(record_embeddings, axis=0)
                    embeddings_list.append(avg_embedding)
                else:
                    # Нулевой вектор для записей без эмбеддингов
                    embedding_dim = self.vectorizer.config.embedding_dim
                    embeddings_list.append(np.zeros(embedding_dim))
            else:
                # Нулевой вектор для пустых записей
                embedding_dim = self.vectorizer.config.embedding_dim
                embeddings_list.append(np.zeros(embedding_dim))
        
        self.embeddings_matrix = np.array(embeddings_list)
        logger.info(f"Built embeddings matrix: {self.embeddings_matrix.shape}")
    
    def _parse_token_ids(self, token_vectors_str: str) -> List[int]:
        """Извлечение ID токенов из строки"""
        try:
            # Ищем паттерн "IDs: [1, 2, 3, ...]"
            match = re.search(r'IDs:\s*\[([\d,\s]+)\]', token_vectors_str)
            if match:
                ids_str = match.group(1)
                # Парсим числа
                token_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
                return token_ids
            return []
        except Exception as e:
            logger.warning(f"Error parsing token IDs from '{token_vectors_str}': {e}")
            return []
    
    def search_by_token_ids(self, query_token_ids: List[int], top_k: int = 10) -> List[SearchResult]:
        """
        Поиск по ID токенов
        
        Args:
            query_token_ids: Список ID токенов для поиска
            top_k: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        if not query_token_ids:
            return []
        
        logger.info(f"Searching by token IDs: {query_token_ids}")
        
        # Подсчитываем совпадения для каждой записи
        record_scores = {}
        
        for token_id in query_token_ids:
            if token_id in self.token_id_index:
                for record_idx in self.token_id_index[token_id]:
                    if record_idx not in record_scores:
                        record_scores[record_idx] = {
                            'matches': 0,
                            'matched_tokens': []
                        }
                    record_scores[record_idx]['matches'] += 1
                    record_scores[record_idx]['matched_tokens'].append(token_id)
        
        # Фильтруем по минимальному количеству совпадений
        filtered_records = {
            idx: data for idx, data in record_scores.items()
            if data['matches'] >= self.config.min_token_matches
        }
        
        # Сортируем по количеству совпадений
        sorted_records = sorted(
            filtered_records.items(),
            key=lambda x: x[1]['matches'],
            reverse=True
        )
        
        # Формируем результаты
        results = []
        for record_idx, match_data in sorted_records[:top_k]:
            row = self.data_df.iloc[record_idx]
            
            # Вычисляем оценку
            score = match_data['matches'] / len(query_token_ids)
            
            # Применяем бустинг для технических терминов
            if self._has_technical_terms(row.get('tokenizer', '')):
                score *= self.config.technical_boost
            
            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=score,
                method='token_ids',
                matched_tokens=[str(tid) for tid in match_data['matched_tokens']]
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} results by token IDs")
        return results
    
    def search_by_similarity(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Семантический поиск по эмбеддингам
        
        Args:
            query_embedding: Векторное представление запроса
            top_k: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        if self.embeddings_matrix is None:
            logger.warning("Embeddings matrix not available for semantic search")
            return []
        
        logger.info("Performing semantic search...")
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Фильтруем по порогу сходства
        valid_indices = np.where(similarities >= self.config.similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            logger.info("No results above similarity threshold")
            return []
        
        # Сортируем по убыванию сходства
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Формируем результаты
        results = []
        for idx in sorted_indices[:top_k]:
            row = self.data_df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Применяем бустинг для технических терминов
            if self._has_technical_terms(row.get('tokenizer', '')):
                similarity_score *= self.config.technical_boost
            
            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=similarity_score,
                method='semantic',
                similarity_details={'cosine_similarity': similarities[idx]}
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} results by semantic search")
        return results
    
    def _has_technical_terms(self, tokens_str: str) -> bool:
        """Проверка наличия технических терминов"""
        if not tokens_str:
            return False
        
        for pattern in self.config.technical_patterns:
            if re.search(pattern, tokens_str):
                return True
        return False
