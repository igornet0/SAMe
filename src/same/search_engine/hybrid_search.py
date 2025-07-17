"""
Модуль гибридного поиска, комбинирующий нечеткий и семантический поиск
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from .semantic_search import SemanticSearchEngine, SemanticSearchConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Конфигурация гибридного поиска"""
    # Конфигурации компонентов
    fuzzy_config: FuzzySearchConfig = None
    semantic_config: SemanticSearchConfig = None
    
    # Веса для комбинирования результатов
    fuzzy_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # Пороги для фильтрации
    min_fuzzy_score: float = 0.5
    min_semantic_score: float = 0.4
    
    # Параметры результатов
    max_candidates_per_method: int = 50
    final_top_k: int = 10
    
    # Стратегия комбинирования
    combination_strategy: str = "weighted_sum"  # weighted_sum, rank_fusion, cascade
    
    # Параллельное выполнение
    enable_parallel_search: bool = True
    max_workers: int = 2


class HybridSearchEngine:
    """Гибридный поисковый движок"""
    
    def __init__(self, config: HybridSearchConfig = None):
        self.config = config or HybridSearchConfig()
        
        # Инициализация компонентов
        self.fuzzy_engine = FuzzySearchEngine(self.config.fuzzy_config)
        self.semantic_engine = SemanticSearchEngine(self.config.semantic_config)
        
        self.is_fitted = False
        
        logger.info("HybridSearchEngine initialized")
    
    def fit(self, documents: List[str], document_ids: List[Any] = None):
        """
        Обучение гибридного поискового движка
        
        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        logger.info(f"Fitting hybrid search engine on {len(documents)} documents")
        
        # Обучение компонентов
        if self.config.enable_parallel_search:
            # Параллельное обучение
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                fuzzy_future = executor.submit(self.fuzzy_engine.fit, documents, document_ids)
                semantic_future = executor.submit(self.semantic_engine.fit, documents, document_ids)
                
                # Ждем завершения
                fuzzy_future.result()
                semantic_future.result()
        else:
            # Последовательное обучение
            self.fuzzy_engine.fit(documents, document_ids)
            self.semantic_engine.fit(documents, document_ids)
        
        self.is_fitted = True
        logger.info("Hybrid search engine fitted successfully")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Гибридный поиск
        
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
        
        top_k = top_k or self.config.final_top_k
        
        # Получение результатов от обоих движков
        if self.config.enable_parallel_search:
            fuzzy_results, semantic_results = self._parallel_search(query)
        else:
            fuzzy_results = self.fuzzy_engine.search(query, self.config.max_candidates_per_method)
            semantic_results = self.semantic_engine.search(query, self.config.max_candidates_per_method)
        
        # Комбинирование результатов
        combined_results = self._combine_results(
            fuzzy_results, 
            semantic_results, 
            query,
            self.config.combination_strategy
        )
        
        # Сортировка и отбор топ-K
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined_results[:top_k]
    
    def _parallel_search(self, query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Параллельный поиск в обоих движках"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            fuzzy_future = executor.submit(
                self.fuzzy_engine.search, 
                query, 
                self.config.max_candidates_per_method
            )
            semantic_future = executor.submit(
                self.semantic_engine.search, 
                query, 
                self.config.max_candidates_per_method
            )
            
            fuzzy_results = fuzzy_future.result()
            semantic_results = semantic_future.result()
        
        return fuzzy_results, semantic_results
    
    def _combine_results(self, 
                        fuzzy_results: List[Dict[str, Any]], 
                        semantic_results: List[Dict[str, Any]], 
                        query: str,
                        strategy: str) -> List[Dict[str, Any]]:
        """Комбинирование результатов разных методов поиска"""
        
        if strategy == "weighted_sum":
            return self._weighted_sum_combination(fuzzy_results, semantic_results)
        elif strategy == "rank_fusion":
            return self._rank_fusion_combination(fuzzy_results, semantic_results)
        elif strategy == "cascade":
            return self._cascade_combination(fuzzy_results, semantic_results)
        else:
            raise ValueError(f"Unknown combination strategy: {strategy}")
    
    def _weighted_sum_combination(self, 
                                 fuzzy_results: List[Dict[str, Any]], 
                                 semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Взвешенное суммирование скоров"""
        # Создаем индекс результатов по document_id
        fuzzy_index = {r['document_id']: r for r in fuzzy_results 
                      if r.get('combined_score', 0) >= self.config.min_fuzzy_score}
        
        semantic_index = {r['document_id']: r for r in semantic_results 
                         if r.get('similarity_score', 0) >= self.config.min_semantic_score}
        
        # Объединяем все уникальные документы
        all_doc_ids = set(fuzzy_index.keys()) | set(semantic_index.keys())
        
        combined_results = []
        
        for doc_id in all_doc_ids:
            fuzzy_result = fuzzy_index.get(doc_id)
            semantic_result = semantic_index.get(doc_id)
            
            # Нормализованные скоры
            fuzzy_score = fuzzy_result.get('combined_score', 0) if fuzzy_result else 0
            semantic_score = semantic_result.get('similarity_score', 0) if semantic_result else 0
            
            # Взвешенная комбинация
            hybrid_score = (
                self.config.fuzzy_weight * fuzzy_score +
                self.config.semantic_weight * semantic_score
            )
            
            # Выбираем базовый результат (приоритет семантическому)
            base_result = semantic_result or fuzzy_result
            
            # Создаем комбинированный результат
            combined_result = {
                **base_result,
                'hybrid_score': hybrid_score,
                'fuzzy_score': fuzzy_score,
                'semantic_score': semantic_score,
                'search_method': 'hybrid',
                'combination_strategy': 'weighted_sum'
            }
            
            # Добавляем детали от обоих методов
            if fuzzy_result:
                combined_result['fuzzy_details'] = {
                    'cosine_score': fuzzy_result.get('cosine_score'),
                    'best_fuzzy_score': fuzzy_result.get('best_fuzzy_score'),
                    'levenshtein_score': fuzzy_result.get('levenshtein_score')
                }
            
            if semantic_result:
                combined_result['semantic_details'] = {
                    'raw_score': semantic_result.get('raw_score'),
                    'rank': semantic_result.get('rank')
                }
            
            combined_results.append(combined_result)
        
        return combined_results
    
    def _rank_fusion_combination(self, 
                                fuzzy_results: List[Dict[str, Any]], 
                                semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Комбинирование на основе рангов (Reciprocal Rank Fusion)"""
        # Создаем индексы с рангами
        fuzzy_ranks = {r['document_id']: i + 1 for i, r in enumerate(fuzzy_results)}
        semantic_ranks = {r['document_id']: i + 1 for i, r in enumerate(semantic_results)}
        
        # Индексы результатов
        fuzzy_index = {r['document_id']: r for r in fuzzy_results}
        semantic_index = {r['document_id']: r for r in semantic_results}
        
        # Все уникальные документы
        all_doc_ids = set(fuzzy_ranks.keys()) | set(semantic_ranks.keys())
        
        combined_results = []
        k = 60  # Константа для RRF
        
        for doc_id in all_doc_ids:
            fuzzy_rank = fuzzy_ranks.get(doc_id, float('inf'))
            semantic_rank = semantic_ranks.get(doc_id, float('inf'))
            
            # Reciprocal Rank Fusion score
            rrf_score = 0
            if fuzzy_rank != float('inf'):
                rrf_score += 1 / (k + fuzzy_rank)
            if semantic_rank != float('inf'):
                rrf_score += 1 / (k + semantic_rank)
            
            # Базовый результат
            base_result = semantic_index.get(doc_id) or fuzzy_index.get(doc_id)
            
            combined_result = {
                **base_result,
                'hybrid_score': rrf_score,
                'fuzzy_rank': fuzzy_rank if fuzzy_rank != float('inf') else None,
                'semantic_rank': semantic_rank if semantic_rank != float('inf') else None,
                'search_method': 'hybrid',
                'combination_strategy': 'rank_fusion'
            }
            
            combined_results.append(combined_result)
        
        return combined_results
    
    def _cascade_combination(self, 
                            fuzzy_results: List[Dict[str, Any]], 
                            semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Каскадное комбинирование: сначала семантический, затем нечеткий"""
        combined_results = []
        used_doc_ids = set()
        
        # Сначала добавляем высококачественные семантические результаты
        for result in semantic_results:
            if result.get('similarity_score', 0) >= 0.7:  # Высокий порог для семантики
                result['hybrid_score'] = result['similarity_score']
                result['search_method'] = 'hybrid'
                result['combination_strategy'] = 'cascade'
                result['primary_method'] = 'semantic'
                
                combined_results.append(result)
                used_doc_ids.add(result['document_id'])
        
        # Затем добавляем нечеткие результаты, которых нет в семантических
        for result in fuzzy_results:
            if (result['document_id'] not in used_doc_ids and 
                result.get('combined_score', 0) >= self.config.min_fuzzy_score):
                
                result['hybrid_score'] = result['combined_score'] * 0.8  # Штраф за нечеткий поиск
                result['search_method'] = 'hybrid'
                result['combination_strategy'] = 'cascade'
                result['primary_method'] = 'fuzzy'
                
                combined_results.append(result)
                used_doc_ids.add(result['document_id'])
        
        # Добавляем оставшиеся семантические результаты
        for result in semantic_results:
            if (result['document_id'] not in used_doc_ids and 
                result.get('similarity_score', 0) >= self.config.min_semantic_score):
                
                result['hybrid_score'] = result['similarity_score']
                result['search_method'] = 'hybrid'
                result['combination_strategy'] = 'cascade'
                result['primary_method'] = 'semantic'
                
                combined_results.append(result)
        
        return combined_results
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """Пакетный гибридный поиск"""
        results = []
        for query in queries:
            try:
                query_results = self.search(query, top_k)
                results.append(query_results)
            except Exception as e:
                logger.error(f"Error in hybrid search for query '{query}': {e}")
                results.append([])
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики гибридного движка"""
        stats = {
            'status': 'fitted' if self.is_fitted else 'not_fitted',
            'combination_strategy': self.config.combination_strategy,
            'weights': {
                'fuzzy_weight': self.config.fuzzy_weight,
                'semantic_weight': self.config.semantic_weight
            },
            'thresholds': {
                'min_fuzzy_score': self.config.min_fuzzy_score,
                'min_semantic_score': self.config.min_semantic_score
            }
        }
        
        if self.is_fitted:
            stats['fuzzy_engine'] = self.fuzzy_engine.get_statistics()
            stats['semantic_engine'] = self.semantic_engine.get_statistics()
        
        return stats
    
    def save_model(self, filepath: str):
        """Сохранение гибридной модели"""
        import pickle
        
        model_data = {
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        # Сохраняем основные данные
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Сохраняем компоненты отдельно
        from pathlib import Path
        base_path = Path(filepath).parent
        
        fuzzy_path = base_path / f"{Path(filepath).stem}_fuzzy.pkl"
        semantic_path = base_path / f"{Path(filepath).stem}_semantic.pkl"
        
        self.fuzzy_engine.save_model(str(fuzzy_path))
        self.semantic_engine.save_model(str(semantic_path))
        
        logger.info(f"Hybrid model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка гибридной модели"""
        import pickle
        
        # Загружаем основные данные
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        # Загружаем компоненты
        from pathlib import Path
        base_path = Path(filepath).parent
        
        fuzzy_path = base_path / f"{Path(filepath).stem}_fuzzy.pkl"
        semantic_path = base_path / f"{Path(filepath).stem}_semantic.pkl"
        
        self.fuzzy_engine = FuzzySearchEngine(self.config.fuzzy_config)
        self.fuzzy_engine.load_model(str(fuzzy_path))
        
        self.semantic_engine = SemanticSearchEngine(self.config.semantic_config)
        self.semantic_engine.load_model(str(semantic_path))
        
        logger.info(f"Hybrid model loaded from {filepath}")
