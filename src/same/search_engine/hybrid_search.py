"""
Модуль гибридного поиска, комбинирующий нечеткий и семантический поиск
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from .semantic_search import SemanticSearchEngine, SemanticSearchConfig
from ..categorization import CategoryClassifier, CategoryClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Конфигурация гибридного поиска"""
    # Конфигурации компонентов
    fuzzy_config: Optional[FuzzySearchConfig] = None
    semantic_config: Optional[SemanticSearchConfig] = None
    
    # Веса для комбинирования результатов
    fuzzy_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # Пороги для фильтрации
    min_fuzzy_score: float = 0.5
    min_semantic_score: float = 0.4
    
    # Параметры результатов
    max_candidates_per_method: int = 50
    final_top_k: int = 10
    max_results: int = 10  # Alias for final_top_k for notebook compatibility
    similarity_threshold: float = 0.4  # Overall similarity threshold for notebook compatibility

    # Стратегия комбинирования
    combination_strategy: str = "weighted_sum"  # weighted_sum, rank_fusion, cascade

    # Категориальная фильтрация
    enable_category_filtering: bool = True
    category_config: Optional[CategoryClassifierConfig] = None

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

        # Инициализация классификатора категорий
        if self.config.enable_category_filtering:
            self.category_classifier = CategoryClassifier(self.config.category_config)
        else:
            self.category_classifier = None

        self.is_fitted = False
        
        logger.info("HybridSearchEngine initialized")
    
    def fit(self, documents: List[str], document_ids: List[Any] = None, metadata: List[Dict[str, Any]] = None):
        """
        Обучение гибридного поискового движка

        Args:
            documents: Список текстов для индексации
            document_ids: Список ID документов (опционально)
            metadata: Список метаданных для каждого документа
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        logger.info(f"Fitting hybrid search engine on {len(documents)} documents")

        # Классификация документов по категориям если нужно
        if self.config.enable_category_filtering and self.category_classifier:
            logger.info("Classifying documents by categories...")
            if not metadata:
                metadata = []
                for i, doc in enumerate(documents):
                    category, confidence = self.category_classifier.classify(doc)
                    metadata.append({
                        'category': category,
                        'category_confidence': confidence
                    })
            else:
                # Дополняем существующие метаданные категориями
                for i, (doc, meta) in enumerate(zip(documents, metadata)):
                    if 'category' not in meta:
                        category, confidence = self.category_classifier.classify(doc)
                        meta['category'] = category
                        meta['category_confidence'] = confidence

        # Обучение компонентов
        if self.config.enable_parallel_search:
            # Параллельное обучение
            try:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    fuzzy_future = executor.submit(self.fuzzy_engine.fit, documents, document_ids)
                    semantic_future = executor.submit(self.semantic_engine.fit, documents, document_ids, metadata)

                    # Ждем завершения
                    fuzzy_future.result()
                    semantic_future.result()
            except Exception as e:
                logger.warning(f"Parallel training failed: {e}, falling back to sequential")
                # Fallback к последовательному обучению
                self.fuzzy_engine.fit(documents, document_ids)
                self.semantic_engine.fit(documents, document_ids, metadata)
        else:
            # Последовательное обучение
            self.fuzzy_engine.fit(documents, document_ids)
            self.semantic_engine.fit(documents, document_ids, metadata)
        
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

        # Определяем категорию запроса для фильтрации
        category_filter = None
        if self.config.enable_category_filtering and self.category_classifier:
            category_filter, confidence = self.category_classifier.classify(query)
            if confidence < 0.5:  # Если уверенность низкая, не применяем фильтр
                category_filter = None
            logger.debug(f"Query category: {category_filter} (confidence: {confidence:.2f})")

        # Получение результатов от обоих движков
        if self.config.enable_parallel_search:
            fuzzy_results, semantic_results = self._parallel_search(query, category_filter)
        else:
            fuzzy_results = self.fuzzy_engine.search(query, self.config.max_candidates_per_method)
            semantic_results = self.semantic_engine.search(query, self.config.max_candidates_per_method, category_filter)
        
        # Комбинирование результатов
        combined_results = self._combine_results(
            fuzzy_results,
            semantic_results,
            query,
            self.config.combination_strategy
        )

        # Сортировка и отбор топ-K с защитой от отсутствующих ключей
        # ИСПРАВЛЕНИЕ: Добавляем защиту от KeyError при сортировке
        try:
            combined_results.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        except (KeyError, TypeError) as e:
            logger.warning(f"Error sorting hybrid results: {e}. Using fallback sorting.")
            # Fallback: сортируем по любому доступному скору
            combined_results.sort(key=lambda x: x.get('hybrid_score',
                                                    x.get('similarity_score',
                                                    x.get('combined_score', 0.0))), reverse=True)

        # Добавляем ранги к результатам
        final_results = combined_results[:top_k]
        for i, result in enumerate(final_results):
            result['rank'] = i + 1

        return final_results
    
    def _parallel_search(self, query: str, category_filter: str = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                self.config.max_candidates_per_method,
                category_filter
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
        """Оптимизированное взвешенное суммирование скоров"""
        if not fuzzy_results and not semantic_results:
            return []

        # Быстрая обработка случаев с одним типом результатов
        # ИСПРАВЛЕНИЕ: Всегда добавляем hybrid_score для консистентности
        if not fuzzy_results:
            results = []
            for r in semantic_results:
                if r.get('similarity_score', 0) >= self.config.min_semantic_score:
                    # Создаем копию результата с hybrid_score
                    result_copy = {**r}
                    result_copy['hybrid_score'] = r.get('similarity_score', 0) * self.config.semantic_weight
                    result_copy['fuzzy_score'] = 0.0
                    result_copy['semantic_score'] = r.get('similarity_score', 0)
                    result_copy['search_method'] = 'hybrid'
                    result_copy['combination_strategy'] = 'weighted_sum'
                    result_copy['primary_method'] = 'semantic'
                    results.append(result_copy)
            return results

        if not semantic_results:
            results = []
            for r in fuzzy_results:
                if r.get('combined_score', 0) >= self.config.min_fuzzy_score:
                    # Создаем копию результата с hybrid_score
                    result_copy = {**r}
                    result_copy['hybrid_score'] = r.get('combined_score', 0) * self.config.fuzzy_weight
                    result_copy['fuzzy_score'] = r.get('combined_score', 0)
                    result_copy['semantic_score'] = 0.0
                    result_copy['search_method'] = 'hybrid'
                    result_copy['combination_strategy'] = 'weighted_sum'
                    result_copy['primary_method'] = 'fuzzy'
                    results.append(result_copy)
            return results

        # Создаем индексы результатов по document_id с предварительной фильтрацией
        fuzzy_index = {}
        for r in fuzzy_results:
            score = r.get('combined_score', 0)
            if score >= self.config.min_fuzzy_score:
                fuzzy_index[r['document_id']] = r

        semantic_index = {}
        for r in semantic_results:
            score = r.get('similarity_score', 0)
            if score >= self.config.min_semantic_score:
                semantic_index[r['document_id']] = r

        # Быстрый выход если после фильтрации ничего не осталось
        if not fuzzy_index and not semantic_index:
            return []

        # Объединяем все уникальные документы
        all_doc_ids = set(fuzzy_index.keys()) | set(semantic_index.keys())

        combined_results = []

        # Предвычисляем веса для оптимизации
        fuzzy_weight = self.config.fuzzy_weight
        semantic_weight = self.config.semantic_weight

        for doc_id in all_doc_ids:
            fuzzy_result = fuzzy_index.get(doc_id)
            semantic_result = semantic_index.get(doc_id)

            # Нормализованные скоры
            fuzzy_score = fuzzy_result.get('combined_score', 0) if fuzzy_result else 0
            semantic_score = semantic_result.get('similarity_score', 0) if semantic_result else 0

            # Взвешенная комбинация с бонусом за присутствие в обоих результатах
            hybrid_score = fuzzy_weight * fuzzy_score + semantic_weight * semantic_score

            # Бонус за консенсус (присутствие в обоих типах поиска)
            if fuzzy_result and semantic_result:
                consensus_bonus = 0.1 * min(fuzzy_score, semantic_score)
                hybrid_score += consensus_bonus

            # Выбираем базовый результат (приоритет семантическому для метаданных)
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
                'semantic_weight': self.config.semantic_weight,
                'fuzzy': self.config.fuzzy_weight,  # Alias for notebook compatibility
                'semantic': self.config.semantic_weight  # Alias for notebook compatibility
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
