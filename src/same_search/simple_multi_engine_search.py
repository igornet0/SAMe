"""
Упрощенный мульти-движковый поиск с использованием только fuzzy и semantic движков
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from tqdm import tqdm

# Импорты поисковых движков
from .search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from .search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
from .categorization.category_classifier import CategoryClassifier, CategoryClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class SimpleMultiEngineConfig:
    """Конфигурация упрощенного мульти-движкового поиска"""
    # Веса для комбинирования результатов
    fuzzy_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # Пороги схожести
    min_similarity_threshold: float = 0.3
    fuzzy_threshold: float = 0.3
    semantic_threshold: float = 0.3
    
    # Параметры результатов
    max_candidates_per_engine: int = 20
    final_top_k: int = 50
    
    # Параллельная обработка
    enable_parallel_search: bool = True


class SimpleMultiEngineSearch:
    """Упрощенный мульти-движковый поисковый класс"""
    
    def __init__(self, config: SimpleMultiEngineConfig = None):
        self.config = config or SimpleMultiEngineConfig()
        
        # Инициализация движков
        fuzzy_config = FuzzySearchConfig(
            cosine_threshold=self.config.fuzzy_threshold,
            fuzzy_threshold=50,
            top_k_results=self.config.max_candidates_per_engine
        )
        self.fuzzy_engine = FuzzySearchEngine(fuzzy_config)
        
        semantic_config = SemanticSearchConfig(
            similarity_threshold=self.config.semantic_threshold,
            top_k_results=self.config.max_candidates_per_engine
        )
        self.semantic_engine = SemanticSearchEngine(semantic_config)
        
        # Классификатор категорий
        self.category_classifier = CategoryClassifier()
        
        # Флаг готовности
        self.is_fitted = False
        
        logger.info("SimpleMultiEngineSearch initialized")
    
    async def fit(self, catalog_df: pd.DataFrame, text_column: str = 'processed_name'):
        """
        Обучение поисковых движков
        
        Args:
            catalog_df: DataFrame с каталогом товаров
            text_column: Название колонки с текстом для анализа
        """
        logger.info(f"Training simple multi-engine search on {len(catalog_df)} items...")
        
        # Подготовка данных
        documents = catalog_df[text_column].fillna('').astype(str).tolist()
        document_ids = catalog_df.index.tolist()
        
        # Подготовка метаданных
        metadata = []
        for idx, row in catalog_df.iterrows():
            category, confidence = self.category_classifier.classify(row[text_column])
            metadata.append({
                'category': category,
                'category_confidence': confidence,
                'original_index': idx
            })
        
        # Параллельное обучение движков
        if self.config.enable_parallel_search:
            await self._parallel_fit(documents, document_ids, metadata)
        else:
            await self._sequential_fit(documents, document_ids, metadata)
        
        self.is_fitted = True
        logger.info("Simple multi-engine search trained successfully")
    
    async def _parallel_fit(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Параллельное обучение движков"""
        tasks = [
            asyncio.create_task(self._fit_fuzzy_engine(documents, document_ids)),
            asyncio.create_task(self._fit_semantic_engine(documents, document_ids, metadata))
        ]
        
        # Ждем завершения всех задач
        await asyncio.gather(*tasks)
        logger.info("Both engines trained successfully")
    
    async def _sequential_fit(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Последовательное обучение движков"""
        await self._fit_fuzzy_engine(documents, document_ids)
        await self._fit_semantic_engine(documents, document_ids, metadata)
        logger.info("Both engines trained successfully")
    
    async def _fit_fuzzy_engine(self, documents: List[str], document_ids: List[int]):
        """Обучение fuzzy движка"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.fuzzy_engine.fit, documents, document_ids)
        logger.info("Fuzzy engine trained successfully")
    
    async def _fit_semantic_engine(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Обучение semantic движка"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.semantic_engine.fit, documents, document_ids, metadata)
        logger.info("Semantic engine trained successfully")
    
    async def search_analogs(self, query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """
        Поиск аналогов с использованием обоих движков
        
        Args:
            query_idx: Индекс запроса
            query_text: Текст запроса
            
        Returns:
            Список найденных аналогов
        """
        if not self.is_fitted:
            raise ValueError("Search engines are not fitted. Call fit() first.")
        
        # Параллельный поиск в обоих движках
        if self.config.enable_parallel_search:
            fuzzy_results, semantic_results = await self._parallel_search(query_text)
        else:
            fuzzy_results = self.fuzzy_engine.search(query_text, self.config.max_candidates_per_engine)
            semantic_results = self.semantic_engine.search(query_text, self.config.max_candidates_per_engine)
        
        # Комбинирование результатов
        combined_results = self._combine_results(fuzzy_results, semantic_results, query_idx, query_text)
        
        return combined_results[:self.config.final_top_k]
    
    async def _parallel_search(self, query_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Параллельный поиск в обоих движках"""
        loop = asyncio.get_event_loop()
        
        # Запускаем поиск в обоих движках параллельно
        fuzzy_task = loop.run_in_executor(
            None, self.fuzzy_engine.search, query_text, self.config.max_candidates_per_engine
        )
        semantic_task = loop.run_in_executor(
            None, self.semantic_engine.search, query_text, self.config.max_candidates_per_engine
        )
        
        # Ждем завершения обоих задач
        fuzzy_results, semantic_results = await asyncio.gather(fuzzy_task, semantic_task)
        
        return fuzzy_results, semantic_results
    
    def _combine_results(self, fuzzy_results: List[Dict[str, Any]], semantic_results: List[Dict[str, Any]], 
                        query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """Комбинирование результатов от обоих движков"""
        if not fuzzy_results and not semantic_results:
            return []
        
        # Собираем все кандидаты
        all_candidates = {}
        
        # Обрабатываем результаты fuzzy поиска
        for result in fuzzy_results:
            candidate_id = self._get_candidate_id(result)
            if candidate_id not in all_candidates:
                all_candidates[candidate_id] = {
                    'candidate': result,
                    'fuzzy_score': result.get('combined_score', 0.0),
                    'semantic_score': 0.0,
                    'engines': []
                }
            all_candidates[candidate_id]['engines'].append('fuzzy')
        
        # Обрабатываем результаты semantic поиска
        for result in semantic_results:
            candidate_id = self._get_candidate_id(result)
            if candidate_id not in all_candidates:
                all_candidates[candidate_id] = {
                    'candidate': result,
                    'fuzzy_score': 0.0,
                    'semantic_score': result.get('similarity_score', 0.0),
                    'engines': []
                }
            else:
                all_candidates[candidate_id]['semantic_score'] = result.get('similarity_score', 0.0)
            all_candidates[candidate_id]['engines'].append('semantic')
        
        # Формируем финальные результаты
        final_results = []
        for candidate_id, data in all_candidates.items():
            # Вычисляем комбинированный скор
            combined_score = (
                self.config.fuzzy_weight * data['fuzzy_score'] +
                self.config.semantic_weight * data['semantic_score']
            )
            
            # Бонус за консенсус (присутствие в обоих движках)
            if len(data['engines']) > 1:
                consensus_bonus = 0.1 * min(data['fuzzy_score'], data['semantic_score'])
                combined_score += consensus_bonus
            
            if combined_score >= self.config.min_similarity_threshold:
                result = data['candidate'].copy()
                result['multi_engine_score'] = combined_score
                result['fuzzy_score'] = data['fuzzy_score']
                result['semantic_score'] = data['semantic_score']
                result['engine_consensus'] = len(data['engines'])
                result['engines_used'] = data['engines']
                result['search_method'] = 'simple_multi_engine'
                result['combination_strategy'] = 'weighted_consensus'
                
                # Добавляем информацию о запросе
                result['query_index'] = query_idx
                result['query_name'] = query_text
                
                final_results.append(result)
        
        # Сортируем по комбинированному скору
        final_results.sort(key=lambda x: x['multi_engine_score'], reverse=True)
        
        return final_results
    
    def _get_candidate_id(self, candidate: Dict[str, Any]) -> str:
        """Получение уникального идентификатора кандидата"""
        # Пытаемся использовать различные поля для идентификации
        for field in ['document_id', 'index', 'candidate_idx']:
            if field in candidate and candidate[field] is not None:
                return str(candidate[field])
        
        # Если нет уникального ID, используем название
        name = candidate.get('document', candidate.get('content', candidate.get('Candidate_Name', '')))
        return str(name)
    
    async def process_catalog(self, catalog_df: pd.DataFrame, 
                            text_column: str = 'processed_name',
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Обработка всего каталога с упрощенным мульти-движковым поиском
        
        Args:
            catalog_df: DataFrame с каталогом товаров
            text_column: Название колонки с текстом
            output_path: Путь для сохранения результатов
            
        Returns:
            DataFrame с результатами поиска
        """
        logger.info(f"Processing catalog with simple multi-engine search on {len(catalog_df)} items...")
        
        # Обучаем движки
        await self.fit(catalog_df, text_column)
        
        # Обрабатываем каждый товар
        all_results = []
        
        for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Simple multi-engine search"):
            try:
                query_text = str(row[text_column])
                results = await self.search_analogs(idx, query_text)
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue
        
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")
        
        logger.info(f"Simple multi-engine processing completed. Found {len(all_results)} analog relationships")
        return results_df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики упрощенного мульти-движкового поиска"""
        stats = {
            'is_fitted': self.is_fitted,
            'engines': ['fuzzy', 'semantic'],
            'config': {
                'fuzzy_weight': self.config.fuzzy_weight,
                'semantic_weight': self.config.semantic_weight,
                'min_similarity_threshold': self.config.min_similarity_threshold,
                'max_candidates_per_engine': self.config.max_candidates_per_engine
            }
        }
        
        # Статистика по каждому движку
        if self.is_fitted:
            stats['fuzzy_engine'] = self.fuzzy_engine.get_statistics()
            stats['semantic_engine'] = self.semantic_engine.get_statistics()
        
        return stats
