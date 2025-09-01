"""
Мульти-движковый поисковый модуль для максимального покрытия
Комбинирует результаты нескольких поисковых алгоритмов
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from tqdm import tqdm

# Импорты поисковых движков
from .search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from .search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
from .search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
from .hybrid_dbscan_search import HybridDBSCANSearchEngine, HybridDBSCANConfig
from .optimized_dbscan_search import OptimizedHybridDBSCANSearchEngine, OptimizedDBSCANConfig
from .categorization.category_classifier import CategoryClassifier, CategoryClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiEngineConfig:
    """Конфигурация мульти-движкового поиска"""
    # Включение/выключение движков
    enable_fuzzy: bool = True
    enable_semantic: bool = True
    enable_hybrid: bool = True
    enable_dbscan: bool = True
    enable_optimized_dbscan: bool = True
    
    # Веса для каждого движка
    fuzzy_weight: float = 0.2
    semantic_weight: float = 0.3
    hybrid_weight: float = 0.2
    dbscan_weight: float = 0.15
    optimized_dbscan_weight: float = 0.15
    
    # Пороги схожести
    min_similarity_threshold: float = 0.3
    consensus_threshold: float = 0.5  # Порог для консенсуса между движками
    
    # Параметры результатов
    max_candidates_per_engine: int = 20
    final_top_k: int = 50
    
    # Параллельная обработка
    enable_parallel_search: bool = True
    max_workers: int = 3
    
    # Стратегия комбинирования
    combination_strategy: str = "weighted_consensus"  # weighted_consensus, rank_fusion, confidence_based


@dataclass
class SearchResult:
    """Результат поиска от одного движка"""
    engine_name: str
    query_id: int
    candidates: List[Dict[str, Any]]
    confidence: float
    processing_time: float


class MultiEngineSearch:
    """Мульти-движковый поисковый класс"""
    
    def __init__(self, config: MultiEngineConfig = None):
        self.config = config or MultiEngineConfig()
        
        # Инициализация движков
        self.engines = {}
        self._init_engines()
        
        # Классификатор категорий
        self.category_classifier = CategoryClassifier()
        
        # Флаг готовности
        self.is_fitted = False
        
        logger.info("MultiEngineSearch initialized")
    
    def _init_engines(self):
        """Инициализация поисковых движков"""
        if self.config.enable_fuzzy:
            fuzzy_config = FuzzySearchConfig(
                cosine_threshold=0.3,
                fuzzy_threshold=50,
                top_k_results=self.config.max_candidates_per_engine
            )
            self.engines['fuzzy'] = FuzzySearchEngine(fuzzy_config)
        
        if self.config.enable_semantic:
            semantic_config = SemanticSearchConfig(
                similarity_threshold=0.3,
                top_k_results=self.config.max_candidates_per_engine
            )
            self.engines['semantic'] = SemanticSearchEngine(semantic_config)
        
        # Отключаем hybrid engine пока не исправим его
        # if self.config.enable_hybrid:
        #     hybrid_config = HybridSearchConfig(
        #         final_top_k=self.config.max_candidates_per_engine,
        #         similarity_threshold=0.3
        #     )
        #     self.engines['hybrid'] = HybridSearchEngine(hybrid_config)
        
        if self.config.enable_dbscan:
            dbscan_config = HybridDBSCANConfig(
                similarity_threshold=0.3,
                max_candidates=self.config.max_candidates_per_engine
            )
            self.engines['dbscan'] = HybridDBSCANSearchEngine(dbscan_config)
        
        if self.config.enable_optimized_dbscan:
            opt_dbscan_config = OptimizedDBSCANConfig(
                similarity_threshold=0.3,
                max_candidates=self.config.max_candidates_per_engine
            )
            self.engines['optimized_dbscan'] = OptimizedHybridDBSCANSearchEngine(opt_dbscan_config)
        
        logger.info(f"Initialized {len(self.engines)} search engines: {list(self.engines.keys())}")
    
    async def fit(self, catalog_df: pd.DataFrame, text_column: str = 'processed_name'):
        """
        Обучение всех поисковых движков
        
        Args:
            catalog_df: DataFrame с каталогом товаров
            text_column: Название колонки с текстом для анализа
        """
        logger.info(f"Training {len(self.engines)} search engines on {len(catalog_df)} items...")
        
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
        logger.info("All search engines trained successfully")
    
    async def _parallel_fit(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Параллельное обучение движков"""
        tasks = []
        
        for engine_name, engine in self.engines.items():
            if hasattr(engine, 'fit_async'):
                task = engine.fit_async(documents, document_ids, metadata)
            else:
                # Адаптируем вызовы fit() под разные движки
                if engine_name == 'fuzzy':
                    task = asyncio.create_task(self._sync_fit_wrapper(engine, documents, document_ids))
                elif engine_name in ['dbscan', 'optimized_dbscan']:
                    # Для DBSCAN движков создаем DataFrame
                    df = pd.DataFrame({
                        'processed_name': documents,
                        'Код': document_ids,
                        'Raw_Name': documents
                    })
                    task = asyncio.create_task(self._dbscan_fit_wrapper(engine, df))
                else:
                    task = asyncio.create_task(self._sync_fit_wrapper(engine, documents, document_ids, metadata))
            tasks.append((engine_name, task))
        
        # Ждем завершения всех задач
        for engine_name, task in tasks:
            try:
                await task
                logger.info(f"Engine '{engine_name}' trained successfully")
            except Exception as e:
                logger.error(f"Error training engine '{engine_name}': {e}")
    
    async def _sequential_fit(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Последовательное обучение движков"""
        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'fit_async'):
                    await engine.fit_async(documents, document_ids, metadata)
                else:
                    await self._sync_fit_wrapper(engine, documents, document_ids, metadata)
                logger.info(f"Engine '{engine_name}' trained successfully")
            except Exception as e:
                logger.error(f"Error training engine '{engine_name}': {e}")
    
    async def _sync_fit_wrapper(self, engine, documents: List[str], document_ids: List[int], metadata: List[Dict] = None):
        """Обертка для синхронного обучения"""
        loop = asyncio.get_event_loop()
        if metadata is not None:
            await loop.run_in_executor(None, engine.fit, documents, document_ids, metadata)
        else:
            await loop.run_in_executor(None, engine.fit, documents, document_ids)
    
    async def _dbscan_fit_wrapper(self, engine, df: pd.DataFrame):
        """Обертка для обучения DBSCAN движков"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.fit, df)
    
    async def search_analogs(self, query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """
        Поиск аналогов с использованием всех движков
        
        Args:
            query_idx: Индекс запроса
            query_text: Текст запроса
            
        Returns:
            Список найденных аналогов
        """
        if not self.is_fitted:
            raise ValueError("Search engines are not fitted. Call fit() first.")
        
        # Параллельный поиск во всех движках
        if self.config.enable_parallel_search:
            results = await self._parallel_search(query_text)
        else:
            results = await self._sequential_search(query_text)
        
        # Комбинирование результатов
        combined_results = self._combine_results(results, query_idx, query_text)
        
        return combined_results[:self.config.final_top_k]
    
    async def _parallel_search(self, query_text: str) -> List[SearchResult]:
        """Параллельный поиск во всех движках"""
        tasks = []
        
        for engine_name, engine in self.engines.items():
            task = asyncio.create_task(self._search_with_engine(engine_name, engine, query_text))
            tasks.append(task)
        
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel search: {e}")
        
        return results
    
    async def _sequential_search(self, query_text: str) -> List[SearchResult]:
        """Последовательный поиск во всех движках"""
        results = []
        
        for engine_name, engine in self.engines.items():
            try:
                result = await self._search_with_engine(engine_name, engine, query_text)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in sequential search for engine '{engine_name}': {e}")
        
        return results
    
    async def _search_with_engine(self, engine_name: str, engine, query_text: str) -> Optional[SearchResult]:
        """Поиск с одним движком"""
        import time
        start_time = time.time()
        
        try:
            if hasattr(engine, 'search_analogs'):
                # Для DBSCAN движков
                candidates = await engine.search_analogs(0)  # query_idx не используется
            elif hasattr(engine, 'search'):
                # Для обычных поисковых движков
                candidates = engine.search(query_text, self.config.max_candidates_per_engine)
            else:
                logger.warning(f"Engine '{engine_name}' has no search method")
                return None
            
            processing_time = time.time() - start_time
            
            # Вычисляем уверенность на основе количества и качества результатов
            confidence = self._calculate_engine_confidence(candidates)
            
            return SearchResult(
                engine_name=engine_name,
                query_id=0,  # Будет обновлено позже
                candidates=candidates,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error searching with engine '{engine_name}': {e}")
            return None
    
    def _calculate_engine_confidence(self, candidates: List[Dict[str, Any]]) -> float:
        """Вычисление уверенности движка на основе результатов"""
        if not candidates:
            return 0.0
        
        # Базовая уверенность на основе количества результатов
        base_confidence = min(len(candidates) / self.config.max_candidates_per_engine, 1.0)
        
        # Дополнительная уверенность на основе качества результатов
        if candidates:
            # Используем средний скор схожести
            scores = []
            for candidate in candidates:
                score = candidate.get('similarity_score', 
                                   candidate.get('similarity', 
                                               candidate.get('combined_score', 0.0)))
                if isinstance(score, (int, float)):
                    scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                quality_confidence = avg_score
            else:
                quality_confidence = 0.5
        else:
            quality_confidence = 0.0
        
        # Комбинированная уверенность
        final_confidence = 0.6 * base_confidence + 0.4 * quality_confidence
        return min(final_confidence, 1.0)
    
    def _combine_results(self, results: List[SearchResult], query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """Комбинирование результатов от всех движков"""
        if not results:
            return []
        
        if self.config.combination_strategy == "weighted_consensus":
            return self._weighted_consensus_combination(results, query_idx, query_text)
        elif self.config.combination_strategy == "rank_fusion":
            return self._rank_fusion_combination(results, query_idx, query_text)
        elif self.config.combination_strategy == "confidence_based":
            return self._confidence_based_combination(results, query_idx, query_text)
        else:
            return self._weighted_consensus_combination(results, query_idx, query_text)
    
    def _weighted_consensus_combination(self, results: List[SearchResult], query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """Взвешенное консенсусное комбинирование"""
        # Собираем все кандидаты
        all_candidates = {}
        
        for result in results:
            engine_name = result.engine_name
            weight = getattr(self.config, f"{engine_name}_weight", 0.1)
            confidence = result.confidence
            
            for candidate in result.candidates:
                # Получаем уникальный идентификатор кандидата
                candidate_id = self._get_candidate_id(candidate)
                
                if candidate_id not in all_candidates:
                    all_candidates[candidate_id] = {
                        'candidate': candidate,
                        'engines': [],
                        'total_score': 0.0,
                        'engine_count': 0
                    }
                
                # Добавляем информацию о движке
                all_candidates[candidate_id]['engines'].append({
                    'name': engine_name,
                    'weight': weight,
                    'confidence': confidence,
                    'score': candidate.get('similarity_score', 
                                        candidate.get('similarity', 
                                                    candidate.get('combined_score', 0.0)))
                })
                
                # Обновляем общий скор
                engine_score = candidate.get('similarity_score', 
                                           candidate.get('similarity', 
                                                       candidate.get('combined_score', 0.0)))
                if isinstance(engine_score, (int, float)):
                    all_candidates[candidate_id]['total_score'] += engine_score * weight * confidence
                    all_candidates[candidate_id]['engine_count'] += 1
        
        # Формируем финальные результаты
        final_results = []
        for candidate_id, data in all_candidates.items():
            if data['engine_count'] > 0:
                # Нормализуем скор
                normalized_score = data['total_score'] / data['engine_count']
                
                # Бонус за консенсус (наличие в нескольких движках)
                consensus_bonus = 0.1 * (data['engine_count'] - 1)
                final_score = min(normalized_score + consensus_bonus, 1.0)
                
                if final_score >= self.config.min_similarity_threshold:
                    result = data['candidate'].copy()
                    result['multi_engine_score'] = final_score
                    result['engine_consensus'] = data['engine_count']
                    result['engines_used'] = [e['name'] for e in data['engines']]
                    result['search_method'] = 'multi_engine'
                    result['combination_strategy'] = 'weighted_consensus'
                    
                    final_results.append(result)
        
        # Сортируем по финальному скору
        final_results.sort(key=lambda x: x['multi_engine_score'], reverse=True)
        
        return final_results
    
    def _rank_fusion_combination(self, results: List[SearchResult], query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """Комбинирование на основе рангов (Reciprocal Rank Fusion)"""
        # Собираем ранги от всех движков
        candidate_ranks = {}
        
        for result in results:
            engine_name = result.engine_name
            weight = getattr(self.config, f"{engine_name}_weight", 0.1)
            
            for rank, candidate in enumerate(result.candidates):
                candidate_id = self._get_candidate_id(candidate)
                
                if candidate_id not in candidate_ranks:
                    candidate_ranks[candidate_id] = {
                        'candidate': candidate,
                        'rrf_score': 0.0,
                        'engines': []
                    }
                
                # RRF скор
                rrf_score = weight / (60 + rank + 1)  # k=60 для RRF
                candidate_ranks[candidate_id]['rrf_score'] += rrf_score
                candidate_ranks[candidate_id]['engines'].append(engine_name)
        
        # Формируем результаты
        final_results = []
        for candidate_id, data in candidate_ranks.items():
            if data['rrf_score'] > 0:
                result = data['candidate'].copy()
                result['multi_engine_score'] = data['rrf_score']
                result['engine_consensus'] = len(data['engines'])
                result['engines_used'] = data['engines']
                result['search_method'] = 'multi_engine'
                result['combination_strategy'] = 'rank_fusion'
                
                final_results.append(result)
        
        # Сортируем по RRF скору
        final_results.sort(key=lambda x: x['multi_engine_score'], reverse=True)
        
        return final_results
    
    def _confidence_based_combination(self, results: List[SearchResult], query_idx: int, query_text: str) -> List[Dict[str, Any]]:
        """Комбинирование на основе уверенности движков"""
        # Сортируем результаты по уверенности
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Берем результаты от наиболее уверенных движков
        final_results = []
        used_candidates = set()
        
        for result in results:
            if result.confidence < 0.3:  # Пропускаем не уверенные движки
                continue
            
            for candidate in result.candidates:
                candidate_id = self._get_candidate_id(candidate)
                
                if candidate_id not in used_candidates:
                    final_result = candidate.copy()
                    final_result['multi_engine_score'] = candidate.get('similarity_score', 
                                                                     candidate.get('similarity', 
                                                                                 candidate.get('combined_score', 0.0)))
                    final_result['engine_confidence'] = result.confidence
                    final_result['engine_used'] = result.engine_name
                    final_result['search_method'] = 'multi_engine'
                    final_result['combination_strategy'] = 'confidence_based'
                    
                    final_results.append(final_result)
                    used_candidates.add(candidate_id)
        
        # Сортируем по скору
        final_results.sort(key=lambda x: x['multi_engine_score'], reverse=True)
        
        return final_results
    
    def _get_candidate_id(self, candidate: Dict[str, Any]) -> str:
        """Получение уникального идентификатора кандидата"""
        # Пытаемся использовать различные поля для идентификации
        for field in ['Candidate_Код', 'candidate_code', 'document_id', 'index']:
            if field in candidate and candidate[field]:
                return str(candidate[field])
        
        # Если нет уникального ID, используем название
        name = candidate.get('Candidate_Name', candidate.get('document', candidate.get('content', '')))
        return str(name)
    
    async def process_catalog(self, catalog_df: pd.DataFrame, 
                            text_column: str = 'processed_name',
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Обработка всего каталога с мульти-движковым поиском
        
        Args:
            catalog_df: DataFrame с каталогом товаров
            text_column: Название колонки с текстом
            output_path: Путь для сохранения результатов
            
        Returns:
            DataFrame с результатами поиска
        """
        logger.info(f"Processing catalog with multi-engine search on {len(catalog_df)} items...")
        
        # Обучаем все движки
        await self.fit(catalog_df, text_column)
        
        # Обрабатываем каждый товар
        all_results = []
        
        for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Multi-engine search"):
            try:
                query_text = str(row[text_column])
                results = await self.search_analogs(idx, query_text)
                
                # Добавляем информацию о запросе
                for result in results:
                    result['query_index'] = idx
                    result['query_name'] = query_text
                    result['original_code'] = row.get('Код', '')
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue
        
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")
        
        logger.info(f"Multi-engine processing completed. Found {len(all_results)} analog relationships")
        return results_df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики мульти-движкового поиска"""
        stats = {
            'total_engines': len(self.engines),
            'enabled_engines': list(self.engines.keys()),
            'is_fitted': self.is_fitted,
            'config': {
                'combination_strategy': self.config.combination_strategy,
                'weights': {
                    'fuzzy_weight': self.config.fuzzy_weight,
                    'semantic_weight': self.config.semantic_weight,
                    'hybrid_weight': self.config.hybrid_weight,
                    'dbscan_weight': self.config.dbscan_weight,
                    'optimized_dbscan_weight': self.config.optimized_dbscan_weight
                },
                'thresholds': {
                    'min_similarity_threshold': self.config.min_similarity_threshold,
                    'consensus_threshold': self.config.consensus_threshold
                }
            }
        }
        
        # Статистика по каждому движку
        if self.is_fitted:
            stats['engine_stats'] = {}
            for engine_name, engine in self.engines.items():
                if hasattr(engine, 'get_statistics'):
                    stats['engine_stats'][engine_name] = engine.get_statistics()
        
        return stats
