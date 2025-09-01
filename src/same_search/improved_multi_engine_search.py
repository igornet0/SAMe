"""
Улучшенный мульти-движковый поиск с фильтрацией по категориям и токенам
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from tqdm import tqdm
import re

# Импорты поисковых движков
from .search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from .search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
from .categorization.category_classifier import CategoryClassifier, CategoryClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class ImprovedMultiEngineConfig:
    """Улучшенная конфигурация мульти-движкового поиска"""
    # Пороги схожести
    min_similarity_threshold: float = 0.4  # Увеличили общий порог
    fuzzy_threshold: float = 40  # Снизили порог для fuzzy
    semantic_threshold: float = 0.7  # Увеличили порог для семантики
    
    # Веса для комбинирования результатов
    fuzzy_weight: float = 0.6  # Увеличили вес fuzzy
    semantic_weight: float = 0.4  # Уменьшили вес семантики
    
    # Фильтрация по категориям
    enable_category_filtering: bool = True
    category_penalty: float = 0.3  # Штраф за разные категории
    
    # Фильтрация по токенам
    enable_token_filtering: bool = True
    min_token_intersection: int = 1  # Минимальное пересечение токенов
    token_bonus: float = 0.2  # Бонус за пересечение токенов
    
    # Фильтрация по брендам
    enable_brand_filtering: bool = True
    same_brand_bonus: float = 0.15  # Бонус за один бренд
    different_brand_penalty: float = 0.1  # Штраф за разные бренды
    
    # Параметры результатов
    max_candidates_per_engine: int = 30
    final_top_k: int = 20
    
    # Параллельная обработка
    enable_parallel_search: bool = True
    max_workers: int = 4


class ImprovedMultiEngineSearch:
    """Улучшенный мульти-движковый поисковый класс с фильтрацией"""
    
    def __init__(self, config: ImprovedMultiEngineConfig = None):
        self.config = config or ImprovedMultiEngineConfig()
        
        # Инициализация движков с улучшенными настройками
        self.fuzzy_engine = FuzzySearchEngine(FuzzySearchConfig(
            similarity_threshold=self.config.fuzzy_threshold / 100.0,
            fuzzy_threshold=self.config.fuzzy_threshold,
            top_k_results=self.config.max_candidates_per_engine,
            cosine_weight=0.3,  # Снизили вес cosine
            fuzzy_weight=0.4,   # Увеличили вес fuzzy
            levenshtein_weight=0.3
        ))
        
        self.semantic_engine = SemanticSearchEngine(SemanticSearchConfig(
            similarity_threshold=self.config.semantic_threshold,
            top_k_results=self.config.max_candidates_per_engine,
            enable_category_filtering=True,
            category_similarity_threshold=0.6,
            enable_enhanced_scoring=True,
            semantic_weight=0.5,
            lexical_weight=0.3,
            key_term_weight=0.2
        ))
        
        self.category_classifier = CategoryClassifier(CategoryClassifierConfig())
        self.is_fitted = False
        
        logger.info("ImprovedMultiEngineSearch initialized with enhanced filtering")

    async def process_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка каталога с использованием улучшенного мульти-движкового поиска
        """
        logger.info(f"Processing catalog with improved multi-engine search on {len(df)} items...")
        
        documents = df['processed_name'].tolist()
        document_ids = df.index.tolist()
        metadata = df[['category', 'model_brand', 'extracted_parameters']].to_dict('records')

        logger.info(f"Training improved multi-engine search on {len(documents)} items...")
        await self._train_engines(documents, document_ids, metadata)
        logger.info("Improved multi-engine search trained successfully")

        all_results = []
        with tqdm(total=len(df), desc="Improved multi-engine search") as pbar:
            if self.config.enable_parallel_search:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {executor.submit(self._search_single_item, idx, row['processed_name'], row['category'], row['model_brand'], row['extracted_parameters']): (idx, row) for idx, row in df.iterrows()}
                    for future in as_completed(futures):
                        try:
                            results = future.result()
                            all_results.extend(results)
                        except Exception as e:
                            logger.error(f"Error in parallel search: {e}")
                        pbar.update(1)
            else:
                for idx, row in df.iterrows():
                    results = await self._search_single_item_async(idx, row['processed_name'], row['category'], row['model_brand'], row['extracted_parameters'])
                    all_results.extend(results)
                    pbar.update(1)
        
        logger.info(f"Improved multi-engine processing completed. Found {len(all_results)} analog relationships")
        
        if not all_results:
            return pd.DataFrame(columns=['query_index', 'candidate_idx', 'multi_engine_score', 'Relation_Type'])

        results_df = pd.DataFrame(all_results)
        return results_df

    async def _train_engines(self, documents: List[str], document_ids: List[int], metadata: List[Dict]):
        """Обучение движков"""
        loop = asyncio.get_event_loop()
        
        # Обучение FuzzyEngine
        try:
            await loop.run_in_executor(None, self.fuzzy_engine.fit, documents, document_ids)
            logger.info("Fuzzy engine trained successfully")
        except Exception as e:
            logger.error(f"Error training fuzzy engine: {e}")

        # Обучение SemanticEngine
        try:
            await loop.run_in_executor(None, self.semantic_engine.fit, documents, document_ids, metadata)
            logger.info("Semantic engine trained successfully")
        except Exception as e:
            logger.error(f"Error training semantic engine: {e}")
        
        self.is_fitted = True
        logger.info("Both engines trained successfully")

    async def _search_single_item_async(self, query_idx: int, query_text: str, category: Optional[str], brand: Optional[str], parameters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Асинхронный поиск аналогов для одного элемента"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_single_item, query_idx, query_text, category, brand, parameters)

    def _search_single_item(self, query_idx: int, query_text: str, category: Optional[str], brand: Optional[str], parameters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Поиск аналогов для одного элемента с улучшенной фильтрацией"""
        all_candidates: Dict[int, Dict[str, Any]] = {}
        
        # Поиск с FuzzyEngine
        try:
            fuzzy_candidates = self.fuzzy_engine.search(query_text, self.config.max_candidates_per_engine)
            for candidate in fuzzy_candidates:
                doc_id = candidate.get('document_id', candidate.get('id', None))
                score = candidate.get('similarity', candidate.get('combined_score', 0.0))
                if doc_id is None or doc_id == query_idx:
                    continue
                    
                # Фильтрация по fuzzy порогу
                if score >= self.config.fuzzy_threshold / 100.0:
                    all_candidates[doc_id] = {
                        'query_index': query_idx,
                        'candidate_idx': doc_id,
                        'scores': {'fuzzy': score},
                        'multi_engine_score': 0.0,
                        'Relation_Type': 'возможный аналог',
                        'fuzzy_details': candidate
                    }
        except Exception as e:
            logger.error(f"Error searching with fuzzy engine: {e}")

        # Поиск с SemanticEngine
        try:
            semantic_candidates = self.semantic_engine.search(query_text, self.config.max_candidates_per_engine, category)
            for candidate in semantic_candidates:
                doc_id = candidate.get('document_id', candidate.get('id', None))
                score = candidate.get('similarity', candidate.get('similarity_score', 0.0))
                if doc_id is None or doc_id == query_idx:
                    continue
                    
                # Фильтрация по semantic порогу
                if score >= self.config.semantic_threshold:
                    if doc_id not in all_candidates:
                        all_candidates[doc_id] = {
                            'query_index': query_idx,
                            'candidate_idx': doc_id,
                            'scores': {'semantic': score},
                            'multi_engine_score': 0.0,
                            'Relation_Type': 'возможный аналог',
                            'semantic_details': candidate
                        }
                    else:
                        all_candidates[doc_id]['scores']['semantic'] = score
                        all_candidates[doc_id]['semantic_details'] = candidate
        except Exception as e:
            logger.error(f"Error searching with semantic engine: {e}")
        
        # Применяем улучшенную фильтрацию и скоринг
        final_results = []
        for doc_id, data in all_candidates.items():
            score = self._calculate_improved_score(
                query_idx, doc_id, query_text, data, category, brand, parameters
            )
            
            data['multi_engine_score'] = score
            if score >= self.config.min_similarity_threshold:
                final_results.append(data)
        
        # Сортировка и выбор top_k
        final_results.sort(key=lambda x: x['multi_engine_score'], reverse=True)
        return final_results[:self.config.final_top_k]

    def _calculate_improved_score(self, query_idx: int, candidate_idx: int, query_text: str, 
                                data: Dict, query_category: Optional[str], query_brand: Optional[str], 
                                query_parameters: Optional[Dict]) -> float:
        """Расчет улучшенного скора с множественными фильтрами"""
        
        # Базовый комбинированный скор
        combined_score = 0.0
        fuzzy_score = data['scores'].get('fuzzy', 0.0)
        semantic_score = data['scores'].get('semantic', 0.0)
        
        if fuzzy_score > 0:
            combined_score += fuzzy_score * self.config.fuzzy_weight
        if semantic_score > 0:
            combined_score += semantic_score * self.config.semantic_weight
        
        # Получаем информацию о кандидате
        candidate_details = data.get('semantic_details') or data.get('fuzzy_details', {})
        candidate_category = candidate_details.get('category')
        candidate_brand = self._extract_brand(candidate_details)
        candidate_text = candidate_details.get('document', '')
        
        # 1. Фильтрация по категориям
        if self.config.enable_category_filtering and query_category and candidate_category:
            if query_category != candidate_category:
                combined_score *= (1.0 - self.config.category_penalty)
                logger.debug(f"Category penalty applied: {query_category} != {candidate_category}")
        
        # 2. Фильтрация по токенам
        if self.config.enable_token_filtering:
            query_tokens = self._tokenize(query_text)
            candidate_tokens = self._tokenize(candidate_text)
            intersection = query_tokens.intersection(candidate_tokens)
            
            if len(intersection) < self.config.min_token_intersection:
                combined_score *= 0.5  # Сильный штраф за отсутствие общих токенов
                logger.debug(f"Token penalty applied: intersection={intersection}")
            else:
                # Бонус за пересечение токенов
                token_bonus = min(0.3, len(intersection) * 0.05)
                combined_score += token_bonus
                logger.debug(f"Token bonus applied: intersection={intersection}, bonus={token_bonus}")
        
        # 3. Фильтрация по брендам
        if self.config.enable_brand_filtering and query_brand and candidate_brand:
            if query_brand.lower() == candidate_brand.lower():
                combined_score += self.config.same_brand_bonus
                logger.debug(f"Same brand bonus: {query_brand}")
            else:
                combined_score *= (1.0 - self.config.different_brand_penalty)
                logger.debug(f"Different brand penalty: {query_brand} != {candidate_brand}")
        
        # 4. Бонус за консенсус между движками
        if fuzzy_score > 0 and semantic_score > 0:
            consensus_bonus = 0.1 * min(fuzzy_score, semantic_score)
            combined_score += consensus_bonus
            logger.debug(f"Consensus bonus applied: {consensus_bonus}")
        
        return min(1.0, combined_score)

    def _tokenize(self, text: str) -> Set[str]:
        """Токенизация текста"""
        if not text:
            return set()
        # Простая токенизация с фильтрацией коротких токенов
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {token for token in tokens if len(token) >= 3}

    def _extract_brand(self, details: Dict) -> Optional[str]:
        """Извлечение бренда из деталей кандидата"""
        brand = details.get('brand')
        if brand:
            return brand
        
        # Попытка извлечь из метаданных
        metadata = details.get('metadata', {})
        if isinstance(metadata, dict):
            return metadata.get('model_brand')
        
        return None
