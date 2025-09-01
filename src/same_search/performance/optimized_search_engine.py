"""
Оптимизированный движок поиска с улучшенной производительностью
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import pickle
import hashlib
from collections import defaultdict
import time
import gc

from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import scipy.sparse as sp

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Конфигурация производительности"""
    max_workers: int = mp.cpu_count()
    chunk_size: int = 1000
    cache_size: int = 10000
    memory_limit_mb: int = 2048
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    enable_batch_processing: bool = True
    similarity_cache_threshold: float = 0.1

@dataclass
class CacheEntry:
    """Запись кэша"""
    hash_key: str
    result: Any
    timestamp: float
    access_count: int = 0

class OptimizedSimilarityCache:
    """Оптимизированный кэш для вычислений схожести"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, text1: str, text2: str, method: str) -> str:
        """Генерация ключа кэша"""
        # Сортировка для обеспечения симметричности
        if text1 > text2:
            text1, text2 = text2, text1
        
        content = f"{text1}|{text2}|{method}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text1: str, text2: str, method: str) -> Optional[float]:
        """Получение из кэша"""
        key = self._generate_key(text1, text2, method)
        
        if key in self.cache:
            self.cache[key].access_count += 1
            self.hit_count += 1
            return self.cache[key].result
        
        self.miss_count += 1
        return None
    
    def put(self, text1: str, text2: str, method: str, result: float):
        """Сохранение в кэш"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        key = self._generate_key(text1, text2, method)
        self.cache[key] = CacheEntry(
            hash_key=key,
            result=result,
            timestamp=time.time()
        )
        self.access_order.append(key)
    
    def _evict_lru(self):
        """Удаление наименее используемых записей"""
        if not self.access_order:
            return
        
        # Удаляем 10% самых старых записей
        evict_count = max(1, len(self.cache) // 10)
        
        for _ in range(evict_count):
            if self.access_order:
                key = self.access_order.pop(0)
                if key in self.cache:
                    del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

class OptimizedTextProcessor:
    """Оптимизированный обработчик текста"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = OptimizedSimilarityCache(config.cache_size)
        self.precomputed_vectors = {}
        self.vectorizer = None
        
    def precompute_vectors(self, texts: List[str]) -> np.ndarray:
        """Предварительное вычисление векторов"""
        logger.info("Precomputing TF-IDF vectors...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Уменьшено для производительности
            ngram_range=(1, 2),  # Уменьшено для производительности
            stop_words=None,
            lowercase=True,
            min_df=2,  # Игнорируем редкие термины
            max_df=0.95  # Игнорируем слишком частые термины
        )
        
        vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Precomputed vectors shape: {vectors.shape}")
        
        return vectors
    
    def batch_similarity(self, vectors: np.ndarray, 
                        batch_size: int = 1000) -> np.ndarray:
        """Пакетное вычисление схожести"""
        n = vectors.shape[0]
        similarity_matrix = np.zeros((n, n))
        
        logger.info(f"Computing batch similarity for {n} items...")
        
        for i in tqdm(range(0, n, batch_size), desc="Computing similarity batches"):
            end_i = min(i + batch_size, n)
            batch_i = vectors[i:end_i]
            
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                batch_j = vectors[j:end_j]
                
                # Вычисление схожести для батча
                batch_similarity = cosine_similarity(batch_i, batch_j)
                similarity_matrix[i:end_i, j:end_j] = batch_similarity
        
        return similarity_matrix
    
    def optimized_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Оптимизированное нечеткое сравнение с кэшированием"""
        # Проверка кэша
        cached_result = self.cache.get(text1, text2, "fuzzy")
        if cached_result is not None:
            return cached_result
        
        # Быстрая проверка на точное совпадение
        if text1 == text2:
            self.cache.put(text1, text2, "fuzzy", 1.0)
            return 1.0
        
        # Быстрая проверка на полное несовпадение
        if not text1 or not text2:
            self.cache.put(text1, text2, "fuzzy", 0.0)
            return 0.0
        
        # Оптимизированное вычисление
        # Используем только ratio для скорости
        similarity = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        
        # Кэширование результата
        self.cache.put(text1, text2, "fuzzy", similarity)
        
        return similarity

class OptimizedDuplicateDetector:
    """Оптимизированный детектор дубликатов"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.text_processor = OptimizedTextProcessor(config)
        self.similarity_cache = {}
        
    def detect_duplicates_optimized(self, df: pd.DataFrame, 
                                  name_column: str = 'processed_name') -> List[Dict[str, Any]]:
        """Оптимизированный поиск дубликатов"""
        logger.info("Starting optimized duplicate detection...")
        
        # Предварительная фильтрация
        names = df[name_column].fillna('').astype(str).tolist()
        filtered_indices = self._prefilter_duplicates(names)
        
        if not filtered_indices:
            logger.info("No potential duplicates found after prefiltering")
            return []
        
        # Предварительное вычисление векторов
        filtered_names = [names[i] for i in filtered_indices]
        vectors = self.text_processor.precompute_vectors(filtered_names)
        
        # Пакетное вычисление схожести
        similarity_matrix = self.text_processor.batch_similarity(vectors)
        
        # Поиск дубликатов
        duplicates = self._find_duplicates_from_matrix(
            filtered_indices, similarity_matrix, names
        )
        
        logger.info(f"Found {len(duplicates)} duplicate groups")
        return duplicates
    
    def _prefilter_duplicates(self, names: List[str]) -> List[int]:
        """Предварительная фильтрация потенциальных дубликатов"""
        logger.info("Prefiltering potential duplicates...")
        
        # Группировка по длине и первым символам
        length_groups = defaultdict(list)
        for i, name in enumerate(names):
            if len(name) > 3:  # Игнорируем слишком короткие названия
                key = (len(name), name[:3].lower())
                length_groups[key].append(i)
        
        # Отбор групп с несколькими элементами
        candidate_indices = []
        for group in length_groups.values():
            if len(group) > 1:
                candidate_indices.extend(group)
        
        logger.info(f"Prefiltered {len(candidate_indices)} candidates from {len(names)} items")
        return candidate_indices
    
    def _find_duplicates_from_matrix(self, indices: List[int], 
                                   similarity_matrix: np.ndarray,
                                   names: List[str]) -> List[Dict[str, Any]]:
        """Поиск дубликатов из матрицы схожести"""
        duplicates = []
        processed = set()
        
        for i, idx1 in enumerate(indices):
            if idx1 in processed:
                continue
            
            name1 = names[idx1]
            duplicate_group = [idx1]
            
            for j, idx2 in enumerate(indices):
                if i != j and idx2 not in processed:
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= 0.85:  # Порог дубликата
                        duplicate_group.append(idx2)
                        processed.add(idx2)
            
            if len(duplicate_group) > 1:
                duplicates.append({
                    'main_index': idx1,
                    'main_name': name1,
                    'duplicate_indices': [idx for idx in duplicate_group if idx != idx1],
                    'similarity_scores': [similarity_matrix[i, indices.index(idx)] 
                                        for idx in duplicate_group if idx != idx1]
                })
                processed.add(idx1)
        
        return duplicates

class OptimizedAnalogFinder:
    """Оптимизированный поисковик аналогов"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.text_processor = OptimizedTextProcessor(config)
        
    def find_analogs_optimized(self, df: pd.DataFrame,
                             name_column: str = 'processed_name',
                             category_column: str = 'category') -> List[Dict[str, Any]]:
        """Оптимизированный поиск аналогов"""
        logger.info("Starting optimized analog detection...")
        
        names = df[name_column].fillna('').astype(str).tolist()
        categories = df[category_column].fillna('').astype(str).tolist()
        
        # Предварительное вычисление векторов
        vectors = self.text_processor.precompute_vectors(names)
        
        # Пакетное вычисление схожести
        similarity_matrix = self.text_processor.batch_similarity(vectors)
        
        # Поиск аналогов
        analogs = self._find_analogs_from_matrix(
            similarity_matrix, names, categories
        )
        
        logger.info(f"Found {len(analogs)} analog groups")
        return analogs
    
    def _find_analogs_from_matrix(self, similarity_matrix: np.ndarray,
                                names: List[str], categories: List[str]) -> List[Dict[str, Any]]:
        """Поиск аналогов из матрицы схожести"""
        analogs = []
        n = len(names)
        
        for i in tqdm(range(n), desc="Finding analogs"):
            name1 = names[i]
            category1 = categories[i]
            
            # Поиск аналогов для текущего элемента
            item_analogs = []
            
            for j in range(n):
                if i != j:
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= 0.3:  # Минимальный порог
                        analog_type = self._determine_analog_type(similarity)
                        
                        item_analogs.append({
                            'index': j,
                            'name': names[j],
                            'similarity': similarity,
                            'type': analog_type,
                            'category': categories[j]
                        })
            
            if item_analogs:
                # Сортировка и ограничение количества
                item_analogs.sort(key=lambda x: x['similarity'], reverse=True)
                item_analogs = item_analogs[:25]  # Максимум 25 аналогов
                
                analogs.append({
                    'reference_index': i,
                    'reference_name': name1,
                    'analogs': item_analogs
                })
        
        return analogs
    
    def _determine_analog_type(self, similarity: float) -> str:
        """Определение типа аналога"""
        if similarity >= 0.8:
            return "точный аналог"
        elif similarity >= 0.65:
            return "близкий аналог"
        elif similarity >= 0.45:
            return "возможный аналог"
        else:
            return "нет аналогов"

class OptimizedSearchEngine:
    """Оптимизированный движок поиска"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.duplicate_detector = OptimizedDuplicateDetector(self.config)
        self.analog_finder = OptimizedAnalogFinder(self.config)
        
        logger.info("OptimizedSearchEngine initialized")
    
    def process_catalog_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизированная обработка каталога"""
        logger.info("Starting optimized catalog processing...")
        
        start_time = time.time()
        
        # Очистка памяти
        if self.config.enable_memory_optimization:
            gc.collect()
        
        # Поиск дубликатов
        logger.info("Phase 1: Duplicate detection")
        duplicate_start = time.time()
        duplicates = self.duplicate_detector.detect_duplicates_optimized(df)
        duplicate_time = time.time() - duplicate_start
        logger.info(f"Duplicate detection completed in {duplicate_time:.2f}s")
        
        # Очистка памяти между этапами
        if self.config.enable_memory_optimization:
            gc.collect()
        
        # Поиск аналогов
        logger.info("Phase 2: Analog detection")
        analog_start = time.time()
        analogs = self.analog_finder.find_analogs_optimized(df)
        analog_time = time.time() - analog_start
        logger.info(f"Analog detection completed in {analog_time:.2f}s")
        
        # Статистика производительности
        total_time = time.time() - start_time
        
        # Статистика кэша
        cache_stats = self.duplicate_detector.text_processor.cache.get_stats()
        
        results = {
            'duplicates': duplicates,
            'analogs': analogs,
            'performance_stats': {
                'total_time': total_time,
                'duplicate_time': duplicate_time,
                'analog_time': analog_time,
                'cache_stats': cache_stats,
                'memory_usage': self._get_memory_usage()
            }
        }
        
        logger.info(f"Optimized processing completed in {total_time:.2f}s")
        logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        return results
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Получение информации об использовании памяти"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def optimize_for_large_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация для больших датасетов"""
        logger.info("Optimizing for large dataset...")
        
        # Разбиение на чанки
        chunk_size = self.config.chunk_size
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {len(df)} records in {total_chunks} chunks of {chunk_size}")
        
        all_duplicates = []
        all_analogs = []
        
        # Обработка по чанкам
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df))
            
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunk_df.index = range(start_idx, end_idx)  # Сохраняем оригинальные индексы
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_df)} records)")
            
            # Обработка чанка
            chunk_results = self.process_catalog_optimized(chunk_df)
            
            # Корректировка индексов
            for dup in chunk_results['duplicates']:
                dup['main_index'] += start_idx
                dup['duplicate_indices'] = [idx + start_idx for idx in dup['duplicate_indices']]
            
            for analog in chunk_results['analogs']:
                analog['reference_index'] += start_idx
                for item in analog['analogs']:
                    item['index'] += start_idx
            
            all_duplicates.extend(chunk_results['duplicates'])
            all_analogs.extend(chunk_results['analogs'])
            
            # Очистка памяти
            if self.config.enable_memory_optimization:
                del chunk_df
                del chunk_results
                gc.collect()
        
        return {
            'duplicates': all_duplicates,
            'analogs': all_analogs,
            'total_chunks': total_chunks,
            'chunk_size': chunk_size
        }
