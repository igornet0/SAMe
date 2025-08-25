#!/usr/bin/env python3
"""
Оптимизированная версия Hybrid DBSCAN Search Engine
Решает проблемы с памятью и производительностью для больших датасетов
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import asyncio
from pathlib import Path
import gc
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class OptimizedDBSCANConfig:
    """Оптимизированная конфигурация для больших датасетов"""
    # DBSCAN параметры
    eps: float = 0.4  # Увеличен для уменьшения количества кластеров
    min_samples: int = 3  # Увеличен для более строгой кластеризации
    
    # Векторизация (уменьшены для экономии памяти)
    max_features: int = 5000  # Уменьшено с 10000
    ngram_range: Tuple[int, int] = (1, 2)  # Уменьшено с (1, 3)
    
    # SVD для уменьшения размерности
    use_svd: bool = True
    svd_components: int = 100  # Уменьшение размерности до 100 компонент
    
    # Пороги схожести
    similarity_threshold: float = 0.6
    cluster_assignment_threshold: float = 0.4
    
    # Веса для гибридного скоринга
    tfidf_weight: float = 0.7
    normalized_weight: float = 0.3
    
    # Параметры обработки (оптимизированы)
    batch_size: int = 500  # Уменьшено для экономии памяти
    max_candidates: int = 5  # Уменьшено с 10
    
    # Ограничения для больших датасетов
    max_records_for_full_dbscan: int = 10000  # Лимит для полного DBSCAN
    use_sampling: bool = True  # Использовать сэмплирование для больших датасетов
    sample_size: int = 5000  # Размер выборки
    
    # Память
    memory_limit_gb: float = 4.0  # Лимит памяти в GB
    enable_memory_monitoring: bool = True


class OptimizedHybridDBSCANSearchEngine:
    """Оптимизированный Hybrid DBSCAN поисковый движок"""
    
    def __init__(self, config: OptimizedDBSCANConfig = None):
        self.config = config or OptimizedDBSCANConfig()
        
        # Векторизаторы (оптимизированные)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words=None,
            lowercase=True,
            token_pattern=r'\b\w+\b',
            max_df=0.95,  # Игнорируем слишком частые слова
            min_df=2      # Игнорируем слишком редкие слова
        )
        
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=self.config.svd_components) if self.config.use_svd else None
        
        # Модели и данные
        self.dbscan = None
        self.catalog_df = None
        self.processed_texts = None
        self.tfidf_matrix = None
        self.reduced_vectors = None
        self.clusters = None
        self.cluster_centers = None
        self.noise_points = None
        self.sample_indices = None  # Индексы выборки
        
        # Флаг готовности
        self.is_fitted = False
        
        logger.info("OptimizedHybridDBSCANSearchEngine initialized")

    def set_catalog(self, catalog_df: pd.DataFrame):
        """Простая установка каталога без обучения"""
        self.catalog_df = catalog_df.copy()
        logger.info(f"Catalog set with {len(catalog_df)} records")
    
    def _check_memory_usage(self) -> float:
        """Проверка использования памяти"""
        if not self.config.enable_memory_monitoring:
            return 0.0
        
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)
        
        if memory_gb > self.config.memory_limit_gb:
            logger.warning(f"Memory usage high: {memory_gb:.1f}GB > {self.config.memory_limit_gb}GB")
            gc.collect()  # Принудительная сборка мусора
            
        return memory_gb
    
    def normalize_text(self, text: str) -> str:
        """Быстрая нормализация текста"""
        if not text or not isinstance(text, str):
            return ""
        
        import re
        
        # Упрощенная нормализация для экономии времени
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = " ".join(filter(lambda x: len(x) > 2, text.split()))
        
        return text
    
    def _create_sample(self, catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Создание репрезентативной выборки для больших датасетов"""
        if len(catalog_df) <= self.config.max_records_for_full_dbscan:
            return catalog_df, np.arange(len(catalog_df))
        
        logger.info(f"Dataset too large ({len(catalog_df)} records). Creating sample of {self.config.sample_size}")
        
        # Стратифицированная выборка по группам если возможно
        if 'Группа' in catalog_df.columns:
            sample_df = catalog_df.groupby('Группа').apply(
                lambda x: x.sample(min(len(x), max(1, self.config.sample_size // catalog_df['Группа'].nunique())))
            ).reset_index(drop=True)
        else:
            # Случайная выборка
            sample_df = catalog_df.sample(n=min(self.config.sample_size, len(catalog_df))).reset_index(drop=True)
        
        # Сохраняем индексы выборки для последующего использования
        sample_indices = sample_df.index.values
        
        logger.info(f"Created sample with {len(sample_df)} records")
        return sample_df, sample_indices
    
    async def fit(self, catalog_df: pd.DataFrame, text_column: str = 'Normalized_Name') -> None:
        """Обучение модели с оптимизациями для больших датасетов"""
        logger.info(f"Starting optimized DBSCAN clustering on {len(catalog_df)} items...")
        
        # Проверяем память
        self._check_memory_usage()
        
        # Создаем выборку если датасет слишком большой
        if self.config.use_sampling:
            sample_df, self.sample_indices = self._create_sample(catalog_df)
        else:
            sample_df = catalog_df
            self.sample_indices = np.arange(len(catalog_df))
        
        self.catalog_df = catalog_df.copy()
        
        # Проверяем наличие колонки
        if text_column not in sample_df.columns:
            logger.warning(f"Column '{text_column}' not found, using 'Raw_Name' instead")
            text_column = 'Raw_Name'
        
        # Извлекаем и обрабатываем тексты
        texts = sample_df[text_column].fillna('').astype(str).tolist()
        self.processed_texts = [self.normalize_text(text) for text in texts]
        
        # Создаем TF-IDF векторы
        logger.info("Creating optimized TF-IDF vectors...")
        self._check_memory_usage()
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Применяем SVD для уменьшения размерности
        if self.config.use_svd:
            logger.info(f"Applying SVD reduction to {self.config.svd_components} components...")
            self.reduced_vectors = self.svd.fit_transform(self.tfidf_matrix)
            vectors_for_clustering = self.reduced_vectors
            logger.info(f"Reduced vectors shape: {self.reduced_vectors.shape}")
        else:
            vectors_for_clustering = self.tfidf_matrix.toarray()
        
        self._check_memory_usage()
        
        # Выполняем кластеризацию DBSCAN
        logger.info(f"Performing optimized DBSCAN clustering...")
        logger.info(f"Parameters: eps={self.config.eps}, min_samples={self.config.min_samples}")
        
        self.dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric='cosine',
            n_jobs=1  # Ограничиваем количество процессов для экономии памяти
        )
        
        cluster_labels = self.dbscan.fit_predict(vectors_for_clustering)
        
        # Анализируем результаты кластеризации
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        
        # Сохраняем результаты кластеризации
        self.clusters = {}
        self.cluster_centers = {}
        self.noise_points = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Шумовые точки
                self.noise_points.append(i)
            else:
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(i)
        
        # Вычисляем центры кластеров
        for label, indices in self.clusters.items():
            cluster_vectors = vectors_for_clustering[indices]
            center = np.mean(cluster_vectors, axis=0)
            self.cluster_centers[label] = center
        
        # Добавляем информацию о кластерах в DataFrame (только для выборки)
        sample_df['cluster_label'] = cluster_labels
        sample_df['is_noise'] = cluster_labels == -1
        
        # Для полного датасета назначаем кластеры на основе выборки
        if self.config.use_sampling and len(catalog_df) > len(sample_df):
            logger.info("Assigning clusters to full dataset based on sample...")
            self._assign_clusters_to_full_dataset(catalog_df, sample_df, text_column)
        
        self.is_fitted = True
        self._check_memory_usage()
        logger.info("Optimized DBSCAN clustering completed successfully")
    
    def _assign_clusters_to_full_dataset(self, full_df: pd.DataFrame, sample_df: pd.DataFrame, text_column: str):
        """Назначение кластеров полному датасету на основе выборки"""
        # Инициализируем колонки
        full_df['cluster_label'] = -1  # По умолчанию шум
        full_df['is_noise'] = True
        
        # Копируем результаты для выборки
        for idx, sample_idx in enumerate(self.sample_indices):
            if sample_idx < len(full_df):
                full_df.iloc[sample_idx]['cluster_label'] = sample_df.iloc[idx]['cluster_label']
                full_df.iloc[sample_idx]['is_noise'] = sample_df.iloc[idx]['is_noise']
        
        logger.info("Cluster assignment to full dataset completed")
    
    async def search_analogs_optimized(self, query_idx: int, max_results: int = None) -> List[Dict[str, Any]]:
        """Оптимизированный поиск аналогов"""
        # Инициализируем каталог если не инициализирован
        if self.catalog_df is None:
            logger.warning("Catalog not initialized, cannot perform search")
            return []
        
        max_results = max_results or self.config.max_candidates
        
        if query_idx >= len(self.catalog_df):
            raise ValueError(f"Query index {query_idx} is out of range")
        
        query_row = self.catalog_df.iloc[query_idx]
        
        # Простой поиск по текстовому сходству для оптимизации
        query_text = self.normalize_text(str(query_row.get('Normalized_Name', query_row.get('Raw_Name', ''))))
        
        candidates = []
        
        # Быстрый поиск по первым N записям
        search_limit = min(1000, len(self.catalog_df))  # Ограничиваем поиск
        
        for i in range(min(search_limit, len(self.catalog_df))):
            if i == query_idx:
                continue
                
            candidate_row = self.catalog_df.iloc[i]
            candidate_text = self.normalize_text(str(candidate_row.get('Normalized_Name', candidate_row.get('Raw_Name', ''))))
            
            # Простая текстовая схожесть
            similarity = self._simple_text_similarity(query_text, candidate_text)
            
            if similarity >= self.config.similarity_threshold:
                candidates.append({
                    'candidate_idx': i,
                    'similarity': similarity,
                    'same_cluster': False,
                    'cluster_label': -1
                })
        
        # Сортируем и ограничиваем результаты
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        candidates = candidates[:max_results]
        
        # Формируем результаты
        results = []
        for candidate in candidates:
            candidate_row = self.catalog_df.iloc[candidate['candidate_idx']]
            
            relation_type = self._determine_relation_type_simple(candidate['similarity'])
            
            result = {
                'Код': query_row.get('Код', ''),
                'Raw_Name': query_row.get('Raw_Name', ''),
                'Candidate_Name': candidate_row.get('Raw_Name', ''),
                'Similarity_Score': round(candidate['similarity'], 4),
                'Relation_Type': relation_type,
                'Suggested_Category': candidate_row.get('Группа', 'Неопределено'),
                'Final_Decision': '',
                'Comment': f"Текстовая схожесть: {candidate['similarity']:.3f}",
                'Original_Category': query_row.get('Группа', ''),
                'Candidate_Код': candidate_row.get('Код', ''),
                'Original_Code': query_row.get('Код', ''),
                'Search_Engine': 'OptimizedHybridDBSCAN'
            }
            
            results.append(result)
        
        return results
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Быстрое вычисление текстовой схожести"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_relation_type_simple(self, similarity: float) -> str:
        """Упрощенное определение типа отношения"""
        if similarity >= 0.9:
            return "дубль"
        elif similarity >= 0.7:
            return "аналог"
        elif similarity >= 0.6:
            return "близкий аналог"
        elif similarity >= 0.4:
            return "возможный аналог"
        else:
            return "похожий товар"
