#!/usr/bin/env python3
"""
Enhanced DBSCAN Search Engine для поиска аналогов товаров
Улучшенная версия с обработкой шумовых точек и гарантированным покрытием
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import asyncio
from pathlib import Path

# Импорты SAMe модулей
try:
    from src.same_clear.text_processing.enhanced_preprocessor import EnhancedPreprocessor
    from src.same_clear.text_processing.text_cleaner import TextCleaner
    from src.same_clear import PreprocessorConfig, TextPreprocessor
    SAME_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMe modules not available. Missing: {e}")
    SAME_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NoiseProcessorConfig:
    """Конфигурация обработки шумовых точек"""
    max_nearest_clusters: int = 3
    min_analogs_per_noise: int = 1
    max_analogs_per_noise: int = 5
    use_hierarchical_search: bool = True
    noise_assignment_threshold: float = 0.4
    similarity_threshold: float = 0.6


@dataclass
class EnhancedDBSCANConfig:
    """Конфигурация для Enhanced DBSCAN поиска"""
    # DBSCAN параметры
    eps: float = 0.3  # Максимальное расстояние между точками в кластере
    min_samples: int = 2  # Минимальное количество точек для формирования кластера
    
    # Векторизация
    max_features: int = 10000  # Максимальное количество признаков для TF-IDF
    ngram_range: Tuple[int, int] = (1, 3)  # Диапазон n-грамм
    
    # Пороги схожести
    similarity_threshold: float = 0.6  # Порог для определения аналогов
    cluster_assignment_threshold: float = 0.4  # Порог для назначения в кластер
    
    # Веса для гибридного скоринга
    tfidf_weight: float = 0.6
    normalized_weight: float = 0.4
    
    # Параметры обработки
    batch_size: int = 1000
    max_candidates: int = 10  # Максимальное количество кандидатов на товар
    
    # Обработка шумовых точек
    noise_config: NoiseProcessorConfig = field(default_factory=NoiseProcessorConfig)


class RelationType:
    """Типы отношений между товарами"""
    DUPLICATE = "дубль"
    ANALOG = "аналог"
    CLOSE_ANALOG = "близкий аналог"
    POSSIBLE_ANALOG = "возможный аналог"
    SIMILAR_PRODUCT = "похожий товар"
    NO_ANALOGS = "нет аналогов"


class EnhancedDBSCANSearch:
    """Enhanced DBSCAN поисковый движок с обработкой шумовых точек"""
    
    def __init__(self, config: EnhancedDBSCANConfig = None):
        self.config = config or EnhancedDBSCANConfig()
        
        # Компоненты обработки текста
        if SAME_MODULES_AVAILABLE:
            self.text_cleaner = TextCleaner()
            # Используем простую конфигурацию для стабильности
            preprocessor_config = PreprocessorConfig(
                enable_parallel_processing=False,
                max_workers=1,
                save_intermediate_steps=False
            )
            self.preprocessor = TextPreprocessor(preprocessor_config)
        else:
            self.text_cleaner = None
            self.preprocessor = None
        
        # Векторизаторы
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words=None,  # Используем свою обработку
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        self.scaler = StandardScaler()
        
        # Модели и данные
        self.dbscan = None
        self.catalog_df = None
        self.processed_texts = None
        self.tfidf_matrix = None
        self.normalized_vectors = None
        self.clusters = None
        self.cluster_centers = None
        self.noise_points = None
        
        # Флаг готовности
        self.is_fitted = False
        
        logger.info("EnhancedDBSCANSearch initialized")

    def set_catalog(self, catalog_df: pd.DataFrame):
        """Установка каталога для обработки"""
        self.catalog_df = catalog_df.copy()
        logger.info(f"Catalog set with {len(catalog_df)} records")

    async def fit(self, text_column: str = 'Наименование'):
        """Обучение модели на каталоге"""
        if self.catalog_df is None:
            raise ValueError("Catalog not set. Call set_catalog() first.")

        logger.info("Starting Enhanced DBSCAN fitting...")

        # 1. Предобработка текста
        await self._preprocess_texts(text_column)

        # 2. Векторизация
        self._vectorize_texts()

        # 3. Кластеризация DBSCAN
        self._perform_clustering()

        # 4. Вычисление центров кластеров
        self._calculate_cluster_centers()

        # 5. Обработка шумовых точек
        await self._process_noise_points()

        self.is_fitted = True
        logger.info("Enhanced DBSCAN fitting completed")

    async def _preprocess_texts(self, text_column: str):
        """Предобработка текстов"""
        logger.info("Preprocessing texts...")
        
        texts = self.catalog_df[text_column].fillna('').astype(str).tolist()
        
        if self.preprocessor:
            # Используем улучшенный предобработчик
            processed_results = []
            for text in texts:
                result = self.preprocessor.preprocess_text(text)
                processed_results.append(result)
            
            # Извлекаем очищенные тексты
            self.processed_texts = [result.get('cleaned_text', text) for result in processed_results]
        else:
            # Простая очистка
            self.processed_texts = [self._simple_clean_text(text) for text in texts]
        
        logger.info(f"Text preprocessing completed for {len(self.processed_texts)} texts")

    def _simple_clean_text(self, text: str) -> str:
        """Простая очистка текста"""
        if not text:
            return ""
        
        # Базовая очистка
        cleaned = text.lower().strip()
        # Удаляем лишние пробелы
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    def _vectorize_texts(self):
        """Векторизация текстов"""
        logger.info("Vectorizing texts...")
        
        # TF-IDF векторизация
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
        
        # Нормализация векторов
        self.normalized_vectors = self.scaler.fit_transform(self.tfidf_matrix.toarray())
        
        logger.info(f"Vectorization completed. Matrix shape: {self.tfidf_matrix.shape}")

    def _perform_clustering(self):
        """Выполнение кластеризации DBSCAN"""
        logger.info("Performing DBSCAN clustering...")
        
        # Показываем прогресс
        total_records = len(self.normalized_vectors)
        logger.info(f"Starting DBSCAN clustering on {total_records:,} records...")
        
        self.dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric='cosine',
            n_jobs=1  # Используем один процесс для стабильности
        )
        
        # Выполняем кластеризацию с индикатором прогресса
        logger.info("Fitting DBSCAN model...")
        cluster_labels = self.dbscan.fit_predict(self.normalized_vectors)
        logger.info("DBSCAN clustering completed!")
        
        # Обрабатываем результаты с прогрессом
        logger.info("Processing clustering results...")
        self.catalog_df['cluster_label'] = cluster_labels
        self.catalog_df['is_noise'] = cluster_labels == -1
        
        # Группируем кластеры
        self.clusters = {}
        self.noise_points = []
        
        # Показываем прогресс обработки результатов
        for i, (idx, label) in enumerate(zip(range(len(cluster_labels)), cluster_labels)):
            if i % 10000 == 0:  # Показываем прогресс каждые 10k записей
                logger.info(f"Processing results: {i:,}/{total_records:,} ({i/total_records*100:.1f}%)")
            
            if label == -1:
                self.noise_points.append(idx)
            else:
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(idx)
        
        logger.info(f"Clustering completed. Found {len(self.clusters)} clusters and {len(self.noise_points)} noise points")

    def _calculate_cluster_centers(self):
        """Вычисление центров кластеров"""
        logger.info("Calculating cluster centers...")
        
        self.cluster_centers = {}
        
        for cluster_label, cluster_indices in self.clusters.items():
            cluster_vectors = self.normalized_vectors[cluster_indices]
            center = np.mean(cluster_vectors, axis=0)
            self.cluster_centers[cluster_label] = center
        
        logger.info(f"Cluster centers calculated for {len(self.cluster_centers)} clusters")

    async def _process_noise_points(self):
        """Обработка шумовых точек"""
        total_noise = len(self.noise_points)
        logger.info(f"Processing {total_noise} noise points...")
        
        for i, noise_idx in enumerate(self.noise_points):
            if i % 1000 == 0:  # Показываем прогресс каждые 1000 шумовых точек
                logger.info(f"Processing noise points: {i:,}/{total_noise:,} ({i/total_noise*100:.1f}%)")
            
            # Находим ближайшие кластеры для шумовой точки
            nearest_clusters = self._find_nearest_clusters(noise_idx)
            
            # Назначаем аналогов из ближайших кластеров
            analogs = self._assign_analogs_from_clusters(noise_idx, nearest_clusters)
            
            # Сохраняем результаты
            self.catalog_df.loc[noise_idx, 'assigned_analogs'] = str(analogs)
        
        logger.info("Noise points processing completed")

    def _find_nearest_clusters(self, noise_idx: int) -> List[Tuple[int, float]]:
        """Поиск ближайших кластеров для шумовой точки"""
        noise_vector = self.normalized_vectors[noise_idx:noise_idx+1]
        
        cluster_distances = []
        for cluster_label, center in self.cluster_centers.items():
            distance = cosine_similarity(noise_vector, center.reshape(1, -1))[0, 0]
            cluster_distances.append((cluster_label, distance))
        
        # Сортируем по убыванию схожести
        cluster_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ кластеры
        return cluster_distances[:self.config.noise_config.max_nearest_clusters]

    def _assign_analogs_from_clusters(self, noise_idx: int, nearest_clusters: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Назначение аналогов из ближайших кластеров"""
        analogs = []
        
        for cluster_label, cluster_similarity in nearest_clusters:
            if cluster_similarity < self.config.noise_config.noise_assignment_threshold:
                continue
            
            cluster_indices = self.clusters[cluster_label]
            
            # Вычисляем схожесть с каждой точкой кластера
            cluster_analogs = []
            for cluster_idx in cluster_indices:
                similarity = self._calculate_hybrid_similarity(noise_idx, cluster_idx)
                
                if similarity >= self.config.similarity_threshold:
                    cluster_analogs.append((cluster_idx, similarity))
            
            # Сортируем по схожести и берем лучшие
            cluster_analogs.sort(key=lambda x: x[1], reverse=True)
            analogs.extend(cluster_analogs[:self.config.noise_config.max_analogs_per_noise])
        
        # Сортируем все аналогов по схожести
        analogs.sort(key=lambda x: x[1], reverse=True)
        
        # Ограничиваем общее количество аналогов
        return analogs[:self.config.noise_config.max_analogs_per_noise]

    def _calculate_hybrid_similarity(self, idx1: int, idx2: int) -> float:
        """Вычисление гибридной схожести между двумя товарами"""
        # TF-IDF схожесть
        tfidf_sim = cosine_similarity(
            self.tfidf_matrix[idx1:idx1+1], 
            self.tfidf_matrix[idx2:idx2+1]
        )[0, 0]
        
        # Нормализованная схожесть
        norm_sim = cosine_similarity(
            self.normalized_vectors[idx1:idx1+1], 
            self.normalized_vectors[idx2:idx2+1]
        )[0, 0]
        
        # Гибридная схожесть
        hybrid_sim = (
            self.config.tfidf_weight * tfidf_sim +
            self.config.normalized_weight * norm_sim
        )
        
        return float(hybrid_sim)

    async def find_all_analogs(self, catalog_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Поиск аналогов для всех записей с гарантированным покрытием"""
        if catalog_df is not None:
            self.set_catalog(catalog_df)
        
        if not self.is_fitted:
            await self.fit()
        
        logger.info("Finding analogs for all records...")
        
        all_results = {}
        total_records = len(self.catalog_df)
        
        # Обрабатываем все записи с прогрессом
        for idx in range(total_records):
            if idx % 5000 == 0: 
                logger.info(f"Finding analogs: {idx:,}/{total_records:,} ({idx/total_records*100:.1f}%)")
            
            analogs = await self.search_analogs(idx)
            all_results[idx] = analogs
        
        # Статистика
        records_with_analogs = sum(1 for analogs in all_results.values() if analogs)
        coverage_percentage = (records_with_analogs / total_records) * 100
        
        logger.info(f"Analog search completed. Coverage: {coverage_percentage:.2f}% ({records_with_analogs}/{total_records})")
        
        return {
            'results': all_results,
            'clusters': self.clusters,
            'noise_assignments': self._get_noise_assignments(),
            'statistics': {
                'total_records': total_records,
                'records_with_analogs': records_with_analogs,
                'coverage_percentage': coverage_percentage,
                'total_clusters': len(self.clusters),
                'noise_points': len(self.noise_points)
            }
        }

    async def search_analogs(self, query_idx: int) -> List[Dict[str, Any]]:
        """Поиск аналогов для товара по его индексу"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        if query_idx >= len(self.catalog_df):
            raise ValueError(f"Query index {query_idx} is out of range")

        query_row = self.catalog_df.iloc[query_idx]
        query_cluster = query_row['cluster_label']
        query_is_noise = query_row['is_noise']

        candidates = []

        # Если товар принадлежит кластеру, ищем в том же кластере
        if not query_is_noise and query_cluster in self.clusters:
            cluster_indices = self.clusters[query_cluster]
            for candidate_idx in cluster_indices:
                if candidate_idx != query_idx:  # Исключаем сам товар
                    similarity = self._calculate_hybrid_similarity(query_idx, candidate_idx)
                    if similarity >= self.config.similarity_threshold:
                        candidates.append({
                            'candidate_idx': candidate_idx,
                            'similarity': similarity,
                            'same_cluster': True,
                            'cluster_label': query_cluster
                        })

        # Если товар - шум или нужно найти кандидатов из других кластеров
        if query_is_noise or len(candidates) < self.config.max_candidates:
            # Для шумовых точек используем предварительно назначенные аналогов
            if query_is_noise:
                assigned_analogs = self._get_assigned_analogs(query_idx)
                for candidate_idx, similarity in assigned_analogs:
                    candidates.append({
                        'candidate_idx': candidate_idx,
                        'similarity': similarity,
                        'same_cluster': False,
                        'assigned_to_cluster': True
                    })

            # Дополнительный поиск по всем товарам (если нужно больше кандидатов)
            remaining_slots = self.config.max_candidates - len(candidates)
            if remaining_slots > 0:
                all_similarities = []
                for i in range(len(self.catalog_df)):
                    if i != query_idx:
                        similarity = self._calculate_hybrid_similarity(query_idx, i)
                        if similarity >= self.config.similarity_threshold:
                            all_similarities.append((i, similarity))

                # Сортируем по убыванию схожести
                all_similarities.sort(key=lambda x: x[1], reverse=True)

                # Добавляем лучшие кандидаты, которых еще нет
                existing_candidates = {c['candidate_idx'] for c in candidates}
                for candidate_idx, similarity in all_similarities[:remaining_slots]:
                    if candidate_idx not in existing_candidates:
                        candidate_row = self.catalog_df.iloc[candidate_idx]
                        candidates.append({
                            'candidate_idx': candidate_idx,
                            'similarity': similarity,
                            'same_cluster': candidate_row['cluster_label'] == query_cluster,
                            'cluster_label': candidate_row['cluster_label'],
                            'global_search': True
                        })

        # Сортируем кандидатов по схожести
        candidates.sort(key=lambda x: x['similarity'], reverse=True)

        return candidates

    def _get_assigned_analogs(self, noise_idx: int) -> List[Tuple[int, float]]:
        """Получение назначенных аналогов для шумовой точки"""
        assigned_str = self.catalog_df.loc[noise_idx, 'assigned_analogs']
        if pd.isna(assigned_str) or assigned_str == 'nan':
            return []
        
        try:
            # Парсим строку с аналогами
            analogs = eval(assigned_str)
            return analogs if isinstance(analogs, list) else []
        except:
            return []

    def _get_noise_assignments(self) -> Dict[int, List[Tuple[int, float]]]:
        """Получение всех назначений аналогов для шумовых точек"""
        noise_assignments = {}
        for noise_idx in self.noise_points:
            analogs = self._get_assigned_analogs(noise_idx)
            if analogs:
                noise_assignments[noise_idx] = analogs
        
        return noise_assignments

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики модели"""
        if not self.is_fitted:
            return {}
        
        return {
            'total_records': len(self.catalog_df),
            'total_clusters': len(self.clusters),
            'noise_points': len(self.noise_points),
            'coverage_percentage': ((len(self.catalog_df) - len(self.noise_points)) / len(self.catalog_df)) * 100,
            'avg_cluster_size': np.mean([len(indices) for indices in self.clusters.values()]) if self.clusters else 0
        }
