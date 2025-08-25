#!/usr/bin/env python3
"""
Hybrid DBSCAN Search Engine для поиска аналогов товаров
Использует кластеризацию DBSCAN для группировки похожих товаров,
а затем распределяет неклассифицированные товары между найденными кластерами.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import asyncio
from pathlib import Path

# Импорты SAMe модулей
try:
    from src.same_clear.text_processing.enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessorConfig
    from src.same_clear.text_processing.text_cleaner import TextCleaner
    SAME_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMe modules not available. Missing: {e}")
    SAME_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HybridDBSCANConfig:
    """Конфигурация для Hybrid DBSCAN поиска"""
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


class RelationType:
    """Типы отношений между товарами"""
    DUPLICATE = "дубль"
    ANALOG = "аналог"
    CLOSE_ANALOG = "близкий аналог"
    POSSIBLE_ANALOG = "возможный аналог"
    SIMILAR_PRODUCT = "похожий товар"
    NO_ANALOGS = "нет аналогов"


class HybridDBSCANSearchEngine:
    """Hybrid DBSCAN поисковый движок"""
    
    def __init__(self, config: HybridDBSCANConfig = None):
        self.config = config or HybridDBSCANConfig()
        
        # Компоненты обработки текста
        if SAME_MODULES_AVAILABLE:
            self.text_cleaner = TextCleaner()
            enhanced_config = EnhancedPreprocessorConfig(
                enable_units_processing=True,
                enable_synonyms_processing=True,
                enable_tech_codes_processing=True,
                parallel_processing=False,  # Отключаем для стабильности
                max_workers=1
            )
            self.preprocessor = EnhancedPreprocessor(enhanced_config)
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
        
        logger.info("HybridDBSCANSearchEngine initialized")
    
    def normalize_text(self, text: str) -> str:
        """Нормализация текста с использованием функции из excel_processor"""
        if not text or not isinstance(text, str):
            return ""
        
        import re
        
        # Применяем ту же нормализацию что и в excel_processor
        delete = [")", "(", ":", ";", "!", "?", "№", "#", "%", "/",
                  ".", ",", "-", "для", "гост", "ГОСТ"]
        
        for d in delete:
            text = text.replace(d, "")
        
        text = re.sub(r'<[^>]*>', '', text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("color", "").replace("num", "")
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = " ".join(filter(lambda x: not any([c.isdigit() for c in x]), text.split()))
        text = " ".join(filter(lambda x: len(x) > 2, text.split()))
        
        return text
    
    async def fit(self, catalog_df: pd.DataFrame, text_column: str = 'Normalized_Name') -> None:
        """
        Обучение модели на каталоге товаров
        
        Args:
            catalog_df: DataFrame с каталогом товаров
            text_column: Название колонки с текстом для анализа
        """
        logger.info(f"Starting DBSCAN clustering on {len(catalog_df)} items...")
        
        self.catalog_df = catalog_df.copy()
        
        # Проверяем наличие колонки
        if text_column not in catalog_df.columns:
            logger.warning(f"Column '{text_column}' not found, using 'Raw_Name' instead")
            text_column = 'Raw_Name'
        
        # Извлекаем и обрабатываем тексты
        texts = catalog_df[text_column].fillna('').astype(str).tolist()
        
        # Дополнительная нормализация
        self.processed_texts = [self.normalize_text(text) for text in texts]
        
        # Создаем TF-IDF векторы
        logger.info("Creating TF-IDF vectors...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
        
        # Нормализуем векторы
        self.normalized_vectors = self.scaler.fit_transform(self.tfidf_matrix.toarray())
        
        # Выполняем кластеризацию DBSCAN
        logger.info(f"Performing DBSCAN clustering (eps={self.config.eps}, min_samples={self.config.min_samples})...")
        self.dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric='cosine'
        )
        
        cluster_labels = self.dbscan.fit_predict(self.normalized_vectors)
        
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
            cluster_vectors = self.normalized_vectors[indices]
            center = np.mean(cluster_vectors, axis=0)
            self.cluster_centers[label] = center
        
        # Добавляем информацию о кластерах в DataFrame
        self.catalog_df['cluster_label'] = cluster_labels
        self.catalog_df['is_noise'] = cluster_labels == -1
        
        self.is_fitted = True
        logger.info("DBSCAN clustering completed successfully")
    
    def _determine_relation_type(self, similarity_score: float, same_cluster: bool, 
                                is_duplicate: bool) -> str:
        """Определение типа отношения на основе схожести и кластерной принадлежности"""
        if is_duplicate:
            return RelationType.DUPLICATE
        elif same_cluster:
            if similarity_score >= 0.9:
                return RelationType.ANALOG
            elif similarity_score >= 0.7:
                return RelationType.CLOSE_ANALOG
            else:
                return RelationType.POSSIBLE_ANALOG
        else:
            if similarity_score >= 0.8:
                return RelationType.CLOSE_ANALOG
            elif similarity_score >= 0.6:
                return RelationType.POSSIBLE_ANALOG
            elif similarity_score >= 0.4:
                return RelationType.SIMILAR_PRODUCT
            else:
                return RelationType.NO_ANALOGS
    
    def _calculate_hybrid_similarity(self, idx1: int, idx2: int) -> float:
        """Вычисление гибридной схожести между двумя товарами"""
        # TF-IDF схожесть
        tfidf_sim = cosine_similarity(
            self.tfidf_matrix[idx1:idx1+1],
            self.tfidf_matrix[idx2:idx2+1]
        )[0, 0]
        
        # Схожесть нормализованных векторов
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

    async def search_analogs(self, query_idx: int) -> List[Dict[str, Any]]:
        """
        Поиск аналогов для товара по его индексу

        Args:
            query_idx: Индекс товара в каталоге

        Returns:
            Список найденных аналогов
        """
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
            # Ищем ближайшие кластеры
            if query_is_noise:
                # Для шумовых точек ищем ближайший кластер
                query_vector = self.normalized_vectors[query_idx:query_idx+1]
                best_cluster = None
                best_distance = float('inf')

                for cluster_label, center in self.cluster_centers.items():
                    distance = cosine_similarity(query_vector, center.reshape(1, -1))[0, 0]
                    if distance > best_distance:
                        best_distance = distance
                        best_cluster = cluster_label

                # Добавляем кандидатов из лучшего кластера
                if best_cluster is not None and best_distance >= self.config.cluster_assignment_threshold:
                    cluster_indices = self.clusters[best_cluster]
                    for candidate_idx in cluster_indices:
                        similarity = self._calculate_hybrid_similarity(query_idx, candidate_idx)
                        if similarity >= self.config.similarity_threshold:
                            candidates.append({
                                'candidate_idx': candidate_idx,
                                'similarity': similarity,
                                'same_cluster': False,
                                'cluster_label': best_cluster,
                                'assigned_to_cluster': True
                            })

            # Дополнительный поиск по всем товарам (если нужно больше кандидатов)
            remaining_slots = self.config.max_candidates - len(candidates)
            if remaining_slots > 0:
                all_similarities = []
                query_vector = self.normalized_vectors[query_idx:query_idx+1]

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

        # Формируем результаты
        results = []
        for candidate in candidates[:self.config.max_candidates]:
            candidate_row = self.catalog_df.iloc[candidate['candidate_idx']]

            # Проверяем на дубликаты
            is_duplicate = (
                query_row.get('Raw_Name', '') == candidate_row.get('Raw_Name', '') or
                query_row.get('Normalized_Name', '') == candidate_row.get('Normalized_Name', '')
            )

            relation_type = self._determine_relation_type(
                candidate['similarity'],
                candidate['same_cluster'],
                is_duplicate
            )

            result = {
                'Код': query_row.get('Код', ''),
                'Raw_Name': query_row.get('Raw_Name', ''),
                'Candidate_Name': candidate_row.get('Raw_Name', ''),
                'Similarity_Score': round(candidate['similarity'], 4),
                'Relation_Type': relation_type,
                'Suggested_Category': self._suggest_category(candidate_row),
                'Final_Decision': '',  # Заполняется пользователем
                'Comment': self._generate_comment(candidate, query_row, candidate_row),
                'Original_Category': query_row.get('Группа', ''),
                'Candidate_Код': candidate_row.get('Код', ''),
                'Original_Code': query_row.get('Код', ''),
                'Search_Engine': 'HybridDBSCAN'
            }

            results.append(result)

        return results

    def _suggest_category(self, candidate_row: pd.Series) -> str:
        """Предложение категории на основе данных кандидата"""
        return candidate_row.get('Группа', 'Неопределено')

    def _generate_comment(self, candidate: Dict, query_row: pd.Series,
                         candidate_row: pd.Series) -> str:
        """Генерация комментария для результата поиска"""
        comments = []

        if candidate.get('same_cluster', False):
            comments.append(f"Тот же кластер #{candidate['cluster_label']}")
        elif candidate.get('assigned_to_cluster', False):
            comments.append(f"Назначен в кластер #{candidate['cluster_label']}")
        elif candidate.get('global_search', False):
            comments.append("Глобальный поиск")

        # Добавляем информацию о схожести
        similarity = candidate['similarity']
        if similarity >= 0.9:
            comments.append("Очень высокая схожесть")
        elif similarity >= 0.7:
            comments.append("Высокая схожесть")
        elif similarity >= 0.5:
            comments.append("Средняя схожесть")
        else:
            comments.append("Низкая схожесть")

        return "; ".join(comments)

    async def process_catalog(self, catalog_df: pd.DataFrame,
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Обработка всего каталога для поиска аналогов

        Args:
            catalog_df: DataFrame с каталогом товаров
            output_path: Путь для сохранения результатов

        Returns:
            DataFrame с результатами поиска
        """
        logger.info(f"Processing catalog with {len(catalog_df)} items...")

        # Обучаем модель
        await self.fit(catalog_df)

        # Обрабатываем каждый товар
        all_results = []
        batch_size = max(1, self.config.batch_size)  # Убеждаемся что batch_size >= 1

        for i in range(0, len(catalog_df), batch_size):
            batch_end = min(i + batch_size, len(catalog_df))
            logger.info(f"Processing batch {i//batch_size + 1}: items {i}-{batch_end-1}")

            batch_tasks = []
            for idx in range(i, batch_end):
                batch_tasks.append(self.search_analogs(idx))

            # Выполняем поиск для батча
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for idx, results in enumerate(batch_results):
                if isinstance(results, Exception):
                    logger.error(f"Error processing item {i + idx}: {results}")
                    continue

                all_results.extend(results)

        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(all_results)

        if output_path:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")

        logger.info(f"Processing completed. Found {len(all_results)} analog relationships")
        return results_df
