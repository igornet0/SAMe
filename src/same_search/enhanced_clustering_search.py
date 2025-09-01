"""
Улучшенный кластеризационный поиск с адаптивными параметрами DBSCAN
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import asyncio
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EnhancedClusteringConfig:
    """Конфигурация улучшенной кластеризации"""
    # Адаптивные параметры DBSCAN
    eps_range: Tuple[float, float] = (0.2, 0.5)  # Диапазон eps для адаптации
    min_samples_range: Tuple[int, int] = (2, 5)  # Диапазон min_samples
    
    # Векторизация
    max_features: int = 8000
    ngram_range: Tuple[int, int] = (1, 3)
    use_svd: bool = True
    svd_components: int = 150
    
    # Пороги схожести
    similarity_threshold: float = 0.4
    cluster_assignment_threshold: float = 0.3
    
    # Обработка шума
    noise_processing: bool = True
    max_noise_analogs: int = 10
    
    # Параметры результатов
    max_candidates: int = 15


class EnhancedClusteringSearch:
    """Улучшенный кластеризационный поиск"""
    
    def __init__(self, config: EnhancedClusteringConfig = None):
        self.config = config or EnhancedClusteringConfig()
        
        # Векторизаторы
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words=None,
            lowercase=True,
            token_pattern=r'\b\w+\b',
            max_df=0.95,
            min_df=2
        )
        
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=self.config.svd_components) if self.config.use_svd else None
        
        # Модели
        self.dbscan = None
        self.catalog_df = None
        self.processed_texts = None
        self.tfidf_matrix = None
        self.reduced_vectors = None
        self.clusters = None
        self.cluster_centers = None
        self.noise_points = None
        self.optimal_params = None
        
        self.is_fitted = False
        
        logger.info("EnhancedClusteringSearch initialized")
    
    def _find_optimal_dbscan_params(self, vectors: np.ndarray) -> Tuple[float, int]:
        """Поиск оптимальных параметров DBSCAN"""
        logger.info("Searching for optimal DBSCAN parameters...")
        
        best_eps = 0.3
        best_min_samples = 2
        best_score = -1
        
        eps_values = np.linspace(self.config.eps_range[0], self.config.eps_range[1], 5)
        min_samples_values = range(self.config.min_samples_range[0], self.config.min_samples_range[1] + 1)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(vectors)
                    
                    # Оценка качества кластеризации
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters > 0:
                        # Комбинированная оценка: больше кластеров, меньше шума
                        noise_ratio = n_noise / len(labels)
                        cluster_ratio = n_clusters / len(labels)
                        
                        # Оценка: максимизируем количество кластеров, минимизируем шум
                        score = cluster_ratio * (1 - noise_ratio)
                        
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                            
                except Exception as e:
                    logger.debug(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        logger.info(f"Optimal DBSCAN parameters: eps={best_eps:.3f}, min_samples={best_min_samples}, score={best_score:.3f}")
        return best_eps, best_min_samples
    
    async def fit(self, catalog_df: pd.DataFrame, text_column: str = 'processed_name'):
        """Обучение модели с адаптивными параметрами"""
        logger.info(f"Training enhanced clustering on {len(catalog_df)} items...")
        
        self.catalog_df = catalog_df.copy()
        
        # Извлекаем и обрабатываем тексты
        texts = catalog_df[text_column].fillna('').astype(str).tolist()
        self.processed_texts = [self._normalize_text(text) for text in texts]
        
        # Создаем TF-IDF векторы
        logger.info("Creating TF-IDF vectors...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
        
        # Применяем SVD для уменьшения размерности
        if self.config.use_svd:
            logger.info(f"Applying SVD reduction to {self.config.svd_components} components...")
            self.reduced_vectors = self.svd.fit_transform(self.tfidf_matrix)
            vectors_for_clustering = self.reduced_vectors
        else:
            vectors_for_clustering = self.tfidf_matrix.toarray()
        
        # Нормализуем векторы
        vectors_for_clustering = self.scaler.fit_transform(vectors_for_clustering)
        
        # Находим оптимальные параметры DBSCAN
        optimal_eps, optimal_min_samples = self._find_optimal_dbscan_params(vectors_for_clustering)
        self.optimal_params = (optimal_eps, optimal_min_samples)
        
        # Выполняем кластеризацию с оптимальными параметрами
        logger.info(f"Performing DBSCAN clustering with optimal parameters...")
        self.dbscan = DBSCAN(
            eps=optimal_eps,
            min_samples=optimal_min_samples,
            metric='cosine'
        )
        
        cluster_labels = self.dbscan.fit_predict(vectors_for_clustering)
        
        # Анализируем результаты
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        
        # Сохраняем результаты кластеризации
        self._process_clustering_results(cluster_labels, vectors_for_clustering)
        
        # Добавляем информацию о кластерах в DataFrame
        self.catalog_df['cluster_label'] = cluster_labels
        self.catalog_df['is_noise'] = cluster_labels == -1
        
        self.is_fitted = True
        logger.info("Enhanced clustering training completed successfully")
    
    def _process_clustering_results(self, cluster_labels: np.ndarray, vectors: np.ndarray):
        """Обработка результатов кластеризации"""
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
            cluster_vectors = vectors[indices]
            center = np.mean(cluster_vectors, axis=0)
            self.cluster_centers[label] = center
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        if not text or not isinstance(text, str):
            return ""
        
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = " ".join(filter(lambda x: len(x) > 2, text.split()))
        
        return text
    
    async def search_analogs(self, query_idx: int) -> List[Dict[str, Any]]:
        """Поиск аналогов для товара"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        if query_idx >= len(self.catalog_df):
            raise ValueError(f"Query index {query_idx} is out of range")
        
        query_row = self.catalog_df.iloc[query_idx]
        query_cluster = query_row['cluster_label']
        query_is_noise = query_row['is_noise']
        
        candidates = []
        
        # Поиск в том же кластере
        if not query_is_noise and query_cluster in self.clusters:
            cluster_indices = self.clusters[query_cluster]
            for candidate_idx in cluster_indices:
                if candidate_idx != query_idx:
                    similarity = self._calculate_similarity(query_idx, candidate_idx)
                    if similarity >= self.config.similarity_threshold:
                        candidates.append({
                            'candidate_idx': candidate_idx,
                            'similarity': similarity,
                            'same_cluster': True,
                            'cluster_label': query_cluster
                        })
        
        # Обработка шумовых точек
        if query_is_noise and self.config.noise_processing:
            noise_candidates = self._find_noise_analogs(query_idx)
            candidates.extend(noise_candidates)
        
        # Дополнительный поиск если нужно больше кандидатов
        if len(candidates) < self.config.max_candidates:
            additional_candidates = self._find_additional_candidates(query_idx, candidates)
            candidates.extend(additional_candidates)
        
        # Сортируем и ограничиваем результаты
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        candidates = candidates[:self.config.max_candidates]
        
        # Формируем результаты
        results = []
        for candidate in candidates:
            candidate_row = self.catalog_df.iloc[candidate['candidate_idx']]
            
            relation_type = self._determine_relation_type(candidate['similarity'], candidate.get('same_cluster', False))
            
            result = {
                'Код': query_row.get('Код', ''),
                'Raw_Name': query_row.get('Raw_Name', ''),
                'Candidate_Name': candidate_row.get('Raw_Name', ''),
                'Similarity_Score': round(candidate['similarity'], 4),
                'Relation_Type': relation_type,
                'Suggested_Category': candidate_row.get('Группа', 'Неопределено'),
                'Final_Decision': '',
                'Comment': self._generate_comment(candidate, query_row, candidate_row),
                'Original_Category': query_row.get('Группа', ''),
                'Candidate_Код': candidate_row.get('Код', ''),
                'Original_Code': query_row.get('Код', ''),
                'Search_Engine': 'EnhancedClustering'
            }
            
            results.append(result)
        
        return results
    
    def _find_noise_analogs(self, query_idx: int) -> List[Dict[str, Any]]:
        """Поиск аналогов для шумовых точек"""
        candidates = []
        
        # Ищем ближайшие кластеры
        if self.reduced_vectors is not None:
            query_vector = self.reduced_vectors[query_idx:query_idx+1]
        else:
            query_vector = self.tfidf_matrix[query_idx:query_idx+1].toarray()
        
        # Находим ближайшие кластеры
        cluster_distances = []
        for cluster_label, center in self.cluster_centers.items():
            distance = cosine_similarity(query_vector, center.reshape(1, -1))[0, 0]
            cluster_distances.append((cluster_label, distance))
        
        # Сортируем по убыванию схожести
        cluster_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Берем кандидатов из ближайших кластеров
        for cluster_label, distance in cluster_distances[:3]:  # Топ-3 кластера
            if distance >= self.config.cluster_assignment_threshold:
                cluster_indices = self.clusters[cluster_label]
                for candidate_idx in cluster_indices[:5]:  # До 5 кандидатов из кластера
                    if candidate_idx != query_idx:
                        similarity = self._calculate_similarity(query_idx, candidate_idx)
                        if similarity >= self.config.similarity_threshold:
                            candidates.append({
                                'candidate_idx': candidate_idx,
                                'similarity': similarity,
                                'same_cluster': False,
                                'cluster_label': cluster_label,
                                'assigned_to_cluster': True
                            })
        
        return candidates[:self.config.max_noise_analogs]
    
    def _find_additional_candidates(self, query_idx: int, existing_candidates: List[Dict]) -> List[Dict[str, Any]]:
        """Поиск дополнительных кандидатов"""
        existing_indices = {c['candidate_idx'] for c in existing_candidates}
        candidates = []
        
        # Простой поиск по всем товарам
        for i in range(len(self.catalog_df)):
            if i != query_idx and i not in existing_indices:
                similarity = self._calculate_similarity(query_idx, i)
                if similarity >= self.config.similarity_threshold:
                    candidate_row = self.catalog_df.iloc[i]
                    candidates.append({
                        'candidate_idx': i,
                        'similarity': similarity,
                        'same_cluster': False,
                        'cluster_label': candidate_row['cluster_label'],
                        'global_search': True
                    })
        
        # Сортируем и возвращаем лучших
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:self.config.max_candidates - len(existing_candidates)]
    
    def _calculate_similarity(self, idx1: int, idx2: int) -> float:
        """Вычисление схожести между двумя товарами"""
        if self.reduced_vectors is not None:
            sim = cosine_similarity(
                self.reduced_vectors[idx1:idx1+1],
                self.reduced_vectors[idx2:idx2+1]
            )[0, 0]
        else:
            sim = cosine_similarity(
                self.tfidf_matrix[idx1:idx1+1],
                self.tfidf_matrix[idx2:idx2+1]
            )[0, 0]
        
        return float(sim)
    
    def _determine_relation_type(self, similarity: float, same_cluster: bool) -> str:
        """Определение типа отношения"""
        if similarity >= 0.9:
            return "дубль"
        elif same_cluster and similarity >= 0.7:
            return "аналог"
        elif similarity >= 0.7:
            return "близкий аналог"
        elif similarity >= 0.5:
            return "возможный аналог"
        elif similarity >= 0.4:
            return "похожий товар"
        else:
            return "нет аналогов"
    
    def _generate_comment(self, candidate: Dict, query_row: pd.Series, candidate_row: pd.Series) -> str:
        """Генерация комментария"""
        comments = []
        
        if candidate.get('same_cluster', False):
            comments.append(f"Тот же кластер #{candidate['cluster_label']}")
        elif candidate.get('assigned_to_cluster', False):
            comments.append(f"Назначен в кластер #{candidate['cluster_label']}")
        elif candidate.get('global_search', False):
            comments.append("Глобальный поиск")
        
        similarity = candidate['similarity']
        if similarity >= 0.8:
            comments.append("Очень высокая схожесть")
        elif similarity >= 0.6:
            comments.append("Высокая схожесть")
        elif similarity >= 0.4:
            comments.append("Средняя схожесть")
        else:
            comments.append("Низкая схожесть")
        
        return "; ".join(comments)
    
    async def process_catalog(self, catalog_df: pd.DataFrame, 
                            text_column: str = 'processed_name',
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """Обработка всего каталога"""
        logger.info(f"Processing catalog with enhanced clustering on {len(catalog_df)} items...")
        
        # Обучаем модель
        await self.fit(catalog_df, text_column)
        
        # Обрабатываем каждый товар
        all_results = []
        
        for idx in tqdm(range(len(catalog_df)), desc="Enhanced clustering search"):
            try:
                results = await self.search_analogs(idx)
                
                # Добавляем информацию о запросе
                for result in results:
                    result['query_index'] = idx
                    result['query_name'] = catalog_df.iloc[idx][text_column]
                    result['original_code'] = catalog_df.iloc[idx].get('Код', '')
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue
        
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to {output_path}")
        
        logger.info(f"Enhanced clustering processing completed. Found {len(all_results)} analog relationships")
        return results_df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        stats = {
            'is_fitted': self.is_fitted,
            'total_items': len(self.catalog_df) if self.catalog_df is not None else 0,
            'optimal_params': self.optimal_params,
            'config': {
                'max_features': self.config.max_features,
                'similarity_threshold': self.config.similarity_threshold,
                'use_svd': self.config.use_svd,
                'svd_components': self.config.svd_components
            }
        }
        
        if self.is_fitted:
            stats['clustering_stats'] = {
                'n_clusters': len(self.clusters),
                'n_noise_points': len(self.noise_points),
                'noise_ratio': len(self.noise_points) / len(self.catalog_df) if len(self.catalog_df) > 0 else 0
            }
        
        return stats
