#!/usr/bin/env python3
"""
Оптимизированный процессор для больших датасетов с DBSCAN
Стратегия: иерархическая обработка + сэмплирование + параллелизация
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from collections import Counter, defaultdict
import gc
import psutil
import os
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import joblib
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_text_column(df: pd.DataFrame) -> str:
    """Определяет колонку с текстом для поиска.
    Предпочтение: 'Normalized_Name' -> 'Raw_Name' -> 'Наименование' -> 'Name' -> первая строковая колонка -> первый столбец.
    """
    preferred_columns = ['Normalized_Name', 'Raw_Name', 'Наименование', 'Name']
    for column_name in preferred_columns:
        if column_name in df.columns:
            logger.info(f"Using column: {column_name}")
            return column_name
    # Поиск первой колонки строкового типа
    for column_name in df.columns:
        try:
            if pd.api.types.is_object_dtype(df[column_name]) or pd.api.types.is_string_dtype(df[column_name]):
                logger.info(f"Using inferred text column: {column_name}")
                return column_name
        except Exception:
            continue
    # Фолбэк: первый столбец, если ничего не подошло
    if len(df.columns):
        fallback = df.columns[0]
        logger.info(f"Using fallback column: {fallback}")
        return fallback
    return 'text'


class LargeDatasetProcessor:
    """Процессор для больших датасетов с оптимизированным DBSCAN"""
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 sample_size: int = 5000,
                 eps: float = 0.4,
                 min_samples: int = 3,
                 similarity_threshold: float = 0.5,
                 max_features: int = 3000,
                 n_components: int = 50,
                 n_jobs: int = None):
        
        self.chunk_size = chunk_size
        self.sample_size = sample_size
        self.eps = eps
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.n_components = n_components
        self.n_jobs = n_jobs or min(4, mp.cpu_count())
        
        # Компоненты
        self.vectorizer = None
        self.svd = None
        self.dbscan = None
        self.cluster_centers = {}
        self.word_index = {}
        
        logger.info(f"LargeDatasetProcessor initialized:")
        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Sample size: {sample_size}")
        logger.info(f"  DBSCAN eps: {eps}, min_samples: {min_samples}")
        logger.info(f"  Max features: {max_features}, SVD components: {n_components}")
        logger.info(f"  Parallel jobs: {self.n_jobs}")
    
    def normalize_text_fast(self, text: str) -> str:
        """Быстрая нормализация текста"""
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = " ".join(filter(lambda x: len(x) > 2, text.split()))
        
        return text
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Быстрое вычисление схожести Жаккара"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def determine_relation_type(self, similarity: float) -> str:
        """Определение типа отношения"""
        if similarity >= 0.9:
            return "дубль"
        elif similarity >= 0.7:
            return "аналог"
        elif similarity >= 0.6:
            return "близкий аналог"
        elif similarity >= 0.4:
            return "возможный аналог"
        elif similarity >= 0.2:
            return "похожий товар"
        else:
            return "нет аналогов"
    
    def create_representative_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание репрезентативной выборки"""
        logger.info(f"Creating representative sample from {len(df)} records...")
        
        # Стратифицированная выборка по группам
        if 'Группа' in df.columns and df['Группа'].nunique() > 1:
            # Пропорциональная выборка по группам
            group_counts = df['Группа'].value_counts()
            sample_per_group = max(1, self.sample_size // len(group_counts))
            
            sample_dfs = []
            for group in group_counts.index:
                group_df = df[df['Группа'] == group]
                sample_size = min(len(group_df), sample_per_group)
                if sample_size > 0:
                    sample_dfs.append(group_df.sample(n=sample_size))
            
            sample_df = pd.concat(sample_dfs, ignore_index=True)
        else:
            # Простая случайная выборка
            sample_size = min(self.sample_size, len(df))
            sample_df = df.sample(n=sample_size)
        
        logger.info(f"Created sample with {len(sample_df)} records")
        return sample_df.reset_index(drop=True)
    
    def train_clustering_model(self, sample_df: pd.DataFrame, text_column: str = 'Normalized_Name'):
        """Обучение модели кластеризации на выборке"""
        logger.info("Training clustering model on sample...")
        
        # Подготовка текстов
        texts = sample_df[text_column].fillna('').astype(str).tolist()
        normalized_texts = [self.normalize_text_fast(text) for text in texts]
        
        # TF-IDF векторизация
        logger.info("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
            stop_words=None
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(normalized_texts)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # SVD для уменьшения размерности
        logger.info(f"Applying SVD reduction to {self.n_components} components...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        reduced_vectors = self.svd.fit_transform(tfidf_matrix)
        logger.info(f"Reduced vectors shape: {reduced_vectors.shape}")
        
        # DBSCAN кластеризация
        logger.info("Performing DBSCAN clustering...")
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine',
            n_jobs=1  # Ограничиваем для стабильности
        )
        
        cluster_labels = self.dbscan.fit_predict(reduced_vectors)
        
        # Анализ результатов
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        
        # Сохраняем центры кластеров
        for label in unique_labels:
            if label != -1:  # Исключаем шумовые точки
                cluster_mask = cluster_labels == label
                cluster_vectors = reduced_vectors[cluster_mask]
                center = np.mean(cluster_vectors, axis=0)
                self.cluster_centers[label] = center
        
        logger.info(f"Saved {len(self.cluster_centers)} cluster centers")
        
        # Создаем инвертированный индекс для быстрого поиска
        self.create_word_index(sample_df, text_column)
        
        return cluster_labels
    
    def create_word_index(self, df: pd.DataFrame, text_column: str):
        """Создание инвертированного индекса"""
        logger.info("Creating word index...")
        
        self.word_index = defaultdict(list)
        
        for idx, row in df.iterrows():
            text = self.normalize_text_fast(str(row.get(text_column, '')))
            words = text.split()
            
            for word in words:
                self.word_index[word].append(idx)
        
        logger.info(f"Word index created with {len(self.word_index)} unique words")
    
    def find_candidates_fast(self, query_text: str, df: pd.DataFrame, max_candidates: int = 50) -> List[int]:
        """Быстрый поиск кандидатов"""
        query_words = self.normalize_text_fast(query_text).split()
        
        if not query_words:
            return []
        
        candidate_scores = Counter()
        
        for word in query_words:
            if word in self.word_index:
                for idx in self.word_index[word]:
                    if idx < len(df):  # Проверяем границы
                        candidate_scores[idx] += 1
        
        # Возвращаем топ кандидатов
        candidates = [idx for idx, score in candidate_scores.most_common(max_candidates)]
        return candidates
    
    def process_chunk(self, chunk_df: pd.DataFrame, chunk_num: int, 
                     text_column: str = 'Normalized_Name') -> List[Dict[str, Any]]:
        """Обработка одного чанка данных"""
        logger.info(f"Processing chunk {chunk_num}: {len(chunk_df)} records")
        
        results = []
        
        for idx, row in chunk_df.iterrows():
            query_text = str(row.get(text_column, ''))
            
            if not query_text or query_text == 'nan':
                continue
            
            # Быстрый поиск кандидатов
            candidates = self.find_candidates_fast(query_text, chunk_df, max_candidates=30)
            
            # Вычисляем схожесть для кандидатов
            for candidate_idx in candidates:
                if candidate_idx == idx:  # Пропускаем сам товар
                    continue
                
                if candidate_idx >= len(chunk_df):  # Проверяем границы
                    continue
                
                candidate_row = chunk_df.iloc[candidate_idx]
                candidate_text = str(candidate_row.get(text_column, ''))
                
                # Вычисляем схожесть
                similarity = self.jaccard_similarity(
                    self.normalize_text_fast(query_text),
                    self.normalize_text_fast(candidate_text)
                )
                
                if similarity >= self.similarity_threshold:
                    relation_type = self.determine_relation_type(similarity)
                    
                    result = {
                        'Код': row.get('Код', ''),
                        'Raw_Name': row.get('Raw_Name', ''),
                        'Candidate_Name': candidate_row.get('Raw_Name', ''),
                        'Similarity_Score': round(similarity, 4),
                        'Relation_Type': relation_type,
                        'Suggested_Category': candidate_row.get('Группа', 'Неопределено'),
                        'Final_Decision': '',
                        'Comment': f"Chunk {chunk_num}; Жаккар: {similarity:.3f}",
                        'Original_Category': row.get('Группа', ''),
                        'Candidate_Код': candidate_row.get('Код', ''),
                        'Original_Code': row.get('Код', ''),
                        'Search_Engine': 'LargeDatasetDBSCAN'
                    }
                    
                    results.append(result)
        
        logger.info(f"Chunk {chunk_num} completed: {len(results)} relationships found")
        return results
    
    def process_large_dataset(self, input_file: str, output_file: str, 
                            text_column: str = 'Normalized_Name') -> str:
        """Основной метод обработки большого датасета"""
        logger.info(f"🚀 Starting large dataset processing: {input_file}")
        
        # Загружаем данные
        logger.info("Loading dataset...")
        df = pd.read_csv(input_file)
        total_records = len(df)
        logger.info(f"Loaded {total_records} records")
        
        # Определяем колонку текста
        selected_text_column = text_column if text_column in df.columns else detect_text_column(df)
        logger.info(f"Using text column: {selected_text_column}")
        
        # Проверяем доступную память
        memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Available memory: {memory_gb:.1f}GB")
        
        # Создаем репрезентативную выборку и обучаем модель
        sample_df = self.create_representative_sample(df)
        self.train_clustering_model(sample_df, selected_text_column)
        
        # Обрабатываем данные чанками
        all_results = []
        total_chunks = (total_records + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Processing {total_records} records in {total_chunks} chunks...")
        
        for chunk_num in range(total_chunks):
            start_idx = chunk_num * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_records)
            
            # Извлекаем чанк
            chunk_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            
            # Обновляем индекс для текущего чанка
            self.create_word_index(chunk_df, selected_text_column)
            
            # Обрабатываем чанк
            chunk_results = self.process_chunk(chunk_df, chunk_num + 1, selected_text_column)
            all_results.extend(chunk_results)
            
            # Мониторинг памяти
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            logger.info(f"Progress: {end_idx}/{total_records} ({end_idx/total_records*100:.1f}%), "
                       f"Memory: {current_memory:.1f}GB, "
                       f"Results so far: {len(all_results)}")
            
            # Сборка мусора каждые 10 чанков
            if chunk_num % 10 == 0:
                gc.collect()
        
        # Сохраняем результаты
        logger.info(f"Saving {len(all_results)} results to {output_file}")
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✅ Results saved to: {output_file}")
            
            # Статистика
            relation_counts = results_df['Relation_Type'].value_counts()
            logger.info(f"📊 Results summary:")
            for relation_type, count in relation_counts.items():
                logger.info(f"  {relation_type}: {count}")
        else:
            logger.warning("No results found!")
        
        logger.info(f"🎉 Large dataset processing completed!")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Large dataset processor with optimized DBSCAN')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file')
    parser.add_argument('-c', '--chunk-size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('-s', '--sample-size', type=int, default=5000, help='Sample size for model training')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--eps', type=float, default=0.4, help='DBSCAN eps parameter')
    parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples parameter')
    parser.add_argument('--max-features', type=int, default=3000, help='Max TF-IDF features')
    parser.add_argument('--n-components', type=int, default=50, help='SVD components')
    parser.add_argument('-j', '--jobs', type=int, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Определяем выходной файл
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"large_dbscan_results_{input_path.stem}_{timestamp}.csv"
    
    # Создаем процессор
    processor = LargeDatasetProcessor(
        chunk_size=args.chunk_size,
        sample_size=args.sample_size,
        eps=args.eps,
        min_samples=args.min_samples,
        similarity_threshold=args.threshold,
        max_features=args.max_features,
        n_components=args.n_components,
        n_jobs=args.jobs
    )
    
    # Запускаем обработку
    result_file = processor.process_large_dataset(args.input_file, output_file)
    print(f"✅ Processing completed! Results saved to: {result_file}")


if __name__ == "__main__":
    main()
