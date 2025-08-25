#!/usr/bin/env python3
"""
Параллельный процессор для обработки всего датасета с DBSCAN
Использует многопроцессорность для максимальной производительности
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
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_text_fast(text: str) -> str:
    """Быстрая нормализация текста"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join(filter(lambda x: len(x) > 2, text.split()))
    
    return text


def jaccard_similarity(text1: str, text2: str) -> float:
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


def determine_relation_type(similarity: float) -> str:
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


def create_word_index(df_chunk: pd.DataFrame, text_column: str) -> Dict[str, List[int]]:
    """Создание инвертированного индекса для чанка"""
    word_index = defaultdict(list)
    
    for idx, row in df_chunk.iterrows():
        text = normalize_text_fast(str(row.get(text_column, '')))
        words = text.split()
        
        for word in words:
            word_index[word].append(idx)
    
    return dict(word_index)


def find_candidates_fast(query_text: str, word_index: Dict[str, List[int]], 
                        df_chunk: pd.DataFrame, max_candidates: int = 50) -> List[int]:
    """Быстрый поиск кандидатов через инвертированный индекс"""
    query_words = normalize_text_fast(query_text).split()
    
    if not query_words:
        return []
    
    candidate_scores = Counter()
    
    for word in query_words:
        if word in word_index:
            for idx in word_index[word]:
                if idx < len(df_chunk):
                    candidate_scores[idx] += 1
    
    candidates = [idx for idx, score in candidate_scores.most_common(max_candidates)]
    return candidates


def process_chunk_parallel(args: Tuple[pd.DataFrame, int, str, float, int]) -> List[Dict[str, Any]]:
    """Обработка одного чанка в отдельном процессе"""
    chunk_df, chunk_num, text_column, similarity_threshold, max_candidates = args
    
    # Создаем локальный индекс для чанка
    word_index = create_word_index(chunk_df, text_column)
    
    results = []
    processed_count = 0
    
    for idx, row in chunk_df.iterrows():
        query_text = str(row.get(text_column, ''))
        
        if not query_text or query_text == 'nan':
            continue
        
        # Быстрый поиск кандидатов
        candidates = find_candidates_fast(query_text, word_index, chunk_df, max_candidates)
        
        # Вычисляем схожесть для кандидатов
        for candidate_idx in candidates:
            if candidate_idx == idx:  # Пропускаем сам товар
                continue
            
            if candidate_idx >= len(chunk_df):  # Проверяем границы
                continue
            
            candidate_row = chunk_df.iloc[candidate_idx]
            candidate_text = str(candidate_row.get(text_column, ''))
            
            # Вычисляем схожесть
            similarity = jaccard_similarity(
                normalize_text_fast(query_text),
                normalize_text_fast(candidate_text)
            )
            
            if similarity >= similarity_threshold:
                relation_type = determine_relation_type(similarity)
                
                result = {
                    'Код': row.get('Код', ''),
                    'Raw_Name': row.get('Raw_Name', ''),
                    'Candidate_Name': candidate_row.get('Raw_Name', ''),
                    'Similarity_Score': round(similarity, 4),
                    'Relation_Type': relation_type,
                    'Suggested_Category': candidate_row.get('Группа', 'Неопределено'),
                    'Final_Decision': '',
                    'Comment': f"Chunk {chunk_num}; Parallel; Жаккар: {similarity:.3f}",
                    'Original_Category': row.get('Группа', ''),
                    'Candidate_Код': candidate_row.get('Код', ''),
                    'Original_Code': row.get('Код', ''),
                    'Search_Engine': 'ParallelDBSCAN'
                }
                
                results.append(result)
        
        processed_count += 1
        
        # Логируем прогресс каждые 1000 записей
        if processed_count % 1000 == 0:
            print(f"Chunk {chunk_num}: processed {processed_count}/{len(chunk_df)} records, found {len(results)} relationships")
    
    print(f"Chunk {chunk_num} completed: {len(results)} relationships found")
    return results


class ParallelDBSCANProcessor:
    """Параллельный процессор для больших датасетов"""
    
    def __init__(self, 
                 chunk_size: int = 8000,
                 similarity_threshold: float = 0.5,
                 max_candidates: int = 30,
                 n_workers: int = None,
                 overlap_size: int = 1000):
        
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # Ограничиваем до 8 процессов
        self.overlap_size = overlap_size  # Перекрытие между чанками
        
        logger.info(f"ParallelDBSCANProcessor initialized:")
        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Overlap size: {overlap_size}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Max candidates per query: {max_candidates}")
        logger.info(f"  Workers: {self.n_workers}")
    
    def create_overlapping_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Создание чанков с перекрытием для лучшего покрытия"""
        chunks = []
        total_records = len(df)
        
        for start_idx in range(0, total_records, self.chunk_size):
            # Основной чанк
            end_idx = min(start_idx + self.chunk_size, total_records)
            
            # Добавляем перекрытие
            overlap_start = max(0, start_idx - self.overlap_size // 2)
            overlap_end = min(total_records, end_idx + self.overlap_size // 2)
            
            chunk = df.iloc[overlap_start:overlap_end].copy().reset_index(drop=True)
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} overlapping chunks")
        return chunks
    
    def process_dataset_parallel(self, input_file: str, output_file: str, 
                               text_column: str = 'Normalized_Name') -> str:
        """Параллельная обработка всего датасета"""
        logger.info(f"🚀 Starting parallel processing: {input_file}")
        
        # Загружаем данные
        logger.info("Loading dataset...")
        df = pd.read_csv(input_file)
        total_records = len(df)
        logger.info(f"Loaded {total_records} records")
        
        # Проверяем доступную память
        memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Available memory: {memory_gb:.1f}GB")
        
        # Создаем чанки с перекрытием
        chunks = self.create_overlapping_chunks(df)
        
        # Подготавливаем аргументы для параллельной обработки
        chunk_args = []
        for i, chunk in enumerate(chunks):
            args = (chunk, i + 1, text_column, self.similarity_threshold, self.max_candidates)
            chunk_args.append(args)
        
        # Параллельная обработка
        logger.info(f"Starting parallel processing with {self.n_workers} workers...")
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Отправляем задачи
            future_to_chunk = {
                executor.submit(process_chunk_parallel, args): i 
                for i, args in enumerate(chunk_args)
            }
            
            # Собираем результаты по мере готовности
            completed_chunks = 0
            for future in as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    completed_chunks += 1
                    
                    # Мониторинг прогресса
                    progress = completed_chunks / len(chunks) * 100
                    current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                    
                    logger.info(f"Progress: {completed_chunks}/{len(chunks)} chunks ({progress:.1f}%), "
                               f"Memory: {current_memory:.1f}GB, "
                               f"Total results: {len(all_results)}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {e}")
        
        # Удаляем дубликаты результатов (из-за перекрытий)
        logger.info("Removing duplicate results from overlapping chunks...")
        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            # Удаляем дубликаты по ключевым полям
            before_dedup = len(results_df)
            results_df = results_df.drop_duplicates(
                subset=['Код', 'Candidate_Код', 'Similarity_Score'], 
                keep='first'
            )
            after_dedup = len(results_df)
            logger.info(f"Removed {before_dedup - after_dedup} duplicate results")
        
        # Сохраняем результаты
        logger.info(f"Saving {len(results_df)} results to {output_file}")
        
        if not results_df.empty:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✅ Results saved to: {output_file}")
            
            # Статистика
            relation_counts = results_df['Relation_Type'].value_counts()
            logger.info(f"📊 Results summary:")
            for relation_type, count in relation_counts.items():
                logger.info(f"  {relation_type}: {count}")
        else:
            logger.warning("No results found!")
        
        logger.info(f"🎉 Parallel processing completed!")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Parallel DBSCAN processor for large datasets')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file')
    parser.add_argument('-c', '--chunk-size', type=int, default=8000, help='Chunk size for processing')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--max-candidates', type=int, default=30, help='Max candidates per query')
    parser.add_argument('-j', '--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--overlap', type=int, default=1000, help='Overlap size between chunks')
    
    args = parser.parse_args()
    
    # Определяем выходной файл
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"parallel_dbscan_results_{input_path.stem}_{timestamp}.csv"
    
    # Создаем процессор
    processor = ParallelDBSCANProcessor(
        chunk_size=args.chunk_size,
        similarity_threshold=args.threshold,
        max_candidates=args.max_candidates,
        n_workers=args.workers,
        overlap_size=args.overlap
    )
    
    # Запускаем обработку
    result_file = processor.process_dataset_parallel(args.input_file, output_file)
    print(f"✅ Parallel processing completed! Results saved to: {result_file}")


if __name__ == "__main__":
    main()
