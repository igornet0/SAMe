#!/usr/bin/env python3
"""
Экстренный поисковый скрипт для больших датасетов
Быстрое решение проблемы с памятью в DBSCAN
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
from collections import Counter
import gc
import psutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def create_word_index(df: pd.DataFrame, text_column: str) -> Dict[str, List[int]]:
    """Создание инвертированного индекса для быстрого поиска"""
    logger.info("Creating word index for fast search...")
    
    word_index = {}
    
    for idx, row in df.iterrows():
        text = normalize_text_fast(str(row.get(text_column, '')))
        words = text.split()
        
        for word in words:
            if word not in word_index:
                word_index[word] = []
            word_index[word].append(idx)
    
    logger.info(f"Word index created with {len(word_index)} unique words")
    return word_index


def find_candidates_fast(query_text: str, word_index: Dict[str, List[int]], 
                        df: pd.DataFrame, max_candidates: int = 100) -> List[int]:
    """Быстрый поиск кандидатов через инвертированный индекс"""
    query_words = normalize_text_fast(query_text).split()
    
    if not query_words:
        return []
    
    # Находим записи, содержащие слова из запроса
    candidate_scores = Counter()
    
    for word in query_words:
        if word in word_index:
            for idx in word_index[word]:
                candidate_scores[idx] += 1
    
    # Сортируем кандидатов по количеству совпадающих слов
    candidates = [idx for idx, score in candidate_scores.most_common(max_candidates)]
    
    return candidates


def emergency_search(input_file: str, output_file: str, similarity_threshold: float = 0.4, 
                    max_records: int = None, batch_size: int = 1000):
    """Экстренный поиск аналогов без DBSCAN"""
    
    logger.info(f"🚨 Emergency search started for: {input_file}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    
    # Загружаем данные
    logger.info("Loading data...")
    df = pd.read_csv(input_file)
    
    if max_records:
        df = df.head(max_records)
        logger.info(f"Limited to {max_records} records")
    
    logger.info(f"Loaded {len(df)} records")
    
    # Определяем колонку для поиска
    text_column = 'Normalized_Name' if 'Normalized_Name' in df.columns else 'Raw_Name'
    logger.info(f"Using column: {text_column}")
    
    # Создаем инвертированный индекс
    word_index = create_word_index(df, text_column)
    
    # Результаты
    all_results = []
    
    # Обрабатываем батчами
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches}: records {start_idx}-{end_idx-1}")
        
        batch_results = []
        
        for idx in range(start_idx, end_idx):
            query_row = df.iloc[idx]
            query_text = str(query_row.get(text_column, ''))
            
            if not query_text or query_text == 'nan':
                continue
            
            # Быстрый поиск кандидатов
            candidates = find_candidates_fast(query_text, word_index, df, max_candidates=50)
            
            # Вычисляем схожесть для кандидатов
            for candidate_idx in candidates:
                if candidate_idx == idx:  # Пропускаем сам товар
                    continue
                
                candidate_row = df.iloc[candidate_idx]
                candidate_text = str(candidate_row.get(text_column, ''))
                
                # Вычисляем схожесть
                similarity = jaccard_similarity(
                    normalize_text_fast(query_text),
                    normalize_text_fast(candidate_text)
                )
                
                if similarity >= similarity_threshold:
                    relation_type = determine_relation_type(similarity)
                    
                    result = {
                        'Код': query_row.get('Код', ''),
                        'Raw_Name': query_row.get('Raw_Name', ''),
                        'Candidate_Name': candidate_row.get('Raw_Name', ''),
                        'Similarity_Score': round(similarity, 4),
                        'Relation_Type': relation_type,
                        'Suggested_Category': candidate_row.get('Группа', 'Неопределено'),
                        'Final_Decision': '',
                        'Comment': f"Жаккар схожесть: {similarity:.3f}; Быстрый поиск",
                        'Original_Category': query_row.get('Группа', ''),
                        'Candidate_Код': candidate_row.get('Код', ''),
                        'Original_Code': query_row.get('Код', ''),
                        'Search_Engine': 'EmergencySearch'
                    }
                    
                    batch_results.append(result)
        
        all_results.extend(batch_results)
        
        # Проверяем память
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        logger.info(f"Memory usage: {memory_gb:.1f}GB, Found {len(batch_results)} relationships in batch")
        
        if memory_gb > 6.0:  # Если память больше 6GB, делаем сборку мусора
            gc.collect()
    
    # Сохраняем результаты
    logger.info(f"Saving {len(all_results)} results to {output_file}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✅ Results saved to: {output_file}")
    else:
        logger.warning("No results found!")
    
    logger.info(f"🎉 Emergency search completed! Found {len(all_results)} analog relationships")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Emergency analog search for large datasets')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Similarity threshold')
    parser.add_argument('-l', '--limit', type=int, help='Limit number of records to process')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Определяем выходной файл
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = f"emergency_search_results_{input_path.stem}.csv"
    
    # Запускаем экстренный поиск
    emergency_search(
        input_file=args.input_file,
        output_file=output_file,
        similarity_threshold=args.threshold,
        max_records=args.limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
