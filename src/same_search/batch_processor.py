"""
Пакетный процессор для обработки больших датасетов
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Generator
from pathlib import Path
import asyncio
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Пакетный процессор для обработки больших датасетов"""
    
    def __init__(self, batch_size: int = 1000, overlap_size: int = 50):
        """
        Args:
            batch_size: Размер пакета для обработки
            overlap_size: Размер перекрытия между пакетами для поиска аналогов
        """
        self.batch_size = batch_size
        self.overlap_size = overlap_size
        
    def create_batches(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Создание пакетов с перекрытием"""
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, self.batch_size - self.overlap_size):
            end_idx = min(start_idx + self.batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # Добавляем информацию о пакете
            batch_df = batch_df.copy()
            batch_df['_batch_start'] = start_idx
            batch_df['_batch_end'] = end_idx
            batch_df['_batch_id'] = start_idx // (self.batch_size - self.overlap_size)
            
            yield batch_df
    
    async def process_batch(self, batch_df: pd.DataFrame, processor, 
                          similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """Обработка одного пакета"""
        try:
            # Получаем batch_id безопасным способом
            try:
                batch_id = batch_df.get('_batch_id', pd.Series([0])).iloc[0]
                batch_start = batch_df.get('_batch_start', pd.Series([0])).iloc[0]
                batch_end = batch_df.get('_batch_end', pd.Series([0])).iloc[0]
            except (IndexError, KeyError, AttributeError):
                batch_id = 0
                batch_start = 0
                batch_end = len(batch_df)
            
            logger.info(f"Processing batch {batch_id} "
                       f"({len(batch_df)} records)")
            
            # Обработка пакета
            processor.processed_df = batch_df
            duplicate_groups, analog_groups = await processor.find_duplicates_and_analogs(
                use_optimized=True, use_multi_engine=False
            )
            
            # Принудительная очистка памяти
            gc.collect()
            
            return {
                'batch_id': batch_id,
                'batch_start': batch_start,
                'batch_end': batch_end,
                'duplicate_groups': duplicate_groups,
                'analog_groups': analog_groups,
                'trees': [],
                'processed_df': batch_df
            }
        
        except Exception as e:
            # Получаем batch_id безопасным способом
            try:
                batch_id = batch_df.get('_batch_id', pd.Series([0])).iloc[0]
                batch_start = batch_df.get('_batch_start', pd.Series([0])).iloc[0]
                batch_end = batch_df.get('_batch_end', pd.Series([0])).iloc[0]
            except (IndexError, KeyError, AttributeError):
                batch_id = 0
                batch_start = 0
                batch_end = len(batch_df)
            
            logger.error(f"Error processing batch {batch_id}: {e}")
            return {
                'batch_id': batch_id,
                'batch_start': batch_start,
                'batch_end': batch_end,
                'duplicate_groups': [],
                'analog_groups': [],
                'trees': [],
                'processed_df': batch_df,
                'error': str(e)
            }
    
    async def process_large_dataset(self, df: pd.DataFrame, processor,
                                  similarity_threshold: float = 0.3,
                                  max_concurrent_batches: int = 2) -> Dict[str, Any]:
        """Обработка большого датасета пакетами"""
        logger.info(f"Starting batch processing of {len(df)} records "
                   f"in batches of {self.batch_size}")
        
        all_results = {
            'duplicate_groups': [],
            'analog_groups': [],
            'processed_data': [],
            'batch_stats': []
        }
        
        # Создаем пакеты
        batches = list(self.create_batches(df))
        total_batches = len(batches)
        
        logger.info(f"Created {total_batches} batches for processing")
        
        # Проверяем, что пакеты созданы корректно
        if total_batches == 0:
            logger.warning("No batches created, returning empty results")
            return {
                'duplicate_groups': [],
                'analog_groups': [],
                'final_processed_df': df,
                'batch_stats': []
            }
        
        # Обрабатываем пакеты с ограничением параллельности
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def process_with_semaphore(batch_df):
            try:
                async with semaphore:
                    return await self.process_batch(batch_df, processor, similarity_threshold)
            except Exception as e:
                logger.error(f"Error in process_with_semaphore: {e}")
                # Получаем batch_id безопасным способом
                try:
                    batch_id = batch_df.get('_batch_id', pd.Series([0])).iloc[0]
                    batch_start = batch_df.get('_batch_start', pd.Series([0])).iloc[0]
                    batch_end = batch_df.get('_batch_end', pd.Series([0])).iloc[0]
                except (IndexError, KeyError, AttributeError):
                    batch_id = 0
                    batch_start = 0
                    batch_end = len(batch_df)
                
                return {
                    'batch_id': batch_id,
                    'batch_start': batch_start,
                    'batch_end': batch_end,
                    'duplicate_groups': [],
                    'analog_groups': [],
                    'trees': [],
                    'processed_df': batch_df,
                    'error': str(e)
                }
        
        # Обрабатываем пакеты
        tasks = [process_with_semaphore(batch) for batch in batches]
        
        for i, task in enumerate(tqdm(asyncio.as_completed(tasks), 
                                    total=total_batches, 
                                    desc="Processing batches")):
            try:
                result = await task
                
                # Проверяем, что результат валиден
                if not isinstance(result, dict):
                    logger.warning(f"Invalid result type: {type(result)}")
                    continue
                
                # Проверяем наличие ошибки в результате
                if result.get('error'):
                    logger.warning(f"Batch {result.get('batch_id', 'unknown')} had error: {result.get('error')}")
                
                # Собираем результаты
                all_results['duplicate_groups'].extend(result.get('duplicate_groups', []))
                all_results['analog_groups'].extend(result.get('analog_groups', []))
                all_results['processed_data'].append(result.get('processed_df', pd.DataFrame()))
                
                # Статистика пакета
                batch_stats = {
                    'batch_id': result.get('batch_id', 0),
                    'records_processed': len(result.get('processed_df', pd.DataFrame())),
                    'duplicates_found': len(result.get('duplicate_groups', [])),
                    'analogs_found': len(result.get('analog_groups', [])),
                    'error': result.get('error')
                }
                all_results['batch_stats'].append(batch_stats)
                
                # Принудительная очистка памяти после каждого пакета
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                continue
        
        # Объединяем все обработанные данные
        if all_results['processed_data']:
            try:
                all_results['final_processed_df'] = pd.concat(
                    all_results['processed_data'], ignore_index=True
                )
                # Удаляем служебные колонки
                all_results['final_processed_df'] = all_results['final_processed_df'].drop(
                    columns=['_batch_start', '_batch_end', '_batch_id'], errors='ignore'
                )
            except Exception as e:
                logger.error(f"Error concatenating processed data: {e}")
                all_results['final_processed_df'] = pd.DataFrame()
        else:
            all_results['final_processed_df'] = pd.DataFrame()
        
        logger.info(f"Batch processing completed. "
                   f"Total duplicates: {len(all_results['duplicate_groups'])}, "
                   f"Total analogs: {len(all_results['analog_groups'])}, "
                   f"Batches with errors: {len([b for b in all_results['batch_stats'] if b.get('error')])}")
        
        return all_results
    
    def get_memory_usage_stats(self) -> Dict[str, float]:
        """Получение статистики использования памяти"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def estimate_processing_time(self, total_records: int) -> Dict[str, Any]:
        """Оценка времени обработки"""
        total_batches = (total_records + self.batch_size - self.overlap_size - 1) // (self.batch_size - self.overlap_size)
        
        # Примерная оценка времени (можно настроить на основе реальных данных)
        estimated_time_per_batch = 30  # секунд
        total_estimated_time = total_batches * estimated_time_per_batch
        
        return {
            'total_batches': total_batches,
            'estimated_time_seconds': total_estimated_time,
            'estimated_time_minutes': total_estimated_time / 60,
            'estimated_time_hours': total_estimated_time / 3600,
            'batch_size': self.batch_size,
            'overlap_size': self.overlap_size
        }
