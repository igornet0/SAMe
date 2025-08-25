#!/usr/bin/env python3
"""
Многопроцессная версия для генерации иерархического дерева аналогов на больших наборах данных
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import List, Dict, Tuple, Any
import pickle
import tempfile
import os

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from search_engine.tree_generator import TreeGenerator, TreeGeneratorConfig
from search_engine.enhanced_dbscan_search import EnhancedDBSCANConfig, NoiseProcessorConfig
from search_engine.hierarchy_tree_builder import TreeBuilderConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiprocess_tree_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def process_chunk_worker(chunk_data: Tuple[int, pd.DataFrame, Dict]) -> Dict[str, Any]:
    """Worker функция для обработки части данных"""
    chunk_id, chunk_df, config_dict = chunk_data
    
    try:
        # Восстанавливаем конфигурацию
        search_config = EnhancedDBSCANConfig(
            eps=config_dict['search_config']['eps'],
            min_samples=config_dict['search_config']['min_samples'],
            similarity_threshold=config_dict['search_config']['similarity_threshold'],
            max_features=config_dict['search_config']['max_features'],
            max_candidates=config_dict['search_config']['max_candidates'],
            noise_config=NoiseProcessorConfig(
                max_nearest_clusters=config_dict['search_config']['noise_config']['max_nearest_clusters'],
                min_analogs_per_noise=config_dict['search_config']['noise_config']['min_analogs_per_noise'],
                max_analogs_per_noise=config_dict['search_config']['noise_config']['max_analogs_per_noise'],
                noise_assignment_threshold=config_dict['search_config']['noise_config']['noise_assignment_threshold'],
                similarity_threshold=config_dict['search_config']['noise_config']['similarity_threshold']
            )
        )
        tree_config = TreeBuilderConfig(
            min_similarity_for_tree=config_dict['tree_config']['min_similarity_for_tree'],
            max_tree_depth=config_dict['tree_config']['max_tree_depth'],
            max_children_per_node=config_dict['tree_config']['max_children_per_node'],
            representative_selection=config_dict['tree_config']['representative_selection'],
            include_noise_points=config_dict['tree_config']['include_noise_points']
        )
        
        # Создаем генератор для чанка
        chunk_config = TreeGeneratorConfig(
            search_config=search_config,
            tree_config=tree_config,
            output_dir=f"temp_chunk_{chunk_id}",
            tree_filename=f"chunk_{chunk_id}_trees.txt",
            save_intermediate_results=False
        )
        
        generator = TreeGenerator(chunk_config)
        
        # Обрабатываем чанк
        logger.info(f"Worker {chunk_id}: Processing {len(chunk_df)} records...")
        
        # Создаем временную директорию для чанка
        temp_dir = Path(f"temp_chunk_{chunk_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Сохраняем чанк во временный файл
        chunk_file = temp_dir / f"chunk_{chunk_id}.xlsx"
        chunk_df.to_excel(chunk_file, index=False)
        
        # Обрабатываем чанк
        results = asyncio.run(generator.process_catalog_file(str(chunk_file), str(temp_dir)))
        
        # Очищаем временные файлы
        chunk_file.unlink(missing_ok=True)
        
        return {
            'chunk_id': chunk_id,
            'success': True,
            'results': results,
            'temp_dir': str(temp_dir)
        }
        
    except Exception as e:
        logger.error(f"Worker {chunk_id} failed: {e}")
        return {
            'chunk_id': chunk_id,
            'success': False,
            'error': str(e),
            'temp_dir': f"temp_chunk_{chunk_id}" if 'temp_dir' in locals() else None
        }


def merge_chunk_results(chunk_results: List[Dict], output_dir: str) -> Dict[str, Any]:
    """Объединяет результаты обработки чанков"""
    logger.info("Merging chunk results...")
    
    # Собираем все результаты
    all_search_results = []
    all_trees = []
    total_records = 0
    total_analogs = 0
    
    for result in chunk_results:
        if result['success']:
            chunk_data = result['results']
            all_search_results.extend(chunk_data['search_results'])
            all_trees.extend(chunk_data['trees'])
            total_records += chunk_data['statistics']['total_records']
            total_analogs += chunk_data['statistics']['records_with_analogs']
    
    # Создаем объединенную статистику
    merged_stats = {
        'total_records': total_records,
        'records_with_analogs': total_analogs,
        'search_coverage': (total_analogs / total_records * 100) if total_records > 0 else 0,
        'total_chunks': len([r for r in chunk_results if r['success']]),
        'failed_chunks': len([r for r in chunk_results if not r['success']])
    }
    
    # Сохраняем объединенные результаты
    merged_file = Path(output_dir) / "merged_search_results.pkl"
    with open(merged_file, 'wb') as f:
        pickle.dump({
            'search_results': all_search_results,
            'trees': all_trees,
            'statistics': merged_stats
        }, f)
    
    logger.info(f"Merged results saved to {merged_file}")
    return merged_stats


async def generate_multiprocess_search_trees():
    """Генерация деревьев с использованием многопроцессной обработки"""
    
    logger.info("Starting multiprocess search trees generation...")
    
    # Проверяем существование входного файла
    input_file = "src/data/input/main_dataset.xlsx"
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    try:
        # Загружаем полный набор данных
        logger.info("Loading full dataset...")
        catalog_df = pd.read_excel(input_file)
        logger.info(f"Loaded {len(catalog_df):,} records from {input_file}")
        
        # Определяем количество worker'ов
        num_workers = min(mp.cpu_count(), 8)  # Ограничиваем максимум 8 worker'ами
        logger.info(f"Using {num_workers} workers for processing")
        
        # Разбиваем данные на чанки
        chunk_size = len(catalog_df) // num_workers
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(catalog_df)
            chunk_df = catalog_df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk_df)
            logger.info(f"Chunk {i}: {len(chunk_df)} records")
        
        # Создаем конфигурацию для worker'ов
        search_config = EnhancedDBSCANConfig(
            eps=0.3,
            min_samples=2,
            similarity_threshold=0.5,
            max_features=2000,
            max_candidates=5,
            noise_config=NoiseProcessorConfig(
                max_nearest_clusters=3,
                min_analogs_per_noise=1,
                max_analogs_per_noise=5,
                noise_assignment_threshold=0.2,
                similarity_threshold=0.4
            )
        )
        
        tree_config = TreeBuilderConfig(
            min_similarity_for_tree=0.4,
            max_tree_depth=3,
            max_children_per_node=10,
            representative_selection="shortest",
            include_noise_points=True
        )
        
        # Подготавливаем конфигурацию для передачи в worker'ы
        config_dict = {
            'search_config': {
                'eps': search_config.eps,
                'min_samples': search_config.min_samples,
                'similarity_threshold': search_config.similarity_threshold,
                'max_features': search_config.max_features,
                'max_candidates': search_config.max_candidates,
                'noise_config': {
                    'max_nearest_clusters': search_config.noise_config.max_nearest_clusters,
                    'min_analogs_per_noise': search_config.noise_config.min_analogs_per_noise,
                    'max_analogs_per_noise': search_config.noise_config.max_analogs_per_noise,
                    'noise_assignment_threshold': search_config.noise_config.noise_assignment_threshold,
                    'similarity_threshold': search_config.noise_config.similarity_threshold
                }
            },
            'tree_config': {
                'min_similarity_for_tree': tree_config.min_similarity_for_tree,
                'max_tree_depth': tree_config.max_tree_depth,
                'max_children_per_node': tree_config.max_children_per_node,
                'representative_selection': tree_config.representative_selection,
                'include_noise_points': tree_config.include_noise_points
            }
        }
        
        # Подготавливаем данные для worker'ов
        worker_data = [(i, chunk, config_dict) for i, chunk in enumerate(chunks)]
        
        # Запускаем многопроцессную обработку
        logger.info("Starting multiprocess processing...")
        chunk_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Запускаем все задачи
            future_to_chunk = {
                executor.submit(process_chunk_worker, data): data[0] 
                for data in worker_data
            }
            
            # Собираем результаты по мере завершения
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    if result['success']:
                        logger.info(f"Chunk {chunk_id} completed successfully")
                    else:
                        logger.error(f"Chunk {chunk_id} failed: {result['error']}")
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} generated exception: {e}")
                    chunk_results.append({
                        'chunk_id': chunk_id,
                        'success': False,
                        'error': str(e),
                        'temp_dir': None
                    })
        
        # Объединяем результаты
        output_dir = "output_multiprocess"
        Path(output_dir).mkdir(exist_ok=True)
        
        merged_stats = merge_chunk_results(chunk_results, output_dir)
        
        # Создаем финальное дерево из объединенных результатов
        logger.info("Creating final merged tree...")
        
        # Загружаем объединенные результаты
        merged_file = Path(output_dir) / "merged_search_results.pkl"
        with open(merged_file, 'rb') as f:
            merged_data = pickle.load(f)
        
        # Создаем финальный генератор
        final_config = TreeGeneratorConfig(
            search_config=search_config,
            tree_config=tree_config,
            output_dir=output_dir,
            tree_filename="final_search_trees.txt",
            save_intermediate_results=False
        )
        
        final_generator = TreeGenerator(final_config)
        
        # Генерируем финальное дерево
        final_tree = await final_generator.generate_search_trees(
            merged_data['search_results'], 
            output_dir
        )
        
        # Выводим финальные результаты
        logger.info("=" * 80)
        logger.info("MULTIPROCESS GENERATION RESULTS:")
        logger.info("=" * 80)
        logger.info(f"  Total records: {merged_stats['total_records']:,}")
        logger.info(f"  Records with analogs: {merged_stats['total_analogs']:,}")
        logger.info(f"  Coverage: {merged_stats['search_coverage']:.2f}%")
        logger.info(f"  Total chunks: {merged_stats['total_chunks']}")
        logger.info(f"  Failed chunks: {merged_stats['failed_chunks']}")
        logger.info(f"  Workers used: {num_workers}")
        
        # Проверяем создание финального файла
        final_tree_file = Path(output_dir) / "final_search_trees.txt"
        if final_tree_file.exists():
            logger.info(f"\nFinal tree file created: {final_tree_file}")
            
            # Показываем статистику файла
            file_size = final_tree_file.stat().st_size
            logger.info(f"Final tree file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            # Показываем первые несколько строк
            with open(final_tree_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:50]
                logger.info("\nFirst 50 lines of final tree:")
                logger.info("-" * 50)
                for line in lines:
                    print(line.rstrip())
                logger.info("-" * 50)
        
        # Очищаем временные директории
        logger.info("Cleaning up temporary files...")
        for result in chunk_results:
            if result['temp_dir'] and Path(result['temp_dir']).exists():
                import shutil
                shutil.rmtree(result['temp_dir'], ignore_errors=True)
        
        logger.info("=" * 80)
        logger.info("MULTIPROCESS GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Multiprocess generation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Устанавливаем метод запуска для multiprocessing
    mp.set_start_method('spawn', force=True)
    
    success = asyncio.run(generate_multiprocess_search_trees())
    sys.exit(0 if success else 1)
