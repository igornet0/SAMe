#!/usr/bin/env python3
"""
Оптимизированный скрипт для генерации иерархического дерева аналогов на больших наборах данных
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys
import gc

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
        logging.FileHandler('optimized_tree_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def generate_optimized_search_trees():
    """Генерация деревьев на оптимизированном наборе данных"""
    
    logger.info("Starting optimized search trees generation...")
    
    # Проверяем существование входного файла
    input_file = "src/data/input/main_dataset.xlsx"
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    try:
        # Загружаем ограниченный набор данных для тестирования
        logger.info("Loading limited dataset for testing...")
        catalog_df = pd.read_excel(input_file, nrows=10000)  # Ограничиваем до 10k записей
        logger.info(f"Loaded {len(catalog_df):,} records from {input_file}")
        
        # Принудительная очистка памяти
        gc.collect()
        
        # Создаем оптимизированную конфигурацию
        search_config = EnhancedDBSCANConfig(
            eps=0.3,  # Увеличиваем eps для лучшей кластеризации
            min_samples=2,  # Уменьшаем для большего количества кластеров
            similarity_threshold=0.5,  # Снижаем порог для большего покрытия
            max_features=2000,  # Уменьшаем для экономии памяти
            max_candidates=5,  # Уменьшаем количество кандидатов
            # Оптимизированные настройки обработки шумовых точек
            noise_config=NoiseProcessorConfig(
                max_nearest_clusters=3,
                min_analogs_per_noise=1,
                max_analogs_per_noise=5,
                noise_assignment_threshold=0.2,
                similarity_threshold=0.4
            )
        )
        
        tree_config = TreeBuilderConfig(
            min_similarity_for_tree=0.4,  # Снижаем для большего количества связей
            max_tree_depth=3,  # Ограничиваем глубину
            max_children_per_node=10,  # Ограничиваем количество детей
            representative_selection="shortest",
            include_noise_points=True
        )
        
        config = TreeGeneratorConfig(
            search_config=search_config,
            tree_config=tree_config,
            output_dir="output_optimized",
            tree_filename="search_trees_optimized.txt",
            save_intermediate_results=False  # Отключаем для экономии памяти
        )
        
        # Создаем генератор
        generator = TreeGenerator(config)
        
        # Обрабатываем данные
        logger.info("Processing optimized dataset...")
        results = await generator.generate_search_trees(catalog_df, "output_optimized")
        
        # Выводим результаты
        stats = results['statistics']
        logger.info("=" * 80)
        logger.info("OPTIMIZED GENERATION RESULTS:")
        logger.info("=" * 80)
        logger.info(f"  Total records: {stats['total_records']:,}")
        logger.info(f"  Records with analogs: {stats['records_with_analogs']:,}")
        logger.info(f"  Coverage: {stats['search_coverage']:.2f}%")
        logger.info(f"  Total clusters: {stats['total_clusters']:,}")
        logger.info(f"  Noise points: {stats['noise_points']:,}")
        logger.info(f"  Tree nodes: {stats['tree_nodes']:,}")
        logger.info(f"  Tree relations: {stats['tree_relations']:,}")
        logger.info(f"  Root nodes: {stats['root_nodes']:,}")
        
        # Выводим типы отношений
        logger.info("\nRelation types:")
        for rel_type, count in stats['relation_types'].items():
            logger.info(f"  {rel_type}: {count:,}")
        
        # Проверяем создание файла
        tree_file = Path(results['tree_file'])
        if tree_file.exists():
            logger.info(f"\nTree file created successfully: {tree_file}")
            
            # Показываем статистику файла
            file_size = tree_file.stat().st_size
            logger.info(f"Tree file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            # Показываем первые несколько строк
            with open(tree_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:50]
                logger.info("\nFirst 50 lines of tree:")
                logger.info("-" * 50)
                for line in lines:
                    print(line.rstrip())
                logger.info("-" * 50)
        else:
            logger.error("Tree file was not created")
            return False
        
        # Генерируем отчет о покрытии
        coverage_report = generator.get_coverage_report(results['search_results'])
        coverage_file = Path("output_optimized") / "coverage_report.md"
        with open(coverage_file, 'w', encoding='utf-8') as f:
            f.write(coverage_report)
        
        logger.info(f"\nCoverage report saved to: {coverage_file}")
        logger.info("=" * 80)
        logger.info("OPTIMIZED GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(generate_optimized_search_trees())
    sys.exit(0 if success else 1)

