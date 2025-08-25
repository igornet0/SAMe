#!/usr/bin/env python3
"""
Финальный скрипт для генерации иерархического дерева аналогов на полном наборе данных
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from search_engine.tree_generator import TreeGenerator, TreeGeneratorConfig
from search_engine.enhanced_dbscan_search import EnhancedDBSCANConfig
from search_engine.hierarchy_tree_builder import TreeBuilderConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_tree_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def generate_full_search_trees():
    """Генерация деревьев на полном наборе данных"""
    
    logger.info("Starting full search trees generation...")
    
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
        
        # Создаем конфигурацию для полного набора данных
        from search_engine.enhanced_dbscan_search import NoiseProcessorConfig
        
        search_config = EnhancedDBSCANConfig(
            eps=0.25,  # Уменьшаем eps для более точной кластеризации
            min_samples=3,  # Увеличиваем минимальное количество образцов
            similarity_threshold=0.6,  # Повышаем порог схожести
            max_features=5000,  # Оптимальное количество признаков
            max_candidates=8,  # Увеличиваем количество кандидатов
            # Настройки обработки шумовых точек
            noise_config=NoiseProcessorConfig(
                max_nearest_clusters=5,
                min_analogs_per_noise=2,
                max_analogs_per_noise=8,
                noise_assignment_threshold=0.3,
                similarity_threshold=0.5
            )
        )
        
        tree_config = TreeBuilderConfig(
            min_similarity_for_tree=0.5,  # Повышаем минимальную схожесть для дерева
            max_tree_depth=4,  # Увеличиваем глубину дерева
            max_children_per_node=15,  # Оптимальное количество детей
            representative_selection="shortest",  # Используем самое короткое название
            include_noise_points=True
        )
        
        config = TreeGeneratorConfig(
            search_config=search_config,
            tree_config=tree_config,
            output_dir="output_full",
            tree_filename="search_trees.txt",
            save_intermediate_results=True
        )
        
        # Создаем генератор
        generator = TreeGenerator(config)
        
        # Обрабатываем данные
        logger.info("Processing full dataset...")
        results = await generator.process_catalog_file(input_file, "output_full")
        
        # Выводим результаты
        stats = results['statistics']
        logger.info("=" * 80)
        logger.info("FULL DATASET GENERATION RESULTS:")
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
                lines = f.readlines()[:30]
                logger.info("\nFirst 30 lines of tree:")
                logger.info("-" * 50)
                for line in lines:
                    print(line.rstrip())
                logger.info("-" * 50)
        else:
            logger.error("Tree file was not created")
            return False
        
        # Генерируем отчет о покрытии
        coverage_report = generator.get_coverage_report(results['search_results'])
        coverage_file = Path("output_full") / "coverage_report.md"
        with open(coverage_file, 'w', encoding='utf-8') as f:
            f.write(coverage_report)
        
        logger.info(f"\nCoverage report saved to: {coverage_file}")
        logger.info("=" * 80)
        logger.info("FULL DATASET GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(generate_full_search_trees())
    sys.exit(0 if success else 1)
