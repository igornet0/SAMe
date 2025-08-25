#!/usr/bin/env python3
"""
Основной скрипт для генерации иерархического дерева аналогов и дублей
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys
import argparse

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
        logging.FileHandler('tree_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_config(args) -> TreeGeneratorConfig:
    """Создание конфигурации на основе аргументов командной строки"""
    
    # Конфигурация поиска
    search_config = EnhancedDBSCANConfig(
        eps=args.eps,
        min_samples=args.min_samples,
        similarity_threshold=args.similarity_threshold,
        max_features=args.max_features,
        max_candidates=args.max_candidates
    )
    
    # Конфигурация дерева
    tree_config = TreeBuilderConfig(
        min_similarity_for_tree=args.min_similarity_for_tree,
        max_tree_depth=args.max_tree_depth,
        max_children_per_node=args.max_children_per_node,
        representative_selection=args.representative_selection
    )
    
    # Общая конфигурация
    config = TreeGeneratorConfig(
        search_config=search_config,
        tree_config=tree_config,
        output_dir=args.output_dir,
        tree_filename=args.tree_filename,
        save_intermediate_results=args.save_intermediate
    )
    
    return config


async def main():
    """Основная функция"""
    
    parser = argparse.ArgumentParser(description='Генерация иерархического дерева аналогов')
    
    # Основные параметры
    parser.add_argument('input_file', help='Путь к Excel файлу с каталогом')
    parser.add_argument('--output-dir', default='output', help='Выходная директория')
    parser.add_argument('--tree-filename', default='search_trees.txt', help='Имя файла дерева')
    
    # Параметры DBSCAN
    parser.add_argument('--eps', type=float, default=0.3, help='Параметр eps для DBSCAN')
    parser.add_argument('--min-samples', type=int, default=2, help='Минимальное количество образцов для DBSCAN')
    parser.add_argument('--similarity-threshold', type=float, default=0.6, help='Порог схожести')
    parser.add_argument('--max-features', type=int, default=10000, help='Максимальное количество признаков')
    parser.add_argument('--max-candidates', type=int, default=10, help='Максимальное количество кандидатов')
    
    # Параметры дерева
    parser.add_argument('--min-similarity-for-tree', type=float, default=0.5, help='Минимальная схожесть для включения в дерево')
    parser.add_argument('--max-tree-depth', type=int, default=5, help='Максимальная глубина дерева')
    parser.add_argument('--max-children-per-node', type=int, default=20, help='Максимальное количество детей на узел')
    parser.add_argument('--representative-selection', choices=['shortest', 'most_similar', 'random'], 
                       default='shortest', help='Способ выбора представителя кластера')
    
    # Дополнительные параметры
    parser.add_argument('--save-intermediate', action='store_true', help='Сохранять промежуточные результаты')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting tree generation process...")
    
    # Проверяем существование входного файла
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        # Создаем конфигурацию
        config = create_config(args)
        
        # Создаем генератор
        generator = TreeGenerator(config)
        
        # Обрабатываем файл
        logger.info(f"Processing file: {args.input_file}")
        results = await generator.process_catalog_file(args.input_file, args.output_dir)
        
        # Выводим результаты
        logger.info("=" * 50)
        logger.info("GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        stats = results['statistics']
        logger.info(f"Tree file: {results['tree_file']}")
        logger.info(f"Total records: {stats['total_records']:,}")
        logger.info(f"Records with analogs: {stats['records_with_analogs']:,}")
        logger.info(f"Coverage: {stats['search_coverage']:.2f}%")
        logger.info(f"Total clusters: {stats['total_clusters']:,}")
        logger.info(f"Noise points: {stats['noise_points']:,}")
        logger.info(f"Tree nodes: {stats['tree_nodes']:,}")
        logger.info(f"Tree relations: {stats['tree_relations']:,}")
        logger.info(f"Root nodes: {stats['root_nodes']:,}")
        
        # Выводим типы отношений
        logger.info("\nRelation types:")
        for rel_type, count in stats['relation_types'].items():
            logger.info(f"  {rel_type}: {count:,}")
        
        # Генерируем отчет о покрытии
        coverage_report = generator.get_coverage_report(results['search_results'])
        coverage_file = Path(args.output_dir) / "coverage_report.md"
        with open(coverage_file, 'w', encoding='utf-8') as f:
            f.write(coverage_report)
        
        logger.info(f"\nCoverage report saved to: {coverage_file}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during tree generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

