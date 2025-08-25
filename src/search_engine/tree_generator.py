#!/usr/bin/env python3
"""
Tree Generator для генерации файла search_trees.txt с иерархическим деревом аналогов
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio

from .enhanced_dbscan_search import EnhancedDBSCANSearch, EnhancedDBSCANConfig
from .hierarchy_tree_builder import HierarchyTreeBuilder, TreeBuilderConfig

logger = logging.getLogger(__name__)


@dataclass
class TreeGeneratorConfig:
    """Конфигурация генератора деревьев"""
    # Конфигурации компонентов
    search_config: EnhancedDBSCANConfig = None
    tree_config: TreeBuilderConfig = None
    
    # Параметры генерации
    output_dir: str = "output"
    tree_filename: str = "search_trees.txt"
    include_statistics: bool = True
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        if self.search_config is None:
            self.search_config = EnhancedDBSCANConfig()
        if self.tree_config is None:
            self.tree_config = TreeBuilderConfig()


class TreeGenerator:
    """Генератор иерархических деревьев аналогов"""
    
    def __init__(self, config: TreeGeneratorConfig = None):
        self.config = config or TreeGeneratorConfig()
        self.search_engine = EnhancedDBSCANSearch(self.config.search_config)
        self.tree_builder = HierarchyTreeBuilder(self.config.tree_config)
        
        # Создаем выходную директорию
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("TreeGenerator initialized")

    async def generate_search_trees(self, catalog_df: pd.DataFrame, 
                                  output_path: str = None) -> Dict[str, Any]:
        """Генерация файла search_trees.txt с иерархическим деревом"""
        
        logger.info("Starting search trees generation...")
        
        # Определяем путь к файлу
        if output_path is None:
            output_path = Path(self.config.output_dir) / self.config.tree_filename
        else:
            # Если передан путь к директории, добавляем имя файла
            if Path(output_path).is_dir() or not Path(output_path).suffix:
                output_path = Path(output_path) / self.config.tree_filename
        
        # 1. Выполняем поиск всех аналогов
        logger.info("Step 1: Finding all analogs...")
        search_results = await self.search_engine.find_all_analogs(catalog_df)
        
        # 2. Строим иерархическое дерево
        logger.info("Step 2: Building hierarchy tree...")
        hierarchy_tree = self.tree_builder.build_hierarchy_tree(
            catalog_df, 
            search_results['clusters'], 
            search_results['noise_assignments']
        )
        
        # 3. Генерируем текстовое представление дерева
        logger.info("Step 3: Generating tree text...")
        tree_text = self._generate_tree_text(hierarchy_tree)
        
        # 4. Сохраняем в файл
        logger.info(f"Step 4: Saving tree to {output_path}")
        self._save_tree_to_file(tree_text, str(output_path))
        
        # 5. Генерируем статистику
        statistics = self._generate_statistics(search_results, hierarchy_tree)
        
        # 6. Сохраняем промежуточные результаты если нужно
        if self.config.save_intermediate_results:
            await self._save_intermediate_results(search_results, output_path)
        
        logger.info("Search trees generation completed")
        
        return {
            'tree_file': str(output_path),
            'statistics': statistics,
            'search_results': search_results,
            'hierarchy_tree': hierarchy_tree
        }

    def _generate_tree_text(self, hierarchy_tree: List) -> str:
        """Генерация текстового представления дерева"""
        tree_text = self.tree_builder.generate_tree_text(hierarchy_tree)
        
        # Добавляем заголовок
        header = "# Иерархическое дерево аналогов и дублей\n"
        header += "# Сгенерировано системой SAMe\n\n"
        
        return header + tree_text

    def _save_tree_to_file(self, tree_text: str, output_path: str):
        """Сохранение дерева в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        logger.info(f"Tree saved to: {output_path}")

    def _generate_statistics(self, search_results: Dict, hierarchy_tree: List) -> Dict[str, Any]:
        """Генерация статистики"""
        # Статистика поиска
        search_stats = search_results.get('statistics', {})
        
        # Статистика дерева
        tree_stats = self.tree_builder.get_tree_statistics()
        
        # Общая статистика
        total_statistics = {
            'search_coverage': search_stats.get('coverage_percentage', 0),
            'total_records': search_stats.get('total_records', 0),
            'records_with_analogs': search_stats.get('records_with_analogs', 0),
            'total_clusters': search_stats.get('total_clusters', 0),
            'noise_points': search_stats.get('noise_points', 0),
            'tree_nodes': tree_stats.get('total_nodes', 0),
            'tree_relations': tree_stats.get('total_relations', 0),
            'tree_max_depth': tree_stats.get('max_depth', 0),
            'relation_types': tree_stats.get('relation_types', {}),
            'root_nodes': tree_stats.get('root_nodes', 0)
        }
        
        return total_statistics

    async def _save_intermediate_results(self, search_results: Dict, output_path: str):
        """Сохранение промежуточных результатов"""
        base_path = Path(output_path).parent
        
        # Функция для сериализации numpy типов
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Сохраняем результаты поиска
        search_file = base_path / "search_results.json"
        import json
        serializable_results = convert_numpy_types(search_results)
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Сохраняем статистику
        stats_file = base_path / "statistics.json"
        serializable_stats = convert_numpy_types(search_results.get('statistics', {}))
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    async def process_catalog_file(self, input_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Обработка каталога из файла"""
        logger.info(f"Processing catalog file: {input_path}")
        
        # Загружаем данные
        catalog_df = pd.read_excel(input_path)
        logger.info(f"Loaded {len(catalog_df)} records from {input_path}")
        
        # Устанавливаем выходную директорию
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Генерируем дерево
        results = await self.generate_search_trees(catalog_df, output_dir)
        
        return results

    def get_coverage_report(self, search_results: Dict) -> str:
        """Генерация отчета о покрытии"""
        stats = search_results.get('statistics', {})
        
        report = f"""
# Отчет о покрытии поиска аналогов

## Общая статистика
- Всего записей: {stats.get('total_records', 0):,}
- Записей с аналогами: {stats.get('records_with_analogs', 0):,}
- Покрытие: {stats.get('coverage_percentage', 0):.2f}%

## Кластеризация
- Всего кластеров: {stats.get('total_clusters', 0):,}
- Шумовых точек: {stats.get('noise_points', 0):,}
- Средний размер кластера: {stats.get('avg_cluster_size', 0):.1f}

## Качество результатов
- Высокая схожесть (>0.8): {self._count_high_similarity(search_results):,}
- Средняя схожесть (0.6-0.8): {self._count_medium_similarity(search_results):,}
- Низкая схожесть (<0.6): {self._count_low_similarity(search_results):,}
"""
        
        return report

    def _count_high_similarity(self, search_results: Dict) -> int:
        """Подсчет результатов с высокой схожестью"""
        count = 0
        for analogs in search_results.get('results', {}).values():
            for analog in analogs:
                if analog.get('similarity', 0) > 0.8:
                    count += 1
        return count

    def _count_medium_similarity(self, search_results: Dict) -> int:
        """Подсчет результатов со средней схожестью"""
        count = 0
        for analogs in search_results.get('results', {}).values():
            for analog in analogs:
                similarity = analog.get('similarity', 0)
                if 0.6 <= similarity <= 0.8:
                    count += 1
        return count

    def _count_low_similarity(self, search_results: Dict) -> int:
        """Подсчет результатов с низкой схожестью"""
        count = 0
        for analogs in search_results.get('results', {}).values():
            for analog in analogs:
                if analog.get('similarity', 0) < 0.6:
                    count += 1
        return count


# Функция для быстрого запуска
async def generate_search_trees_from_file(input_path: str, output_dir: str = "output", 
                                        config: TreeGeneratorConfig = None) -> Dict[str, Any]:
    """Быстрая функция для генерации деревьев из файла"""
    generator = TreeGenerator(config)
    return await generator.process_catalog_file(input_path, output_dir)

