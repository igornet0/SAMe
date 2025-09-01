
import asyncio
import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import pickle
import time
import tempfile
import os
import re
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from data_manager.DataManager import DataManager
from same_clear.text_processing.enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessorConfig
from same_clear.parameter_extraction.regex_extractor import RegexParameterExtractor
from same_clear.parameter_extraction.enhanced_parameter_extractor import EnhancedParameterExtractor
from same_search.performance.optimized_search_engine import OptimizedSearchEngine, PerformanceConfig
from same_search.categorization.category_classifier import CategoryClassifier, CategoryClassifierConfig
from same_search.duplicate_analog_search import (
    DuplicateAnalogSearchEngine, 
    DuplicateSearchConfig, 
    AnalogSearchConfig
)
from same_search.multi_engine_search import MultiEngineSearch, MultiEngineConfig
from same_search.simple_multi_engine_search import SimpleMultiEngineSearch, SimpleMultiEngineConfig
from same_search.improved_multi_engine_search import ImprovedMultiEngineSearch, ImprovedMultiEngineConfig
from same_search.batch_processor import BatchProcessor
from same_search.tree_optimizer import TreeOptimizer
from same_api.export.excel_exporter import ExcelExporter

logger = logging.getLogger(__name__)

# Список моделей для извлечения
MODEL_BRANDS = [
    'neox', 'osairous', 'yealink', 'sanfor', 'санфор', 'биолан', 'нэфис',
    'персил', 'dallas', 'премиум', 'маяк', 'chint', 'andeli', 'grass',
    'kraft', 'reoflex', 'керхер', 'huawei', 'honor', 'ВЫСОТА', 'ugreen',
    'alisafox', 'маякавто', 'техноавиа', 'восток-сервис', 'attache', 'камаз',
    'зубр', 'hp', 'ekf', 'dexp', 'matrix', 'siemens', 'комус', 'gigant',
    'hyundai', 'iveco', 'stayer', 'brauberg', 'makita', 'bentec', 'сибртех',
    'bosch', 'rexant', 'sampa', 'kyocera', 'avrora', 'derrick', 'cummins',
    'economy', 'samsung', 'ofite', 'professional', 'caterpillar', 'intel',
    'proxima', 'core', 'shantui', 'king', 'office', 'петролеум', 'трейл',
    'skf', 'форвелд', 'скаймастер', 'tony', 'kentek', 'ресанта', 'dexter',
    'electric', 'оттм'
]

@dataclass
class ProcessingConfig:
    """Конфигурация обработки"""
    similarity_threshold: float = 0.4
    duplicate_threshold: float = 0.95
    analog_threshold: float = 0.7
    possible_analog_threshold: float = 0.5
    batch_size: int = 1000
    max_workers: int = 4

@dataclass
class DuplicateGroup:
    """Группа дубликатов"""
    main_index: int
    main_name: str
    duplicate_indices: List[int]
    duplicate_names: List[str]
    similarity_scores: List[float]

@dataclass
class AnalogGroup:
    """Группа аналогов"""
    reference_index: int
    reference_name: str
    analogs: List[Dict[str, Any]]  # index, name, similarity, type

@dataclass
class ProductTree:
    """Дерево товаров"""
    root_index: int
    root_name: str
    duplicates: List[Dict[str, Any]]
    exact_analogs: List[Dict[str, Any]]
    close_analogs: List[Dict[str, Any]]
    possible_analogs: List[Dict[str, Any]]

class DuplicateAnalogProcessor:
    """Процессор поиска дубликатов и аналогов"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.data_manager = DataManager()
        
        # Инициализация компонентов обработки
        self.preprocessor = EnhancedPreprocessor(EnhancedPreprocessorConfig())
        self.parameter_extractor = RegexParameterExtractor()
        self.enhanced_parameter_extractor = EnhancedParameterExtractor()
        self.category_classifier = CategoryClassifier(CategoryClassifierConfig())
        
        # Инициализация оптимизированного движка
        performance_config = PerformanceConfig(
            max_workers=4,
            chunk_size=1000,
            cache_size=10000,
            enable_caching=True,
            enable_parallel_processing=True,
            enable_memory_optimization=True
        )
        self.optimized_engine = OptimizedSearchEngine(performance_config)
        
        # Инициализация специализированного движка с улучшенной конфигурацией
        duplicate_config = DuplicateSearchConfig(
            fuzzy_match_threshold=0.70,  # Снижен для тестирования
            parameter_similarity_threshold=0.60,
            semantic_similarity_threshold=0.50,
            enable_semantic_check=True,
            enable_parameter_check=True,
            enable_brand_check=True
        )
        analog_config = AnalogSearchConfig(
            exact_analog_threshold=0.70,
            close_analog_threshold=0.50,
            possible_analog_threshold=0.30,  # Снижен для тестирования
            max_analogs_per_item=25,
            enable_hierarchical_search=True,
            enable_parameter_priority=True,
            # Улучшенные весовые коэффициенты
            semantic_weight=0.35,
            fuzzy_weight=0.30,
            parameter_weight=0.20,
            brand_weight=0.10,
            category_weight=0.05
        )
        self.search_engine = DuplicateAnalogSearchEngine(duplicate_config, analog_config)
        
        # Инициализация упрощенного мульти-движкового поиска
        simple_multi_engine_config = SimpleMultiEngineConfig(
            min_similarity_threshold=0.4,  # Будет обновлено при запуске
            final_top_k=50,
            enable_parallel_search=True
        )
        self.simple_multi_engine_search = SimpleMultiEngineSearch(simple_multi_engine_config)
        
        # Инициализация улучшенного мульти-движкового поиска
        improved_multi_engine_config = ImprovedMultiEngineConfig(
            min_similarity_threshold=0.4,
            fuzzy_threshold=40,
            semantic_threshold=0.7,
            fuzzy_weight=0.6,
            semantic_weight=0.4,
            enable_category_filtering=True,
            enable_token_filtering=True,
            enable_brand_filtering=True
        )
        self.improved_multi_engine_search = ImprovedMultiEngineSearch(improved_multi_engine_config)
        
        # Экспорт
        self.excel_exporter = ExcelExporter()
        
        # Данные
        self.catalog_df = None
        self.processed_df = None
        self.duplicate_groups = []
        self.analog_groups = []
        self.product_trees = []
    
    def update_multi_engine_config(self, threshold: float):
        """Обновление конфигурации мульти-движкового поиска"""
        self.simple_multi_engine_search.config.min_similarity_threshold = threshold
        self.improved_multi_engine_search.config.min_similarity_threshold = threshold
        
        logger.info("DuplicateAnalogProcessor initialized")
    
    def extract_model_brand(self, text: str) -> Optional[str]:
        """Извлечение модели/бренда из текста"""
        if not text:
            return None
        
        text_lower = text.lower()
        words = text_lower.split()
        
        for brand in MODEL_BRANDS:
            if brand in text:
                return brand
    
        return None
    
    async def load_and_preprocess_data(self, input_file: str) -> pd.DataFrame:
        """Загрузка и предобработка данных"""
        logger.info(f"Loading data from {input_file}")
        
        # Загрузка данных
        if input_file.endswith('.csv'):
            self.catalog_df = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            self.catalog_df = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
        
        logger.info(f"Loaded {len(self.catalog_df)} records")
        
        # Создание копии для обработки
        self.processed_df = self.catalog_df.copy()
        
        # Добавление новых колонок
        self.processed_df['processed_name'] = ''
        self.processed_df['model_brand'] = ''
        self.processed_df['extracted_parameters'] = ''
        self.processed_df['category'] = ''
        self.processed_df['duplicate_count'] = 0
        self.processed_df['duplicate_indices'] = ''
        
        # Предобработка с прогресс-баром
        logger.info("Starting text preprocessing...")
        for idx in tqdm(self.processed_df.index, desc="Preprocessing text"):
            try:
                # Определение правильной колонки с названиями
                name_column = None
                for col in self.processed_df.columns:
                    if 'наименование' in col.lower() or 'название' in col.lower() or 'name' in col.lower():
                        name_column = col
                        break
                
                if name_column is None:
                    # Fallback на первую колонку если не найдена колонка с названиями
                    name_column = self.processed_df.columns[0]
                
                original_name = str(self.processed_df.loc[idx, name_column])
                
                # Предобработка текста
                processed_result = self.preprocessor.preprocess_text(original_name)
                processed_text = processed_result.get('normalized', original_name)
                self.processed_df.loc[idx, 'processed_name'] = processed_text
                
                # Извлечение модели/бренда
                model_brand = self.extract_model_brand(original_name)
                self.processed_df.loc[idx, 'model_brand'] = model_brand if model_brand else ''
                
                # Извлечение параметров (расширенное)
                basic_parameters = self.parameter_extractor.extract_parameters(original_name)
                enhanced_parameters = self.enhanced_parameter_extractor.extract_parameters(original_name)
                
                # Объединение параметров
                all_parameters = basic_parameters + enhanced_parameters
                param_str = '; '.join([f"{p.name}: {p.value}" for p in all_parameters])
                self.processed_df.loc[idx, 'extracted_parameters'] = param_str
                
                # Классификация категории
                try:
                    category, confidence = self.category_classifier.classify(original_name)
                    self.processed_df.loc[idx, 'category'] = category
                except Exception as e:
                    logger.warning(f"Error classifying category for row {idx}: {e}")
                    self.processed_df.loc[idx, 'category'] = 'общие_товары'
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        logger.info("Text preprocessing completed")
        return self.processed_df
    
    async def find_duplicates_and_analogs(self, use_optimized: bool = False, use_multi_engine: bool = False, use_improved: bool = False) -> Tuple[List[Any], List[Any]]:
        """Поиск дубликатов и аналогов с использованием специализированного движка"""
        logger.info("Starting duplicate and analog detection...")
        
        if use_improved:
            logger.info("Using improved multi-engine search with enhanced filtering...")
            # Использование улучшенного мульти-движкового поиска
            results_df = await self.improved_multi_engine_search.process_catalog(self.processed_df)
            
            # Преобразуем результаты в формат, совместимый с существующей логикой
            results = {
                'duplicates': [],
                'analogs': [],
                'trees': []
            }
            
            # Группируем результаты по запросам с улучшенной обработкой данных
            if 'query_index' in results_df.columns and len(results_df) > 0:
                # Фильтруем валидные записи
                valid_results = results_df.dropna(subset=['query_index', 'candidate_idx'])
                
                if len(valid_results) > 0:
                    query_groups = valid_results.groupby('query_index')
                    for query_idx, group in query_groups:
                        analogs = []
                        for _, row in group.iterrows():
                            candidate_idx = row.get('candidate_idx')
                            similarity_score = row.get('multi_engine_score', 0.0)
                            
                            # Проверяем валидность данных
                            if candidate_idx is not None and not pd.isna(candidate_idx) and similarity_score > 0:
                                analog_result = {
                                    'original_index': int(query_idx),
                                    'similar_index': int(candidate_idx),
                                    'similarity_score': float(similarity_score),
                                    'analog_type': self._determine_analog_type(similarity_score)
                                }
                                analogs.append(analog_result)
                        
                        if analogs:
                            analog_group = {
                                'main_index': int(query_idx),
                                'analogs': analogs
                            }
                            results['analogs'].append(analog_group)
                        
            logger.info(f"Improved multi-engine search found {len(results['analogs'])} analog groups")
            
        elif use_multi_engine:
            logger.info("Using simple multi-engine search for maximum coverage...")
            # Использование упрощенного мульти-движкового поиска
            results_df = await self.simple_multi_engine_search.process_catalog(self.processed_df)
            
            # Преобразуем результаты в формат, совместимый с существующей логикой
            results = {
                'duplicates': [],
                'analogs': [],
                'trees': []
            }
            
            # Группируем результаты по запросам
            if 'query_index' in results_df.columns:
                query_groups = results_df.groupby('query_index')
                for query_idx, group in query_groups:
                    analogs = []
                    for _, row in group.iterrows():
                        analog_result = {
                            'original_index': query_idx,
                            'similar_index': row.get('document_id', row.get('index', 0)),
                            'similarity_score': row.get('multi_engine_score', 0.0),
                            'analog_type': 'возможный аналог'  # Упрощаем тип
                        }
                        analogs.append(analog_result)
                    
                    if analogs:
                        results['analogs'].append({
                            'main_index': query_idx,
                            'analogs': analogs
                        })
            else:
                # Если нет query_index, создаем результаты на основе индексов
                for idx, row in results_df.iterrows():
                    analog_result = {
                        'original_index': idx,
                        'similar_index': row.get('document_id', row.get('index', 0)),
                        'similarity_score': row.get('multi_engine_score', 0.0),
                        'analog_type': 'возможный аналог'
                    }
                    
                    results['analogs'].append({
                        'main_index': idx,
                        'analogs': [analog_result]
                    })
            
        elif use_optimized and len(self.processed_df) > 1000:
            logger.info("Using optimized engine for large dataset...")
            # Использование оптимизированного движка для больших датасетов
            results = self.optimized_engine.optimize_for_large_dataset(self.processed_df)
        else:
            # Использование стандартного специализированного движка
            results = await self.search_engine.process_catalog(self.processed_df)
        
        # Обновление DataFrame с результатами дубликатов
        for dup_result in results.get('duplicates', []):
            if isinstance(dup_result, dict):
                main_index = dup_result.get('main_index', 0)
                duplicate_indices = dup_result.get('duplicate_indices', [])
            else:
                main_index = dup_result.main_index
                duplicate_indices = dup_result.duplicate_indices
            
            self.processed_df.loc[main_index, 'duplicate_count'] = len(duplicate_indices)
            self.processed_df.loc[main_index, 'duplicate_indices'] = ','.join(map(str, duplicate_indices))
            
            # Отметка дубликатов как обработанных
            for dup_idx in duplicate_indices:
                self.processed_df.loc[dup_idx, 'duplicate_count'] = -1
        
        self.duplicate_groups = results.get('duplicates', [])
        self.analog_groups = results.get('analogs', [])
        
        # Создаем простые деревья для мульти-движкового поиска
        if (use_multi_engine or use_improved) and not results.get('trees'):
            self.product_trees = self._create_simple_trees_from_analogs()
        else:
            self.product_trees = results.get('trees', [])
        
        logger.info(f"Found {len(self.duplicate_groups)} duplicate groups and {len(self.analog_groups)} analog groups")
        return self.duplicate_groups, self.analog_groups
    
    def _create_simple_trees_from_analogs(self) -> List[Dict[str, Any]]:
        """Создание оптимизированных деревьев из результатов аналогов"""
        logger.info("Creating optimized trees from analog results...")
        
        if not self.analog_groups:
            logger.warning("No analog groups found, returning empty trees list")
            return []
        
        logger.info(f"Processing {len(self.analog_groups)} analog groups for tree creation")
        
        # Сначала попробуем создать простые деревья без оптимизатора
        simple_trees = []
        processed_groups = 0
        valid_groups = 0
        
        for group in self.analog_groups:
            processed_groups += 1
            if isinstance(group, dict):
                main_index = group.get('main_index', 0)
                analogs = group.get('analogs', [])
            else:
                main_index = group.reference_index
                analogs = group.analogs
            
            if analogs:
                valid_groups += 1
                tree = {
                    'root_index': main_index,
                    'exact_analogs': [],
                    'close_analogs': [],
                    'possible_analogs': [],
                    'tree_depth': 1,
                    'total_nodes': len(analogs) + 1
                }
                
                for analog in analogs:
                    if isinstance(analog, dict):
                        analog_index = analog.get('similar_index', analog.get('index', 0))
                        similarity = analog.get('similarity_score', analog.get('similarity', 0.0))
                        analog_type = analog.get('analog_type', analog.get('type', 'возможный аналог'))
                    else:
                        analog_index = analog.index
                        similarity = analog.similarity
                        analog_type = analog.type
                    
                    analog_data = {
                        'index': analog_index,
                        'similarity': similarity
                    }
                    
                    if similarity >= 0.8:
                        tree['exact_analogs'].append(analog_data)
                    elif similarity >= 0.6:
                        tree['close_analogs'].append(analog_data)
                    else:
                        tree['possible_analogs'].append(analog_data)
                
                simple_trees.append(tree)
        
        logger.info(f"Tree creation stats: processed {processed_groups} groups, {valid_groups} had analogs, created {len(simple_trees)} simple trees")
        
        # Если у нас достаточно групп, попробуем оптимизировать
        if len(self.analog_groups) > 5:
            try:
                # Используем оптимизатор деревьев для устранения циклов
                tree_optimizer = TreeOptimizer(max_tree_depth=4, min_similarity_for_parent=0.3)
                
                # Анализируем граф аналогов
                graph_analysis = tree_optimizer.create_graph_analysis(self.analog_groups)
                logger.info(f"Graph analysis: {graph_analysis['total_nodes']} nodes, "
                           f"{graph_analysis['total_edges']} edges, "
                           f"{graph_analysis['connected_components']} components, "
                           f"{graph_analysis['components_with_cycles']} cycles detected")
                
                # Создаем оптимизированные деревья
                optimized_trees = tree_optimizer.optimize_trees(self.analog_groups, self.processed_df)
                
                if optimized_trees:
                    # Конвертируем в старый формат для совместимости
                    converted_trees = []
                    for tree in optimized_trees:
                        converted_tree = {
                            'root_index': tree['root_index'],
                            'exact_analogs': tree.get('exact_analogs', []),
                            'close_analogs': tree.get('close_analogs', []),
                            'possible_analogs': tree.get('possible_analogs', []),
                            'tree_depth': tree.get('tree_depth', 0),
                            'total_nodes': tree.get('total_nodes', 0)
                        }
                        converted_trees.append(converted_tree)
                    
                    logger.info(f"Created {len(converted_trees)} optimized trees "
                               f"(reduced from {len(self.analog_groups)} analog groups)")
                    return converted_trees
                    
            except Exception as e:
                logger.warning(f"Tree optimization failed: {e}, using simple trees")
        
        logger.info(f"Created {len(simple_trees)} simple trees from {len(self.analog_groups)} analog groups")
        return simple_trees
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Расчет схожести между двумя текстами"""
        if not text1 or not text2:
            return 0.0
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        if text1_lower == text2_lower:
            return 1.0
        
        # Расчет схожести на основе общих слов
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _determine_analog_type(self, similarity: float) -> str:
        """Определение типа аналога по схожести"""
        if similarity >= self.config.duplicate_threshold:
            return "дубль"
        elif similarity >= self.config.analog_threshold:
            return "аналог"
        elif similarity >= self.config.possible_analog_threshold:
            return "возможный аналог"
        else:
            return "нет аналогов"
    
    async def save_results(self, output_dir: Path):
        """Сохранение результатов"""
        logger.info(f"Saving results to {output_dir}")
        
        # Создание папки с текущей датой
        date_folder = output_dir / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        date_folder.mkdir(parents=True, exist_ok=True)
        
        # 1. Сохранение обработанных данных БЕЗ дубликатов
        # Исключаем строки с duplicate_count = -1 (это дубликаты)
        unique_df = self.processed_df[self.processed_df['duplicate_count'] != -1].copy()
        
        processed_file = date_folder / "processed_data_with_duplicates.csv"
        unique_df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data (without duplicates) to {processed_file}")
        logger.info(f"Excluded {len(self.processed_df) - len(unique_df)} duplicate records from output")
        
        # 2. Создание файла с аналогами
        # Определение правильной колонки с названиями
        name_column = None
        for col in self.processed_df.columns:
            if 'наименование' in col.lower() or 'название' in col.lower() or 'name' in col.lower():
                name_column = col
                break
        
        if name_column is None:
            name_column = self.processed_df.columns[0]
        
        analogs_data = []
        for group in self.analog_groups:
            if isinstance(group, dict):
                main_index = group.get('main_index', 0)
                analogs = group.get('analogs', [])
            else:
                main_index = group.reference_index
                analogs = group.analogs
            
            for analog in analogs:
                if isinstance(analog, dict):
                    analog_index = analog.get('similar_index', analog.get('index', 0))
                    similarity = analog.get('similarity_score', analog.get('similarity', 0.0))
                    analog_type = analog.get('analog_type', analog.get('type', 'возможный аналог'))
                else:
                    analog_index = analog.index
                    similarity = analog.similarity
                    analog_type = analog.type
                
                analogs_data.append({
                    'index': analog_index,
                    'original_name': self.processed_df.loc[main_index, name_column],
                    'similar_name': self.processed_df.loc[analog_index, name_column],
                    'similarity_coefficient': similarity,
                    'type': analog_type
                })
        
        analogs_df = pd.DataFrame(analogs_data)
        analogs_file = date_folder / "analogs_search_results.csv"
        analogs_df.to_csv(analogs_file, index=False)
        logger.info(f"Saved analogs data to {analogs_file}")
        
        # 3. Создание файла с оптимизированными деревьями
        trees_file = date_folder / "product_trees.txt"
        logger.info(f"Saving {len(self.product_trees)} product trees to {trees_file}")
        
        with open(trees_file, 'w', encoding='utf-8') as f:
            f.write("ДЕРЕВЬЯ АНАЛОГОВ\n")
            f.write("=" * 60 + "\n\n")
            
            if not self.product_trees:
                f.write("❌ ДЕРЕВЬЯ НЕ НАЙДЕНЫ\n")
                f.write("Возможные причины:\n")
                f.write("- Порог схожести слишком высокий\n")
                f.write("- Недостаточно качественных аналогов\n")
                f.write("- Ошибка в алгоритме группировки\n\n")
                logger.warning("No product trees generated - writing diagnostic information")
            
            for i, tree in enumerate(self.product_trees, 1):
                # Получение оригинального названия
                original_name = self.processed_df.loc[tree['root_index'], name_column]
                tree_depth = tree.get('tree_depth', 0)
                total_nodes = tree.get('total_nodes', 0)
                
                f.write(f"ДЕРЕВО {i} (глубина: {tree_depth}, узлов: {total_nodes})\n")
                f.write(f"└── [КОРЕНЬ] {tree['root_index']} | {original_name}\n")
                
                # Дубликаты (если есть)
                duplicates = tree.get('duplicates', [])
                if duplicates:
                    f.write("    ├── [ДУБЛИКАТЫ]\n")
                    for j, dup in enumerate(duplicates):
                        dup_name = self.processed_df.loc[dup['index'], name_column]
                        connector = "├──" if j < len(duplicates) - 1 else "└──"
                        f.write(f"    │   {connector} {dup['index']} | {dup_name} ({dup['similarity']:.4f})\n")
                
                # Точные аналоги
                exact_analogs = tree.get('exact_analogs', [])
                if exact_analogs:
                    f.write("    ├── [ТОЧНЫЕ АНАЛОГИ]\n")
                    for j, analog in enumerate(exact_analogs):
                        analog_name = self.processed_df.loc[analog['index'], name_column]
                        connector = "├──" if j < len(exact_analogs) - 1 else "└──"
                        f.write(f"    │   {connector} {analog['index']} | {analog_name} ({analog['similarity']:.4f})\n")
                
                # Близкие аналоги
                close_analogs = tree.get('close_analogs', [])
                if close_analogs:
                    f.write("    ├── [БЛИЗКИЕ АНАЛОГИ]\n")
                    for j, analog in enumerate(close_analogs):
                        analog_name = self.processed_df.loc[analog['index'], name_column]
                        connector = "├──" if j < len(close_analogs) - 1 else "└──"
                        f.write(f"    │   {connector} {analog['index']} | {analog_name} ({analog['similarity']:.4f})\n")
                
                # Возможные аналоги
                possible_analogs = tree.get('possible_analogs', [])
                if possible_analogs:
                    f.write("    └── [ВОЗМОЖНЫЕ АНАЛОГИ]\n")
                    for j, analog in enumerate(possible_analogs):
                        analog_name = self.processed_df.loc[analog['index'], name_column]
                        connector = "├──" if j < len(possible_analogs) - 1 else "└──"
                        f.write(f"        {connector} {analog['index']} | {analog_name} ({analog['similarity']:.4f})\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
        
        logger.info(f"Saved product trees to {trees_file}")
        
        # 4. Создание сводного отчета
        report_file = date_folder / "processing_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ОБ ОБРАБОТКЕ КАТАЛОГА\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего записей в исходном каталоге: {len(self.processed_df)}\n")
            f.write(f"Уникальных записей (без дубликатов): {len(unique_df)}\n")
            f.write(f"Исключено дубликатов: {len(self.processed_df) - len(unique_df)}\n")
            f.write(f"Найдено групп дубликатов: {len(self.duplicate_groups)}\n")
            f.write(f"Найдено групп аналогов: {len(self.analog_groups)}\n")
            f.write(f"Построено деревьев: {len(self.product_trees)}\n\n")
            
            # Статистика по типам аналогов
            type_counts = {}
            for group in self.analog_groups:
                if isinstance(group, dict):
                    analogs = group.get('analogs', [])
                else:
                    analogs = group.analogs if hasattr(group, 'analogs') else []
                
                for analog in analogs:
                    if isinstance(analog, dict):
                        analog_type = analog.get('analog_type', analog.get('type', 'возможный аналог'))
                    else:
                        analog_type = analog.type if hasattr(analog, 'type') else 'возможный аналог'
                    type_counts[analog_type] = type_counts.get(analog_type, 0) + 1
            
            f.write("Статистика по типам аналогов:\n")
            for analog_type, count in type_counts.items():
                f.write(f"  {analog_type}: {count}\n")
        
        logger.info(f"Saved processing report to {report_file}")
        
        return date_folder

async def generate_search_trees(input_file: str, similarity_threshold: float, max_records: int, batch_size: int, use_optimized: bool = False, use_multi_engine: bool = False, use_improved: bool = False, force_batch: bool = False, force_full: bool = False) -> bool:
    """Главная функция генерации деревьев поиска"""
    try:
        logger.info("Starting duplicate and analog search system")
        
        # Создание конфигурации
        config = ProcessingConfig(
            similarity_threshold=similarity_threshold,
            batch_size=batch_size
        )
        
        # Инициализация процессора
        processor = DuplicateAnalogProcessor(config)
        
        # Загрузка и предобработка данных
        processed_df = await processor.load_and_preprocess_data(input_file)
        
        # Автоматическое ограничение для больших датасетов
        if force_full:
            logger.info(f"🚀 FORCE FULL MODE: Processing ALL {len(processed_df):,} records without limits!")
            logger.warning("This may take several hours and use significant memory.")
        elif max_records and len(processed_df) > max_records:
            processed_df = processed_df.head(max_records)
            processor.processed_df = processed_df
            logger.info(f"Limited to {max_records} records")
        elif len(processed_df) > 50000 or force_batch:
            # Для очень больших датасетов используем пакетную обработку
            logger.warning(f"Very large dataset detected ({len(processed_df)} records). Using batch processing.")
            batch_processor = BatchProcessor(batch_size=2000, overlap_size=100)
            
            # Оценка времени обработки
            time_estimate = batch_processor.estimate_processing_time(len(processed_df))
            logger.info(f"Estimated processing time: {time_estimate['estimated_time_hours']:.1f} hours")
            
            # Пакетная обработка
            batch_results = await batch_processor.process_large_dataset(
                processed_df, processor, similarity_threshold
            )
            
            # Сохраняем результаты пакетной обработки
            processor.processed_df = batch_results.get('final_processed_df', pd.DataFrame())
            processor.duplicate_groups = batch_results.get('duplicate_groups', [])
            processor.analog_groups = batch_results.get('analog_groups', [])
            
            # Создаем простые деревья
            processor.product_trees = processor._create_simple_trees_from_analogs()
            
            logger.info(f"Batch processing completed. Found {len(processor.duplicate_groups)} duplicate groups and {len(processor.analog_groups)} analog groups")
            
            # Сохранение результатов
            output_dir = Path("src/data/output")
            result_folder = await processor.save_results(output_dir)
            
            logger.info(f"Processing completed successfully. Results saved to {result_folder}")
            return True
        
        # Обновляем конфигурацию мульти-движкового поиска
        if use_multi_engine:
            processor.update_multi_engine_config(similarity_threshold)
        
        # Поиск дубликатов и аналогов
        duplicate_groups, analog_groups = await processor.find_duplicates_and_analogs(use_optimized, use_multi_engine, use_improved)
        
        # Сохранение результатов
        output_dir = Path("src/data/output")
        result_folder = await processor.save_results(output_dir)
        
        logger.info(f"Processing completed successfully. Results saved to {result_folder}")
        return True
        
    except Exception as e:
        logger.error(f"Error in generate_search_trees: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='SAMe - Система поиска дубликатов и аналогов товаров',
        epilog='''
Примеры использования:
  %(prog)s catalog.xlsx --improved                    # Рекомендуемый режим
  %(prog)s data.xlsx -t 0.3 -l 1000 --improved      # С настройками
  %(prog)s big_file.xlsx --batch --improved          # Для больших файлов
  
Документация: см. MAIN_PY_USAGE.md
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', 
                       help='Входной CSV/Excel файл с каталогом товаров')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, 
                       help='Порог схожести (0.0-1.0, по умолчанию: 0.4)')
    parser.add_argument('-l', '--limit', type=int, 
                       help='Лимит записей для обработки (автолимит: 10,000)')
    parser.add_argument('-s', '--batch-size', type=int, default=1000, 
                       help='Размер пакета для обработки (по умолчанию: 1000)')
    parser.add_argument('-o', '--optimized', action='store_true', 
                       help='Оптимизированный движок для больших датасетов')
    parser.add_argument('-m', '--multi-engine', action='store_true', 
                       help='Простой мульти-движковый поиск')
    parser.add_argument('--improved', action='store_true', 
                       help='🔥 РЕКОМЕНДУЕТСЯ: Улучшенный поиск с фильтрацией')
    parser.add_argument('-b', '--batch', action='store_true', 
                       help='Принудительная пакетная обработка')
    parser.add_argument('--force-full', action='store_true',
                       help='🚀 Обработать ВСЕ записи без ограничений (медленно!)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting processing with parameters:")
    logger.info(f"  Input file: {args.input_file}")
    logger.info(f"  Similarity threshold: {args.threshold}")
    logger.info(f"  Record limit: {args.limit}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Optimized engine: {args.optimized}")
    
    result = asyncio.run(generate_search_trees(
        input_file=args.input_file, 
        similarity_threshold=args.threshold, 
        max_records=args.limit, 
        batch_size=args.batch_size,
        use_optimized=args.optimized,
        use_multi_engine=args.multi_engine,
        use_improved=args.improved,
        force_batch=args.batch,
        force_full=args.force_full
    ))
    
    if result:
        logger.info("Duplicate and analog search completed successfully")
    else:
        logger.error("Duplicate and analog search failed")

if __name__ == "__main__":
    main()