"""
Главный модуль системы поиска аналогов SAMe
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import asyncio
import time

# Импорты модулей системы - обновлено для новой архитектуры
from same_clear.text_processing import TextPreprocessor, PreprocessorConfig
from same_core.interfaces import (
    AnalogSearchEngineInterface, TextProcessorInterface as TextPreprocessorInterface,
    SearchEngineInterface
)
# Дополнительные интерфейсы для обратной совместимости
FuzzySearchInterface = SearchEngineInterface
SemanticSearchInterface = SearchEngineInterface
HybridSearchInterface = SearchEngineInterface

from same_search.search_engine import FuzzySearchEngine, SemanticSearchEngine, HybridSearchEngine
from same_clear.parameter_extraction import RegexParameterExtractor, ParameterParser, ParameterParserConfig
from same_clear.parameter_extraction.parameter_utils import ParameterFormatter, ParameterAnalyzer, ParameterDataFrameUtils
from same_api.export import ExcelExporter, ExportConfig
from src.data_manager.DataManager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class AnalogSearchConfig:
    """Конфигурация системы поиска аналогов"""
    # Конфигурации компонентов
    preprocessor_config: PreprocessorConfig = None
    export_config: ExportConfig = None
    
    # Параметры поиска
    search_method: str = "hybrid"  # fuzzy, semantic, hybrid
    similarity_threshold: float = 0.6
    max_results_per_query: int = 100
    
    # Параметры обработки
    batch_size: int = 100
    enable_parameter_extraction: bool = True
    
    # Пути к данным
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/output")


class AnalogSearchEngine(AnalogSearchEngineInterface):
    """Главный класс системы поиска аналогов с dependency injection"""

    def __init__(self, config: AnalogSearchConfig = None,
                 preprocessor: TextPreprocessorInterface = None,
                 fuzzy_engine: FuzzySearchInterface = None,
                 semantic_engine: SemanticSearchInterface = None,
                 hybrid_engine: HybridSearchInterface = None):
        self.config = config or AnalogSearchConfig()

        # Инициализация компонентов через dependency injection
        self._preprocessor = preprocessor or TextPreprocessor(self.config.preprocessor_config)
        self._fuzzy_engine = fuzzy_engine
        self._semantic_engine = semantic_engine
        self._hybrid_engine = hybrid_engine

        # Инициализация парсера параметров
        parameter_config = ParameterParserConfig(
            use_regex=self.config.enable_parameter_extraction,
            use_ml=False,  
            min_confidence=0.5,
            remove_duplicates=True
        )
        self.parameter_parser = ParameterParser(parameter_config)

        # Для обратной совместимости
        self.parameter_extractor = self.parameter_parser.regex_extractor

        self.exporter = ExcelExporter(self.config.export_config)
        self.data_manager = DataManager()

        # Поисковые движки (используем injected или создаем по умолчанию)
        self.fuzzy_engine = self._fuzzy_engine
        self.semantic_engine = self._semantic_engine
        self.hybrid_engine = self._hybrid_engine
        
        # Данные
        self.catalog_data = None
        self.processed_catalog = None
        self.is_ready = False
        # Report about preprocessing issues
        self.processing_report: Dict[str, Any] = {
            'failed_count': 0,
            'failed_rows': []  # list of {code, name, error}
        }
        
        logger.info("AnalogSearchEngine initialized")

    def set_preprocessor(self, preprocessor: TextPreprocessorInterface) -> None:
        """Установка предобработчика через dependency injection"""
        self._preprocessor = preprocessor
        logger.info("Preprocessor updated via dependency injection")

    def set_search_engine(self, engine: SearchEngineInterface, engine_type: str) -> None:
        """
        Установка поискового движка через dependency injection

        Args:
            engine: Экземпляр поискового движка
            engine_type: Тип движка ('fuzzy', 'semantic', 'hybrid')
        """
        if engine_type == 'fuzzy':
            if not isinstance(engine, FuzzySearchInterface):
                raise TypeError("Engine must implement FuzzySearchInterface")
            self.fuzzy_engine = engine
            self._fuzzy_engine = engine
        elif engine_type == 'semantic':
            if not isinstance(engine, SemanticSearchInterface):
                raise TypeError("Engine must implement SemanticSearchInterface")
            self.semantic_engine = engine
            self._semantic_engine = engine
        elif engine_type == 'hybrid':
            if not isinstance(engine, HybridSearchInterface):
                raise TypeError("Engine must implement HybridSearchInterface")
            self.hybrid_engine = engine
            self._hybrid_engine = engine
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        logger.info(f"Search engine '{engine_type}' updated via dependency injection")

    @property
    def preprocessor(self) -> TextPreprocessorInterface:
        """Получение текущего предобработчика"""
        return self._preprocessor

    async def initialize(self, catalog_data: Union[pd.DataFrame, List[Dict[str, Any]], str]):
        """
        Инициализация системы с каталогом данных
        
        Args:
            catalog_data: Каталог данных (DataFrame, список словарей или путь к файлу)
        """
        try:
            # Загрузка данных
            # Поддержка путей к файлам/списков/готового DataFrame
            self.catalog_data = await self._load_catalog_data(catalog_data)
            # logger.info(f"Loaded catalog with {len(self.catalog_data)} items")
            
            # Предобработка данных
            await self._preprocess_catalog()
            
            # Инициализация поисковых движков
            await self._initialize_search_engines()
            
            self.is_ready = True
            logger.info("AnalogSearchEngine initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing AnalogSearchEngine: {e}")
            raise
    
    async def _load_catalog_data(self, data_source: Union[pd.DataFrame, List[Dict[str, Any]], str]) -> pd.DataFrame:
        """Загрузка каталога данных"""
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        
        elif isinstance(data_source, list):
            return pd.DataFrame(data_source)
        
        elif isinstance(data_source, str):
            file_path = Path(data_source)
            if file_path.suffix.lower() == '.xlsx':
                return pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        else:
            raise ValueError("Invalid data source type")
    
    async def _preprocess_catalog(self):
        """Предобработка каталога данных"""
        logger.info("Starting catalog preprocessing...")

        # Определяем колонку с наименованиями
        name_column = self._find_name_column(self.catalog_data)

        # Предобработка текстов с использованием async версии
        self.processed_catalog = await self._preprocess_dataframe_async(
            self.catalog_data,
            name_column,
            output_columns={
                'final': 'processed_name'
            }
        )

        # Извлечение параметров если включено
        if self.config.enable_parameter_extraction:
            await self._extract_parameters()

        logger.info("Catalog preprocessing completed")

    async def _preprocess_dataframe_async(self, df: pd.DataFrame, text_column: str,
                                        output_columns: Dict[str, str] = None) -> pd.DataFrame:
        """
        Асинхронная предобработка DataFrame с текстами

        Args:
            df: DataFrame с данными
            text_column: Название колонки с текстом
            output_columns: Маппинг названий выходных колонок

        Returns:
            DataFrame с добавленными колонками обработанного текста
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Настройки выходных колонок по умолчанию
        default_columns = {
            'cleaned': f'{text_column}_cleaned',
            'normalized': f'{text_column}_normalized',
            'lemmatized': f'{text_column}_lemmatized',
            'final': f'{text_column}_processed'
        }

        if output_columns:
            default_columns.update(output_columns)

        # Получаем тексты для обработки (всегда списком в порядке строк, чтобы не терять дубликаты ключей)
        code_series = df['Код'] if 'Код' in df.columns else df.index
        name_series = df[text_column].fillna('').astype(str)
        texts: List[str] = name_series.tolist()

        # Обрабатываем тексты асинхронно
        self.preprocessor: TextPreprocessor
        results = await self.preprocessor.preprocess_batch_async(texts)

        # Проверяем соответствие длин и исправляем если нужно
        if len(results) != len(texts):
            logger.warning(f"Results length ({len(results)}) doesn't match texts length ({len(texts)}). Attempting to fix...")
            
            # Если результатов меньше, добавляем недостающие
            if len(results) < len(texts):
                missing_count = len(texts) - len(results)
                logger.warning(f"Adding {missing_count} missing results")
                for i in range(missing_count):
                    results.append({
                        'original': '',
                        'final_text': '',
                        'processing_successful': False,
                        'error': 'Missing result from processing'
                    })
            
            # Если результатов больше, обрезаем лишние
            elif len(results) > len(texts):
                excess_count = len(results) - len(texts)
                logger.warning(f"Truncating {excess_count} excess results")
                results = results[:len(texts)]

        # Добавляем результаты в DataFrame
        df_result = df.copy()

        if self.preprocessor.config.save_intermediate_steps:
            df_result[default_columns['cleaned']] = [r.get('cleaning', {}).get('normalized', '') for r in results]
            df_result[default_columns['normalized']] = [r.get('normalization', {}).get('final_normalized', '') for r in results]
            df_result[default_columns['lemmatized']] = [r.get('lemmatization', {}).get('lemmatized', '') for r in results]

        df_result[default_columns['final']] = [r.get('final_text', '') for r in results]

        # Добавляем статистику
        success_list = [r.get('processing_successful', False) for r in results]
        # Выравниваем длину при рассинхронизации
        if len(success_list) != len(df_result):
            logger.warning(
                f"Processing success length mismatch: got {len(success_list)}, expected {len(df_result)}. "
                "Auto-correcting by padding/truncating and marking errors."
            )
            if len(success_list) < len(df_result):
                pad = len(df_result) - len(success_list)
                success_list = success_list + [False] * pad
                # Обновляем отчет
                for i in range(min(pad, 1000)):
                    idx = len(success_list) - pad + i
                    code_series = df['Код'] if 'Код' in df.columns else df.index
                    try:
                        code_val = code_series.iloc[idx]
                        name_val = name_series.iloc[idx]
                    except Exception:
                        code_val, name_val = '', ''
                    self.processing_report['failed_rows'].append({
                        'code': code_val,
                        'name': name_val,
                        'error': 'Processing success missing result'
                    })
                self.processing_report['failed_count'] = int(self.processing_report.get('failed_count', 0)) + pad
            else:
                success_list = success_list[:len(df_result)]
        df_result[f'{text_column}_processing_success'] = success_list

        # Обновляем отчёт о сбоях обработки
        failed_rows = []
        # Ограничим размер сохраняемого списка, чтобы не раздувать ответ
        MAX_FAILED_ROWS = 1000
        for i, ok in enumerate(success_list):
            if not ok and len(failed_rows) < MAX_FAILED_ROWS:
                failed_rows.append({
                    'code': code_series.iloc[i].item() if hasattr(code_series.iloc[i], 'item') else code_series.iloc[i],
                    'name': name_series.iloc[i],
                    'error': results[i].get('error')
                })
        self.processing_report = {
            'failed_count': int(len([x for x in success_list if not x])),
            'failed_rows': failed_rows
        }

        return df_result

    def _find_name_column(self, df: pd.DataFrame) -> str:
        """Поиск колонки с наименованиями"""
        possible_names = ['name', 'наименование', 'название', 'item_name', 'product_name']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Если не найдено, берем первую текстовую колонку
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"Using column '{col}' as name column")
                return col
        
        raise ValueError("Could not find suitable name column in catalog")
    
    async def _extract_parameters(self):
        """Извлечение параметров из наименований"""
        logger.info("Extracting parameters from catalog items...")

        name_column = self._find_name_column(self.catalog_data)
        names = self.catalog_data[name_column].fillna('').astype(str).tolist()

        # Пакетное извлечение параметров с защитой от рассинхронизации длин
        try:
            extracted_params = self.parameter_parser.parse_batch(names) or []
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            extracted_params = []

        expected_len = len(self.processed_catalog)
        actual_len = len(extracted_params)

        parameters_processing_success = []

        if actual_len < expected_len:
            # Дополняем недостающие элементы пустыми значениями и помечаем как ошибку
            missing = expected_len - actual_len
            logger.warning(
                f"Parameter extraction length mismatch: got {actual_len}, expected {expected_len}. Padding {missing} items"
            )
            extracted_params = list(extracted_params) + [{} for _ in range(missing)]
            parameters_processing_success = [True] * actual_len + [False] * missing
            # Обновляем отчёт о сбоях
            for i in range(min(missing, 1000)):
                try:
                    code_val = self.processed_catalog.index[actual_len + i]
                    name_val = names[actual_len + i] if (actual_len + i) < len(names) else ''
                except Exception:
                    code_val, name_val = '', ''
                self.processing_report['failed_rows'].append({
                    'code': code_val,
                    'name': name_val,
                    'error': 'Parameter extraction missing result'
                })
            self.processing_report['failed_count'] = int(self.processing_report.get('failed_count', 0)) + missing
        elif actual_len > expected_len:
            # Обрезаем лишние и считаем их как избыточные
            extra = actual_len - expected_len
            logger.warning(
                f"Parameter extraction length mismatch: got {actual_len}, expected {expected_len}. Truncating extra {extra} items"
            )
            extracted_params = extracted_params[:expected_len]
            parameters_processing_success = [True] * expected_len
        else:
            parameters_processing_success = [True] * expected_len

        # Безопасно добавляем результаты в DataFrame
        try:
            self.processed_catalog['extracted_parameters'] = extracted_params
            self.processed_catalog['parameters_processing_success'] = parameters_processing_success
        except Exception as assign_error:
            logger.error(f"Failed to assign extracted parameters to DataFrame: {assign_error}")
            # В крайнем случае заполняем полностью пустыми значениями
            self.processed_catalog['extracted_parameters'] = [{} for _ in range(expected_len)]
            self.processed_catalog['parameters_processing_success'] = [False] * expected_len
            self.processing_report['failed_count'] = int(self.processing_report.get('failed_count', 0)) + expected_len
            for i in range(min(expected_len, 1000)):
                self.processing_report['failed_rows'].append({
                    'code': self.processed_catalog.index[i],
                    'name': names[i] if i < len(names) else '',
                    'error': 'Failed to assign parameters to DataFrame'
                })

        # Добавляем дополнительные колонки с параметрами
        try:
            self.processed_catalog = ParameterDataFrameUtils.add_parameters_columns(
                self.processed_catalog, 'extracted_parameters'
            )
        except Exception as add_cols_error:
            logger.warning(f"Failed to add parameter columns: {add_cols_error}")

        # Анализ и логирование статистики (устойчиво к ошибкам)
        try:
            stats = ParameterAnalyzer.analyze_parameters_batch(self.processed_catalog['extracted_parameters'])
            logger.info(
                f"Parameter extraction completed. Stats: {stats['items_with_parameters']}/{stats['total_items']} items with parameters"
            )
        except Exception as stats_error:
            logger.warning(f"Failed to analyze parameter extraction stats: {stats_error}")

        # Сохраняем данные с параметрами
        if hasattr(self, 'data_manager'):
            try:
                await self.data_manager.save_parameters_data(
                    self.processed_catalog,
                    parameters_column='extracted_parameters',
                    dataset_name='catalog_with_parameters'
                )
            except Exception as e:
                logger.warning(f"Failed to save parameters data: {e}")
    
    async def _initialize_search_engines(self):
        """Инициализация поисковых движков"""
        logger.info("Initializing search engines...")
        
        # Подготовка данных для индексации
        name_col = 'processed_name' if 'processed_name' in self.processed_catalog.columns else self._find_name_column(self.processed_catalog)
        documents = self.processed_catalog[name_col].fillna('').astype(str).tolist()
        document_ids = self.processed_catalog.index.tolist()

        # Подготовка метаданных с категориями
        metadata = []
        for idx, row in self.processed_catalog.iterrows():
            meta = {
                'original_name': row.get('original_name', row.get(name_col, '')),
                'processed_name': row.get('processed_name', row.get(name_col, '')),
                'category': 'unknown'  # Будет определена автоматически
            }
            # Добавляем параметры если есть
            if hasattr(row, 'extracted_parameters') and row['extracted_parameters']:
                meta['parameters'] = row['extracted_parameters']
            metadata.append(meta)

        # Инициализация движков в зависимости от выбранного метода
        if self.config.search_method in ['fuzzy', 'hybrid']:
            self.fuzzy_engine = FuzzySearchEngine()
            self.fuzzy_engine.fit(documents, document_ids)
            logger.info("Fuzzy search engine initialized")

        if self.config.search_method in ['semantic', 'hybrid']:
            try:
                self.semantic_engine = SemanticSearchEngine()
                self.semantic_engine.fit(documents, document_ids, metadata)
                logger.info("Semantic search engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic search engine: {e}")
                logger.info("Continuing with fuzzy search only")
                self.semantic_engine = None

        if self.config.search_method == 'hybrid':
            try:
                self.hybrid_engine = HybridSearchEngine()
                self.hybrid_engine.fit(documents, document_ids, metadata)
                logger.info("Hybrid search engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid search engine: {e}")
                logger.info("Falling back to individual engines")
        
        logger.info("Search engines initialization completed")

    def search_analogs(self, query: str, search_type: str = "hybrid",
                      top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Синхронный поиск аналогов (реализация интерфейса)

        Args:
            query: Поисковый запрос
            search_type: Тип поиска (fuzzy, semantic, hybrid)
            top_k: Количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список найденных аналогов
        """
        try:
            # Используем asyncio.run для вызова асинхронной версии
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если уже в event loop, используем синхронную обработку
                return self._search_analogs_sync(query, search_type, top_k, **kwargs)
            else:
                # Если нет активного loop, можем использовать asyncio.run
                result = asyncio.run(self.search_analogs_async([query], search_type))
                return result.get(query, [])[:top_k]
        except RuntimeError:
            # Fallback к синхронной версии
            return self._search_analogs_sync(query, search_type, top_k, **kwargs)

    async def search_analogs_async(self,
                           queries: Union[str, List[str]],
                           method: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Поиск аналогов для запросов
        
        Args:
            queries: Запрос или список запросов
            method: Метод поиска (fuzzy, semantic, hybrid)
            
        Returns:
            Словарь с результатами поиска
        """
        if not self.is_ready:
            raise ValueError("Engine is not initialized. Call initialize() first.")
        
        # Нормализация входных данных
        if isinstance(queries, str):
            queries = [queries]
        
        method = method or self.config.search_method
        
        logger.info(f"Searching analogs for {len(queries)} queries using {method} method")
        
        # Предобработка запросов
        processed_queries = await self._preprocess_queries(queries)

        # Извлечение параметров из запросов если включено
        if self.config.enable_parameter_extraction:
            enhanced_queries = await self._extract_query_parameters(processed_queries)
            # Извлекаем обработанные запросы для поиска
            processed_queries = [q['processed_query'] if q['processed_query'] is not None else "" for q in enhanced_queries]
        else:
            enhanced_queries = None
        
        # Выполнение поиска
        results = {}

        try:
            if method == 'fuzzy' and self.fuzzy_engine:
                results = await self._fuzzy_search(processed_queries)

            elif method == 'semantic' and self.semantic_engine:
                results = await self._semantic_search(processed_queries)

            elif method == 'hybrid':
                results = await self._hybrid_search(processed_queries)

            else:
                raise ValueError(f"Unsupported search method: {method}")
        except Exception as search_error:
            logger.error(f"Error during search execution: {search_error}")
            raise

        # Обогащение результатов
        try:
            enriched_results = await self._enrich_results(results, queries)
        except Exception as enrich_error:
            logger.error(f"Error during result enrichment: {enrich_error}")
            raise
        
        logger.info(f"Search completed. Found results for {len(enriched_results)} queries")
        
        return enriched_results
    
    async def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """Предобработка поисковых запросов"""
        processed_results = self.preprocessor.preprocess_batch(queries)
        return [result['final_text'] if result['final_text'] is not None else "" for result in processed_results]

    async def _extract_query_parameters(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Извлечение параметров из запросов"""
        logger.info("Extracting parameters from queries...")

        enhanced_queries = []

        for i, query in enumerate(queries):
            # Извлекаем параметры из запроса
            parameters = self.parameter_parser.parse_parameters(query)
            parameter_dict = self.parameter_parser.extract_parameters_dict(query)

            # Проверяем на None значения
            if parameters is None:
                parameters = []
            if parameter_dict is None:
                parameter_dict = {}

            # Создаем расширенные данные запроса
            enhanced_query = {
                'original_query': query,
                'processed_query': query,
                'extracted_parameters': parameters,
                'parameters_dict': parameter_dict,
                'parameters_formatted': ParameterFormatter.format_parameters_list(parameters),
                'query_index': i
            }

            enhanced_queries.append(enhanced_query)

        logger.info(f"Parameters extracted for {len(enhanced_queries)} queries")
        return enhanced_queries

    async def _fuzzy_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Нечеткий поиск"""
        results = {}
        
        for i, query in enumerate(queries):
            if query.strip():
                search_results = self.fuzzy_engine.search(query, self.config.max_results_per_query)
                results[f"query_{i}"] = search_results if search_results is not None else []
            else:
                results[f"query_{i}"] = []
        
        return results
    
    async def _semantic_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Семантический поиск"""
        results = {}
        
        # Пакетный поиск для оптимизации
        batch_results = self.semantic_engine.batch_search(queries, self.config.max_results_per_query)

        # Проверяем, что batch_results не None
        if batch_results is None:
            batch_results = [[] for _ in queries]

        for i, query_results in enumerate(batch_results):
            results[f"query_{i}"] = query_results if query_results is not None else []
        
        return results
    
    async def _hybrid_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Гибридный поиск с улучшенным скорингом и категориальной фильтрацией"""
        results = {}

        # Используем новый гибридный движок если доступен
        if hasattr(self, 'hybrid_engine') and self.hybrid_engine and self.hybrid_engine.is_fitted:
            for i, query in enumerate(queries):
                if query.strip():
                    search_results = self.hybrid_engine.search(query, self.config.max_results_per_query)
                    results[f"query_{i}"] = search_results if search_results is not None else []
                else:
                    results[f"query_{i}"] = []
            return results

        # Fallback к старой логике комбинирования
        # Получаем результаты от нечеткого поиска
        fuzzy_results = await self._fuzzy_search(queries)

        # Получаем результаты от семантического поиска если доступен
        if self.semantic_engine and self.semantic_engine.is_fitted:
            semantic_results = await self._semantic_search(queries)
        else:
            logger.info("Semantic search not available, using fuzzy search only")
            return fuzzy_results

        # Комбинируем результаты
        combined_results = {}
        
        for key in fuzzy_results.keys():
            fuzzy_res = fuzzy_results.get(key, [])
            semantic_res = semantic_results.get(key, [])
            
            # Простая стратегия комбинирования: берем лучшие результаты от каждого метода
            combined = []
            
            # Добавляем семантические результаты с высоким скором
            for result in semantic_res:
                if result.get('similarity_score', 0) >= 0.7:
                    result['search_method'] = 'semantic'
                    combined.append(result)
            
            # Добавляем нечеткие результаты, которых нет в семантических
            semantic_docs = {r.get('document_id') for r in combined}
            
            for result in fuzzy_res:
                if (result.get('document_id') not in semantic_docs and 
                    result.get('combined_score', 0) >= 0.6):
                    result['search_method'] = 'fuzzy'
                    combined.append(result)
            
            # Сортируем по лучшему скору
            combined.sort(key=lambda x: max(
                x.get('similarity_score', 0), 
                x.get('combined_score', 0)
            ), reverse=True)
            
            combined_results[key] = combined[:self.config.max_results_per_query]
        
        return combined_results
    
    async def _enrich_results(self,
                            results: Dict[str, List[Dict[str, Any]]],
                            original_queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Обогащение результатов дополнительной информацией"""
        enriched = {}

        for i, (key, query_results) in enumerate(results.items()):
            original_query = original_queries[i] if i < len(original_queries) else ""
            enriched_results = []

            # Проверяем, что query_results не None и не пустой
            if query_results is None:
                query_results = []

            for j, result in enumerate(query_results):
                try:
                    if result is None:
                        continue

                    # Получаем полную информацию о найденном элементе
                    doc_id = result.get('document_id')

                    if doc_id is not None and self.processed_catalog is not None and doc_id < len(self.processed_catalog):
                        catalog_item = self.processed_catalog.iloc[doc_id].to_dict()

                        # Обогащаем результат
                        enriched_result = {
                            **result,
                            'original_query': original_query,
                            'catalog_item': catalog_item,
                            'timestamp': time.time()
                        }

                        # Добавляем извлеченные параметры если есть
                        if 'extracted_parameters' in catalog_item:
                            enriched_result['extracted_parameters'] = catalog_item['extracted_parameters']

                        enriched_results.append(enriched_result)
                    else:
                        # Skip results that can't be enriched
                        continue
                except Exception as result_error:
                    logger.error(f"Error processing result {j}: {result_error}")
                    continue
            
            enriched[original_query] = enriched_results
        
        return enriched
    
    async def export_results(self, 
                           results: Dict[str, List[Dict[str, Any]]], 
                           filepath: str = None,
                           export_format: str = "excel") -> str:
        """
        Экспорт результатов поиска
        
        Args:
            results: Результаты поиска
            filepath: Путь к выходному файлу
            export_format: Формат экспорта (excel)
            
        Returns:
            Путь к созданному файлу
        """
        if not filepath:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self.config.output_dir / f"analog_search_results_{timestamp}.xlsx"
        
        # Создаем директорию если не существует
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if export_format == "excel":
            # Подготавливаем метаданные
            metadata = {
                'search_method': self.config.search_method,
                'similarity_threshold': self.config.similarity_threshold,
                'total_catalog_items': len(self.catalog_data),
                'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_queries': len(results),
                'total_results': sum(len(query_results) for query_results in results.values())
            }
            
            return self.exporter.export_search_results(results, str(filepath), metadata)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        stats = {
            'is_ready': self.is_ready,
            'catalog_size': len(self.catalog_data) if self.catalog_data is not None else 0,
            'search_method': self.config.search_method,
            'similarity_threshold': self.config.similarity_threshold,
            'processing_failures': self.processing_report.get('failed_count', 0)
        }
        
        if self.fuzzy_engine:
            stats['fuzzy_engine'] = self.fuzzy_engine.get_statistics()
        
        if self.semantic_engine:
            stats['semantic_engine'] = self.semantic_engine.get_statistics()
        
        return stats

    def get_processing_report(self) -> Dict[str, Any]:
        """Отчёт о неуспешно обработанных строках (ограниченный список)."""
        return self.processing_report
    
    async def save_models(self, models_dir: str = None):
        """Сохранение обученных моделей"""
        models_dir = Path(models_dir) if models_dir else self.config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.fuzzy_engine:
            fuzzy_path = models_dir / "fuzzy_search_model.pkl"
            self.fuzzy_engine.save_model(str(fuzzy_path))
        
        if self.semantic_engine:
            semantic_path = models_dir / "semantic_search_model.pkl"
            self.semantic_engine.save_model(str(semantic_path))
        
        logger.info(f"Models saved to {models_dir}")
    
    async def load_models(self, models_dir: str = None):
        """Загрузка сохраненных моделей"""
        models_dir = Path(models_dir) if models_dir else self.config.models_dir
        
        fuzzy_path = models_dir / "fuzzy_search_model.pkl"
        if fuzzy_path.exists():
            self.fuzzy_engine = FuzzySearchEngine()
            self.fuzzy_engine.load_model(str(fuzzy_path))
            logger.info("Fuzzy search model loaded")
        
        semantic_path = models_dir / "semantic_search_model.pkl"
        if semantic_path.exists():
            self.semantic_engine = SemanticSearchEngine()
            self.semantic_engine.load_model(str(semantic_path))
            logger.info("Semantic search model loaded")
        
        self.is_ready = True
        logger.info("Models loaded successfully")

    def _search_analogs_sync(self, query: str, search_type: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Синхронная версия поиска аналогов

        Args:
            query: Поисковый запрос
            search_type: Тип поиска
            top_k: Количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов поиска
        """
        if not self.is_ready:
            raise ValueError("Engine is not initialized. Call initialize() first.")

        try:
            # Предобработка запроса
            processed_query = self._preprocessor.preprocess_text(query)
            final_query = processed_query.get('final_text', query)

            # Выбор поискового движка
            if search_type == "fuzzy" and self.fuzzy_engine:
                if hasattr(self.fuzzy_engine, 'is_fitted'):
                    is_fitted = self.fuzzy_engine.is_fitted() if callable(self.fuzzy_engine.is_fitted) else self.fuzzy_engine.is_fitted
                    if is_fitted:
                        results = self.fuzzy_engine.search(final_query, top_k, **kwargs)
                    else:
                        logger.warning("Fuzzy engine not fitted, falling back to semantic")
                        results = self._fallback_search(final_query, top_k, **kwargs)
                else:
                    logger.warning("Fuzzy engine has no is_fitted attribute, trying search anyway")
                    results = self.fuzzy_engine.search(final_query, top_k, **kwargs)
            elif search_type == "semantic" and self.semantic_engine:
                if hasattr(self.semantic_engine, 'is_fitted'):
                    is_fitted = self.semantic_engine.is_fitted() if callable(self.semantic_engine.is_fitted) else self.semantic_engine.is_fitted
                    if is_fitted:
                        results = self.semantic_engine.search(final_query, top_k, **kwargs)
                    else:
                        logger.warning("Semantic engine not fitted, falling back to fuzzy")
                        results = self._fallback_search(final_query, top_k, **kwargs)
                else:
                    logger.warning("Semantic engine has no is_fitted attribute, trying search anyway")
                    results = self.semantic_engine.search(final_query, top_k, **kwargs)
            elif search_type == "hybrid" and self.hybrid_engine:
                if hasattr(self.hybrid_engine, 'is_fitted'):
                    is_fitted = self.hybrid_engine.is_fitted() if callable(self.hybrid_engine.is_fitted) else self.hybrid_engine.is_fitted
                    if is_fitted:
                        results = self.hybrid_engine.search(final_query, top_k, **kwargs)
                    else:
                        logger.warning("Hybrid engine not fitted, using fallback")
                        results = self._fallback_search(final_query, top_k, **kwargs)
                else:
                    logger.warning("Hybrid engine has no is_fitted attribute, trying search anyway")
                    results = self.hybrid_engine.search(final_query, top_k, **kwargs)
            else:
                # Fallback к доступному движку
                results = self._fallback_search(final_query, top_k, **kwargs)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in synchronous search: {e}")
            return []

    def _fallback_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Fallback поиск при недоступности основных движков

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов поиска
        """
        # Пробуем доступные движки в порядке приоритета
        engines_to_try = [
            ('semantic', self.semantic_engine),
            ('fuzzy', self.fuzzy_engine),
            ('hybrid', self.hybrid_engine)
        ]

        for engine_type, engine in engines_to_try:
            if engine and hasattr(engine, 'is_fitted'):
                try:
                    is_fitted = engine.is_fitted() if callable(engine.is_fitted) else engine.is_fitted
                    if is_fitted:
                        results = engine.search(query, top_k, **kwargs)
                        logger.info(f"Fallback to {engine_type} engine successful")
                        return results
                except Exception as e:
                    logger.warning(f"Fallback {engine_type} engine failed: {e}")
                    continue
            elif engine:
                # Если движок есть, но нет атрибута is_fitted, пробуем поиск
                try:
                    results = engine.search(query, top_k, **kwargs)
                    logger.info(f"Fallback to {engine_type} engine successful (no is_fitted check)")
                    return results
                except Exception as e:
                    logger.warning(f"Fallback {engine_type} engine failed: {e}")
                    continue

        # Последний fallback - простой текстовый поиск
        logger.warning("All engines failed, using simple text matching")
        return self._simple_text_search(query, top_k)

    def _simple_text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Простой текстовый поиск как последний fallback

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов простого поиска
        """
        if not self.processed_catalog:
            return []

        query_lower = query.lower()
        results = []

        for idx, row in self.processed_catalog.iterrows():
            # Простое сравнение по вхождению подстроки
            text_to_search = str(row.get('final_text', '')).lower()
            if query_lower in text_to_search:
                similarity = len(query_lower) / len(text_to_search) if text_to_search else 0
                results.append({
                    'document_id': idx,
                    'document': str(row.get('original_text', '')),
                    'similarity_score': similarity,
                    'rank': len(results) + 1,
                    'search_type': 'simple_text'
                })

        # Сортируем по схожести и возвращаем top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
