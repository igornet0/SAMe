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

# Импорты модулей системы
from core.text_processing import TextPreprocessor, PreprocessorConfig
from core.search_engine import FuzzySearchEngine, SemanticSearchEngine, HybridSearchEngine
from core.parameter_extraction import RegexParameterExtractor
from core.export import ExcelExporter, ExportConfig

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
    max_results_per_query: int = 10
    
    # Параметры обработки
    batch_size: int = 100
    enable_parameter_extraction: bool = True
    
    # Пути к данным
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("data/output")


class AnalogSearchEngine:
    """Главный класс системы поиска аналогов"""
    
    def __init__(self, config: AnalogSearchConfig = None):
        self.config = config or AnalogSearchConfig()
        
        # Инициализация компонентов
        self.preprocessor = TextPreprocessor(self.config.preprocessor_config)
        self.parameter_extractor = RegexParameterExtractor()
        self.exporter = ExcelExporter(self.config.export_config)
        
        # Поисковые движки
        self.fuzzy_engine = None
        self.semantic_engine = None
        self.hybrid_engine = None
        
        # Данные
        self.catalog_data = None
        self.processed_catalog = None
        self.is_ready = False
        
        logger.info("AnalogSearchEngine initialized")
    
    async def initialize(self, catalog_data: Union[pd.DataFrame, List[Dict[str, Any]], str]):
        """
        Инициализация системы с каталогом данных
        
        Args:
            catalog_data: Каталог данных (DataFrame, список словарей или путь к файлу)
        """
        try:
            # Загрузка данных
            self.catalog_data = await self._load_catalog_data(catalog_data)
            logger.info(f"Loaded catalog with {len(self.catalog_data)} items")
            
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
        
        # Предобработка текстов
        self.processed_catalog = self.preprocessor.preprocess_dataframe(
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
        
        # Пакетное извлечение параметров
        extracted_params = self.parameter_extractor.extract_parameters_batch(names)
        
        # Добавляем параметры в DataFrame
        self.processed_catalog['extracted_parameters'] = extracted_params
        
        logger.info("Parameter extraction completed")
    
    async def _initialize_search_engines(self):
        """Инициализация поисковых движков"""
        logger.info("Initializing search engines...")
        
        # Подготовка данных для индексации
        documents = self.processed_catalog['processed_name'].fillna('').astype(str).tolist()
        document_ids = self.processed_catalog.index.tolist()
        
        # Инициализация движков в зависимости от выбранного метода
        if self.config.search_method in ['fuzzy', 'hybrid']:
            self.fuzzy_engine = FuzzySearchEngine()
            self.fuzzy_engine.fit(documents, document_ids)
            logger.info("Fuzzy search engine initialized")
        
        if self.config.search_method in ['semantic', 'hybrid']:
            self.semantic_engine = SemanticSearchEngine()
            self.semantic_engine.fit(documents, document_ids)
            logger.info("Semantic search engine initialized")
        
        if self.config.search_method == 'hybrid':
            # TODO: Реализовать HybridSearchEngine
            logger.info("Hybrid search engine will be implemented")
        
        logger.info("Search engines initialization completed")
    
    async def search_analogs(self, 
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
        
        # Выполнение поиска
        results = {}
        
        if method == 'fuzzy' and self.fuzzy_engine:
            results = await self._fuzzy_search(processed_queries)
        
        elif method == 'semantic' and self.semantic_engine:
            results = await self._semantic_search(processed_queries)
        
        elif method == 'hybrid':
            results = await self._hybrid_search(processed_queries)
        
        else:
            raise ValueError(f"Unsupported search method: {method}")
        
        # Обогащение результатов
        enriched_results = await self._enrich_results(results, queries)
        
        logger.info(f"Search completed. Found results for {len(enriched_results)} queries")
        
        return enriched_results
    
    async def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """Предобработка поисковых запросов"""
        processed_results = self.preprocessor.preprocess_batch(queries)
        return [result['final_text'] for result in processed_results]
    
    async def _fuzzy_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Нечеткий поиск"""
        results = {}
        
        for i, query in enumerate(queries):
            if query.strip():
                search_results = self.fuzzy_engine.search(query, self.config.max_results_per_query)
                results[f"query_{i}"] = search_results
            else:
                results[f"query_{i}"] = []
        
        return results
    
    async def _semantic_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Семантический поиск"""
        results = {}
        
        # Пакетный поиск для оптимизации
        batch_results = self.semantic_engine.batch_search(queries, self.config.max_results_per_query)
        
        for i, query_results in enumerate(batch_results):
            results[f"query_{i}"] = query_results
        
        return results
    
    async def _hybrid_search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Гибридный поиск (комбинация нечеткого и семантического)"""
        # Получаем результаты от обоих движков
        fuzzy_results = await self._fuzzy_search(queries)
        semantic_results = await self._semantic_search(queries)
        
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
            
            for result in query_results:
                # Получаем полную информацию о найденном элементе
                doc_id = result.get('document_id')
                if doc_id is not None and doc_id < len(self.processed_catalog):
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
            'similarity_threshold': self.config.similarity_threshold
        }
        
        if self.fuzzy_engine:
            stats['fuzzy_engine'] = self.fuzzy_engine.get_statistics()
        
        if self.semantic_engine:
            stats['semantic_engine'] = self.semantic_engine.get_statistics()
        
        return stats
    
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
