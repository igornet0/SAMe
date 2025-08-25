"""
Базовые интерфейсы для всех модулей SAMe
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Protocol
import pandas as pd
from pathlib import Path

from .types import ProcessingResult, SearchResult, ParameterData


class TextProcessorInterface(ABC):
    """Базовый интерфейс для компонентов обработки текста"""
    
    @abstractmethod
    def process_text(self, text: str) -> ProcessingResult:
        """
        Обработка одного текста
        
        Args:
            text: Входной текст
            
        Returns:
            Результат обработки
        """
        pass
    
    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[ProcessingResult]:
        """
        Пакетная обработка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Список результатов обработки
        """
        pass


class SearchEngineInterface(ABC):
    """Интерфейс для поисковых движков"""
    
    @abstractmethod
    def fit(self, documents: List[str], document_ids: List[str], 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Обучение поискового движка на документах
        
        Args:
            documents: Список документов для индексации
            document_ids: Идентификаторы документов
            metadata: Дополнительные метаданные
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Поиск по запросу
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            filters: Фильтры поиска
            
        Returns:
            Список результатов поиска
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Сохранение модели"""
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Загрузка модели"""
        pass


class AnalogSearchEngineInterface(Protocol):
    """Интерфейс для главного движка поиска аналогов"""
    
    def initialize(self, catalog_data: Union[pd.DataFrame, List[Dict[str, Any]], str]) -> None:
        """Инициализация с каталогом данных"""
        pass
    
    def search_analogs(self, queries: List[str], method: str = "hybrid") -> Dict[str, List[SearchResult]]:
        """Поиск аналогов"""
        pass
    
    def export_results(self, results: Dict[str, Any], format: str = "excel") -> str:
        """Экспорт результатов"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        pass


class ExporterInterface(ABC):
    """Интерфейс для экспорта данных"""
    
    @abstractmethod
    def export_data(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                   filepath: Path, **kwargs) -> None:
        """
        Экспорт данных в файл
        
        Args:
            data: Данные для экспорта
            filepath: Путь к файлу
            **kwargs: Дополнительные параметры
        """
        pass


class DataManagerInterface(ABC):
    """Интерфейс для управления данными"""
    
    @abstractmethod
    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        """Загрузка данных из источника"""
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, destination: Union[str, Path]) -> None:
        """Сохранение данных"""
        pass
    
    @abstractmethod
    def get_path(self, path_type: str, **kwargs) -> Path:
        """Получение пути к данным"""
        pass


class ParameterExtractorInterface(ABC):
    """Интерфейс для извлечения параметров"""
    
    @abstractmethod
    def extract_parameters(self, text: str) -> List[ParameterData]:
        """
        Извлечение параметров из текста
        
        Args:
            text: Входной текст
            
        Returns:
            Список извлеченных параметров
        """
        pass
    
    @abstractmethod
    def extract_batch(self, texts: List[str]) -> List[List[ParameterData]]:
        """Пакетное извлечение параметров"""
        pass
