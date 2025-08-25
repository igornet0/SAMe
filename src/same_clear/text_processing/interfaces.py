"""
Абстрактные интерфейсы для компонентов обработки текста
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class TextProcessorInterface(ABC):
    """Базовый интерфейс для всех компонентов обработки текста"""
    
    @abstractmethod
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Обработка одного текста
        
        Args:
            text: Входной текст
            
        Returns:
            Результат обработки
        """
        pass
    
    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Пакетная обработка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Список результатов обработки
        """
        pass


class TextCleanerInterface(TextProcessorInterface):
    """Интерфейс для очистки текста"""
    
    @abstractmethod
    def clean_text(self, text: str) -> Dict[str, Any]:
        """Очистка одного текста"""
        pass
    
    @abstractmethod
    def clean_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная очистка текстов"""
        pass


class LemmatizerInterface(TextProcessorInterface):
    """Интерфейс для лемматизации"""
    
    @abstractmethod
    def lemmatize_text(self, text: str) -> Dict[str, Any]:
        """Лемматизация одного текста"""
        pass
    
    @abstractmethod
    def lemmatize_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная лемматизация"""
        pass
    
    @abstractmethod
    async def lemmatize_text_async(self, text: str) -> Dict[str, Any]:
        """Асинхронная лемматизация одного текста"""
        pass
    
    @abstractmethod
    async def lemmatize_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Асинхронная пакетная лемматизация"""
        pass


class NormalizerInterface(TextProcessorInterface):
    """Интерфейс для нормализации текста"""
    
    @abstractmethod
    def normalize_text(self, text: str) -> Dict[str, Any]:
        """Нормализация одного текста"""
        pass
    
    @abstractmethod
    def normalize_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная нормализация"""
        pass


class SearchEngineInterface(ABC):
    """Базовый интерфейс для поисковых движков"""
    
    @abstractmethod
    def fit(self, documents: List[str], document_ids: List[Any] = None, 
            metadata: List[Dict[str, Any]] = None) -> None:
        """
        Обучение поискового движка
        
        Args:
            documents: Список документов
            document_ids: Список ID документов
            metadata: Метаданные документов
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Поиск по запросу
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список результатов поиска
        """
        pass
    
    @abstractmethod
    def is_fitted(self) -> bool:
        """Проверка, обучен ли движок"""
        pass


class FuzzySearchInterface(SearchEngineInterface):
    """Интерфейс для нечеткого поиска"""
    pass


class SemanticSearchInterface(SearchEngineInterface):
    """Интерфейс для семантического поиска"""
    
    @abstractmethod
    async def fit_async(self, documents: List[str], document_ids: List[Any] = None,
                       metadata: List[Dict[str, Any]] = None) -> None:
        """Асинхронное обучение"""
        pass


class HybridSearchInterface(SearchEngineInterface):
    """Интерфейс для гибридного поиска"""
    
    @abstractmethod
    def set_fuzzy_engine(self, engine: FuzzySearchInterface) -> None:
        """Установка движка нечеткого поиска"""
        pass
    
    @abstractmethod
    def set_semantic_engine(self, engine: SemanticSearchInterface) -> None:
        """Установка движка семантического поиска"""
        pass


class TextPreprocessorInterface(ABC):
    """Интерфейс для главного предобработчика"""
    
    @abstractmethod
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Полная предобработка одного текста"""
        pass
    
    @abstractmethod
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная предобработка"""
        pass
    
    @abstractmethod
    async def preprocess_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Асинхронная пакетная предобработка"""
        pass
    
    @abstractmethod
    def set_cleaner(self, cleaner: TextCleanerInterface) -> None:
        """Установка компонента очистки"""
        pass
    
    @abstractmethod
    def set_lemmatizer(self, lemmatizer: LemmatizerInterface) -> None:
        """Установка компонента лемматизации"""
        pass
    
    @abstractmethod
    def set_normalizer(self, normalizer: NormalizerInterface) -> None:
        """Установка компонента нормализации"""
        pass


class AnalogSearchEngineInterface(ABC):
    """Интерфейс для главного движка поиска аналогов"""
    
    @abstractmethod
    def search_analogs(self, query: str, search_type: str = "hybrid", 
                      top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Поиск аналогов
        
        Args:
            query: Поисковый запрос
            search_type: Тип поиска (fuzzy, semantic, hybrid)
            top_k: Количество результатов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список найденных аналогов
        """
        pass
    
    @abstractmethod
    def set_preprocessor(self, preprocessor: TextPreprocessorInterface) -> None:
        """Установка предобработчика"""
        pass
    
    @abstractmethod
    def set_search_engine(self, engine: SearchEngineInterface, engine_type: str) -> None:
        """Установка поискового движка"""
        pass
