"""
Общие типы данных для модулей SAMe
"""

from typing import Dict, List, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class ProcessingStage(Enum):
    """Этапы обработки текста"""
    RAW = "raw"
    CLEANED = "cleaned"
    NORMALIZED = "normalized"
    LEMMATIZED = "lemmatized"
    ENHANCED = "enhanced"


class ParameterType(Enum):
    """Типы параметров"""
    NUMERIC = "numeric"
    UNIT = "unit"
    MATERIAL = "material"
    STANDARD = "standard"
    DIMENSION = "dimension"
    TECHNICAL_CODE = "technical_code"
    OTHER = "other"


@dataclass
class ProcessingResult:
    """Результат обработки текста"""
    original: str
    processed: str
    stages: Dict[ProcessingStage, str]
    metadata: Dict[str, Any]
    processing_time: float
    
    def get_stage(self, stage: ProcessingStage) -> str:
        """Получить результат определенного этапа обработки"""
        return self.stages.get(stage, self.processed)


@dataclass
class SearchResult:
    """Результат поиска"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")


@dataclass
class ParameterData:
    """Данные извлеченного параметра"""
    name: str
    value: Union[str, float, int]
    parameter_type: ParameterType
    confidence: float
    position: Optional[tuple] = None  # (start, end) позиция в тексте
    unit: Optional[str] = None
    normalized_value: Optional[Union[str, float, int]] = None
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


class SearchMethod(Enum):
    """Методы поиска"""
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class SearchConfig:
    """Конфигурация поиска"""
    method: SearchMethod = SearchMethod.HYBRID
    similarity_threshold: float = 0.6
    max_results: int = 10
    enable_caching: bool = True
    filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Валидация конфигурации"""
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"Similarity threshold must be between 0 and 1")
        if self.max_results < 1:
            raise ValueError(f"Max results must be >= 1")


@dataclass
class ExportConfig:
    """Конфигурация экспорта"""
    format: str = "excel"
    include_metadata: bool = True
    include_scores: bool = True
    max_rows: Optional[int] = None
    custom_columns: Optional[List[str]] = None


# Типы для совместимости с существующим кодом
TextProcessingResult = ProcessingResult
SearchResultItem = SearchResult
ExtractedParameter = ParameterData

# Алиасы для Union типов
CatalogData = Union[pd.DataFrame, List[Dict[str, Any]], str]
ProcessingInput = Union[str, List[str]]
SearchQuery = Union[str, List[str]]
