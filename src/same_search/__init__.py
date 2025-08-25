"""
SAMe Search - Модуль поиска и индексации

Этот модуль содержит поисковые алгоритмы, управление ML моделями,
кэширование и мониторинг поисковых операций.
"""

__version__ = "1.0.0"
__author__ = "igornet0"

# Импорты из search_engine
from .search_engine import (
    FuzzySearchEngine, FuzzySearchConfig,
    SemanticSearchEngine, SemanticSearchConfig,
    HybridSearchEngine, HybridSearchConfig,
    SearchIndexer, IndexConfig
)

# Импорты из models
from .models import (
    get_model_manager,
    ModelManager,
    MemoryMonitor,
    QuantizationConfig
)

# Импорты из caching
from .caching import (
    AdvancedCache
)

# Импорты из monitoring
from .monitoring import (
    SearchAnalytics
)

from .searg_processor import AnalogSearchProcessor

__all__ = [
    # Search Engine
    "FuzzySearchEngine", "FuzzySearchConfig",
    "SemanticSearchEngine", "SemanticSearchConfig", 
    "HybridSearchEngine", "HybridSearchConfig",
    "SearchIndexer", "IndexConfig",
    
    # Models
    "get_model_manager",
    "ModelManager",
    "MemoryMonitor", 
    "QuantizationConfig",
    
    # Caching
    "AdvancedCache",
    
    # Monitoring
    "SearchAnalytics",

    # Search Processor
    "AnalogSearchProcessor"
]
