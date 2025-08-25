"""
SAMe Core - Общие интерфейсы и типы для модулей SAMe

Этот модуль содержит базовые интерфейсы, типы данных и исключения,
используемые всеми модулями системы SAMe.
"""

__version__ = "1.0.0"
__author__ = "igornet0"

from .interfaces import (
    TextProcessorInterface,
    SearchEngineInterface, 
    AnalogSearchEngineInterface,
    ExporterInterface,
    DataManagerInterface
)

from .types import (
    ProcessingResult,
    SearchResult,
    ParameterData
)

from .exceptions import (
    SAMeError,
    ProcessingError,
    SearchError,
    ConfigurationError
)

from .integration import (
    deprecated_import,
    create_proxy_import,
    ModuleRegistry,
    module_registry,
    validate_module_structure,
    setup_backward_compatibility
)

__all__ = [
    # Интерфейсы
    "TextProcessorInterface",
    "SearchEngineInterface",
    "AnalogSearchEngineInterface",
    "ExporterInterface",
    "DataManagerInterface",

    # Типы данных
    "ProcessingResult",
    "SearchResult",
    "ParameterData",

    # Исключения
    "SAMeError",
    "ProcessingError",
    "SearchError",
    "ConfigurationError",

    # Интеграция
    "deprecated_import",
    "create_proxy_import",
    "ModuleRegistry",
    "module_registry",
    "validate_module_structure",
    "setup_backward_compatibility"
]
