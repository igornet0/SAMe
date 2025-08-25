"""
SAMe Clear - Модуль обработки и очистки текста

Этот модуль содержит все компоненты для предобработки, очистки,
нормализации текста и извлечения параметров.
"""

__version__ = "1.0.0"
__author__ = "igornet0"

# Импорты из text_processing (новые модули классификации токенов)
from .text_processing import (
    TokenClassifier, create_token_classifier,
    ProductNamePreprocessor, create_product_preprocessor
)

# Импорты из text_processing/preprocessor
from .text_processing.preprocessor import (
    TextPreprocessor, PreprocessorConfig
)

# Импорты из parameter_extraction
from .parameter_extraction import (
    RegexParameterExtractor,
    ParameterParser, ParameterParserConfig,
    ParameterFormatter, ParameterAnalyzer, ParameterDataFrameUtils,
    ParameterPattern, ParameterType, ExtractedParameter
)

# Импорты утилит
from .utils import (
    case_converter
)

__all__ = [
    # Text Processing (новые модули)
    "TokenClassifier", "create_token_classifier",
    "ProductNamePreprocessor", "create_product_preprocessor",
    
    # Text Processing (preprocessor)
    "TextPreprocessor", "PreprocessorConfig",
    
    # Parameter Extraction
    "RegexParameterExtractor",
    "ParameterParser", "ParameterParserConfig",
    "ParameterFormatter", "ParameterAnalyzer", "ParameterDataFrameUtils",
    "ParameterPattern", "ParameterType", "ExtractedParameter",
    
    # Utils
    "case_converter"
]
