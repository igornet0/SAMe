"""
Модуль для обработки текста и классификации токенов.

Предоставляет инструменты для:
- Классификации токенов товарных наименований
- Предобработки названий товаров
- Нормализации текста для поиска аналогов
"""

from .token_classifier import TokenClassifier, create_token_classifier
from .product_name_preprocessor import ProductNamePreprocessor, create_product_preprocessor
from .preprocessor import TextPreprocessor, PreprocessorConfig

__all__ = [
    'TokenClassifier',
    'create_token_classifier',
    'ProductNamePreprocessor', 
    'create_product_preprocessor',
    'TextPreprocessor',
    'PreprocessorConfig'
]

__version__ = '1.0.0'
__author__ = 'SAMe Team'
