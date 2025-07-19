"""
SAMe - Search Analog Model Engine

Система поиска аналогов материально-технических ресурсов (МТР)
с использованием современных методов машинного обучения и обработки естественного языка.
"""

__version__ = "1.2.0"
__author__ = "igornet0"
__email__ = "93836464+igornet0@users.noreply.github.com"

# Note: AnalogSearchEngine can be imported directly from same.analog_search_engine
# Temporarily disabled from main __init__ to avoid circular imports during development

from same.database import get_db_helper
from same.settings import settings

__all__ = [
    "get_db_helper",
    "settings",
    # "AnalogSearchEngine",  # Available via: from same.analog_search_engine import AnalogSearchEngine
]