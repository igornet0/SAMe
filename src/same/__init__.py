"""
SAMe - Search Analog Model Engine

Система поиска аналогов материально-технических ресурсов (МТР)
с использованием современных методов машинного обучения и обработки естественного языка.
"""

__version__ = "0.1.0"
__author__ = "igornet0"
__email__ = "93836464+igornet0@users.noreply.github.com"

from .analog_search_engine import AnalogSearchEngine
from .same_model import SAMeModel
from .same_process import SAMeProcess

__all__ = [
    "AnalogSearchEngine",
    "SAMeModel",
    "SAMeProcess",
]