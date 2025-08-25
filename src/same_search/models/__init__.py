"""
Модуль для работы с ML моделями
"""

from .model_manager import AdvancedModelManager, ModelType, ModelConfig, get_model_manager
from .memory_monitor import MemoryMonitor
from .exceptions import ModelLoadError, ModelNotFoundError, MemoryLimitExceededError
from .quantization import QuantizationConfig, QuantizationType

# Алиас для обратной совместимости
ModelManager = AdvancedModelManager

__all__ = [
    "AdvancedModelManager",
    "ModelManager",  # Алиас
    "ModelType",
    "ModelConfig",
    "get_model_manager",
    "MemoryMonitor",
    "ModelLoadError",
    "ModelNotFoundError",
    "MemoryLimitExceededError",
    "QuantizationConfig",
    "QuantizationType"
]
