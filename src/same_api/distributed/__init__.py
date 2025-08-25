"""
Модуль распределенной обработки для масштабирования SAMe системы
"""

from .processor import (
    DistributedProcessor,
    DistributedConfig,
    ProcessingMode,
    TaskResult,
    distributed_processor
)

__all__ = [
    "DistributedProcessor",
    "DistributedConfig", 
    "ProcessingMode",
    "TaskResult",
    "distributed_processor"
]
