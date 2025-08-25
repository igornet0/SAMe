"""
Интеграционный модуль для всех оптимизаций SAMe системы
"""

from .integration import (
    SAMeOptimizer,
    OptimizationSuite,
    same_optimizer
)

from .phase3_integration import (
    Phase3Optimizer,
    Phase3Config,
    phase3_optimizer
)

__all__ = [
    "SAMeOptimizer",
    "OptimizationSuite", 
    "same_optimizer",
    "Phase3Optimizer",
    "Phase3Config",
    "phase3_optimizer"
]
