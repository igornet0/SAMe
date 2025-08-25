"""
SAMe API - Модуль API и интеграций

Этот модуль содержит FastAPI приложение, работу с базой данных,
экспорт данных и управление данными.
"""

__version__ = "1.0.0"
__author__ = "igornet0"

# Импорты из api
from .api import (
    create_app
)

# Импорты из database
from .database import (
    Database, get_db_helper, select_working_url,
    Base,
    User, Item, ItemParameter
)

# Импорты из export
from .export import (
    ExcelExporter, ExportConfig,
    ReportGenerator
)

# Импорты из settings
from .settings import (
    settings, get_settings
)

# Импорты из distributed
from .distributed import (
    DistributedProcessor, DistributedConfig, ProcessingMode,
    TaskResult, distributed_processor
)

# Импорты из realtime
from .realtime import (
    RealTimeProcessor, StreamEvent, WebSocketConnection,
    EventType, StreamingMode, realtime_processor
)

# Импорты из optimizations
from .optimizations import (
    SAMeOptimizer, OptimizationSuite, same_optimizer,
    Phase3Optimizer, Phase3Config, phase3_optimizer
)

__all__ = [
    # API
    "create_app",

    # Database
    "Database", "get_db_helper", "select_working_url",
    "Base",
    "User", "Item", "ItemParameter",

    # Export
    "ExcelExporter", "ExportConfig",
    "ReportGenerator",

    # Settings
    "settings", "get_settings",

    # Distributed Processing
    "DistributedProcessor", "DistributedConfig", "ProcessingMode",
    "TaskResult", "distributed_processor",

    # Real-time Processing
    "RealTimeProcessor", "StreamEvent", "WebSocketConnection",
    "EventType", "StreamingMode", "realtime_processor",

    # Optimizations
    "SAMeOptimizer", "OptimizationSuite", "same_optimizer",
    "Phase3Optimizer", "Phase3Config", "phase3_optimizer"
]
