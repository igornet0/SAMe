"""
Модуль мониторинга и аналитики поиска
"""

from .analytics import PerformanceMonitor, MetricCollector, AlertManager

# Алиас для обратной совместимости
SearchAnalytics = PerformanceMonitor

__all__ = [
    "PerformanceMonitor",
    "SearchAnalytics",  # Алиас
    "MetricCollector",
    "AlertManager"
]
