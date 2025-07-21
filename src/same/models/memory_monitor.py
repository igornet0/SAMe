"""
Мониторинг памяти для системы управления моделями
"""

import psutil
import gc
import logging
import threading
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Статистика использования памяти"""
    total_memory: float  # GB
    used_memory: float   # GB
    available_memory: float  # GB
    memory_percent: float
    model_memory: float  # GB (приблизительная оценка)
    timestamp: datetime


class MemoryMonitor:
    """Монитор памяти с автоматической очисткой"""
    
    def __init__(self, 
                 memory_limit_gb: float = 8.0,
                 warning_threshold: float = 0.8,
                 cleanup_threshold: float = 0.9,
                 monitoring_interval: float = 30.0):
        """
        Args:
            memory_limit_gb: Лимит памяти в GB
            warning_threshold: Порог предупреждения (0.0-1.0)
            cleanup_threshold: Порог автоочистки (0.0-1.0)
            monitoring_interval: Интервал мониторинга в секундах
        """
        self.memory_limit_gb = memory_limit_gb
        self.warning_threshold = warning_threshold
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._cleanup_callbacks: Dict[str, Callable] = {}
        self._model_memory_estimates: Dict[str, float] = {}
        
        self._last_stats: Optional[MemoryStats] = None
        self._warning_callbacks: Dict[str, Callable] = {}
        
    def start_monitoring(self):
        """Запуск мониторинга памяти"""
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.warning("Memory monitoring already running")
                return
                
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="MemoryMonitor"
            )
            self._monitoring_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Остановка мониторинга памяти"""
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._stop_monitoring.set()
                self._monitoring_thread.join(timeout=5.0)
                if self._monitoring_thread.is_alive():
                    logger.warning("Memory monitoring thread did not stop gracefully")
                else:
                    logger.info("Memory monitoring stopped")
            self._monitoring_thread = None
    
    def get_memory_stats(self) -> MemoryStats:
        """Получение текущей статистики памяти"""
        memory = psutil.virtual_memory()
        
        total_gb = memory.total / (1024**3)
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        
        # Приблизительная оценка памяти моделей
        model_memory_gb = sum(self._model_memory_estimates.values())
        
        stats = MemoryStats(
            total_memory=total_gb,
            used_memory=used_gb,
            available_memory=available_gb,
            memory_percent=memory.percent,
            model_memory=model_memory_gb,
            timestamp=datetime.now()
        )
        
        self._last_stats = stats
        return stats
    
    def register_model_memory(self, model_name: str, memory_gb: float):
        """Регистрация использования памяти моделью"""
        with self._lock:
            self._model_memory_estimates[model_name] = memory_gb
            logger.debug(f"Registered memory usage for {model_name}: {memory_gb:.2f}GB")
    
    def unregister_model_memory(self, model_name: str):
        """Отмена регистрации памяти модели"""
        with self._lock:
            if model_name in self._model_memory_estimates:
                memory_gb = self._model_memory_estimates.pop(model_name)
                logger.debug(f"Unregistered memory usage for {model_name}: {memory_gb:.2f}GB")
    
    def register_cleanup_callback(self, name: str, callback: Callable):
        """Регистрация callback для автоочистки"""
        with self._lock:
            self._cleanup_callbacks[name] = callback
            logger.debug(f"Registered cleanup callback: {name}")
    
    def register_warning_callback(self, name: str, callback: Callable):
        """Регистрация callback для предупреждений"""
        with self._lock:
            self._warning_callbacks[name] = callback
            logger.debug(f"Registered warning callback: {name}")
    
    def force_cleanup(self):
        """Принудительная очистка памяти"""
        logger.info("Starting forced memory cleanup")
        
        # Вызываем все callback'и очистки
        with self._lock:
            for name, callback in self._cleanup_callbacks.items():
                try:
                    logger.debug(f"Executing cleanup callback: {name}")
                    callback()
                except Exception as e:
                    logger.error(f"Error in cleanup callback {name}: {e}")
        
        # Принудительная сборка мусора
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Обновляем статистику
        stats = self.get_memory_stats()
        logger.info(f"Memory after cleanup: {stats.used_memory:.2f}GB / {stats.total_memory:.2f}GB")
    
    def check_memory_limit(self) -> bool:
        """Проверка превышения лимита памяти"""
        stats = self.get_memory_stats()
        return stats.used_memory > self.memory_limit_gb
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        logger.info("Memory monitoring loop started")
        
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                stats = self.get_memory_stats()
                usage_ratio = stats.used_memory / self.memory_limit_gb
                
                # Проверяем пороги
                if usage_ratio >= self.cleanup_threshold:
                    logger.warning(f"Memory usage critical: {usage_ratio:.1%} of limit")
                    self.force_cleanup()
                elif usage_ratio >= self.warning_threshold:
                    logger.warning(f"Memory usage high: {usage_ratio:.1%} of limit")
                    self._trigger_warning_callbacks(stats)
                
                # Логируем статистику каждые 5 минут
                if int(time.time()) % 300 == 0:
                    logger.info(f"Memory stats: {stats.used_memory:.2f}GB used, "
                              f"{stats.model_memory:.2f}GB models, "
                              f"{usage_ratio:.1%} of limit")
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
    
    def _trigger_warning_callbacks(self, stats: MemoryStats):
        """Вызов callback'ов предупреждений"""
        with self._lock:
            for name, callback in self._warning_callbacks.items():
                try:
                    callback(stats)
                except Exception as e:
                    logger.error(f"Error in warning callback {name}: {e}")
