"""
Продвинутый менеджер моделей с поддержкой потокобезопасности,
мониторинга памяти и ленивой загрузки
"""

import asyncio
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, Union, Callable, Set
from pathlib import Path

import spacy
import torch
from sentence_transformers import SentenceTransformer

from .memory_monitor import MemoryMonitor, MemoryStats
from .exceptions import (
    ModelLoadError, ModelNotFoundError, MemoryLimitExceededError,
    ModelInitializationError, ThreadSafetyError
)
from ..settings import get_settings

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Типы поддерживаемых моделей"""
    SPACY = "spacy"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    name: str
    model_type: ModelType
    model_path: str
    device: str = "cpu"
    cache_size_gb: float = 0.5  # Приблизительный размер в памяти
    lazy_load: bool = True
    max_idle_time: int = 3600  # Время в секундах до выгрузки неиспользуемой модели
    initialization_timeout: int = 300  # Таймаут загрузки в секундах
    
    # SpaCy специфичные настройки
    spacy_disable_pipes: Optional[list] = None
    
    # SentenceTransformer специфичные настройки
    normalize_embeddings: bool = True
    use_gpu: bool = False


class ModelInstance:
    """Обертка для экземпляра модели с метаданными"""
    
    def __init__(self, model: Any, config: ModelConfig):
        self.model = model
        self.config = config
        self.load_time = time.time()
        self.last_access_time = time.time()
        self.access_count = 0
        self.is_loading = False
        self._lock = threading.RLock()
    
    def access(self) -> Any:
        """Получение доступа к модели с обновлением статистики"""
        with self._lock:
            self.last_access_time = time.time()
            self.access_count += 1
            return self.model
    
    def is_idle(self) -> bool:
        """Проверка, не используется ли модель длительное время"""
        return (time.time() - self.last_access_time) > self.config.max_idle_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики использования модели"""
        return {
            "load_time": self.load_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "idle_time": time.time() - self.last_access_time,
            "memory_size_gb": self.config.cache_size_gb
        }


class AdvancedModelManager:
    """
    Продвинутый менеджер моделей с поддержкой:
    - Потокобезопасности
    - Ленивой загрузки
    - Мониторинга памяти
    - Автоматической выгрузки неиспользуемых моделей
    - Асинхронной загрузки
    """
    
    _instance: Optional['AdvancedModelManager'] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> 'AdvancedModelManager':
        """Singleton pattern с потокобезопасностью"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация менеджера моделей"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.settings = get_settings()
        
        # Хранилища моделей и конфигураций
        self._models: Dict[str, ModelInstance] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._loading_futures: Dict[str, asyncio.Future] = {}
        
        # Потокобезопасность
        self._models_lock = threading.RLock()
        self._loading_lock = threading.RLock()
        
        # Мониторинг памяти
        memory_limit = self._parse_memory_limit(self.settings.performance.memory_limit)
        self.memory_monitor = MemoryMonitor(
            memory_limit_gb=memory_limit,
            warning_threshold=0.8,
            cleanup_threshold=0.9
        )
        
        # Executor для асинхронной загрузки
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ModelLoader"
        )
        
        # Регистрируем callback'и для очистки памяти
        self.memory_monitor.register_cleanup_callback(
            "model_manager", self._cleanup_idle_models
        )
        self.memory_monitor.register_warning_callback(
            "model_manager", self._on_memory_warning
        )
        
        # Запускаем мониторинг
        self.memory_monitor.start_monitoring()
        
        # Регистрируем стандартные конфигурации
        self._register_default_configs()
        
        logger.info("AdvancedModelManager initialized")
    
    def _parse_memory_limit(self, memory_limit: str) -> float:
        """Парсинг лимита памяти из строки"""
        if memory_limit.upper().endswith('GB'):
            return float(memory_limit[:-2])
        elif memory_limit.upper().endswith('MB'):
            return float(memory_limit[:-2]) / 1024
        else:
            return float(memory_limit)
    
    def _register_default_configs(self):
        """Регистрация стандартных конфигураций моделей"""
        # SpaCy модели
        spacy_models = [
            "ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"
        ]
        
        for model_name in spacy_models:
            config = ModelConfig(
                name=model_name,
                model_type=ModelType.SPACY,
                model_path=model_name,
                cache_size_gb=0.1 if "sm" in model_name else 0.3 if "md" in model_name else 0.8,
                device="cpu"
            )
            self.register_model_config(config)
        
        # SentenceTransformer модели
        st_config = ModelConfig(
            name="default_sentence_transformer",
            model_type=ModelType.SENTENCE_TRANSFORMER,
            model_path=self.settings.ml.semantic_model,
            cache_size_gb=1.2,
            device="cuda" if self.settings.ml.use_gpu and torch.cuda.is_available() else "cpu",
            use_gpu=self.settings.ml.use_gpu
        )
        self.register_model_config(st_config)
    
    def register_model_config(self, config: ModelConfig):
        """Регистрация конфигурации модели"""
        with self._models_lock:
            self._model_configs[config.name] = config
            logger.debug(f"Registered model config: {config.name}")
    
    async def get_spacy_model(self, model_name: str = None) -> spacy.Language:
        """
        Получение SpaCy модели с ленивой загрузкой
        
        Args:
            model_name: Имя модели (по умолчанию из настроек)
            
        Returns:
            Загруженная SpaCy модель
        """
        model_name = model_name or self.settings.ml.spacy_model
        
        if model_name not in self._model_configs:
            # Создаем конфигурацию на лету
            config = ModelConfig(
                name=model_name,
                model_type=ModelType.SPACY,
                model_path=model_name,
                cache_size_gb=0.5,
                device="cpu"
            )
            self.register_model_config(config)
        
        model_instance = await self._get_or_load_model(model_name)
        return model_instance.access()
    
    async def get_sentence_transformer(self, model_name: str = None) -> SentenceTransformer:
        """
        Получение SentenceTransformer модели с ленивой загрузкой
        
        Args:
            model_name: Имя модели (по умолчанию из настроек)
            
        Returns:
            Загруженная SentenceTransformer модель
        """
        model_name = model_name or "default_sentence_transformer"
        
        if model_name not in self._model_configs:
            # Создаем конфигурацию на лету
            config = ModelConfig(
                name=model_name,
                model_type=ModelType.SENTENCE_TRANSFORMER,
                model_path=self.settings.ml.semantic_model,
                cache_size_gb=1.2,
                device="cuda" if self.settings.ml.use_gpu and torch.cuda.is_available() else "cpu",
                use_gpu=self.settings.ml.use_gpu
            )
            self.register_model_config(config)
        
        model_instance = await self._get_or_load_model(model_name)
        return model_instance.access()

    def get_sentence_transformer_sync(self, model_name: str = None) -> SentenceTransformer:
        """Синхронное получение SentenceTransformer модели"""
        model_name = model_name or "default_sentence_transformer"

        # Простая синхронная загрузка для обратной совместимости
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.settings.ml.semantic_model)

    async def _get_or_load_model(self, model_name: str) -> ModelInstance:
        """Получение или загрузка модели с потокобезопасностью"""
        # Проверяем, есть ли уже загруженная модель
        with self._models_lock:
            if model_name in self._models:
                return self._models[model_name]

        # Проверяем, не загружается ли модель в данный момент
        with self._loading_lock:
            if model_name in self._loading_futures:
                # Ждем завершения загрузки
                return await self._loading_futures[model_name]

            # Создаем future для загрузки
            future = asyncio.Future()
            self._loading_futures[model_name] = future

        try:
            # Загружаем модель
            model_instance = await self._load_model(model_name)

            # Сохраняем в кэше
            with self._models_lock:
                self._models[model_name] = model_instance

            # Регистрируем использование памяти
            self.memory_monitor.register_model_memory(
                model_name, model_instance.config.cache_size_gb
            )

            # Завершаем future
            future.set_result(model_instance)

            return model_instance

        except Exception as e:
            # В случае ошибки завершаем future с исключением
            future.set_exception(e)
            raise
        finally:
            # Удаляем future из списка загружающихся
            with self._loading_lock:
                self._loading_futures.pop(model_name, None)

    async def _load_model(self, model_name: str) -> ModelInstance:
        """Загрузка модели в отдельном потоке"""
        config = self._model_configs.get(model_name)
        if not config:
            raise ModelNotFoundError(model_name)

        # Проверяем лимит памяти
        if self.memory_monitor.check_memory_limit():
            # Пытаемся освободить память
            self._cleanup_idle_models()
            if self.memory_monitor.check_memory_limit():
                stats = self.memory_monitor.get_memory_stats()
                raise MemoryLimitExceededError(
                    stats.used_memory, self.memory_monitor.memory_limit_gb
                )

        logger.info(f"Loading model: {model_name} ({config.model_type.value})")

        try:
            # Загружаем модель в executor'е
            loop = asyncio.get_event_loop()
            model = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, self._load_model_sync, config
                ),
                timeout=config.initialization_timeout
            )

            model_instance = ModelInstance(model, config)
            logger.info(f"Successfully loaded model: {model_name}")

            return model_instance

        except asyncio.TimeoutError:
            raise ModelLoadError(model_name, f"Loading timeout ({config.initialization_timeout}s)")
        except Exception as e:
            raise ModelLoadError(model_name, str(e))

    def _load_model_sync(self, config: ModelConfig) -> Any:
        """Синхронная загрузка модели"""
        if config.model_type == ModelType.SPACY:
            return self._load_spacy_model(config)
        elif config.model_type == ModelType.SENTENCE_TRANSFORMER:
            return self._load_sentence_transformer_model(config)
        else:
            raise ModelLoadError(config.name, f"Unsupported model type: {config.model_type}")

    def _load_spacy_model(self, config: ModelConfig) -> spacy.Language:
        """Загрузка SpaCy модели"""
        try:
            # Пытаемся загрузить основную модель
            nlp = spacy.load(config.model_path)

            # Отключаем ненужные компоненты для экономии памяти
            if config.spacy_disable_pipes:
                for pipe in config.spacy_disable_pipes:
                    if pipe in nlp.pipe_names:
                        nlp.disable_pipe(pipe)

            logger.debug(f"SpaCy model loaded: {config.model_path}")
            return nlp

        except OSError:
            # Пытаемся загрузить fallback модель
            fallback_models = ["ru_core_news_sm", "ru_core_news_md"]

            for fallback in fallback_models:
                if fallback != config.model_path:
                    try:
                        logger.warning(f"Fallback to {fallback} for {config.model_path}")
                        return spacy.load(fallback)
                    except OSError:
                        continue

            raise ModelLoadError(
                config.name,
                f"SpaCy model not found: {config.model_path}. "
                f"Install with: python -m spacy download {config.model_path}"
            )

    def _load_sentence_transformer_model(self, config: ModelConfig) -> SentenceTransformer:
        """Загрузка SentenceTransformer модели"""
        try:
            device = config.device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            model = SentenceTransformer(config.model_path, device=device)

            logger.debug(f"SentenceTransformer model loaded: {config.model_path} on {device}")
            return model

        except Exception as e:
            raise ModelLoadError(config.name, f"Failed to load SentenceTransformer: {str(e)}")

    def _cleanup_idle_models(self):
        """Очистка неиспользуемых моделей"""
        logger.info("Starting cleanup of idle models")

        with self._models_lock:
            idle_models = []

            for model_name, model_instance in self._models.items():
                if model_instance.is_idle():
                    idle_models.append(model_name)

            for model_name in idle_models:
                logger.info(f"Unloading idle model: {model_name}")
                model_instance = self._models.pop(model_name)

                # Отменяем регистрацию памяти
                self.memory_monitor.unregister_model_memory(model_name)

                # Удаляем ссылку на модель для сборки мусора
                del model_instance

        if idle_models:
            logger.info(f"Unloaded {len(idle_models)} idle models")

    def _on_memory_warning(self, stats: MemoryStats):
        """Обработка предупреждения о высоком использовании памяти"""
        logger.warning(f"High memory usage detected: {stats.used_memory:.2f}GB")

        # Принудительная очистка неиспользуемых моделей
        self._cleanup_idle_models()

    def get_model_stats(self) -> Dict[str, Any]:
        """Получение статистики всех моделей"""
        with self._models_lock:
            stats = {}
            for model_name, model_instance in self._models.items():
                stats[model_name] = model_instance.get_stats()

            return {
                "loaded_models": len(self._models),
                "total_memory_gb": sum(
                    instance.config.cache_size_gb
                    for instance in self._models.values()
                ),
                "models": stats,
                "memory_stats": self.memory_monitor.get_memory_stats().__dict__
            }

    def unload_model(self, model_name: str) -> bool:
        """Принудительная выгрузка модели"""
        with self._models_lock:
            if model_name in self._models:
                logger.info(f"Manually unloading model: {model_name}")
                model_instance = self._models.pop(model_name)
                self.memory_monitor.unregister_model_memory(model_name)
                del model_instance
                return True
            return False

    def shutdown(self):
        """Завершение работы менеджера"""
        logger.info("Shutting down AdvancedModelManager")

        # Останавливаем мониторинг памяти
        self.memory_monitor.stop_monitoring()

        # Выгружаем все модели
        with self._models_lock:
            for model_name in list(self._models.keys()):
                self.unload_model(model_name)

        # Завершаем executor
        self._executor.shutdown(wait=True)

        logger.info("AdvancedModelManager shutdown complete")

    def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизация использования памяти"""
        import gc

        logger.info("Starting memory optimization")

        # Получаем статистику до оптимизации
        stats_before = self.memory_monitor.get_memory_stats()

        # Принудительная сборка мусора
        collected = gc.collect()

        # Очистка кэшей PyTorch если доступен
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Получаем статистику после оптимизации
        stats_after = self.memory_monitor.get_memory_stats()

        optimization_result = {
            'memory_freed_mb': (stats_before.used_memory - stats_after.used_memory) / (1024 * 1024),
            'gc_collected_objects': collected,
            'memory_before_mb': stats_before.used_memory / (1024 * 1024),
            'memory_after_mb': stats_after.used_memory / (1024 * 1024),
            'optimization_timestamp': time.time()
        }

        logger.info(f"Memory optimization completed: freed {optimization_result['memory_freed_mb']:.1f} MB")
        return optimization_result

    def __del__(self):
        """Деструктор"""
        try:
            self.shutdown()
        except:
            pass


# Глобальный экземпляр менеджера моделей
_model_manager: Optional[AdvancedModelManager] = None


def get_model_manager() -> AdvancedModelManager:
    """Получение глобального экземпляра менеджера моделей"""
    global _model_manager
    if _model_manager is None:
        _model_manager = AdvancedModelManager()
    return _model_manager
