"""
Утилиты для интеграции между модулями SAMe
"""

import warnings
from typing import Dict, Any, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def deprecated_import(old_path: str, new_path: str, version: str = "2.0.0"):
    """
    Декоратор для отметки импортов как устаревших
    
    Args:
        old_path: Старый путь импорта
        new_path: Новый путь импорта
        version: Версия, в которой старый импорт будет удален
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Importing from '{old_path}' is deprecated and will be removed in version {version}. "
                f"Use '{new_path}' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_proxy_import(module_name: str, old_module: str, new_module: str):
    """
    Создает прокси-импорт для обратной совместимости
    
    Args:
        module_name: Имя импортируемого модуля/класса
        old_module: Старый путь модуля
        new_module: Новый путь модуля
    """
    try:
        # Пытаемся импортировать из нового модуля
        exec(f"from {new_module} import {module_name}")
        logger.debug(f"Successfully imported {module_name} from {new_module}")
        return True
    except ImportError:
        try:
            # Fallback на старый модуль
            exec(f"from {old_module} import {module_name}")
            logger.warning(f"Fallback import {module_name} from {old_module}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import {module_name} from both {new_module} and {old_module}: {e}")
            return False


class ModuleRegistry:
    """Реестр модулей для управления зависимостями"""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._dependencies: Dict[str, list] = {}
    
    def register_module(self, name: str, module: Any, dependencies: Optional[list] = None):
        """
        Регистрация модуля
        
        Args:
            name: Имя модуля
            module: Объект модуля
            dependencies: Список зависимостей
        """
        self._modules[name] = module
        self._dependencies[name] = dependencies or []
        logger.info(f"Registered module: {name}")
    
    def get_module(self, name: str) -> Any:
        """Получение модуля по имени"""
        if name not in self._modules:
            raise KeyError(f"Module {name} not registered")
        return self._modules[name]
    
    def check_dependencies(self, name: str) -> bool:
        """Проверка зависимостей модуля"""
        if name not in self._dependencies:
            return True
        
        for dep in self._dependencies[name]:
            if dep not in self._modules:
                logger.error(f"Missing dependency {dep} for module {name}")
                return False
        return True
    
    def get_load_order(self) -> list:
        """Получение порядка загрузки модулей с учетом зависимостей"""
        loaded = set()
        order = []
        
        def load_module(name: str):
            if name in loaded:
                return
            
            # Сначала загружаем зависимости
            for dep in self._dependencies.get(name, []):
                load_module(dep)
            
            order.append(name)
            loaded.add(name)
        
        for module_name in self._modules:
            load_module(module_name)
        
        return order


# Глобальный реестр модулей
module_registry = ModuleRegistry()


def validate_module_structure(module_path: str, required_components: list) -> bool:
    """
    Валидация структуры модуля
    
    Args:
        module_path: Путь к модулю
        required_components: Список обязательных компонентов
        
    Returns:
        True если структура корректна
    """
    try:
        import importlib
        module = importlib.import_module(module_path)
        
        for component in required_components:
            if not hasattr(module, component):
                logger.error(f"Missing required component {component} in {module_path}")
                return False
        
        logger.info(f"Module {module_path} structure is valid")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        return False


def setup_backward_compatibility():
    """Настройка обратной совместимости для всех модулей"""
    
    # Регистрируем модули в правильном порядке зависимостей
    module_registry.register_module("same_core", None, [])
    module_registry.register_module("same_clear", None, ["same_core"])
    module_registry.register_module("same_search", None, ["same_core", "same_clear"])
    module_registry.register_module("same_api", None, ["same_core", "same_clear", "same_search"])
    
    logger.info("Backward compatibility setup completed")


# Автоматическая настройка при импорте модуля
setup_backward_compatibility()
