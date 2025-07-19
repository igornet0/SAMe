"""
Исключения для системы управления моделями
"""


class ModelError(Exception):
    """Базовое исключение для ошибок моделей"""
    pass


class ModelLoadError(ModelError):
    """Ошибка загрузки модели"""
    def __init__(self, model_name: str, error: str):
        self.model_name = model_name
        self.error = error
        super().__init__(f"Failed to load model '{model_name}': {error}")


class ModelNotFoundError(ModelError):
    """Модель не найдена"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not found")


class MemoryLimitExceededError(ModelError):
    """Превышен лимит памяти"""
    def __init__(self, current_usage: float, limit: float):
        self.current_usage = current_usage
        self.limit = limit
        super().__init__(f"Memory limit exceeded: {current_usage:.2f}GB > {limit:.2f}GB")


class ModelInitializationError(ModelError):
    """Ошибка инициализации модели"""
    def __init__(self, model_name: str, error: str):
        self.model_name = model_name
        self.error = error
        super().__init__(f"Failed to initialize model '{model_name}': {error}")


class ThreadSafetyError(ModelError):
    """Ошибка потокобезопасности"""
    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(f"Thread safety violation in operation: {operation}")
