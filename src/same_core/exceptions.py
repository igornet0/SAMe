"""
Общие исключения для модулей SAMe
"""


class SAMeError(Exception):
    """Базовое исключение для всех ошибок SAMe"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class ProcessingError(SAMeError):
    """Ошибки обработки текста"""
    pass


class SearchError(SAMeError):
    """Ошибки поиска"""
    pass


class ConfigurationError(SAMeError):
    """Ошибки конфигурации"""
    pass


class ModelError(SAMeError):
    """Ошибки работы с моделями"""
    pass


class DataError(SAMeError):
    """Ошибки работы с данными"""
    pass


class ExportError(SAMeError):
    """Ошибки экспорта"""
    pass


class ValidationError(SAMeError):
    """Ошибки валидации"""
    pass


class InitializationError(SAMeError):
    """Ошибки инициализации"""
    pass


# Специфичные исключения для обратной совместимости
class TextProcessingError(ProcessingError):
    """Ошибки обработки текста (алиас для ProcessingError)"""
    pass


class ParameterExtractionError(ProcessingError):
    """Ошибки извлечения параметров"""
    pass


class SearchEngineError(SearchError):
    """Ошибки поискового движка"""
    pass


class DatabaseError(DataError):
    """Ошибки базы данных"""
    pass
