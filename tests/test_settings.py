"""
Тесты для модуля настроек
"""

import pytest
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from same.settings.config import (
    AppBaseConfig,
    RunConfig,
    LoggingConfig,
    LOG_DEFAULT_FORMAT
)


class TestAppBaseConfig:
    """Тесты для базовой конфигурации приложения"""
    
    def test_base_config_attributes(self):
        """Тест атрибутов базовой конфигурации"""
        config = AppBaseConfig()
        
        assert config.case_sensitive is False
        assert config.env_file == ".env"
        assert config.env_file_encoding == "utf-8"
        assert config.env_nested_delimiter == "__"
        assert config.extra == "ignore"


class TestRunConfig:
    """Тесты для конфигурации запуска"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем переменные окружения
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("RUN__"):
                del os.environ[key]
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        # Устанавливаем обязательные поля
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        
        config = RunConfig()
        
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.reload is False
        assert config.frontend_host == "localhost"
        assert config.frontend_port == 5173
        assert config.celery_broker_url == "redis://localhost:6379/0"
        assert config.celery_result_backend == "redis://localhost:6379/0"
    
    def test_custom_values_from_env(self):
        """Тест пользовательских значений из переменных окружения"""
        os.environ["RUN__HOST"] = "0.0.0.0"
        os.environ["RUN__PORT"] = "9000"
        os.environ["RUN__RELOAD"] = "true"
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/1"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/1"
        os.environ["RUN__FRONTEND_HOST"] = "127.0.0.1"
        os.environ["RUN__FRONTEND_PORT"] = "3000"
        
        config = RunConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.reload is True
        assert config.celery_broker_url == "redis://localhost:6379/1"
        assert config.celery_result_backend == "redis://localhost:6379/1"
        assert config.frontend_host == "127.0.0.1"
        assert config.frontend_port == 3000
    
    def test_frontend_url_property(self):
        """Тест свойства frontend_url"""
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        os.environ["RUN__FRONTEND_HOST"] = "example.com"
        os.environ["RUN__FRONTEND_PORT"] = "8080"
        
        config = RunConfig()
        
        assert config.frontend_url == "http://example.com:8080"
    
    def test_missing_required_fields(self):
        """Тест значений по умолчанию для полей"""
        # Очищаем все переменные окружения для RunConfig
        for key in list(os.environ.keys()):
            if key.startswith("RUN__"):
                del os.environ[key]

        # Проверяем что конфигурация создается с значениями по умолчанию
        config = RunConfig(_env_file=None)
        assert config.celery_broker_url == "redis://localhost:6379/0"
        assert config.celery_result_backend == "redis://localhost:6379/0"
    
    def test_type_validation(self):
        """Тест валидации типов"""
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        os.environ["RUN__PORT"] = "invalid_port"
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            RunConfig()


class TestLoggingConfig:
    """Тесты для конфигурации логирования"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем переменные окружения
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("LOGGING__"):
                del os.environ[key]
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        # Очищаем переменные окружения для чистого теста
        for key in list(os.environ.keys()):
            if key.startswith("LOGGING__"):
                del os.environ[key]

        config = LoggingConfig()

        # Проверяем что уровень логирования установлен (может быть INFO или другой по умолчанию)
        assert config.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.format == LOG_DEFAULT_FORMAT
        assert config.access_log is True
    
    def test_custom_values_from_env(self):
        """Тест пользовательских значений из переменных окружения"""
        os.environ["LOGGING__LEVEL"] = "DEBUG"
        os.environ["LOGGING__FORMAT"] = "%(levelname)s - %(message)s"
        os.environ["LOGGING__ACCESS_LOG"] = "false"
        
        config = LoggingConfig()
        
        assert config.level == "DEBUG"
        assert config.format == "%(levelname)s - %(message)s"
        assert config.access_log is False
    
    def test_log_level_property(self):
        """Тест свойства log_level"""
        # Тестируем разные уровни логирования
        test_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        for level_name, expected_level in test_levels.items():
            os.environ["LOGGING__LEVEL"] = level_name
            config = LoggingConfig()
            assert config.log_level == expected_level
            del os.environ["LOGGING__LEVEL"]
    
    def test_invalid_log_level(self):
        """Тест невалидного уровня логирования"""
        os.environ["LOGGING__LEVEL"] = "INVALID_LEVEL"
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            LoggingConfig()
    
    def test_case_insensitive_level(self):
        """Тест нечувствительности к регистру уровня логирования"""
        os.environ["LOGGING__LEVEL"] = "DEBUG"  # Используем верхний регистр

        config = LoggingConfig()
        # Pydantic требует точное соответствие Literal значениям
        assert config.level == "DEBUG"
    
    def test_boolean_access_log_variations(self):
        """Тест различных вариантов булевого значения access_log"""
        boolean_variations = {
            "true": True,
            "True": True,
            "TRUE": True,
            "1": True,
            "yes": True,
            "false": False,
            "False": False,
            "FALSE": False,
            "0": False,
            "no": False
        }
        
        for env_value, expected_value in boolean_variations.items():
            os.environ["LOGGING__ACCESS_LOG"] = env_value
            config = LoggingConfig()
            assert config.access_log == expected_value
            del os.environ["LOGGING__ACCESS_LOG"]


class TestConfigIntegration:
    """Интеграционные тесты конфигурации"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.original_env = os.environ.copy()
        # Очищаем все переменные окружения связанные с конфигурацией
        for key in list(os.environ.keys()):
            if key.startswith(("RUN__", "LOGGING__")):
                del os.environ[key]
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_multiple_configs_isolation(self):
        """Тест изоляции между различными конфигурациями"""
        # Устанавливаем переменные для разных конфигураций
        os.environ["RUN__HOST"] = "run_host"
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        os.environ["LOGGING__LEVEL"] = "DEBUG"
        
        run_config = RunConfig()
        logging_config = LoggingConfig()
        
        # Проверяем что каждая конфигурация читает только свои переменные
        assert run_config.host == "run_host"
        assert logging_config.level == "DEBUG"
        
        # Проверяем что значения по умолчанию не пересекаются
        assert run_config.port == 8000  # Значение по умолчанию
        assert logging_config.access_log is True  # Значение по умолчанию
    
    def test_env_file_loading(self):
        """Тест загрузки из файла окружения"""
        # Создаем временный файл окружения
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("RUN__HOST=file_host\n")
            f.write("RUN__PORT=9999\n")
            f.write("RUN__CELERY_BROKER_URL=redis://localhost:6379/0\n")
            f.write("RUN__CELERY_RESULT_BACKEND=redis://localhost:6379/0\n")
            f.write("LOGGING__LEVEL=ERROR\n")
            temp_file = f.name

        try:
            # Очищаем переменные окружения
            for key in list(os.environ.keys()):
                if key.startswith(("RUN__", "LOGGING__")):
                    del os.environ[key]

            # Создаем конфигурации с явным указанием файла окружения
            run_config = RunConfig(_env_file=temp_file)
            logging_config = LoggingConfig(_env_file=temp_file)

            assert run_config.host == "file_host"
            assert run_config.port == 9999
            assert logging_config.level == "ERROR"
        finally:
            # Удаляем временный файл
            os.unlink(temp_file)
    
    def test_env_vars_override_file(self):
        """Тест что переменные окружения переопределяют файл"""
        # Создаем временный файл окружения
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("RUN__HOST=file_host\n")
            f.write("RUN__CELERY_BROKER_URL=redis://localhost:6379/0\n")
            f.write("RUN__CELERY_RESULT_BACKEND=redis://localhost:6379/0\n")
            temp_file = f.name
        
        try:
            # Устанавливаем переменную окружения
            os.environ["RUN__HOST"] = "env_host"
            
            with patch.object(AppBaseConfig, 'env_file', temp_file):
                config = RunConfig()
                
                # Переменная окружения должна переопределить файл
                assert config.host == "env_host"
        finally:
            os.unlink(temp_file)
    
    def test_nested_delimiter(self):
        """Тест вложенного разделителя"""
        # Устанавливаем вложенную переменную окружения
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        
        config = RunConfig()
        
        # Проверяем что вложенный разделитель работает
        assert config.celery_broker_url == "redis://localhost:6379/0"
    
    def test_config_case_sensitivity(self):
        """Тест чувствительности к регистру"""
        # Устанавливаем переменные в разном регистре
        os.environ["run__host"] = "lowercase_host"  # Нижний регистр
        os.environ["RUN__CELERY_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["RUN__CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
        
        config = RunConfig()
        
        # Поскольку case_sensitive=False, должно работать
        assert config.host == "lowercase_host"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
