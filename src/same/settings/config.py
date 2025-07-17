"""
Конфигурация системы SAMe
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_DEFAULT_FORMAT = '[%(asctime)s] %(name)-35s:%(lineno)-3d - %(levelname)-7s - %(message)s'

class AppBaseConfig:
    """Базовый класс для конфигурации с общими настройками"""
    case_sensitive = False
    env_file = ".env"
    env_file_encoding = "utf-8"
    env_nested_delimiter = "__"
    extra = "ignore"


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="RUN__")
    
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)

    celery_broker_url: str = Field(...)
    celery_result_backend: str = Field(...)

    frontend_host: str = Field(default="localhost")
    frontend_port: int = Field(default=5173)

    @property
    def frontend_url(self):
        return f"http://{self.frontend_host}:{self.frontend_port}"


class LoggingConfig(BaseSettings):
    
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="LOGGING__")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = LOG_DEFAULT_FORMAT
    
    access_log: bool = Field(default=True)

    @property
    def log_level(self) -> int:
        return getattr(logging, self.level)


class DatabaseConfig(BaseSettings):

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="DB__")
    
    user: str = Field(default=...)
    password: str = Field(default=...)
    host: str = Field(default="localhost")
    host_alt: str = Field(default="localhost")
    db_name: str = Field(default="db_name")
    port: int = Field(default=5432)

    echo: bool = Field(default=False)
    echo_pool: bool = Field(default=False)
    pool_size: int = Field(default=20)
    max_overflow: int = 10
    pool_timeout: int = 30

    naming_convention: dict[str, str] = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
    
    def get_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
    
    def get_url_alt(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host_alt}:{self.port}/{self.db_name}"


class RabbitMQConfig(BaseSettings):

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="RABBITMQ__")
    
    host: str = Field(default="localhost")
    port: int = Field(default=5672)
    user: str = Field(default="guest")
    password: str = Field(default="guest")


class PathSettings(BaseSettings):
    """Настройки путей"""
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="PATH__")

    data_dir: Path = Field(default=Path("./data"))
    input_dir: Path = Field(default=Path("./data/input"))
    processed_dir: Path = Field(default=Path("./data/processed"))
    output_dir: Path = Field(default=Path("./data/output"))
    models_dir: Path = Field(default=Path("./models"))
    logs_dir: Path = Field(default=Path("./logs"))
    temp_dir: Path = Field(default=Path("./temp"))

    @validator("*", pre=True)
    def convert_to_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class SearchSettings(BaseSettings):
    """Настройки поиска"""
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="SEARCH__")

    default_method: str = Field(default="hybrid")
    similarity_threshold: float = Field(default=0.6)
    max_results_per_query: int = Field(default=10)
    batch_size: int = Field(default=1000)


class MLSettings(BaseSettings):
    """Настройки машинного обучения"""
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="ML__")

    spacy_model: str = Field(default="ru_core_news_lg")
    semantic_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    use_gpu: bool = Field(default=False)
    batch_size: int = Field(default=32)


class PerformanceSettings(BaseSettings):
    """Настройки производительности"""
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="PERF__")

    enable_multiprocessing: bool = Field(default=True)
    max_workers: int = Field(default=4)
    enable_caching: bool = Field(default=True)
    cache_size: int = Field(default=10000)
    cache_ttl: int = Field(default=3600)
    memory_limit: str = Field(default="8GB")


class SecuritySettings(BaseSettings):
    """Настройки безопасности"""
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, env_prefix="SEC__")

    secret_key: str = Field(default="your-super-secret-key-change-this-in-production")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)


class Settings(BaseSettings):
    """Главные настройки приложения"""

    model_config = SettingsConfigDict(**AppBaseConfig.__dict__)

    # Основные настройки
    app_name: str = "SAMe - Search Analog Model Engine"
    version: str = "0.1.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # Подсистемы настроек
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rbmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathSettings = Field(default_factory=PathSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Создаем необходимые директории
        self._create_directories()

    def _create_directories(self):
        """Создать необходимые директории"""
        for path_attr in ["data_dir", "input_dir", "processed_dir", "output_dir",
                         "models_dir", "logs_dir", "temp_dir"]:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (с кэшированием)"""
    return Settings()


# Глобальный экземпляр настроек для обратной совместимости
settings = get_settings()