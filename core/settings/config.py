from typing import Literal
from pydantic import Field
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

LOG_DEFAULT_FORMAT = '[%(asctime)s] %(name)-35s:%(lineno)-3d - %(levelname)-7s - %(message)s'

class AppBaseConfig:
    """Базовый класс для конфигурации с общими настройками"""
    case_sensitive = False
    env_file = "./settings/prod.env"
    env_file_encoding = "utf-8"
    env_nested_delimiter="__"
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


class Config(BaseSettings):

    model_config = SettingsConfigDict(
        **AppBaseConfig.__dict__,
    )

    debug: bool = Field(default=True)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rbmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    run: RunConfig = Field(default_factory=RunConfig)

settings = Config()