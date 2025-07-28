import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.exc import OperationalError, TimeoutError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker
)

from same.database.models.main_models import Base
from ..settings import get_settings

settings = get_settings()

import logging

logger = logging.getLogger("Database")

async def test_connection(url: str, timeout: float = 10.0, retries: int = 3) -> bool:
    """Проверяет доступность базы данных по URL с таймаутом и повторами."""
    test_engine = create_async_engine(url, connect_args={"timeout": timeout})
    for attempt in range(1, retries + 1):
        try:
            async with test_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info(f"Connection to {url} successful (attempt {attempt}/{retries})")
            return True
        except (OperationalError, TimeoutError) as e:
            logger.warning(f"Connection attempt {attempt}/{retries} to {url} failed: {str(e)}")
            if attempt < retries:
                await asyncio.sleep(2)  # Задержка между попытками
        except Exception as e:
            logger.error(f"Unexpected error during connection to {url}: {str(e)}")
            break

    return False

async def select_working_url() -> str:
    """Выбирает рабочий URL из доступных вариантов."""
    urls = [
        ("primary", settings.db.get_url()),
        ("alternative", settings.db.get_url_alt())
    ]
    
    for name, url in urls:
        logger.info(f"Testing {name} database connection: {url}")
        if await test_connection(url):
            logger.info(f"Using {name} database: {url}")
            return url

    raise ConnectionError("All database connections failed. Exiting application.")

class Database:

    def __init__(self,
                url: str,
                echo: bool = False,
                echo_pool: bool = False,
                pool_size: int = 5,
                max_overflow: int = 10,
    ) -> None:
        self.engine: AsyncEngine = create_async_engine(
            url=url,
            echo=echo,
            echo_pool=echo_pool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True
        )
        
        self.async_session: AsyncSession = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )

        self._url = url

    def get_url(self) -> str:
        return self._url

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def init_db(self):
        await self._create_tables()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.async_session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def _create_tables(self):
        
        async with self.engine.begin() as conn:
            logger.info("Creating tables")
            await conn.run_sync(Base.metadata.create_all)

# Глобальная переменная для db_helper
db_helper: Optional[Database] = None

# Асинхронная инициализация db_helper
async def initialize_db_helper():
    global db_helper
    if db_helper is None:
        working_url = await select_working_url()
        db_helper = Database(
            url=working_url,
            echo=settings.db.echo,
            echo_pool=settings.db.echo_pool,
            pool_size=settings.db.pool_size,
            max_overflow=settings.db.max_overflow
        )
    return db_helper

# Функция для получения db_helper с ленивой инициализацией
async def get_db_helper() -> Database:
    global db_helper
    if db_helper is None:
        db_helper = await initialize_db_helper()
    return db_helper