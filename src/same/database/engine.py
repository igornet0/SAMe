import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.exc import OperationalError, TimeoutError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker
)

from core.database.models.main_models import Base
from core.settings import settings

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
    
    logger.critical("All database connections failed. Exiting application.")
    sys.exit(1)

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
        # from core import data_helper
        # from backend.Dataset import Indicators
        # from core.database.orm_query import (orm_add_coin, orm_add_feature, 
        #                                      orm_get_feature_by_name,
        #                                      orm_add_feature_argument)

        async with self.engine.begin() as conn:
            logger.info("Creating tables")
            await conn.run_sync(Base.metadata.create_all)

        # async with self.async_session() as session:
        #     for coin in data_helper.coin_list:
        #         logger.info(f"Adding coin {coin}")
        #         await orm_add_coin(session, coin)

        #     for indicator_name, iunput_args in Indicators.indicators_input.items():

        #         feature = await orm_get_feature_by_name(session, indicator_name)

        #         if feature:
        #             continue

        #         logger.info(f"Adding indicator {indicator_name}")

        #         await orm_add_feature(session, indicator_name)
        #         feature = await orm_get_feature_by_name(session, indicator_name)

        #         for iunput_arg, type in iunput_args.items():
        #             await orm_add_feature_argument(session, feature.id, iunput_arg, type)

# Асинхронная инициализация db_helper
async def initialize_db_helper():
    working_url = await select_working_url()

    return Database(
        url=working_url,
        echo=settings.db.echo,
        echo_pool=settings.db.echo_pool,
        pool_size=settings.db.pool_size,
        max_overflow=settings.db.max_overflow
    )

# Создаем экземпляр db_helper асинхронно
loop = asyncio.get_event_loop()
db_helper = loop.run_until_complete(initialize_db_helper())