import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

# from .tasks import tasks
from .rabbitmq_server import rabbit
from same import get_db_helper

import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Менеджер жизненного цикла приложения"""
    try:
        logger.info("Initializing database...")
        db_helper = await get_db_helper()
        await db_helper.init_db()
        logger.info("Starting application...")
        # await rabbit.setup_dlx()
        # asyncio.create_task(rabbit.consume_messages("process_queue", tasks.start_process_train))
        logger.info("Application startup complete")
        yield
    finally:
        logger.info("Shutting down application...")
        db_helper = await get_db_helper()
        await db_helper.dispose()
        # await rabbit.close()
        logger.info("Application shutdown complete")