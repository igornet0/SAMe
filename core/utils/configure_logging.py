import logging
from logging.handlers import RotatingFileHandler
from core.settings import settings

logging.getLogger('passlib').setLevel(logging.ERROR)

def setup_logging():
    from core import data_helper

    # Очищаем все существующие обработчики
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.handlers = []

    level = logging.DEBUG if settings.debug else logging.INFO

    # Корневой логгер (все сообщения)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Форматтер для всех логов
    formatter = logging.Formatter(settings.logging.format)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Общий файловый хендлерs
    common_handler = RotatingFileHandler(data_helper["log"] / "all.log", maxBytes=1e6, backupCount=3)
    common_handler.setFormatter(formatter)

    app_fastapi_logger = logging.getLogger("app_fastapi")
    app_fastapi_handler = RotatingFileHandler(data_helper["log"] / "app_fastapi.log", maxBytes=1e6, backupCount=3)
    app_fastapi_handler.setFormatter(formatter)
    app_fastapi_handler.setLevel(logging.DEBUG)
    app_fastapi_logger.addHandler(app_fastapi_handler)

    process_logger = logging.getLogger("process_logger")
    process_handler = RotatingFileHandler(data_helper["log"] / "process_logger.log", maxBytes=1e6, backupCount=3)
    process_handler.setFormatter(formatter)
    process_handler.setLevel(logging.DEBUG)
    process_logger.addHandler(process_handler)

    # root_logger.addHandler(app_fastapi_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(common_handler)

    # Настройка логгеров Uvicorn
    uvicorn_handler = RotatingFileHandler(data_helper["log"] / "uvicorn.log", maxBytes=1e6, backupCount=3)
    uvicorn_handler.setFormatter(formatter)

    uvicorn_access_handler = RotatingFileHandler(data_helper["log"] / "uvicorn_access.log", maxBytes=5e6, backupCount=5)
    access_formatter = logging.Formatter('%(asctime)s] %(name)-35s:%(client_addr)s - "%(request_line)s" %(status_code)s')
    uvicorn_access_handler.setFormatter(access_formatter)

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [uvicorn_handler]
    uvicorn_logger.propagate = False
    
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = [uvicorn_access_handler]
    uvicorn_access_logger.propagate = False
    
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = [uvicorn_handler]
    uvicorn_error_logger.propagate = False