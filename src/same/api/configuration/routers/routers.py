from dataclasses import dataclass
from fastapi import FastAPI, APIRouter
from typing import Tuple
import importlib
from pathlib import Path

import logging

logger = logging.getLogger("app_fastapi.routers")

@dataclass(frozen=True)
class Routers:

    routers: tuple

    def register(self, app: FastAPI):
        for router in self.routers:
            app.include_router(router)

    @classmethod
    def _discover_routers(cls) -> Tuple[APIRouter, ...]:
        routers = []
        try:
            # Получаем абсолютный путь к папке routers
            routers_dir = Path(__file__).parent.parent.parent / 'routers'
            
            # Ищем все подпапки в routers
            for subdir in routers_dir.iterdir():
                if subdir.is_dir():
                    router_file = subdir / 'router.py'
                    if router_file.exists():
                        module_name = f'backend.app.routers.{subdir.name}.router'
                        try:
                            module = importlib.import_module(module_name)
                            if hasattr(module, 'router'):
                                router = getattr(module, 'router')
                                if isinstance(router, APIRouter):
                                    routers.append(router)
                        except ImportError as e:
                            logger.error(f"Error importing {module_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering routes: {e}")

        from backend.app.router_main import router as main_router
        routers.append(main_router)
        
        return tuple(routers)