import asyncio

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

from same.api.configuration.routers import Routers
from same import settings
from same import get_db_helper

class Server:

    __app: FastAPI

    # templates = Jinja2Templates(directory="same/api/front/templates")
    # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # http_bearer = HTTPBearer(auto_error=False)

    frontend_url = settings.run.frontend_url

    def __init__(self, app: FastAPI):

        self.__app = app
        self.__register_routers(app)
        self.__regist_middleware(app)

    @staticmethod
    async def get_db() -> AsyncGenerator[AsyncSession, None]:
        db_helper = await get_db_helper()
        async with db_helper.get_session() as session:
            yield session

    def get_app(self) -> FastAPI:
        return self.__app

    @staticmethod
    def __register_routers(app: FastAPI):

        Routers(Routers._discover_routers()).register(app)

    @staticmethod
    def __regist_middleware(app: FastAPI):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",    # React dev server
                "https://localhost:3000",
                "http://127.0.0.1:3000",
                "https://127.0.0.1:3000",
                "http://localhost:5173",    # Vite dev server
                "https://localhost:5173",
                "http://127.0.0.1:5173",
                "https://127.0.0.1:5173",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

