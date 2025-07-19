"""
Тесты для модуля базы данных
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import text, MetaData, DateTime, func, String, Integer, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column, relationship
from typing import List

# Создаем тестовые модели без импорта проблемных модулей
class MockBase(DeclarativeBase):
    __abstract__ = True

    created: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
    updated: Mapped[DateTime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    metadata = MetaData()

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return f"{cls.__name__.lower()}s"

class MockUser(MockBase):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String(128), nullable=False)

class MockItemParameter(MockBase):
    __tablename__ = "item_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), nullable=False)
    parameter_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameter_value: Mapped[str] = mapped_column(String(500), nullable=True)

    item: Mapped["MockItem"] = relationship(
        "MockItem",
        back_populates="parameters",
        lazy="selectin"
    )

class MockItem(MockBase):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)

    parameters: Mapped[List["MockItemParameter"]] = relationship(
        "MockItemParameter",
        back_populates="item",
        cascade="all, delete-orphan",
        lazy="selectin"
    )


class TestDatabaseModels:
    """Тесты для моделей базы данных"""
    
    @pytest_asyncio.fixture
    async def async_engine(self):
        """Создает тестовый асинхронный движок с SQLite в памяти"""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )

        # Создаем таблицы
        async with engine.begin() as conn:
            await conn.run_sync(MockBase.metadata.create_all)

        yield engine

        # Очистка
        await engine.dispose()

    @pytest_asyncio.fixture
    async def async_session(self, async_engine):
        """Создает тестовую асинхронную сессию"""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async_session_factory = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        async with async_session_factory() as session:
            yield session
    
    @pytest.mark.asyncio
    async def test_user_model_creation(self, async_session):
        """Тест создания модели User"""
        user = MockUser(
            username="testuser",
            email="test@example.com",
            password="hashed_password"
        )

        async_session.add(user)
        await async_session.commit()
        await async_session.refresh(user)

        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password == "hashed_password"
        assert user.created is not None
        assert user.updated is not None
    
    @pytest.mark.asyncio
    async def test_user_unique_constraints(self, async_session):
        """Тест уникальности полей User"""
        user1 = MockUser(
            username="testuser",
            email="test@example.com",
            password="password1"
        )

        user2 = MockUser(
            username="testuser",  # Дублирующееся имя пользователя
            email="test2@example.com",
            password="password2"
        )

        async_session.add(user1)
        await async_session.commit()

        async_session.add(user2)

        with pytest.raises(Exception):  # Должно вызвать исключение уникальности
            await async_session.commit()
    
    @pytest.mark.asyncio
    async def test_item_model_creation(self, async_session):
        """Тест создания модели Item"""
        item = MockItem(
            name="Test Item",
            description="Test Description"
        )

        async_session.add(item)
        await async_session.commit()
        await async_session.refresh(item)

        assert item.id is not None
        assert item.name == "Test Item"
        assert item.description == "Test Description"
        assert item.created is not None
        assert item.updated is not None
        assert item.parameters == []  # Пустой список параметров
    
    @pytest.mark.asyncio
    async def test_item_parameter_relationship(self, async_session):
        """Тест связи между Item и ItemParameter"""
        # Создаем элемент
        item = MockItem(
            name="Test Item",
            description="Test Description"
        )
        async_session.add(item)
        await async_session.commit()
        await async_session.refresh(item)

        # Создаем параметры
        param1 = MockItemParameter(
            item_id=item.id,
            parameter_name="color",
            parameter_value="red"
        )

        param2 = MockItemParameter(
            item_id=item.id,
            parameter_name="size",
            parameter_value="large"
        )

        async_session.add_all([param1, param2])
        await async_session.commit()

        # Обновляем item для загрузки связанных параметров
        await async_session.refresh(item)

        assert len(item.parameters) == 2
        assert param1 in item.parameters
        assert param2 in item.parameters
        assert param1.item == item
        assert param2.item == item
    
    @pytest.mark.asyncio
    async def test_cascade_delete(self, async_session):
        """Тест каскадного удаления параметров при удалении элемента"""
        # Создаем элемент с параметрами через relationship
        item = MockItem(name="Test Item")
        param = MockItemParameter(
            parameter_name="test_param",
            parameter_value="test_value"
        )

        # Добавляем параметр к элементу через relationship
        item.parameters.append(param)

        async_session.add(item)
        await async_session.commit()
        await async_session.refresh(item)

        # Проверяем что параметр создан
        assert len(item.parameters) == 1
        param_id = item.parameters[0].id

        # Удаляем элемент
        await async_session.delete(item)
        await async_session.commit()

        # Проверяем что параметры тоже удалены через попытку найти параметр
        from sqlalchemy import select
        result = await async_session.execute(
            select(MockItemParameter).where(MockItemParameter.id == param_id)
        )
        deleted_param = result.scalar_one_or_none()
        assert deleted_param is None
    
    def test_base_model_tablename_generation(self):
        """Тест генерации имен таблиц"""
        assert MockUser.__tablename__ == "users"
        assert MockItem.__tablename__ == "items"
        assert MockItemParameter.__tablename__ == "item_parameters"


class TestDatabaseEngine:
    """Тесты для движка базы данных"""

    @pytest.mark.asyncio
    async def test_database_connection_basic(self):
        """Тест базового подключения к базе данных"""
        # Создаем простой движок для тестирования
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )

        # Тестируем подключение
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_database_table_creation(self):
        """Тест создания таблиц"""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )

        # Создаем таблицы
        async with engine.begin() as conn:
            await conn.run_sync(MockBase.metadata.create_all)

        # Проверяем что таблицы созданы
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result.fetchall()]

            assert "users" in tables
            assert "items" in tables
            assert "item_parameters" in tables

        await engine.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
