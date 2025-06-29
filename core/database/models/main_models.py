# модели для БД
from typing import List
from sqlalchemy import ForeignKey, String, BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

class User(Base):

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String(128), nullable=False)

class ItemParameter(Base):

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("item.id"), nullable=False)
    parameter_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameter_value: Mapped[str] = mapped_column(String(500), nullable=True)

    item: Mapped["Item"] = relationship(
        "Item",
        back_populates="parameters",
        lazy="selectin"
    )

class Item(Base):

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    
    parameters: Mapped[List["ItemParameter"]] = relationship(
        "ItemParameter",
        back_populates="item",
        cascade="all, delete-orphan",
        lazy="selectin"
    )