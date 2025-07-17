from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base

class Processe(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(20), unique=True)
    data: Mapped[str] = mapped_column(String(20))
    is_started: Mapped[bool] = mapped_column(Boolean, default=False)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    