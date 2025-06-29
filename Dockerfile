# Используем официальный образ Python
FROM python:3.12-alpine as builder

WORKDIR /app

RUN apk add --no-cache gcc musl-dev libffi-dev libpq-dev

# Устанавливаем Poetry
ENV POETRY_VERSION=2.0.0
RUN pip install "poetry==$POETRY_VERSION"

# Копируем зависимости
COPY pyproject.toml ./

# Устанавливаем зависимости системы
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main --no-interaction --no-ansi

# Финальный образ
FROM python:3.12-alpine

WORKDIR /app

# Копируем виртуальное окружение и исходный код
COPY --from=builder /app/.venv ./.venv
COPY . .

# Активируем виртуальное окружение
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Копируем настройки
COPY ./settings/prod.env  /app/settings/prod.env 

# Порт приложения
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]