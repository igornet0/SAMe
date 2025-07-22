# Multi-stage build для оптимизации размера образа
FROM python:3.11-slim AS builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR=/opt/poetry/.cache
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Копирование файлов зависимостей
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Установка зависимостей
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root

# Production stage
FROM python:3.11-slim AS production

# Создание пользователя для безопасности
RUN groupadd -r same && useradd -r -g same same

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование виртуального окружения из builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Установка рабочей директории
WORKDIR /app

# Копирование исходного кода
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY config/.env.example ./.env

# Загрузка моделей ML
RUN python scripts/download_models.py

# Создание необходимых директорий с правильными правами
RUN mkdir -p data logs temp models cache && \
    mkdir -p /app/models/transformers /app/models/sentence_transformers && \
    mkdir -p /tmp/matplotlib && \
    chown -R same:same /app && \
    chown -R same:same /tmp/matplotlib

# Переключение на непривилегированного пользователя
USER same

# Настройка переменных окружения
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE="/app/models/transformers"
ENV SENTENCE_TRANSFORMERS_HOME="/app/models/sentence_transformers"
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV HF_HOME="/app/models/transformers"

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/search/health || exit 1

# Экспорт портов
EXPOSE 8000

# Команда по умолчанию
CMD ["uvicorn", "same.api.create_app:create_app", "--host", "0.0.0.0", "--port", "8000"]