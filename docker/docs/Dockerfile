FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка MkDocs и зависимостей
RUN pip install --no-cache-dir \
    mkdocs \
    mkdocs-material \
    mkdocstrings[python] \
    mkdocs-mermaid2-plugin \
    pymdown-extensions

# Установка рабочей директории
WORKDIR /docs

# Команда по умолчанию
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]
