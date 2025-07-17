#!/bin/bash

# Скрипт для настройки окружения разработки SAMe

set -e  # Выход при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Настройка окружения разработки SAMe...${NC}"

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 не найден. Установите Python 3.9+${NC}"
    exit 1
fi

# Проверка версии Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Требуется Python $REQUIRED_VERSION+, найден $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION найден${NC}"

# Проверка наличия Poetry
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}⚠️  Poetry не найден. Устанавливаем...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v poetry &> /dev/null; then
        echo -e "${RED}❌ Не удалось установить Poetry${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Poetry найден${NC}"

# Установка зависимостей
echo -e "${BLUE}📦 Установка зависимостей...${NC}"
poetry install

# Создание .env файла если его нет
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Создание .env файла...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ Файл .env создан из .env.example${NC}"
    echo -e "${YELLOW}⚠️  Не забудьте настроить переменные в .env файле${NC}"
fi

# Установка pre-commit хуков
echo -e "${BLUE}🔧 Установка pre-commit хуков...${NC}"
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg

# Создание необходимых директорий
echo -e "${BLUE}📁 Создание директорий...${NC}"
mkdir -p data/{input,processed,output} logs temp models/{configs,logs,models_pth}

# Установка SpaCy модели
echo -e "${BLUE}🧠 Установка SpaCy модели для русского языка...${NC}"
poetry run python -m spacy download ru_core_news_lg

# Проверка структуры проекта
echo -e "${BLUE}🔍 Проверка структуры проекта...${NC}"
poetry run python scripts/check_structure.py

# Запуск первичных проверок
echo -e "${BLUE}🧪 Запуск первичных проверок...${NC}"

echo -e "${BLUE}  - Форматирование кода...${NC}"
poetry run black src/ tests/ || echo -e "${YELLOW}⚠️  Найдены проблемы с форматированием${NC}"

echo -e "${BLUE}  - Сортировка импортов...${NC}"
poetry run isort src/ tests/ || echo -e "${YELLOW}⚠️  Найдены проблемы с импортами${NC}"

echo -e "${BLUE}  - Проверка линтером...${NC}"
poetry run flake8 src/ tests/ || echo -e "${YELLOW}⚠️  Найдены проблемы с кодом${NC}"

echo -e "${BLUE}  - Проверка типов...${NC}"
poetry run mypy src/ || echo -e "${YELLOW}⚠️  Найдены проблемы с типами${NC}"

echo -e "${BLUE}  - Проверка безопасности...${NC}"
poetry run bandit -r src/ || echo -e "${YELLOW}⚠️  Найдены проблемы с безопасностью${NC}"

# Запуск тестов
echo -e "${BLUE}🧪 Запуск тестов...${NC}"
if poetry run pytest tests/ --tb=short; then
    echo -e "${GREEN}✅ Все тесты прошли${NC}"
else
    echo -e "${YELLOW}⚠️  Некоторые тесты не прошли${NC}"
fi

echo -e "${GREEN}🎉 Настройка окружения разработки завершена!${NC}"
echo -e "${BLUE}📋 Следующие шаги:${NC}"
echo -e "  1. Настройте переменные в файле .env"
echo -e "  2. Запустите базу данных: ${YELLOW}make docker-dev${NC}"
echo -e "  3. Примените миграции: ${YELLOW}make db-upgrade${NC}"
echo -e "  4. Запустите приложение: ${YELLOW}make run${NC}"
echo -e "  5. Откройте документацию: ${YELLOW}make docs-serve${NC}"

echo -e "${BLUE}🔗 Полезные команды:${NC}"
echo -e "  - ${YELLOW}make help${NC} - показать все доступные команды"
echo -e "  - ${YELLOW}make test${NC} - запустить тесты"
echo -e "  - ${YELLOW}make format${NC} - отформатировать код"
echo -e "  - ${YELLOW}make lint${NC} - проверить код линтерами"
echo -e "  - ${YELLOW}make demo${NC} - запустить демо notebook"
