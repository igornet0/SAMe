#!/bin/bash

# Скрипт для проверки готовности кода к коммиту

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Проверка готовности кода к коммиту...${NC}"

# Счетчик ошибок
ERRORS=0

# Функция для вывода результата
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

# Проверка форматирования с black
echo -e "${BLUE}📝 Проверка форматирования кода...${NC}"
poetry run black --check src/ tests/ > /dev/null 2>&1
check_result $? "Форматирование кода (black)"

# Проверка сортировки импортов
echo -e "${BLUE}📦 Проверка сортировки импортов...${NC}"
poetry run isort --check-only src/ tests/ > /dev/null 2>&1
check_result $? "Сортировка импортов (isort)"

# Проверка линтером flake8
echo -e "${BLUE}🔍 Проверка линтером...${NC}"
poetry run flake8 src/ tests/ > /dev/null 2>&1
check_result $? "Линтинг кода (flake8)"

# Проверка типов с mypy
echo -e "${BLUE}🏷️  Проверка типов...${NC}"
poetry run mypy src/ > /dev/null 2>&1
check_result $? "Проверка типов (mypy)"

# Проверка безопасности с bandit
echo -e "${BLUE}🔒 Проверка безопасности...${NC}"
poetry run bandit -r src/ > /dev/null 2>&1
check_result $? "Проверка безопасности (bandit)"

# Проверка зависимостей с safety
echo -e "${BLUE}🛡️  Проверка уязвимостей в зависимостях...${NC}"
poetry run safety check > /dev/null 2>&1
check_result $? "Проверка уязвимостей (safety)"

# Запуск тестов
echo -e "${BLUE}🧪 Запуск тестов...${NC}"
poetry run pytest tests/ --tb=no -q > /dev/null 2>&1
check_result $? "Тесты"

# Проверка покрытия тестами
echo -e "${BLUE}📊 Проверка покрытия тестами...${NC}"
COVERAGE=$(poetry run pytest tests/ --cov=src/same --cov-report=term-missing --tb=no -q | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
if [ -n "$COVERAGE" ] && [ "$COVERAGE" -ge 80 ]; then
    echo -e "${GREEN}✅ Покрытие тестами: ${COVERAGE}%${NC}"
else
    echo -e "${RED}❌ Покрытие тестами: ${COVERAGE}% (требуется минимум 80%)${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Проверка документации
echo -e "${BLUE}📚 Проверка документации...${NC}"
if [ -f "mkdocs.yml" ]; then
    poetry run mkdocs build --strict > /dev/null 2>&1
    check_result $? "Сборка документации"
else
    echo -e "${YELLOW}⚠️  mkdocs.yml не найден, пропускаем проверку документации${NC}"
fi

# Проверка структуры проекта
echo -e "${BLUE}🏗️  Проверка структуры проекта...${NC}"
python scripts/check_structure.py > /dev/null 2>&1
check_result $? "Структура проекта"

# Итоговый результат
echo -e "\n${BLUE}📋 Результат проверки:${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}🎉 Все проверки пройдены! Код готов к коммиту.${NC}"
    exit 0
else
    echo -e "${RED}💥 Найдено ошибок: $ERRORS${NC}"
    echo -e "${YELLOW}Исправьте ошибки перед коммитом.${NC}"
    echo -e "\n${BLUE}Полезные команды для исправления:${NC}"
    echo -e "  - ${YELLOW}make format${NC} - автоматическое форматирование"
    echo -e "  - ${YELLOW}make lint${NC} - подробная проверка линтерами"
    echo -e "  - ${YELLOW}make test${NC} - запуск тестов"
    echo -e "  - ${YELLOW}make security${NC} - проверка безопасности"
    exit 1
fi
