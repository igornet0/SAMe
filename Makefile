# SAMe - Search Analog Model Engine
# Makefile для автоматизации задач разработки

.PHONY: help install install-dev test test-cov lint format clean run docker-build docker-run setup-dev

# Переменные
PYTHON := python3
PIP := pip3
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Цвета для вывода
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Помощь
help: ## Показать это сообщение помощи
	@echo "$(BLUE)SAMe - Search Analog Model Engine$(NC)"
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Установка зависимостей
install: ## Установить основные зависимости
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	$(POETRY) install --only=main

install-dev: ## Установить зависимости для разработки
	@echo "$(BLUE)Установка зависимостей для разработки...$(NC)"
	$(POETRY) install
	$(POETRY) run pre-commit install

# Настройка окружения разработки
setup-dev: install-dev ## Полная настройка окружения разработки
	@echo "$(BLUE)Настройка окружения разработки...$(NC)"
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(YELLOW)Создан файл .env из .env.example$(NC)"; fi
	@echo "$(GREEN)Окружение разработки настроено!$(NC)"
	@echo "$(YELLOW)Не забудьте:$(NC)"
	@echo "  1. Настроить .env файл"
	@echo "  2. Установить SpaCy модель: python -m spacy download ru_core_news_lg"
	@echo "  3. Запустить базу данных"

# Тестирование
test: ## Запустить тесты
	@echo "$(BLUE)Запуск тестов...$(NC)"
	$(POETRY) run pytest tests/ -v

test-cov: ## Запустить тесты с покрытием
	@echo "$(BLUE)Запуск тестов с покрытием...$(NC)"
	$(POETRY) run pytest tests/ --cov=src/same --cov-report=html --cov-report=term

test-unit: ## Запустить только unit тесты
	@echo "$(BLUE)Запуск unit тестов...$(NC)"
	$(POETRY) run pytest tests/ -m "unit" -v

test-integration: ## Запустить только integration тесты
	@echo "$(BLUE)Запуск integration тестов...$(NC)"
	$(POETRY) run pytest tests/ -m "integration" -v

# Линтинг и форматирование
lint: ## Проверить код линтерами
	@echo "$(BLUE)Проверка кода...$(NC)"
	$(POETRY) run flake8 src/ tests/
	$(POETRY) run mypy src/
	$(POETRY) run bandit -r src/

format: ## Отформатировать код
	@echo "$(BLUE)Форматирование кода...$(NC)"
	$(POETRY) run black src/ tests/
	$(POETRY) run isort src/ tests/

format-check: ## Проверить форматирование без изменений
	@echo "$(BLUE)Проверка форматирования...$(NC)"
	$(POETRY) run black --check src/ tests/
	$(POETRY) run isort --check-only src/ tests/

security: ## Проверить безопасность кода
	@echo "$(BLUE)Проверка безопасности...$(NC)"
	$(POETRY) run bandit -r src/
	$(POETRY) run safety check

# Очистка
clean: ## Очистить временные файлы
	@echo "$(BLUE)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Запуск приложения
run: ## Запустить приложение локально
	@echo "$(BLUE)Запуск приложения...$(NC)"
	$(POETRY) run uvicorn same.api.create_app:create_app --host 0.0.0.0 --port 8000 --reload

run-prod: ## Запустить приложение в продакшн режиме
	@echo "$(BLUE)Запуск приложения в продакшн режиме...$(NC)"
	$(POETRY) run uvicorn same.api.create_app:create_app --host 0.0.0.0 --port 8000 --workers 4

# Docker команды
docker-build: ## Собрать Docker образ
	@echo "$(BLUE)Сборка Docker образа...$(NC)"
	$(DOCKER) build -t same:latest .

docker-run: ## Запустить приложение в Docker
	@echo "$(BLUE)Запуск приложения в Docker...$(NC)"
	$(DOCKER_COMPOSE) up -d

docker-dev: ## Запустить окружение разработки в Docker
	@echo "$(BLUE)Запуск окружения разработки в Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up -d

docker-prod: ## Запустить продакшн окружение в Docker
	@echo "$(BLUE)Запуск продакшн окружения в Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

docker-stop: ## Остановить Docker контейнеры
	@echo "$(BLUE)Остановка Docker контейнеров...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## Показать логи Docker контейнеров
	$(DOCKER_COMPOSE) logs -f

# База данных
db-upgrade: ## Применить миграции базы данных
	@echo "$(BLUE)Применение миграций...$(NC)"
	$(POETRY) run alembic upgrade head

db-downgrade: ## Откатить миграции базы данных
	@echo "$(BLUE)Откат миграций...$(NC)"
	$(POETRY) run alembic downgrade -1

db-migration: ## Создать новую миграцию
	@echo "$(BLUE)Создание миграции...$(NC)"
	@read -p "Введите название миграции: " name; \
	$(POETRY) run alembic revision --autogenerate -m "$$name"

# Утилиты
spacy-model: ## Установить SpaCy модель для русского языка
	@echo "$(BLUE)Установка SpaCy модели...$(NC)"
	$(POETRY) run python -m spacy download ru_core_news_lg

check-deps: ## Проверить зависимости на уязвимости
	@echo "$(BLUE)Проверка зависимостей...$(NC)"
	$(POETRY) run safety check

update-deps: ## Обновить зависимости
	@echo "$(BLUE)Обновление зависимостей...$(NC)"
	$(POETRY) update

# CI/CD команды
ci-test: format-check lint security test-cov ## Запустить все проверки CI
	@echo "$(GREEN)Все проверки CI пройдены!$(NC)"

pre-commit: ## Запустить pre-commit хуки
	@echo "$(BLUE)Запуск pre-commit хуков...$(NC)"
	$(POETRY) run pre-commit run --all-files

pre-commit-install: ## Установить pre-commit хуки
	@echo "$(BLUE)Установка pre-commit хуков...$(NC)"
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit install --hook-type commit-msg

pre-commit-update: ## Обновить pre-commit хуки
	@echo "$(BLUE)Обновление pre-commit хуков...$(NC)"
	$(POETRY) run pre-commit autoupdate

# Документация
docs-serve: ## Запустить сервер документации
	@echo "$(BLUE)Запуск сервера документации...$(NC)"
	$(POETRY) run mkdocs serve

docs-build: ## Собрать документацию
	@echo "$(BLUE)Сборка документации...$(NC)"
	$(POETRY) run mkdocs build

# Демо и примеры
demo: ## Запустить демо notebook
	@echo "$(BLUE)Запуск демо notebook...$(NC)"
	$(POETRY) run jupyter notebook SAMe_Demo.ipynb

demo-simple: ## Запустить простое демо
	@echo "$(BLUE)Запуск простого демо...$(NC)"
	$(POETRY) run jupyter notebook SAMe_Demo_Simple.ipynb

# По умолчанию показываем help
.DEFAULT_GOAL := help
