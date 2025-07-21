#!/bin/bash

# Быстрый запуск для разработки (с кэшем)

set -e

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Переход в директорию проекта
cd "$(dirname "$0")/.."

MODE=${1:-"dev"}
COMPOSE_FILE="docker/docker-compose.dev.yml"

log "🚀 Быстрый запуск в режиме разработки..."

# Создание необходимых директорий
mkdir -p data/input data/output data/processed logs models temp cache

# Остановка и запуск с кэшем
log "Перезапуск сервисов..."
docker-compose -f "$COMPOSE_FILE" down
docker-compose -f "$COMPOSE_FILE" up -d --build

# Краткая информация
echo ""
log "Сервисы запущены:"
echo -e "${BLUE}🚀 Backend API:${NC} http://localhost:8000"
echo -e "${BLUE}🌐 Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}📚 Документация:${NC} http://localhost:8080"
echo ""
log "Для просмотра логов: docker-compose -f $COMPOSE_FILE logs -f"
