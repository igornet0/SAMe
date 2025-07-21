#!/bin/bash

# Скрипт для запуска SAMe проекта в Docker

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    error "Docker не установлен. Пожалуйста, установите Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose не установлен. Пожалуйста, установите Docker Compose."
    exit 1
fi

# Переход в директорию проекта
cd "$(dirname "$0")/.."

# Определение режима запуска и опций
MODE=${1:-"dev"}
BUILD_OPTION=${2:-"cache"}

case $MODE in
    "dev"|"development")
        COMPOSE_FILE="docker/docker-compose.dev.yml"
        log "Запуск в режиме разработки..."
        ;;
    "prod"|"production")
        COMPOSE_FILE="docker/docker-compose.prod.yml"
        log "Запуск в режиме production..."
        ;;
    "basic")
        COMPOSE_FILE="docker/docker-compose.yml"
        log "Запуск базовой конфигурации..."
        ;;
    *)
        error "Неизвестный режим: $MODE"
        echo "Использование: $0 [dev|prod|basic] [cache|no-cache|rebuild]"
        echo "  cache    - использовать кэш Docker (быстро, по умолчанию)"
        echo "  no-cache - пересобрать без кэша (медленно, но гарантированно свежее)"
        echo "  rebuild  - пересобрать только измененные сервисы"
        exit 1
        ;;
esac

# Проверка наличия файла конфигурации
if [ ! -f "$COMPOSE_FILE" ]; then
    error "Файл конфигурации $COMPOSE_FILE не найден"
    exit 1
fi

# Создание необходимых директорий
log "Создание необходимых директорий..."
mkdir -p data/input data/output data/processed logs models temp cache

# Проверка наличия frontend
if [ ! -d "frontend/same-frontend" ]; then
    warn "Директория frontend не найдена. Frontend не будет запущен."
fi

# Остановка существующих контейнеров
log "Остановка существующих контейнеров..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans

# Сборка образов в зависимости от опции
case $BUILD_OPTION in
    "no-cache")
        log "Сборка Docker образов без кэша (может занять много времени)..."
        docker-compose -f "$COMPOSE_FILE" build --no-cache
        ;;
    "rebuild")
        log "Пересборка измененных образов..."
        docker-compose -f "$COMPOSE_FILE" build --pull
        ;;
    "cache"|*)
        log "Сборка Docker образов с использованием кэша..."
        docker-compose -f "$COMPOSE_FILE" build
        ;;
esac

# Запуск сервисов
log "Запуск сервисов..."
docker-compose -f "$COMPOSE_FILE" up -d

# Ожидание готовности сервисов
log "Ожидание готовности сервисов..."
sleep 10

# Проверка статуса сервисов
log "Проверка статуса сервисов..."
docker-compose -f "$COMPOSE_FILE" ps

# Вывод информации о доступных сервисах
echo ""
log "Сервисы запущены и доступны по следующим адресам:"
echo ""

if [ "$MODE" = "dev" ] || [ "$MODE" = "development" ]; then
    echo -e "${BLUE}🚀 Backend API:${NC} http://localhost:8000"
    echo -e "${BLUE}🌐 Frontend:${NC} http://localhost:3000"
    echo -e "${BLUE}📊 PgAdmin:${NC} http://localhost:5050"
    echo -e "${BLUE}🔍 Redis Commander:${NC} http://localhost:8081"
    echo -e "${BLUE}📧 Mailhog:${NC} http://localhost:8025"
    echo -e "${BLUE}📚 Документация:${NC} http://localhost:8080"
    echo -e "${BLUE}📓 Jupyter:${NC} http://localhost:8888"
elif [ "$MODE" = "prod" ] || [ "$MODE" = "production" ]; then
    echo -e "${BLUE}🚀 Backend API:${NC} http://localhost:8000"
    echo -e "${BLUE}🌐 Frontend:${NC} http://localhost:3000"
    echo -e "${BLUE}🌐 Nginx (Main):${NC} http://localhost:80"
    echo -e "${BLUE}📊 Grafana:${NC} http://localhost:3000"
    echo -e "${BLUE}📈 Prometheus:${NC} http://localhost:9090"
    echo -e "${BLUE}🗄️ MinIO:${NC} http://localhost:9000"
else
    echo -e "${BLUE}🚀 Backend API:${NC} http://localhost:8000"
    echo -e "${BLUE}🌐 Frontend:${NC} http://localhost:3000"
    echo -e "${BLUE}📊 PgAdmin:${NC} http://localhost:5050"
fi

echo ""
log "Для просмотра логов используйте: docker-compose -f $COMPOSE_FILE logs -f [service_name]"
log "Для остановки используйте: docker-compose -f $COMPOSE_FILE down"

# Показать логи в реальном времени (опционально)
read -p "Показать логи в реальном времени? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f "$COMPOSE_FILE" logs -f
fi
