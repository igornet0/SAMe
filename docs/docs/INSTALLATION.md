# Установка и настройка системы поиска аналогов SAMe

## Требования к системе

- Python 3.9+
- Минимум 8 GB RAM
- 5 GB свободного места на диске
- Интернет-соединение для загрузки моделей

## Установка зависимостей

### 1. Установка основных зависимостей

```bash
# Установка через poetry (рекомендуется)
poetry install

# Или через make
make install-dev
```

### 2. Установка SpaCy модели для русского языка

```bash
# Загрузка русской языковой модели
python -m spacy download ru_core_news_lg

# Альтернативно, если модель недоступна
python -m spacy download ru_core_news_lg
```

### 3. Установка дополнительных зависимостей для ML

```bash
# Для работы с FAISS на CPU
pip install faiss-cpu

# Для работы с GPU (опционально)
pip install faiss-gpu

# Для работы с трансформерами
pip install sentence-transformers transformers torch
```

## Структура проекта после установки

```
SAMe/
├── src/same/                     # Основная логика SAMe
│   ├── __init__.py
│   ├── analog_search_engine.py   # Главный движок поиска аналогов
│   ├── text_processing/          # Модули предобработки текста
│   │   ├── __init__.py
│   │   ├── text_cleaner.py       # Очистка текста
│   │   ├── lemmatizer.py         # Лемматизация
│   │   ├── normalizer.py         # Нормализация
│   │   └── preprocessor.py       # Главный предобработчик
│   ├── search_engine/            # Поисковые алгоритмы
│   │   ├── __init__.py
│   │   ├── fuzzy_search.py       # Нечеткий поиск
│   │   ├── semantic_search.py    # Семантический поиск
│   │   ├── hybrid_search.py      # Гибридный поиск
│   │   └── indexer.py            # Индексация
│   ├── parameter_extraction/     # Извлечение параметров
│   │   ├── __init__.py
│   │   ├── regex_extractor.py    # Regex-извлечение
│   │   ├── ml_extractor.py       # ML-извлечение
│   │   ├── parameter_parser.py   # Парсер параметров
│   │   └── parameter_utils.py    # Утилиты для параметров
│   ├── export/                   # Экспорт результатов
│   │   ├── __init__.py
│   │   ├── excel_exporter.py     # Экспорт в Excel
│   │   └── report_generator.py   # Генерация отчетов
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   ├── create_app.py         # Создание FastAPI приложения
│   │   ├── router_main.py        # Основные роуты
│   │   ├── configuration/        # Конфигурация API
│   │   ├── middleware/           # Middleware компоненты
│   │   └── routers/              # API роутеры
│   ├── database/                 # Работа с базой данных
│   │   ├── __init__.py
│   │   ├── base.py               # Базовые модели
│   │   ├── engine.py             # Движок БД
│   │   ├── optimizations.py      # Оптимизации БД
│   │   ├── models/               # ORM модели
│   │   └── orm/                  # ORM операции
│   ├── models/                   # ML модели и управление
│   │   ├── __init__.py
│   │   ├── model_manager.py      # Менеджер моделей
│   │   ├── memory_monitor.py     # Мониторинг памяти
│   │   ├── quantization.py       # Квантизация моделей
│   │   └── exceptions.py         # Исключения моделей
│   ├── data_manager/             # Управление данными
│   │   ├── __init__.py
│   │   └── DataManager.py        # Основной менеджер данных
│   ├── caching/                  # Кэширование
│   │   └── advanced_cache.py     # Продвинутое кэширование
│   ├── monitoring/               # Мониторинг системы
│   │   └── analytics.py          # Аналитика и метрики
│   ├── realtime/                 # Реальное время
│   │   └── streaming.py          # Потоковая обработка
│   ├── distributed/              # Распределенная обработка
│   │   └── processor.py          # Распределенный процессор
│   ├── optimizations/            # Оптимизации производительности
│   │   ├── integration.py        # Интеграционные оптимизации
│   │   └── phase3_integration.py # Фаза 3 интеграции
│   ├── settings/                 # Конфигурация
│   │   ├── __init__.py
│   │   └── config.py             # Настройки системы
│   ├── utils/                    # Утилиты
│   │   ├── __init__.py
│   │   ├── case_converter.py     # Конвертер регистра
│   │   └── configure_logging.py  # Настройка логирования
│   └── alembic/                  # Миграции БД
├── data/
│   ├── input/                    # Входные данные
│   ├── processed/                # Обработанные данные
│   └── output/                   # Результаты
├── models/                       # Сохраненные модели
├── config/                       # Конфигурационные файлы
└── tests/                        # Тесты
    ├── test_text_processing.py
    ├── test_search_engine.py
    └── test_parameter_extraction.py
```

## Конфигурация

### 1. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
# Настройки базы данных
DATABASE_URL=postgresql+asyncpg://user:password@localhost/same_db

# Настройки API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Настройки поиска
DEFAULT_SEARCH_METHOD=hybrid
SIMILARITY_THRESHOLD=0.6
MAX_RESULTS_PER_QUERY=10

# Пути к данным
DATA_DIR=./data
MODELS_DIR=./models
OUTPUT_DIR=./data/output
```

### 2. Настройка логирования

Создайте файл `logging.conf`:

```ini
[loggers]
keys=root,same

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_same]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=same
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('logs/same.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Запуск системы

### 1. Запуск API сервера

```bash
# Через make (рекомендуется)
make run

# Или через uvicorn напрямую
uvicorn same.api.create_app:create_app --host 0.0.0.0 --port 8000 --reload
```

### 2. Инициализация поискового движка

```python
from same.analog_search_engine import AnalogSearchEngine
from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
import pandas as pd
import asyncio

# Создание движка
engine = AnalogSearchEngine()

# Загрузка тестовых данных
test_data = pd.DataFrame({
    'name': [
        'Болт М10х50 ГОСТ 7798-70',
        'Гайка М10 ГОСТ 5915-70',
        'Шайба 10 ГОСТ 11371-78'
    ]
})

# Асинхронная функция для инициализации
async def main():
    # Инициализация
    await engine.initialize(test_data)

    # Поиск аналогов
    results = await engine.search_analogs(['болт м10'])

    # Экспорт результатов
    config = ExcelExportConfig(include_statistics=True)
    exporter = ExcelExporter(config)
    exporter.export_search_results(results, 'results.xlsx')

# Запуск
asyncio.run(main())
```

### 3. Использование через API

```bash
# Инициализация движка
curl -X POST "http://localhost:8000/search/initialize" \
     -H "Content-Type: application/json" \
     -d '{"search_method": "hybrid"}'

# Поиск аналогов
curl -X POST "http://localhost:8000/search/search-analogs" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["болт м10"], "method": "hybrid"}'
```

## Тестирование

### 1. Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Тесты с покрытием
pytest tests/ --cov=src/same --cov-report=html

# Тесты конкретного модуля
pytest tests/test_text_processing.py -v
```

### 2. Создание тестовых данных

```python
# Создание тестового каталога
import pandas as pd

test_catalog = pd.DataFrame({
    'id': range(1, 101),
    'name': [f'Тестовое изделие {i}' for i in range(1, 101)],
    'category': ['Крепеж', 'Трубы', 'Электрика'] * 33 + ['Крепеж']
})

test_catalog.to_excel('data/input/test_catalog.xlsx', index=False)
```

## Оптимизация производительности

### 1. Настройка FAISS для больших данных

```python
# Для больших каталогов (>100K позиций)
from same.search_engine import SemanticSearchConfig

config = SemanticSearchConfig(
    index_type="ivf",  # Использовать IVF индекс
    nlist=1000,        # Количество кластеров
    nprobe=50          # Количество кластеров для поиска
)
```

### 2. Пакетная обработка

```python
# Обработка больших файлов по частям
config = AnalogSearchConfig(
    batch_size=500,  # Размер пакета
    max_results_per_query=20
)
```

## Устранение неполадок

### 1. Проблемы с установкой SpaCy

```bash
# Если модель не загружается
pip install --upgrade spacy
python -m spacy validate

# Ручная загрузка модели
pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.7.0/ru_core_news_lg-3.7.0-py3-none-any.whl
```

### 2. Проблемы с FAISS

```bash
# Для macOS с Apple Silicon
conda install -c conda-forge faiss-cpu

# Для Linux
pip install faiss-cpu --no-cache-dir
```

### 3. Проблемы с памятью

```python
# Уменьшение размера модели
config = SemanticSearchConfig(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=16  # Уменьшить размер пакета
)
```

## Мониторинг и логирование

### 1. Проверка состояния системы

```bash
# Проверка API
curl http://localhost:8000/search/health

# Получение статистики
curl http://localhost:8000/search/statistics
```

### 2. Анализ логов

```bash
# Просмотр логов в реальном времени
tail -f logs/same.log

# Поиск ошибок
grep ERROR logs/same.log
```

## Развертывание в продакшене

### 1. Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install poetry && poetry install --no-dev
RUN python -m spacy download ru_core_news_lg

EXPOSE 8000
CMD ["uvicorn", "same.api.create_app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Настройка nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
