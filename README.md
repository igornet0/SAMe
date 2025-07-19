# SAMe - Search Analog Model Engine

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Система поиска аналогов материально-технических ресурсов с использованием современных методов машинного обучения и обработки естественного языка.

## 🚀 Возможности

- **Интеллектуальный поиск аналогов** - Нечеткий, семантический и гибридный поиск
- **Обработка естественного языка** - Предобработка и нормализация технических наименований с использованием SpaCy
- **Извлечение параметров** - Автоматическое извлечение технических характеристик (regex и ML подходы)
- **Масштабируемость** - Поддержка больших каталогов с параллельной обработкой
- **REST API** - Простая интеграция с существующими системами через FastAPI
- **Экспорт результатов** - Детальные отчеты в Excel с визуализацией и статистикой
- **База данных** - Поддержка PostgreSQL с асинхронными операциями
- **Jupyter Notebooks** - Интерактивные демо и примеры использования

## 🏗️ Архитектура

```
SAMe/
├── src/same/                      # Основной пакет
│   ├── api/                       # FastAPI приложение
│   │   ├── routers/               # API роутеры
│   │   ├── configuration/         # Конфигурация API
│   │   └── middleware/            # Middleware компоненты
│   ├── text_processing/           # Предобработка текста
│   │   ├── lemmatizer.py          # Лемматизация с SpaCy
│   │   ├── normalizer.py          # Нормализация текста
│   │   ├── preprocessor.py        # Общая предобработка
│   │   └── text_cleaner.py        # Очистка текста
│   ├── search_engine/             # Поисковые алгоритмы
│   │   ├── fuzzy_search.py        # Нечеткий поиск
│   │   ├── semantic_search.py     # Семантический поиск
│   │   ├── hybrid_search.py       # Гибридный поиск
│   │   └── indexer.py             # Индексация данных
│   ├── parameter_extraction/      # Извлечение параметров
│   │   ├── regex_extractor.py     # Regex-извлечение
│   │   ├── ml_extractor.py        # ML-извлечение
│   │   ├── parameter_parser.py    # Парсинг параметров
│   │   └── parameter_utils.py     # Утилиты для параметров
│   ├── export/                    # Экспорт результатов
│   │   ├── excel_exporter.py      # Excel экспорт
│   │   └── report_generator.py    # Генерация отчетов
│   ├── database/                  # База данных
│   │   ├── models/                # ORM модели
│   │   ├── orm/                   # ORM операции
│   │   ├── base.py                # Базовые модели
│   │   ├── engine.py              # Движок БД
│   │   └── optimizations.py       # Оптимизации БД
│   ├── models/                    # ML модели и управление
│   │   ├── model_manager.py       # Менеджер моделей
│   │   ├── memory_monitor.py      # Мониторинг памяти
│   │   ├── quantization.py        # Квантизация моделей
│   │   └── exceptions.py          # Исключения моделей
│   ├── data_manager/              # Управление данными
│   │   └── DataManager.py         # Основной менеджер данных
│   ├── caching/                   # Кэширование
│   │   └── advanced_cache.py      # Продвинутое кэширование
│   ├── monitoring/                # Мониторинг системы
│   │   └── analytics.py           # Аналитика и метрики
│   ├── realtime/                  # Реальное время
│   │   └── streaming.py           # Потоковая обработка
│   ├── distributed/               # Распределенная обработка
│   │   └── processor.py           # Распределенный процессор
│   ├── optimizations/             # Оптимизации производительности
│   │   ├── integration.py         # Интеграционные оптимизации
│   │   └── phase3_integration.py  # Фаза 3 интеграции
│   ├── settings/                  # Настройки системы
│   │   └── config.py              # Конфигурация
│   ├── utils/                     # Утилиты
│   │   ├── case_converter.py      # Конвертер регистра
│   │   └── configure_logging.py   # Настройка логирования
│   ├── alembic/                   # Миграции БД
│   └── analog_search_engine.py    # Главный поисковый движок
├── config/                        # Конфигурационные файлы
├── docs/                          # Документация
├── tests/                         # Тесты
├── notebooks/                     # Jupyter notebooks
├── docker/                        # Docker конфигурация
└── scripts/                       # Скрипты автоматизации
```

## 🛠️ Технологии

### Основной стек
- **Python 3.9+** - Основной язык разработки
- **FastAPI 0.115+** - Современный веб-фреймворк для API
- **SQLAlchemy 2.0+** - ORM с поддержкой async/await
- **PostgreSQL** - Основная база данных
- **Alembic** - Миграции базы данных

### Машинное обучение и NLP
- **SpaCy 3.7+** - Обработка естественного языка (модель `ru_core_news_lg`)
- **scikit-learn 1.3+** - Классические алгоритмы ML
- **Sentence Transformers 2.2+** - Семантические эмбеддинги
- **Transformers 4.30+** - Трансформерные модели
- **FAISS** - Быстрый поиск по векторам
- **PyTorch 2.0+** - Deep Learning фреймворк

### Обработка данных
- **Pandas 2.3+** - Анализ и обработка данных
- **NumPy 1.24+** - Численные вычисления
- **OpenPyXL 3.1+** - Работа с Excel файлами
- **RapidFuzz 3.0+** - Быстрое нечеткое сравнение строк

### Инфраструктура
- **Docker** - Контейнеризация
- **Poetry** - Управление зависимостями
- **Pydantic 2.11+** - Валидация данных
- **Uvicorn** - ASGI сервер

## 📦 Быстрая установка

### Через Docker (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/igornet0/SAMe.git
cd SAMe

# Запуск для разработки
make docker-dev

# Или для продакшена
make docker-prod
```

### Локальная установка

```bash
# Установка зависимостей и настройка окружения
make setup-dev

# Установка SpaCy модели
make spacy-model

# Применение миграций БД
make db-upgrade

# Запуск сервера
make run
```

### Быстрый старт с Makefile

```bash
# Показать все доступные команды
make help

# Полная настройка для разработки
make setup-dev

# Запуск тестов
make test

# Форматирование кода
make format

# Запуск всех проверок CI
make ci-test
```

## 🚀 Быстрый старт

### 1. Запуск приложения

```bash
# Локально
make run

# Или в Docker
make docker-dev
```

### 2. Открытие документации API

Перейдите в браузере по адресу:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Работа с Jupyter Notebooks

```bash
# Запуск демо notebook
make demo

# Или запуск простого демо
make demo-simple
```

### 4. Пример использования API

```bash
# Инициализация поискового движка
curl -X POST "http://localhost:8000/search/initialize" \
     -H "Content-Type: application/json" \
     -d '{"search_method": "hybrid"}'

# Загрузка каталога
curl -X POST "http://localhost:8000/search/upload-catalog" \
     -F "file=@your_catalog.xlsx"

# Поиск аналогов
curl -X GET "http://localhost:8000/search/search-single/болт%20м10"
```

### 5. Пример ответа

```json
{
  "query": "болт м10",
  "results": [
    {
      "document_id": 1,
      "document": "Болт М10х50 ГОСТ 7798-70",
      "similarity_score": 0.95,
      "search_method": "hybrid",
      "rank": 1,
      "parameters": {
        "diameter": "М10",
        "length": "50",
        "standard": "ГОСТ 7798-70"
      }
    }
  ],
  "processing_time": 0.045,
  "total_results": 1
}
```

## 📊 Методы поиска

### Нечеткий поиск (Fuzzy Search)
- **TF-IDF векторизация** с настраиваемыми n-граммами (1-3)
- **Косинусное сходство** для сравнения документов
- **RapidFuzz** для быстрого нечеткого сравнения строк
- **Расстояние Левенштейна** для точного сравнения
- **Взвешенное комбинирование** различных метрик

### Семантический поиск (Semantic Search)
- **Sentence Transformers** модели (multilingual MiniLM)
- **Векторные представления** текстов в 384-мерном пространстве
- **FAISS индексация** для быстрого поиска ближайших соседей
- **Нормализация эмбеддингов** для стабильности результатов
- **Пакетная обработка** для оптимизации производительности

### Гибридный поиск (Hybrid Search)
- **Комбинация методов** с настраиваемыми весами (fuzzy: 0.4, semantic: 0.6)
- **Стратегии объединения**: weighted_sum, rank_fusion, cascade
- **Фильтрация кандидатов** по минимальным порогам
- **Ранговое слияние** результатов разных методов
- **Каскадная стратегия** для оптимизации производительности

## 🔧 Конфигурация

Основные настройки в `config/same_config.yaml`:

```yaml
# Поисковый движок
search_engine:
  default_method: "hybrid"  # fuzzy, semantic, hybrid
  similarity_threshold: 0.6
  max_results_per_query: 10
  enable_parallel_search: true
  max_workers: 4

  # Нечеткий поиск
  fuzzy_search:
    tfidf_max_features: 10000
    tfidf_ngram_range: [1, 3]
    cosine_threshold: 0.3
    fuzzy_threshold: 60

  # Семантический поиск
  semantic_search:
    model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: 384
    similarity_threshold: 0.5
    batch_size: 32

  # Гибридный поиск
  hybrid_search:
    fuzzy_weight: 0.4
    semantic_weight: 0.6
    combination_strategy: "weighted_sum"

# Предобработка текста
text_processing:
  lemmatization:
    model_name: "ru_core_news_lg"
    preserve_technical_terms: true
  normalization:
    standardize_units: true
    normalize_abbreviations: true

# Извлечение параметров
parameter_extraction:
  use_regex: true
  use_ml: false
  combination_strategy: "union"
  min_confidence: 0.5
```

## 📈 Производительность

### Бенчмарки

| Размер каталога | Время индексации | Время поиска | Память | Рекомендуемый метод |
|----------------|------------------|--------------|--------|-------------------|
| 1K позиций     | 5 сек           | 50 мс        | 500 MB | Hybrid            |
| 10K позиций    | 30 сек          | 100 мс       | 1.5 GB | Hybrid            |
| 100K позиций   | 5 мин           | 200 мс       | 4 GB   | Fuzzy + Semantic  |
| 1M+ позиций    | 30 мин          | 500 мс       | 16 GB  | Fuzzy             |

### Оптимизация производительности

- **Параллельная обработка**: Включена по умолчанию (4 воркера)
- **Пакетная индексация**: Размер батча 1000 документов
- **Кэширование**: Результаты поиска кэшируются на 1 час
- **Инкрементальные обновления**: Поддержка добавления новых документов
- **Оптимизация памяти**: Автоматическая сборка мусора при превышении лимитов

### Настройки производительности

```yaml
performance:
  enable_multiprocessing: true
  max_workers: 4
  batch_size: 1000
  enable_caching: true
  cache_size: 10000
  memory_limit: "8GB"
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
make test

# Тесты с покрытием
make test-cov

# Только unit тесты
make test-unit

# Только integration тесты
make test-integration

# Конкретный модуль
pytest tests/test_search_engine.py -v
```

### Структура тестов

- `test_search_engine.py` - Тесты поисковых алгоритмов
- `test_text_processing.py` - Тесты предобработки текста
- `test_parameter_extraction.py` - Тесты извлечения параметров
- `test_export.py` - Тесты экспорта результатов
- `test_database.py` - Тесты работы с БД
- `test_backend.py` - Тесты API endpoints

## 📚 Документация

### Основная документация
- [Руководство пользователя](docs/USER_GUIDE.md) - Подробное руководство по использованию
- [API Reference](docs/API_REFERENCE.md) - Документация по API endpoints
- [Установка и настройка](docs/INSTALLATION.md) - Детальная инструкция по установке
- [Notebook Guide](docs/NOTEBOOK_GUIDE.md) - Руководство по работе с Jupyter notebooks

### Интерактивная документация
- **Swagger UI**: http://localhost:8000/docs (после запуска сервера)
- **ReDoc**: http://localhost:8000/redoc (альтернативный интерфейс)

### Jupyter Notebooks
- `notebooks/demo/` - Демонстрационные примеры
- `SAMe_Demo.ipynb` - Основное демо с полным функционалом
- `SAMe_Demo_Simple.ipynb` - Упрощенное демо для быстрого старта

## 🛠️ Разработка

### Настройка окружения разработки

```bash
# Полная настройка
make setup-dev

# Установка pre-commit хуков
make pre-commit-install

# Проверка кода
make format lint security

# Запуск всех проверок CI
make ci-test
```

### Стандарты кода

- **PEP 8** - Стиль кодирования Python
- **Black** - Автоматическое форматирование кода
- **isort** - Сортировка импортов
- **MyPy** - Статическая типизация
- **Flake8** - Линтинг кода
- **Bandit** - Проверка безопасности
- **Type hints** - Обязательны для всех функций
- **Docstrings** - Документация для всех публичных методов
- **Покрытие тестами** - Не менее 80%

### Структура коммитов

Используйте conventional commits:
- `feat:` - новая функциональность
- `fix:` - исправление багов
- `docs:` - изменения в документации
- `style:` - форматирование кода
- `refactor:` - рефакторинг
- `test:` - добавление тестов
- `chore:` - обновление зависимостей, конфигурации

## 🚀 Деплой

### Docker деплой

```bash
# Продакшн сборка
make docker-build

# Запуск в продакшне
make docker-prod

# Просмотр логов
make docker-logs
```

### Переменные окружения

Создайте `.env` файл для настройки окружения:

```bash
# База данных
DATABASE_URL=postgresql+asyncpg://user:password@localhost/same_db

# API настройки
API_HOST=0.0.0.0
API_PORT=8000

# Безопасность
SECRET_KEY=your-secret-key-here
```

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

---

**SAMe** - Делаем поиск аналогов простым и эффективным! 🔍✨
