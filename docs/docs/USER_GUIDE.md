# Руководство пользователя SAMe

## Введение

SAMe (Search Analog Model Engine) - это система поиска аналогов материально-технических ресурсов (МТР), использующая современные методы машинного обучения и обработки естественного языка.

## Быстрый старт

### 1. Запуск системы

```bash
# Запуск через Docker Compose (рекомендуется)
make docker-dev

# Или локальный запуск
make run
```

### 2. Проверка работоспособности

Откройте в браузере: http://localhost:8000/search/health

Ответ должен содержать:
```json
{
  "status": "healthy",
  "search_engine_ready": false,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. Инициализация системы

Перед началом работы необходимо инициализировать поисковый движок:

```bash
curl -X POST "http://localhost:8000/search/initialize" \
     -H "Content-Type: application/json" \
     -d '{"search_method": "hybrid", "similarity_threshold": 0.6}'
```

## Загрузка каталога

### Через API

```bash
curl -X POST "http://localhost:8000/search/upload-catalog" \
     -F "file=@your_catalog.xlsx"
```

### Поддерживаемые форматы

- **Excel (.xlsx)**: Рекомендуемый формат
- **CSV (.csv)**: Альтернативный формат

### Структура каталога

Каталог должен содержать минимум одну колонку с наименованиями МТР:

| name | id | category | description |
|------|----|-----------|-----------| 
| Болт М10х50 ГОСТ 7798-70 | 1 | Крепеж | Болт с метрической резьбой |
| Гайка М10 DIN 934 | 2 | Крепеж | Шестигранная гайка |

**Обязательные колонки:**
- `name` или `наименование` - наименование МТР

**Опциональные колонки:**
- `id` - уникальный идентификатор
- `category` или `категория` - категория МТР
- `description` или `описание` - дополнительное описание

## Поиск аналогов

### Простой поиск

```bash
curl -X GET "http://localhost:8000/search/search-single/болт%20м10"
```

Ответ:
```json
{
  "query": "болт м10",
  "results": [
    {
      "document_id": 1,
      "document": "Болт М10х50 ГОСТ 7798-70",
      "similarity_score": 0.95,
      "search_method": "hybrid",
      "rank": 1
    }
  ],
  "method": "hybrid"
}
```

### Пакетный поиск

```bash
curl -X POST "http://localhost:8000/search/search-analogs" \
     -H "Content-Type: application/json" \
     -d '{
       "queries": ["болт м10", "гайка м10", "шайба 10"],
       "method": "hybrid",
       "max_results": 5
     }'
```

### Методы поиска

1. **fuzzy** - Нечеткий поиск на основе TF-IDF и строковых расстояний
2. **semantic** - Семантический поиск с использованием BERT-подобных моделей
3. **hybrid** - Комбинированный подход (рекомендуется)

### Параметры поиска

- `similarity_threshold` (0.0-1.0) - Минимальный порог схожести
- `max_results` (1-100) - Максимальное количество результатов
- `method` - Метод поиска

## Экспорт результатов

### Экспорт в Excel

```bash
curl -X POST "http://localhost:8000/search/export-results" \
     -H "Content-Type: application/json" \
     -d '{
       "results": {
         "болт м10": [
           {
             "document_id": 1,
             "document": "Болт М10х50 ГОСТ 7798-70",
             "similarity_score": 0.95
           }
         ]
       },
       "format": "excel"
     }' \
     --output results.xlsx
```

### Структура экспорта

Экспортируемый файл содержит:

1. **Лист "Результаты поиска"**
   - Запрос
   - Ранг результата
   - Найденное наименование
   - Оценка схожести
   - Тип поиска

2. **Лист "Статистика"**
   - Общие метрики
   - Статистика по качеству
   - Время обработки

3. **Лист "Метаданные"**
   - Параметры системы
   - Время экспорта

## Интерпретация результатов

### Оценка схожести

- **0.9-1.0** - Точное соответствие
- **0.7-0.9** - Высокая схожесть
- **0.5-0.7** - Средняя схожесть
- **0.3-0.5** - Низкая схожесть
- **0.0-0.3** - Очень низкая схожесть

### Цветовая индикация в Excel

- 🟢 **Зеленый** - Высокое качество (≥0.8)
- 🟡 **Желтый** - Среднее качество (0.6-0.8)
- 🔴 **Красный** - Низкое качество (<0.6)

## Примеры использования

### Поиск крепежа

```bash
# Поиск болтов
curl -X GET "http://localhost:8000/search/search-single/болт%20м12х40"

# Поиск гаек
curl -X GET "http://localhost:8000/search/search-single/гайка%20м12"

# Поиск шайб
curl -X GET "http://localhost:8000/search/search-single/шайба%2012"
```

### Поиск электрооборудования

```bash
# Поиск двигателей
curl -X GET "http://localhost:8000/search/search-single/двигатель%204квт"

# Поиск кабелей
curl -X GET "http://localhost:8000/search/search-single/кабель%202.5мм2"
```

### Поиск трубопроводной арматуры

```bash
# Поиск труб
curl -X GET "http://localhost:8000/search/search-single/труба%2057х3.5"

# Поиск фланцев
curl -X GET "http://localhost:8000/search/search-single/фланец%20ду50"
```

## Оптимизация поиска

### Рекомендации по запросам

1. **Используйте ключевые термины**
   - ✅ "болт м10х50"
   - ❌ "нужен болт размером 10 на 50"

2. **Указывайте технические характеристики**
   - ✅ "двигатель 4квт 1500об/мин"
   - ❌ "электродвигатель"

3. **Используйте стандартные обозначения**
   - ✅ "труба 57х3.5 гост 8732"
   - ❌ "трубка диаметром 57"

### Настройка порогов

Для разных задач рекомендуются разные пороги схожести:

- **Точный поиск**: 0.8-1.0
- **Поиск аналогов**: 0.6-0.8
- **Широкий поиск**: 0.4-0.6

## Мониторинг и статистика

### Получение статистики

```bash
curl -X GET "http://localhost:8000/search/statistics"
```

Ответ содержит:
- Статус системы
- Размер каталога
- Статистику поисковых движков
- Производительность

### Проверка здоровья системы

```bash
curl -X GET "http://localhost:8000/search/health"
```

## Устранение неполадок

### Частые проблемы

1. **Система не инициализирована**
   ```
   Error: Search engine is not initialized
   ```
   **Решение**: Выполните инициализацию через `/search/initialize`

2. **Каталог не загружен**
   ```
   Error: No catalog data available
   ```
   **Решение**: Загрузите каталог через `/search/upload-catalog`

3. **Низкое качество результатов**
   **Решение**: 
   - Проверьте качество данных в каталоге
   - Уменьшите порог схожести
   - Попробуйте другой метод поиска

4. **Медленная работа**
   **Решение**:
   - Уменьшите размер каталога
   - Используйте нечеткий поиск вместо семантического
   - Увеличьте количество воркеров

### Логи системы

Логи находятся в директории `./logs/same.log`

```bash
# Просмотр последних логов
tail -f logs/same.log

# Поиск ошибок
grep ERROR logs/same.log
```

## API Reference

### Основные endpoints

- `GET /search/health` - Проверка состояния
- `POST /search/initialize` - Инициализация системы
- `POST /search/upload-catalog` - Загрузка каталога
- `GET /search/search-single/{query}` - Простой поиск
- `POST /search/search-analogs` - Пакетный поиск
- `POST /search/export-results` - Экспорт результатов
- `GET /search/statistics` - Статистика системы

### Модели данных

#### SearchRequest
```json
{
  "queries": ["string"],
  "method": "hybrid",
  "similarity_threshold": 0.6,
  "max_results": 10
}
```

#### SearchResponse
```json
{
  "results": {
    "query": [
      {
        "document_id": "string",
        "document": "string",
        "similarity_score": 0.95,
        "search_method": "hybrid",
        "rank": 1
      }
    ]
  },
  "statistics": {},
  "processing_time": 0.123
}
```

## Интеграция

### Python

```python
import requests

# Инициализация
response = requests.post('http://localhost:8000/search/initialize', 
                        json={'search_method': 'hybrid'})

# Поиск
response = requests.get('http://localhost:8000/search/search-single/болт м10')
results = response.json()
```

### JavaScript

```javascript
// Поиск аналогов
fetch('http://localhost:8000/search/search-single/болт м10')
  .then(response => response.json())
  .then(data => console.log(data.results));
```

### cURL

```bash
# Полный пример использования
curl -X POST "http://localhost:8000/search/initialize" \
     -H "Content-Type: application/json" \
     -d '{"search_method": "hybrid"}'

curl -X POST "http://localhost:8000/search/upload-catalog" \
     -F "file=@catalog.xlsx"

curl -X GET "http://localhost:8000/search/search-single/болт%20м10"
```

## Поддержка

Для получения поддержки:

1. Проверьте документацию
2. Изучите логи системы
3. Обратитесь к разработчикам

**Контакты:**
- GitHub: https://github.com/igornet0/SAMe
- Документация: См. файлы в папке docs/
