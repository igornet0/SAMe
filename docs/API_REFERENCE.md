# API Reference - SAMe System

## Базовая информация

**Base URL:** `http://localhost:8000`

**Content-Type:** `application/json`

**Версия API:** v1

## Аутентификация

В текущей версии аутентификация не требуется. В продакшене рекомендуется использовать JWT токены.

## Endpoints

### 1. Проверка состояния системы

#### GET /search/health

Проверка работоспособности системы.

**Параметры:** Нет

**Ответ:**
```json
{
  "status": "healthy",
  "search_engine_ready": true,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

**Коды ответов:**
- `200` - Система работает
- `503` - Система недоступна

**Пример:**
```bash
curl -X GET "http://localhost:8000/search/health"
```

---

### 2. Инициализация поискового движка

#### POST /search/initialize

Инициализация системы поиска с заданными параметрами.

**Тело запроса:**
```json
{
  "catalog_file_path": "string",
  "search_method": "hybrid",
  "similarity_threshold": 0.6
}
```

**Параметры:**
- `catalog_file_path` (optional) - Путь к файлу каталога
- `search_method` (optional) - Метод поиска: `fuzzy`, `semantic`, `hybrid`
- `similarity_threshold` (optional) - Порог схожести (0.0-1.0)

**Ответ:**
```json
{
  "status": "success",
  "message": "Search engine initialized successfully",
  "statistics": {
    "total_documents": 1000,
    "search_method": "hybrid",
    "similarity_threshold": 0.6
  }
}
```

**Коды ответов:**
- `200` - Успешная инициализация
- `400` - Неверные параметры
- `500` - Ошибка инициализации

**Пример:**
```bash
curl -X POST "http://localhost:8000/search/initialize" \
     -H "Content-Type: application/json" \
     -d '{
       "search_method": "hybrid",
       "similarity_threshold": 0.6
     }'
```

---

### 3. Загрузка каталога

#### POST /search/upload-catalog

Загрузка каталога МТР из файла.

**Параметры:**
- `file` (form-data) - Файл каталога (.xlsx или .csv)

**Ответ:**
```json
{
  "status": "success",
  "message": "Catalog uploaded successfully. 1000 items loaded.",
  "statistics": {
    "total_documents": 1000,
    "search_engine_ready": true
  }
}
```

**Коды ответов:**
- `200` - Успешная загрузка
- `400` - Неподдерживаемый формат файла
- `500` - Ошибка обработки файла

**Пример:**
```bash
curl -X POST "http://localhost:8000/search/upload-catalog" \
     -F "file=@catalog.xlsx"
```

---

### 4. Поиск одного аналога

#### GET /search/search-single/{query}

Поиск аналогов для одного запроса.

**Параметры пути:**
- `query` (required) - Поисковый запрос

**Query параметры:**
- `method` (optional) - Метод поиска: `fuzzy`, `semantic`, `hybrid`
- `max_results` (optional) - Максимальное количество результатов (1-100)

**Ответ:**
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
      "catalog_item": {
        "id": 1,
        "name": "Болт М10х50 ГОСТ 7798-70",
        "category": "Крепеж"
      }
    }
  ],
  "method": "hybrid"
}
```

**Коды ответов:**
- `200` - Успешный поиск
- `400` - Система не инициализирована
- `500` - Ошибка поиска

**Пример:**
```bash
curl -X GET "http://localhost:8000/search/search-single/болт%20м10?method=hybrid&max_results=5"
```

---

### 5. Пакетный поиск аналогов

#### POST /search/search-analogs

Поиск аналогов для множества запросов.

**Тело запроса:**
```json
{
  "queries": ["болт м10", "гайка м10", "шайба 10"],
  "method": "hybrid",
  "similarity_threshold": 0.6,
  "max_results": 10
}
```

**Параметры:**
- `queries` (required) - Список поисковых запросов
- `method` (optional) - Метод поиска
- `similarity_threshold` (optional) - Порог схожести
- `max_results` (optional) - Максимальное количество результатов на запрос

**Ответ:**
```json
{
  "results": {
    "болт м10": [
      {
        "document_id": 1,
        "document": "Болт М10х50 ГОСТ 7798-70",
        "similarity_score": 0.95,
        "search_method": "hybrid",
        "rank": 1
      }
    ],
    "гайка м10": [
      {
        "document_id": 2,
        "document": "Гайка М10 DIN 934",
        "similarity_score": 0.92,
        "search_method": "hybrid",
        "rank": 1
      }
    ]
  },
  "statistics": {
    "total_queries": 3,
    "total_results": 15,
    "search_method": "hybrid"
  },
  "processing_time": 0.234
}
```

**Коды ответов:**
- `200` - Успешный поиск
- `400` - Неверные параметры или система не инициализирована
- `500` - Ошибка поиска

**Пример:**
```bash
curl -X POST "http://localhost:8000/search/search-analogs" \
     -H "Content-Type: application/json" \
     -d '{
       "queries": ["болт м10", "гайка м10"],
       "method": "hybrid",
       "max_results": 5
     }'
```

---

### 6. Экспорт результатов

#### POST /search/export-results

Экспорт результатов поиска в файл.

**Тело запроса:**
```json
{
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
}
```

**Параметры:**
- `results` (required) - Результаты поиска для экспорта
- `format` (optional) - Формат экспорта: `excel` (по умолчанию)

**Ответ:** Файл Excel

**Коды ответов:**
- `200` - Успешный экспорт
- `400` - Неверные параметры
- `500` - Ошибка экспорта

**Пример:**
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
       }
     }' \
     --output results.xlsx
```

---

### 7. Получение статистики

#### GET /search/statistics

Получение статистики работы системы.

**Параметры:** Нет

**Ответ:**
```json
{
  "is_ready": true,
  "catalog_size": 1000,
  "search_method": "hybrid",
  "similarity_threshold": 0.6,
  "fuzzy_engine": {
    "status": "fitted",
    "total_documents": 1000,
    "vocabulary_size": 5000
  },
  "semantic_engine": {
    "status": "fitted",
    "total_documents": 1000,
    "embedding_dimension": 384,
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  }
}
```

**Коды ответов:**
- `200` - Статистика получена

**Пример:**
```bash
curl -X GET "http://localhost:8000/search/statistics"
```

---

### 8. Сохранение моделей

#### POST /search/save-models

Сохранение обученных моделей на диск.

**Параметры:** Нет

**Ответ:**
```json
{
  "status": "success",
  "message": "Models saved successfully"
}
```

**Коды ответов:**
- `200` - Модели сохранены
- `400` - Система не инициализирована
- `500` - Ошибка сохранения

**Пример:**
```bash
curl -X POST "http://localhost:8000/search/save-models"
```

---

### 9. Загрузка моделей

#### POST /search/load-models

Загрузка ранее сохраненных моделей.

**Параметры:** Нет

**Ответ:**
```json
{
  "status": "success",
  "message": "Models loaded successfully"
}
```

**Коды ответов:**
- `200` - Модели загружены
- `500` - Ошибка загрузки

**Пример:**
```bash
curl -X POST "http://localhost:8000/search/load-models"
```

## Модели данных

### SearchRequest

```json
{
  "queries": ["string"],
  "method": "hybrid",
  "similarity_threshold": 0.6,
  "max_results": 10
}
```

**Поля:**
- `queries` - Список поисковых запросов
- `method` - Метод поиска (`fuzzy`, `semantic`, `hybrid`)
- `similarity_threshold` - Порог схожести (0.0-1.0)
- `max_results` - Максимальное количество результатов (1-100)

### SearchResponse

```json
{
  "results": {
    "query": [
      {
        "document_id": "string|number",
        "document": "string",
        "similarity_score": 0.95,
        "search_method": "string",
        "rank": 1,
        "catalog_item": {}
      }
    ]
  },
  "statistics": {},
  "processing_time": 0.123
}
```

### InitializeRequest

```json
{
  "catalog_file_path": "string",
  "search_method": "hybrid",
  "similarity_threshold": 0.6
}
```

## Коды ошибок

### HTTP статус коды

- `200` - Успешный запрос
- `400` - Неверный запрос
- `404` - Ресурс не найден
- `500` - Внутренняя ошибка сервера
- `503` - Сервис недоступен

### Коды ошибок приложения

```json
{
  "error": {
    "code": "SEARCH_ENGINE_NOT_INITIALIZED",
    "message": "Search engine is not initialized. Call initialize() first.",
    "details": {}
  }
}
```

**Основные коды:**
- `SEARCH_ENGINE_NOT_INITIALIZED` - Система не инициализирована
- `INVALID_FILE_FORMAT` - Неподдерживаемый формат файла
- `CATALOG_NOT_LOADED` - Каталог не загружен
- `INVALID_SEARCH_METHOD` - Неверный метод поиска
- `PROCESSING_ERROR` - Ошибка обработки

## Ограничения

### Размеры запросов

- Максимальный размер файла каталога: 100 MB
- Максимальное количество запросов в пакете: 1000
- Максимальная длина запроса: 1000 символов
- Максимальное количество результатов: 100

### Rate Limiting

- Максимум 100 запросов в минуту на IP
- Максимум 10 одновременных соединений

### Таймауты

- Таймаут запроса: 30 секунд
- Таймаут инициализации: 300 секунд
- Таймаут загрузки каталога: 600 секунд

## Примеры интеграции

### Python

```python
import requests
import json

class SAMeClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def initialize(self, method="hybrid", threshold=0.6):
        response = requests.post(
            f"{self.base_url}/search/initialize",
            json={
                "search_method": method,
                "similarity_threshold": threshold
            }
        )
        return response.json()
    
    def upload_catalog(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/search/upload-catalog",
                files={"file": f}
            )
        return response.json()
    
    def search(self, queries, method="hybrid", max_results=10):
        if isinstance(queries, str):
            queries = [queries]
        
        response = requests.post(
            f"{self.base_url}/search/search-analogs",
            json={
                "queries": queries,
                "method": method,
                "max_results": max_results
            }
        )
        return response.json()

# Использование
client = SAMeClient()
client.initialize()
client.upload_catalog("catalog.xlsx")
results = client.search(["болт м10", "гайка м10"])
```

### JavaScript

```javascript
class SAMeClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async initialize(method = 'hybrid', threshold = 0.6) {
        const response = await fetch(`${this.baseUrl}/search/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                search_method: method,
                similarity_threshold: threshold
            })
        });
        return response.json();
    }
    
    async search(queries, method = 'hybrid', maxResults = 10) {
        if (typeof queries === 'string') {
            queries = [queries];
        }
        
        const response = await fetch(`${this.baseUrl}/search/search-analogs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                queries: queries,
                method: method,
                max_results: maxResults
            })
        });
        return response.json();
    }
}

// Использование
const client = new SAMeClient();
await client.initialize();
const results = await client.search(['болт м10', 'гайка м10']);
```
