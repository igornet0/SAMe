# Продвинутые методы поиска в системе SAMe

Система SAMe теперь поддерживает множество продвинутых алгоритмов поиска для различных сценариев использования. Этот документ описывает каждый метод, его применение и рекомендации по использованию.

## Обзор методов поиска

| Метод | Сложность | Применение | Приоритет |
|-------|-----------|------------|-----------|
| **Trie (префиксные деревья)** | O(m) | Поиск по началу строки | Высокий |
| **Inverted Index** | O(k) | Полнотекстовый поиск | Высокий |
| **MinHash/LSH** | O(1) амортизированная | Масштабируемый поиск похожих | Средний |
| **TF-IDF + Cosine** | O(n) | Базовая метрика схожести | Средний |
| **Advanced Embeddings** | O(d) | Глубокое семантическое понимание | Низкий |
| **Spatial Indexing (FAISS)** | O(log n) | Быстрый поиск ближайших соседей | Средний |
| **Graph-based Search** | O(V + E) | Кластерный анализ | Низкий |

## Детальное описание методов

### 1. Trie (Префиксные деревья)

**Назначение**: Быстрый поиск по началу строки с O(m) сложностью, где m - длина префикса.

**Когда использовать**:
- Поиск аналогов по началу названия товара
- Автодополнение в поисковых запросах
- Поиск по артикулам с известным префиксом

**Пример использования**:
```python
# Поиск всех товаров, начинающихся с "свет"
results = search_engine.prefix_search("свет", top_k=10)

# Через основной интерфейс
results = search_interface.search("свет", method="prefix", top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_trie_search=True,
    trie_min_prefix_length=2,  # Минимальная длина префикса
    trie_weight=0.3           # Вес в гибридном поиске
)
```

### 2. Inverted Index (Обратный индекс)

**Назначение**: Быстрый полнотекстовый поиск по токенам с хранением списка товаров для каждого токена.

**Когда использовать**:
- Поиск по ключевым словам
- Булевы запросы (AND, OR операции)
- Быстрый поиск в больших коллекциях

**Пример использования**:
```python
# Поиск товаров, содержащих "светильник" И "LED"
results = search_engine.inverted_index_search("светильник LED", top_k=10)

# Через основной интерфейс
results = search_interface.search("светильник LED", method="inverted_index", top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_inverted_index=True,
    inverted_index_weight=0.4  # Вес в гибридном поиске
)
```

### 3. TF-IDF + Cosine Similarity

**Назначение**: Векторное представление с TF-IDF весами и косинусным расстоянием для базовой метрики схожести по важности слов.

**Когда использовать**:
- Ранжирование результатов по релевантности
- Поиск документов с похожим содержанием
- Базовая текстовая аналитика

**Пример использования**:
```python
# TF-IDF поиск с ранжированием по релевантности
results = search_engine.tfidf_search("автомат защиты электрический", top_k=10)

# Через основной интерфейс
results = search_interface.search("автомат защиты", method="tfidf", top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_tfidf_search=True,
    tfidf_weight=0.35,
    tfidf_max_features=10000,    # Максимальное количество признаков
    tfidf_ngram_range=(1, 3)     # Диапазон n-грамм
)
```

### 4. MinHash/LSH (Locality-Sensitive Hashing)

**Назначение**: Быстрый поиск по Jaccard/Cosine схожести для масштабируемого сравнения похожих названий.

**Когда использовать**:
- Работа с большими объемами данных (>100k товаров)
- Поиск дубликатов и похожих записей
- Быстрая кластеризация

**Пример использования**:
```python
# LSH поиск похожих товаров
results = search_engine.lsh_search("болт М10 DIN 912", top_k=10)

# Через основной интерфейс
results = search_interface.search("болт М10", method="lsh", top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_lsh_search=True,
    lsh_weight=0.25,
    lsh_threshold=0.6,    # Порог схожести Jaccard
    lsh_num_perm=128      # Количество перестановок для MinHash
)
```

### 5. Spatial Indexing (FAISS)

**Назначение**: Быстрый поиск ближайших соседей в векторном пространстве с использованием FAISS.

**Когда использовать**:
- Оптимизация семантического поиска
- Работа с большими векторными пространствами
- Быстрый поиск похожих эмбеддингов

**Пример использования**:
```python
# Пространственный поиск (требует эмбеддинг запроса)
query_embedding = vectorizer.vectorize_tokens(token_ids)
results = search_engine.spatial_search(query_embedding, top_k=10)

# Через основной интерфейс (автоматически создает эмбеддинг)
results = search_interface.search("кабель силовой", method="spatial", top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_spatial_search=True,
    spatial_weight=0.3,
    faiss_index_type="flat"  # "flat", "ivf", "hnsw"
)
```

### 6. Graph-based Search

**Назначение**: Граф схожих объектов с поиском компонент связности для визуализации и кластерного анализа.

**Когда использовать**:
- Анализ связей между товарами
- Поиск товаров в одной категории
- Визуализация похожих товаров

**Пример использования**:
```python
# Поиск связанных товаров по индексу
results = search_engine.graph_search(reference_idx=100, top_k=10)
```

**Конфигурация**:
```python
config = SearchConfig(
    enable_graph_search=True,  # Внимание: вычислительно затратный
    graph_weight=0.2,
    graph_similarity_threshold=0.7  # Порог для создания рёбер
)
```

### 7. Advanced Hybrid Search

**Назначение**: Комбинирует все доступные методы поиска с взвешенным скорингом.

**Когда использовать**:
- Универсальный поиск для большинства запросов
- Максимальная точность результатов
- Когда неизвестен оптимальный метод

**Пример использования**:
```python
# Продвинутый гибридный поиск
results = search_engine.advanced_hybrid_search("трансформатор понижающий", top_k=10)

# Через основной интерфейс
results = search_interface.search("трансформатор", method="advanced_hybrid", top_k=10)
```

## Рекомендации по выбору метода

### По типу запроса

| Тип запроса | Рекомендуемый метод | Альтернативы | Описание |
|-------------|-------------------|--------------|----------|
| **Общий поиск** | `advanced_hybrid` | `extended_hybrid`, `hybrid` | Универсальный поиск для большинства запросов |
| **Технические термины** | `token_id` | `inverted_index`, `tfidf` | Поиск точных совпадений и технических терминов |
| **Поиск по префиксу** | `prefix` | `inverted_index` | Поиск по началу названия или артикула |
| **Семантически похожие** | `semantic` | `lsh`, `spatial` | Поиск семантически похожих товаров |
| **Быстрый поиск** | `spatial` | `lsh`, `inverted_index` | Быстрый поиск для больших объемов данных |

### По размеру данных

- **< 10k записей**: Любой метод, рекомендуется `advanced_hybrid`
- **10k - 100k записей**: `spatial`, `lsh`, `inverted_index`
- **> 100k записей**: `lsh`, `spatial`, избегать `graph_search`

### По требованиям к скорости

- **Максимальная скорость**: `inverted_index`, `spatial`, `lsh`
- **Баланс скорость/качество**: `hybrid`, `tfidf`
- **Максимальное качество**: `advanced_hybrid`, `extended_hybrid`

## Примеры использования

### Базовое использование

```python
from search_interface import SAMeSearchInterface

# Создание интерфейса
search_interface = SAMeSearchInterface()
search_interface.initialize()

# Поиск с разными методами
results_hybrid = search_interface.search("светильник LED", method="advanced_hybrid")
results_fast = search_interface.search("светильник LED", method="spatial")
results_prefix = search_interface.search("свет", method="prefix")
```

### Бенчмарк методов

```python
# Сравнение производительности всех методов
benchmark_results = search_interface.benchmark_methods("автомат защиты", top_k=10)

for method, result in benchmark_results.items():
    if result['available']:
        print(f"{method}: {result['execution_time']:.4f}s, {result['results_count']} results")
```

### Получение рекомендаций

```python
# Получение рекомендаций по типу запроса
recommendations = search_interface.get_method_recommendations("technical")
print(f"Для технических запросов рекомендуется: {recommendations['primary']}")
```

## Конфигурация и настройка

### Полная конфигурация

```python
config = SearchConfig(
    # Традиционные методы
    token_id_weight=0.6,
    semantic_weight=0.4,
    similarity_threshold=0.3,
    
    # Новые методы
    enable_trie_search=True,
    trie_min_prefix_length=2,
    trie_weight=0.3,
    
    enable_inverted_index=True,
    inverted_index_weight=0.4,
    
    enable_tfidf_search=True,
    tfidf_weight=0.35,
    tfidf_max_features=10000,
    tfidf_ngram_range=(1, 3),
    
    enable_lsh_search=True,
    lsh_weight=0.25,
    lsh_threshold=0.6,
    lsh_num_perm=128,
    
    enable_spatial_search=True,
    spatial_weight=0.3,
    faiss_index_type="flat",
    
    # Вычислительно затратные методы (по умолчанию отключены)
    enable_advanced_embeddings=False,
    enable_graph_search=False,
    
    max_results=100
)
```

### Оптимизация производительности

1. **Для быстрого поиска**: Отключите `graph_search` и `advanced_embeddings`
2. **Для экономии памяти**: Уменьшите `tfidf_max_features` и `lsh_num_perm`
3. **Для больших данных**: Используйте `faiss_index_type="ivf"`

## Устранение неполадок

### Частые проблемы

1. **Метод не возвращает результаты**:
   - Проверьте, что соответствующий индекс построен
   - Убедитесь, что `enable_*_search=True` в конфигурации

2. **Медленная работа**:
   - Отключите неиспользуемые методы
   - Используйте более быстрые индексы (FAISS IVF вместо Flat)

3. **Высокое потребление памяти**:
   - Уменьшите параметры индексов
   - Отключите graph_search для больших данных

### Логирование

Для отладки включите подробное логирование:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Это поможет отследить, какие индексы строятся и как работают методы поиска.

## Заключение

Система SAMe теперь предоставляет мощный набор алгоритмов поиска для различных сценариев использования. Выбор правильного метода зависит от:

- Типа данных и запросов
- Требований к скорости и точности
- Размера коллекции
- Доступных вычислительных ресурсов

Рекомендуется начать с `advanced_hybrid` метода для общих задач и переходить к специализированным методам при необходимости оптимизации производительности или точности.
