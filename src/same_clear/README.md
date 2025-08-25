# same_clear

Модуль обработки и очистки текста для SAMe (Semantic Analog Matching Engine).

## Описание

`same_clear` предоставляет инструменты для предобработки текстовых данных, включая очистку от HTML, нормализацию, лемматизацию и извлечение параметров. Этот модуль является первым этапом в пайплайне обработки данных SAMe.

## Основные возможности

- 🧹 **Очистка текста** - удаление HTML тегов, специальных символов
- 🔤 **Нормализация** - приведение к единому формату
- 📝 **Лемматизация** - приведение слов к начальной форме
- 🔍 **Извлечение параметров** - автоматическое извлечение технических параметров
- ⚡ **Высокая производительность** - оптимизированные алгоритмы
- 🌐 **Многоязычность** - поддержка русского и английского языков

## Установка

```bash
pip install same-clear
```

### Дополнительные возможности

```bash
# Расширенные возможности (pymorphy2, langdetect)
pip install same-clear[enhanced]

# Ускорение производительности (numba, cython)
pip install same-clear[performance]

# Для разработки
pip install same-clear[dev]
```

## Быстрый старт

### Очистка текста

```python
from same_clear.text_processing import TextCleaner

cleaner = TextCleaner()
result = cleaner.clean_text('Болт <b>М10х50</b> &nbsp; ГОСТ 7798-70')

print(result['normalized'])  # "Болт М10х50 ГОСТ 7798-70"
```

### Предобработка текста

```python
from same_clear.text_processing import TextPreprocessor

preprocessor = TextPreprocessor()
result = preprocessor.preprocess_text('БОЛТ М10Х50 ГОСТ 7798-70')

print(result['processed'])  # "болт м10х50 гост 7798-70"
```

### Извлечение параметров

```python
from same_clear.parameter_extraction import RegexParameterExtractor

extractor = RegexParameterExtractor()
parameters = extractor.extract_parameters('Болт М10х50 ГОСТ 7798-70')

for param in parameters:
    print(f"{param.name}: {param.value} ({param.parameter_type})")
```

## Архитектура модуля

### text_processing/

- **TextCleaner** - Очистка от HTML и специальных символов
- **TextNormalizer** - Нормализация регистра и пробелов
- **TextPreprocessor** - Комплексная предобработка
- **EnhancedPreprocessor** - Расширенная предобработка с ML
- **UnitsProcessor** - Обработка единиц измерения
- **SynonymsProcessor** - Обработка синонимов
- **TechCodesProcessor** - Обработка технических кодов

### parameter_extraction/

- **RegexParameterExtractor** - Извлечение параметров через регулярные выражения
- **MLParameterExtractor** - Извлечение параметров через ML
- **ParameterParser** - Парсинг и валидация параметров
- **ParameterUtils** - Утилиты для работы с параметрами

### utils/

- **CaseConverter** - Конвертация регистра
- **TextUtils** - Общие утилиты для работы с текстом

## Примеры использования

### Пакетная обработка

```python
from same_clear.text_processing import TextPreprocessor

preprocessor = TextPreprocessor()

texts = [
    'Болт М10х50 ГОСТ 7798-70',
    'Гайка М10 ГОСТ 5915-70',
    'Шайба 10 ГОСТ 11371-78'
]

# Пакетная обработка
results = [preprocessor.preprocess_text(text) for text in texts]

for result in results:
    print(f"Исходный: {result['original']}")
    print(f"Обработанный: {result['processed']}")
    print("---")
```

### Настройка конфигурации

```python
from same_clear.text_processing import TextCleaner, CleaningConfig

config = CleaningConfig(
    remove_html=True,
    normalize_spaces=True,
    remove_special_chars=True,
    preserve_numbers=True
)

cleaner = TextCleaner(config)
result = cleaner.clean_text('<b>Болт</b> М10х50')
```

### Извлечение конкретных типов параметров

```python
from same_clear.parameter_extraction import RegexParameterExtractor
from same_core.types import ParameterType

extractor = RegexParameterExtractor()
parameters = extractor.extract_parameters('Болт М10х50 ГОСТ 7798-70')

# Фильтрация по типу параметра
numeric_params = [p for p in parameters if p.parameter_type == ParameterType.NUMERIC]
standards = [p for p in parameters if p.parameter_type == ParameterType.STANDARD]

print("Числовые параметры:", numeric_params)
print("Стандарты:", standards)
```

## Интеграция с другими модулями

### С same_search

```python
from same_clear.text_processing import TextPreprocessor
from same_search.search_engine import FuzzySearchEngine

preprocessor = TextPreprocessor()
search_engine = FuzzySearchEngine()

# Предобработка документов перед индексацией
documents = ['Болт М10х50', 'Гайка М10', 'Шайба 10']
processed_docs = []

for doc in documents:
    result = preprocessor.preprocess_text(doc)
    processed_docs.append(result['processed'])

# Индексация обработанных документов
search_engine.fit(processed_docs, ['1', '2', '3'])
```

### С same_api

```python
from same_clear.text_processing import TextCleaner
from same_api.export import ExcelExporter
import pandas as pd

cleaner = TextCleaner()
exporter = ExcelExporter()

# Очистка данных перед экспортом
raw_data = ['<b>Болт</b> М10х50', 'Гайка &nbsp; М10']
cleaned_data = []

for item in raw_data:
    result = cleaner.clean_text(item)
    cleaned_data.append(result['normalized'])

# Экспорт очищенных данных
df = pd.DataFrame({
    'Raw': raw_data,
    'Cleaned': cleaned_data
})

exporter.export_data(df, 'cleaned_data.xlsx')
```

## Производительность

### Бенчмарки

```python
import time
from same_clear.text_processing import TextPreprocessor

preprocessor = TextPreprocessor()
text = "Болт М10х50 ГОСТ 7798-70" * 100

# Измерение производительности
start_time = time.time()
for _ in range(1000):
    result = preprocessor.preprocess_text(text)
end_time = time.time()

print(f"Обработано 1000 текстов за {end_time - start_time:.2f} секунд")
```

### Оптимизация

Для повышения производительности:

1. Используйте пакетную обработку
2. Установите `same-clear[performance]` для ускорения
3. Кэшируйте результаты для повторяющихся текстов

## Конфигурация

### Переменные окружения

```bash
# Язык по умолчанию
SAME_CLEAR_DEFAULT_LANGUAGE=ru

# Уровень логирования
SAME_CLEAR_LOG_LEVEL=INFO

# Путь к кэшу
SAME_CLEAR_CACHE_DIR=/tmp/same_clear_cache
```

### Файл конфигурации

```yaml
# same_clear_config.yaml
text_processing:
  remove_html: true
  normalize_case: true
  remove_extra_spaces: true
  preserve_numbers: true

parameter_extraction:
  confidence_threshold: 0.7
  max_parameters: 50
  extract_units: true
```

## Разработка

### Установка для разработки

```bash
git clone https://github.com/same-project/same-clear.git
cd same-clear
pip install -e .[dev]
```

### Запуск тестов

```bash
pytest tests/
pytest tests/ -v  # подробный вывод
pytest tests/ --cov=same_clear  # с покрытием кода
```

### Линтинг и форматирование

```bash
black same_clear/
isort same_clear/
mypy same_clear/
```

## CLI интерфейс

```bash
# Очистка текста
same-clean "Болт <b>М10х50</b> ГОСТ"

# Извлечение параметров
same-extract "Болт М10х50 ГОСТ 7798-70"

# Пакетная обработка файла
same-clean --file input.txt --output cleaned.txt
```

## Лицензия

MIT License. См. файл [LICENSE](LICENSE) для подробностей.

## Поддержка

- GitHub Issues: https://github.com/same-project/same-clear/issues
- Документация: https://same-clear.readthedocs.io/
- Email: dev@same-project.com
