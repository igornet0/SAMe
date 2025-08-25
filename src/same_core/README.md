# same_core

Базовые интерфейсы и типы данных для SAMe (Semantic Analog Matching Engine).

## Описание

`same_core` предоставляет общие интерфейсы, типы данных и утилиты интеграции, которые используются всеми остальными модулями SAMe. Этот модуль обеспечивает единообразие API и типобезопасность во всей системе.

## Основные компоненты

### Интерфейсы (`interfaces.py`)

- **TextProcessorInterface** - Интерфейс для обработки текста
- **SearchEngineInterface** - Интерфейс для поисковых движков  
- **ExporterInterface** - Интерфейс для экспорта данных
- **DataManagerInterface** - Интерфейс для управления данными
- **ParameterExtractorInterface** - Интерфейс для извлечения параметров
- **AnalogSearchEngineInterface** - Protocol для главного класса

### Типы данных (`types.py`)

- **ProcessingResult** - Результат обработки текста
- **SearchResult** - Результат поиска
- **ParameterData** - Данные извлеченного параметра
- **SearchConfig** - Конфигурация поиска
- **ExportConfig** - Конфигурация экспорта

### Перечисления

- **ProcessingStage** - Этапы обработки текста
- **ParameterType** - Типы параметров
- **SearchMethod** - Методы поиска

## Установка

```bash
pip install same-core
```

## Использование

### Базовые интерфейсы

```python
from same_core.interfaces import TextProcessorInterface, SearchEngineInterface
from same_core.types import ProcessingResult, SearchResult

class MyTextProcessor(TextProcessorInterface):
    def process_text(self, text: str) -> ProcessingResult:
        # Ваша реализация
        return ProcessingResult(
            original=text,
            processed=text.lower(),
            stages={},
            metadata={},
            processing_time=0.1
        )

class MySearchEngine(SearchEngineInterface):
    def fit(self, documents: List[str], document_ids: List[str]) -> None:
        # Ваша реализация
        pass
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Ваша реализация
        return []
```

### Типы данных

```python
from same_core.types import ProcessingStage, ParameterType, SearchConfig

# Конфигурация поиска
config = SearchConfig(
    method=SearchMethod.HYBRID,
    similarity_threshold=0.7,
    max_results=10,
    enable_caching=True
)

# Результат обработки
result = ProcessingResult(
    original="Болт М10х50 ГОСТ 7798-70",
    processed="болт м10х50 гост 7798-70",
    stages={
        ProcessingStage.RAW: "Болт М10х50 ГОСТ 7798-70",
        ProcessingStage.CLEANED: "Болт М10х50 ГОСТ 7798-70",
        ProcessingStage.NORMALIZED: "болт м10х50 гост 7798-70"
    },
    metadata={"language": "ru"},
    processing_time=0.05
)
```

## Интеграция с другими модулями

`same_core` служит основой для всех остальных модулей SAMe:

- **same_clear** - использует интерфейсы для обработки текста
- **same_search** - использует интерфейсы для поиска
- **same_api** - использует интерфейсы для экспорта и управления данными

## Разработка

### Установка для разработки

```bash
git clone https://github.com/same-project/same-core.git
cd same-core
pip install -e .[dev]
```

### Запуск тестов

```bash
pytest tests/
```

### Проверка типов

```bash
mypy same_core/
```

## Лицензия

MIT License. См. файл [LICENSE](LICENSE) для подробностей.

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста, ознакомьтесь с [CONTRIBUTING.md](CONTRIBUTING.md) для получения инструкций.

## Поддержка

- GitHub Issues: https://github.com/same-project/same-core/issues
- Документация: https://same-core.readthedocs.io/
- Email: dev@same-project.com
