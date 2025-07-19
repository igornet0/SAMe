# Руководство по миграции импортов SAMe

## Обзор изменений

В рамках рефакторинга проекта SAMe была изменена структура модулей. Все модули были перенесены из директории `core/` в `src/same/`.

## Таблица соответствия импортов

### Модули предобработки текста
| Старый импорт | Новый импорт |
|---------------|--------------|
| `from core.text_processing.text_cleaner import TextCleaner` | `from same.text_processing.text_cleaner import TextCleaner` |
| `from core.text_processing.lemmatizer import Lemmatizer` | `from same.text_processing.lemmatizer import Lemmatizer` |
| `from core.text_processing.normalizer import TextNormalizer` | `from same.text_processing.normalizer import TextNormalizer` |
| `from core.text_processing.preprocessor import TextPreprocessor` | `from same.text_processing.preprocessor import TextPreprocessor` |

### Поисковые алгоритмы
| Старый импорт | Новый импорт |
|---------------|--------------|
| `from core.search_engine.fuzzy_search import FuzzySearchEngine` | `from same.search_engine.fuzzy_search import FuzzySearchEngine` |
| `from core.search_engine.semantic_search import SemanticSearchEngine` | `from same.search_engine.semantic_search import SemanticSearchEngine` |
| `from core.search_engine.hybrid_search import HybridSearchEngine` | `from same.search_engine.hybrid_search import HybridSearchEngine` |
| `from core.search_engine.indexer import SearchIndexer` | `from same.search_engine.indexer import SearchIndexer` |

### Извлечение параметров
| Старый импорт | Новый импорт |
|---------------|--------------|
| `from core.parameter_extraction.regex_extractor import RegexParameterExtractor` | `from same.parameter_extraction.regex_extractor import RegexParameterExtractor` |
| `from core.parameter_extraction.ml_extractor import MLParameterExtractor` | `from same.parameter_extraction.ml_extractor import MLParameterExtractor` |
| `from core.parameter_extraction.parameter_parser import ParameterParser` | `from same.parameter_extraction.parameter_parser import ParameterParser` |

### Экспорт результатов
| Старый импорт | Новый импорт |
|---------------|--------------|
| `from core.export.excel_exporter import ExcelExporter` | `from same.export.excel_exporter import ExcelExporter` |
| `from core.export.report_generator import ReportGenerator` | `from same.export.report_generator import ReportGenerator` |

## Примеры миграции кода

### До миграции
```python
# Старый код
from core.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
from core.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from core.export.excel_exporter import ExcelExporter, ExcelExportConfig

# Создание компонентов
lemmatizer = Lemmatizer(LemmatizerConfig())
search_engine = FuzzySearchEngine(FuzzySearchConfig())
exporter = ExcelExporter(ExcelExportConfig())
```

### После миграции
```python
# Новый код
from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
from same.export.excel_exporter import ExcelExporter, ExcelExportConfig

# Создание компонентов (логика не изменилась)
lemmatizer = Lemmatizer(LemmatizerConfig())
search_engine = FuzzySearchEngine(FuzzySearchConfig())
exporter = ExcelExporter(ExcelExportConfig())
```

## Автоматическая миграция

Для автоматической замены импортов в ваших файлах можно использовать следующий скрипт:

```bash
# Замена импортов в Python файлах
find . -name "*.py" -type f -exec sed -i 's/from core\./from same\./g' {} \;
find . -name "*.py" -type f -exec sed -i 's/import core\./import same\./g' {} \;
```

## Обратная совместимость

Все API и интерфейсы остались неизменными. Изменились только пути импорта. Функциональность модулей полностью сохранена.

## Проверка миграции

После миграции импортов убедитесь, что:

1. Все тесты проходят успешно:
```bash
poetry run pytest tests/ -v
```

2. Импорты работают корректно:
```python
# Проверочный скрипт
try:
    from same.export.excel_exporter import ExcelExporter
    from same.search_engine.fuzzy_search import FuzzySearchEngine
    from same.text_processing.lemmatizer import Lemmatizer
    print("✅ Все импорты работают корректно")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
```

## Поддержка

Если у вас возникли проблемы с миграцией, обратитесь к:
- [Документации по установке](INSTALLATION.md)
- [Руководству пользователя](USER_GUIDE.md)
- [Справочнику API](API_REFERENCE.md)
