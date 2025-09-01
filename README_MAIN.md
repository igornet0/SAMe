# 🚀 SAMe - Быстрый старт

## Установка и запуск

```bash
# Установка зависимостей
poetry install

# Базовый запуск (рекомендуется)
poetry run python main.py your_catalog.xlsx --improved

# Справка
poetry run python main.py --help
```

## 🔥 Рекомендуемые команды

### Для небольших каталогов (< 1000 товаров)
```bash
poetry run python main.py catalog.xlsx --improved
```

### Для средних каталогов (1000-10000 товаров)
```bash
poetry run python main.py catalog.xlsx -t 0.3 -l 5000 --improved
```

### Для больших каталогов (> 10000 товаров)
```bash
poetry run python main.py catalog.xlsx --batch --improved
```

## 📊 Основные параметры

- `-t 0.3` - Порог схожести (чем меньше, тем больше результатов)
- `-l 1000` - Лимит записей для обработки
- `--improved` - **Рекомендуемый режим** с улучшенной фильтрацией
- `--batch` - Пакетная обработка для больших файлов

## 📁 Результаты

После обработки система создаст папку `src/data/output/ДАТА-ВРЕМЯ/` с файлами:
- `processed_data_with_duplicates.csv` - Обработанный каталог
- `analogs_search_results.csv` - Найденные аналоги
- `product_trees.txt` - Деревья связей
- `processing_report.txt` - Отчет об обработке

## 📖 Полная документация

См. файл `MAIN_PY_USAGE.md` для подробного описания всех возможностей.

---

**💡 Совет:** Начните с команды `poetry run python main.py your_file.xlsx -t 0.3 -l 100 --improved` для первого знакомства с системой.
