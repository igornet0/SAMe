import asyncio
from pathlib import Path
import argparse
import warnings
import logging
warnings.filterwarnings('ignore')

import sys
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(current_dir / "src"))

# NLP библиотеки
try:
    import spacy
    nlp = spacy.load('ru_core_news_lg')
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("⚠️ Русская модель SpaCy недоступна. Некоторые NLP функции будут ограничены.")

from src import data_helper, AnalogSearchProcessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)-45s:%(lineno)-3d - %(levelname)-7s - %(message)s',
    handlers=[
        logging.FileHandler(data_helper["log"] / 'analog_search_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(
        description="Обработка каталога и поиск аналогов с использованием SAMe проекта"
    )

    parser.add_argument(
        "input_csv",
        help="Путь к входному CSV файлу с обработанным каталогом"
    )

    parser.add_argument(
        "-o", "--output",
        help="Путь к выходному Excel файлу (опционально)",
        default=None
    )

    parser.add_argument(
        "-m", "--method",
        choices=[
            "fuzzy", "semantic", "hybrid", "extended_hybrid",
            "token_id", "prefix", "inverted_index", "tfidf",
            "lsh", "spatial", "advanced_hybrid", "hybrid_dbscan", "optimized_dbscan"
        ],
        default="extended_hybrid",
        help="Метод поиска аналогов. Доступные методы:\n"
             "  fuzzy, semantic, hybrid, extended_hybrid - классические методы\n"
             "  token_id - поиск по ID токенов\n"
             "  prefix - префиксный поиск\n"
             "  inverted_index - поиск по инвертированному индексу\n"
             "  tfidf - TF-IDF поиск\n"
             "  lsh - LSH поиск\n"
             "  spatial - пространственный поиск\n"
             "  advanced_hybrid - продвинутый гибридный поиск\n"
             "  hybrid_dbscan - гибридный DBSCAN поиск с кластеризацией\n"
             "  optimized_dbscan - оптимизированный DBSCAN для больших данных\n"
             "(по умолчанию: extended_hybrid)"
    )

    parser.add_argument(
        "--disable-extended-search",
        action="store_true",
        help="Отключить новую систему расширенного поиска (использовать только legacy)"
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.6,
        help="Порог схожести для фильтрации результатов (по умолчанию: 0.6)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод логов"
    )

    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Ограничить количество обрабатываемых записей (для тестирования)"
    )

    parser.add_argument(
        "--max-excel-results",
        type=int,
        default=1000000,
        help="Максимальное количество результатов для записи в Excel (по умолчанию: 1,000,000)"
    )

    args = parser.parse_args()

    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Проверка существования входного файла
        if not Path(args.input_csv).exists():
            print(f"❌ Входной файл не найден: {args.input_csv}")
            return 1

        print(f"🚀 Начало обработки каталога: {args.input_csv}")
        print(f"📊 Метод поиска: {args.method}")

        # Информация о типе поиска
        token_search_methods = {'token_id', 'prefix', 'inverted_index',
                                'tfidf', 'lsh', 'spatial',
                                'advanced_hybrid', 'hybrid'}

        if args.method == 'hybrid_dbscan':
            print(f"🔍 Тип поиска: Гибридный DBSCAN поиск с кластеризацией")
            print(f"🎯 Порог схожести: {args.threshold}")
            print(f"📊 Выходной формат: CSV с детальной информацией о кластерах")
        elif args.method == 'optimized_dbscan':
            print(f"🔍 Тип поиска: Оптимизированный DBSCAN для больших данных")
            print(f"🎯 Порог схожести: {args.threshold}")
            print(f"📊 Выходной формат: CSV с быстрым поиском")
            print(f"⚡ Оптимизация: Сэмплирование и упрощенные алгоритмы")
        elif args.method in token_search_methods:
            method_descriptions = {
                'token_id': 'Поиск по ID токенов',
                'prefix': 'Префиксный поиск (Trie)',
                'inverted_index': 'Поиск по инвертированному индексу',
                'tfidf': 'TF-IDF векторный поиск',
                'lsh': 'LSH приближенный поиск',
                'spatial': 'Пространственный поиск (FAISS)',
                'advanced_hybrid': 'Продвинутый гибридный поиск',
                'hybrid': 'Гибридный поиск (традиционный)'
            }
            print(f"🔍 Тип поиска: {method_descriptions.get(args.method, 'Поиск по токенам')}")
            if args.method == "token_id":
                print(f"🎯 Порог схожести (адаптированный): {args.threshold * 0.5}")
            else:
                print(f"🎯 Порог схожести: {args.threshold}")
        else:
            print(f"🎯 Порог схожести: {args.threshold}")

        print(f"🔍 Расширенный поиск: {'Отключен' if args.disable_extended_search else 'Включен'}")
        if args.limit:
            print(f"⚠️  Ограничение: обработка только первых {args.limit} записей")
        if args.max_excel_results < 1048575:
            print(f"📊 Лимит результатов в Excel: {args.max_excel_results:,}")

        # Создание процессора
        processor = AnalogSearchProcessor(
            search_method=args.method,
            similarity_threshold=args.threshold,
            use_extended_search=not args.disable_extended_search,
            max_excel_results=args.max_excel_results
        )

        # Обработка каталога
        output_path = await processor.process_catalog(
            input_csv_path=args.input_csv,
            output_excel_path=args.output,
            limit_records=args.limit
        )

        print(f"✅ Обработка завершена успешно!")
        print(f"📄 Результаты сохранены в: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        print(f"❌ Ошибка: {e}")
        return 1


if __name__ == "__main__":
    # Проверка аргументов командной строки
    if len(sys.argv) == 1:
        print(f"\n🔧 Для запуска обработки используйте:")
        print(f"python {sys.argv[0]} input.csv")
        print(f"\n📖 Для получения справки:")
        print(f"python {sys.argv[0]} --help")
    else:
        # Запуск основной функции
        exit_code = asyncio.run(main())
        sys.exit(exit_code)