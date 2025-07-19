#!/usr/bin/env python3
"""
Тестовый скрипт для проверки модулей SAMe из notebook
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(os.path.abspath('.'))

def test_imports():
    """Тестирует импорты всех модулей"""
    print("🧪 Тестирование импортов модулей SAMe")
    print("=" * 50)
    
    # Тест 1: Предобработка текста
    print("\n1️⃣ Тестирование модулей предобработки текста...")
    try:
        from same.text_processing.text_cleaner import TextCleaner, CleaningConfig
        print("   ✅ TextCleaner импортирован")
    except ImportError as e:
        print(f"   ❌ TextCleaner: {e}")

    try:
        from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
        print("   ✅ Lemmatizer импортирован")
    except ImportError as e:
        print(f"   ❌ Lemmatizer: {e}")

    try:
        from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
        print("   ✅ TextNormalizer импортирован")
    except ImportError as e:
        print(f"   ❌ TextNormalizer: {e}")
    
    try:
        from same.text_processing.preprocessor import TextPreprocessor, PreprocessorConfig
        print("   ✅ TextPreprocessor импортирован")
    except ImportError as e:
        print(f"   ❌ TextPreprocessor: {e}")
    
    # Тест 2: Поисковые алгоритмы
    print("\n2️⃣ Тестирование поисковых модулей...")
    try:
        from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
        print("   ✅ FuzzySearchEngine импортирован")
    except ImportError as e:
        print(f"   ❌ FuzzySearchEngine: {e}")
    
    try:
        from same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
        print("   ✅ SemanticSearchEngine импортирован")
    except ImportError as e:
        print(f"   ❌ SemanticSearchEngine: {e}")
    
    try:
        from same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
        print("   ✅ HybridSearchEngine импортирован")
    except ImportError as e:
        print(f"   ❌ HybridSearchEngine: {e}")

    try:
        from same.search_engine.indexer import SearchIndexer, IndexConfig
        print("   ✅ SearchIndexer импортирован")
    except ImportError as e:
        print(f"   ❌ SearchIndexer: {e}")

    # Тест 3: Извлечение параметров
    print("\n3️⃣ Тестирование модулей извлечения параметров...")
    try:
        from same.parameter_extraction.regex_extractor import (
            RegexParameterExtractor, ParameterPattern, ParameterType, ExtractedParameter
        )
        print("   ✅ RegexParameterExtractor импортирован")
    except ImportError as e:
        print(f"   ❌ RegexParameterExtractor: {e}")

    try:
        from same.parameter_extraction.ml_extractor import MLParameterExtractor, MLExtractorConfig
        print("   ✅ MLParameterExtractor импортирован")
    except ImportError as e:
        print(f"   ❌ MLParameterExtractor: {e}")

    try:
        from same.parameter_extraction.parameter_parser import ParameterParser, ParameterParserConfig
        print("   ✅ ParameterParser импортирован")
    except ImportError as e:
        print(f"   ❌ ParameterParser: {e}")

    # Тест 4: Экспорт
    print("\n4️⃣ Тестирование модулей экспорта...")
    try:
        from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
        print("   ✅ ExcelExporter импортирован")
    except ImportError as e:
        print(f"   ❌ ExcelExporter: {e}")
    
    try:
        from same.export.report_generator import ReportGenerator, ReportConfig
        print("   ✅ ReportGenerator импортирован")
    except ImportError as e:
        print(f"   ❌ ReportGenerator: {e}")


def test_basic_functionality():
    """Тестирует базовую функциональность модулей"""
    print("\n🔧 Тестирование базовой функциональности")
    print("=" * 50)
    
    # Тестовые данные
    sample_data = [
        "Болт М10×50 ГОСТ 7798-70 оцинкованный",
        "Двигатель асинхронный АИР80В2 1.5кВт 3000об/мин",
        "Труба стальная 57×3.5 ГОСТ 8732-78 бесшовная"
    ]
    
    # Тест TextCleaner
    print("\n🧹 Тестирование TextCleaner...")
    try:
        from same.text_processing.text_cleaner import TextCleaner, CleaningConfig
        
        config = CleaningConfig(
            remove_html=True,
            remove_special_chars=True,
            remove_extra_spaces=True,
            remove_numbers=False
        )
        
        cleaner = TextCleaner(config)
        
        test_text = "<p>Болт М10×50 @#$% ГОСТ 7798-70</p>"
        result = cleaner.clean_text(test_text)
        
        print(f"   Исходный: '{test_text}'")
        print(f"   Очищенный: '{result['normalized']}'")
        print("   ✅ TextCleaner работает")
        
    except Exception as e:
        print(f"   ❌ Ошибка TextCleaner: {e}")
    
    # Тест RegexParameterExtractor
    print("\n🔧 Тестирование RegexParameterExtractor...")
    try:
        from same.parameter_extraction.regex_extractor import RegexParameterExtractor
        
        extractor = RegexParameterExtractor()
        
        test_text = "Болт М10×50 диаметр 10мм длина 50мм"
        parameters = extractor.extract_parameters(test_text)
        
        print(f"   Текст: '{test_text}'")
        print(f"   Найдено параметров: {len(parameters)}")
        
        for param in parameters[:3]:  # Показываем первые 3
            print(f"   - {param.name}: {param.value} {param.unit or ''}")
        
        print("   ✅ RegexParameterExtractor работает")
        
    except Exception as e:
        print(f"   ❌ Ошибка RegexParameterExtractor: {e}")
    
    # Тест FuzzySearchEngine
    print("\n🔍 Тестирование FuzzySearchEngine...")
    try:
        from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
        
        config = FuzzySearchConfig(
            tfidf_max_features=1000,
            similarity_threshold=0.1,
            max_results=3
        )
        
        engine = FuzzySearchEngine(config)
        
        # Обучаем на тестовых данных
        document_ids = list(range(1, len(sample_data) + 1))
        engine.fit(sample_data, document_ids)
        
        # Тестируем поиск
        query = "болт м10"
        results = engine.search(query)
        
        print(f"   Запрос: '{query}'")
        print(f"   Найдено результатов: {len(results)}")
        
        for i, result in enumerate(results[:2], 1):
            print(f"   {i}. {result['document'][:40]}... (скор: {result.get('combined_score', 0):.3f})")
        
        print("   ✅ FuzzySearchEngine работает")
        
    except Exception as e:
        print(f"   ❌ Ошибка FuzzySearchEngine: {e}")


def test_data_creation():
    """Тестирует создание тестовых данных как в notebook"""
    print("\n📋 Тестирование создания тестовых данных")
    print("=" * 50)
    
    def create_sample_mtr_data():
        """Создает тестовые данные МТР"""
        return [
            "Болт М10×50 ГОСТ 7798-70 оцинкованный",
            "Двигатель асинхронный АИР80В2 1.5кВт 3000об/мин",
            "Труба стальная 57×3.5 ГОСТ 8732-78 бесшовная",
            "Гайка М10 шестигранная ГОСТ 5915-70",
            "Кабель ВВГ 3×2.5 мм² 0.66кВ медный"
        ]
    
    sample_data = create_sample_mtr_data()
    
    print(f"✅ Создано {len(sample_data)} образцов МТР")
    print("\n📝 Примеры данных:")
    for i, item in enumerate(sample_data, 1):
        print(f"{i}. {item}")
    
    return sample_data


def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование модулей SAMe Notebook")
    print("=" * 60)
    
    # Проверяем рабочую директорию
    print(f"📁 Рабочая директория: {os.getcwd()}")
    print(f"🐍 Python версия: {sys.version}")
    
    # Запускаем тесты
    test_imports()
    test_data_creation()
    test_basic_functionality()
    
    print("\n🎉 Тестирование завершено!")
    print("\n💡 Рекомендации:")
    print("1. Установите недостающие зависимости: pip install matplotlib seaborn")
    print("2. Для семантического поиска: pip install sentence-transformers faiss-cpu")
    print("3. Для SpaCy: pip install spacy && python -m spacy download ru_core_news_lg")
    print("4. Запустите notebook: jupyter notebook SAMe_Demo.ipynb")


if __name__ == "__main__":
    main()
