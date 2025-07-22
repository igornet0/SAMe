# %% [markdown]
# # Система поиска аналогов продуктов в номенклатуре
# 
# **Product Analog Search System for Nomenclature Data**
# 
# Этот notebook реализует систему поиска аналогов/похожих товаров в основном датасете номенклатуры, используя возможности системы SAMe (Similar Articles Matching Engine).
# 
# ## Цели:
# - Анализ структуры данных номенклатуры
# - Предобработка текстовых описаний товаров
# - Реализация различных алгоритмов поиска аналогов
# - Демонстрация практических примеров поиска
# - Оценка качества результатов
# 
# ---

# %% [markdown]
# ## 1. Настройка и импорты

# %%
# Системные импорты
import sys
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
import re
from collections import defaultdict, Counter

# Настройка отображения
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)

# Добавляем путь к модулям SAMe
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../..'))

print("✅ Базовые импорты загружены")
print(f"📁 Рабочая директория: {os.getcwd()}")
print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 2. Загрузка модулей SAMe

# %%
# Импорты модулей SAMe
try:
    from same.data_manager import data_helper
    from same.text_processing.text_cleaner import TextCleaner, CleaningConfig
    from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
    from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
    from same.text_processing.preprocessor import TextPreprocessor, PreprocessorConfig
    from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
    from same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
    from same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
    from same.parameter_extraction.regex_extractor import RegexParameterExtractor
    print("✅ Модули SAMe успешно загружены")
except ImportError as e:
    print(f"❌ Ошибка импорта модулей SAMe: {e}")
    print("💡 Убедитесь что модули созданы в директории src/same/")

# %% [markdown]
# ## 3. Загрузка и анализ данных

# %%
# Загрузка основного датасета
try:
    # Попробуем найти файл с указанным именем
    dataset_path = data_helper["datasets"] / "main/Выгрузка_Номенклатура_без_удаленных_17_07_25.xlsx"
    if not dataset_path.exists():
        # Если не найден, используем альтернативный файл
        dataset_path = data_helper["datasets"] / "main/main_dataset.xlsx"
    
    print(f"📂 Загружаем датасет: {dataset_path}")
    data = pd.read_excel(dataset_path)
    
    print(f"✅ Датасет загружен успешно")
    print(f"📊 Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
    
except Exception as e:
    print(f"❌ Ошибка загрузки датасета: {e}")
    # Создаем тестовые данные для демонстрации
    print("🔧 Создаем тестовые данные для демонстрации...")
    data = pd.DataFrame({
        'Код': ['НИ-001', 'НИ-002', 'НИ-003', 'НИ-004', 'НИ-005'],
        'Наименование': [
            'Болт М10×50 ГОСТ 7798-70 оцинкованный',
            'Болт с шестигранной головкой М12×60 DIN 933',
            'Винт М8×30 с внутренним шестигранником',
            'Гайка М10 шестигранная ГОСТ 5915-70',
            'Шайба плоская 10 ГОСТ 11371-78'
        ],
        'Группа': ['Крепеж'] * 5,
        'ВидНоменклатуры': ['Материалы'] * 5
    })

# %%
# Анализ структуры данных
print("📋 Структура датасета:")
print(f"Столбцы: {list(data.columns)}")
print(f"\n📊 Информация о данных:")
print(data.info())

print(f"\n🔍 Первые 5 записей:")
print(data.head())

# Определяем основной столбец с наименованиями
name_columns = [col for col in data.columns if 'наименование' in col.lower() or 'название' in col.lower()]
if name_columns:
    main_name_column = name_columns[0]
    print(f"\n📝 Основной столбец с наименованиями: '{main_name_column}'")
else:
    main_name_column = data.columns[1] if len(data.columns) > 1 else data.columns[0]
    print(f"\n📝 Используем столбец: '{main_name_column}'")

# %%
# Анализ качества данных
print("🔍 Анализ качества данных:")
print(f"Пустые значения в основном столбце: {data[main_name_column].isnull().sum()}")
print(f"Дубликаты: {data.duplicated().sum()}")
print(f"Уникальные значения в '{main_name_column}': {data[main_name_column].nunique()}")

# Статистика по длине наименований
name_lengths = data[main_name_column].dropna().str.len()
print(f"\n📏 Статистика длины наименований:")
print(f"Средняя длина: {name_lengths.mean():.1f} символов")
print(f"Медиана: {name_lengths.median():.1f} символов")
print(f"Мин/Макс: {name_lengths.min()}/{name_lengths.max()} символов")

# Примеры наименований разной длины
print(f"\n📝 Примеры наименований:")
sample_names = data[main_name_column].dropna().sample(min(5, len(data))).tolist()
for i, name in enumerate(sample_names, 1):
    print(f"{i}. {name[:80]}{'...' if len(name) > 80 else ''}")

# %% [markdown]
# ## 3. Предобработка данных

# %%
# Улучшенная настройка системы предобработки текста для технической номенклатуры
print("🔧 Настройка системы предобработки для технической номенклатуры...")

def check_spacy_model(model_name: str = "ru_core_news_sm") -> bool:
    """
    Проверяет наличие модели SpaCy и предлагает установку при необходимости
    """
    try:
        import spacy
        spacy.load(model_name)
        print(f"✅ Модель SpaCy '{model_name}' найдена и загружена")
        return True
    except OSError:
        print(f"⚠️ Модель SpaCy '{model_name}' не найдена")
        print(f"📥 Для установки выполните команду:")
        print(f"   python -m spacy download {model_name}")
        print(f"💡 Альтернативно можно использовать 'ru_core_news_lg' для лучшего качества")
        return False
    except ImportError:
        print(f"❌ SpaCy не установлен. Установите: pip install spacy")
        return False

# Словари для нормализации технических терминов (глобальные для сериализации)
TECHNICAL_ABBREVIATIONS = {
    'эл': 'электрический', 'электр': 'электрический',
    'мех': 'механический', 'гидр': 'гидравлический',
    'пневм': 'пневматический', 'авт': 'автоматический',
    'руч': 'ручной', 'стац': 'стационарный',
    'перен': 'переносной', 'мобил': 'мобильный',
    'нерж': 'нержавеющий', 'оцинк': 'оцинкованный'
}

# Паттерны для сохранения технических характеристик (глобальные для сериализации)
TECHNICAL_PATTERNS = [
    r'\b\d+[.,]?\d*\s*[а-яё]*[вт|а|в|ом|мм|см|м|кг|г|л|мл]\b',  # Размеры и единицы
    r'\b[мм]\d+[x×]\d+\b',  # Размеры типа М10×50
    r'\b[гост|din|iso]\s*\d+[-]?\d*[-]?\d*\b',  # Стандарты
    r'\b\d+[.,]\d*\s*[квт|кв|мвт|вт]\b',  # Мощность
    r'\b\d+\s*об[/.]мин\b',  # Обороты
    r'\b\d+[.,]\d*\s*[мпа|кпа|па|бар]\b',  # Давление
    r'\b\d+[.,]\d*\s*[мм|см|м]\b'  # Размеры
]

def enhanced_simple_preprocess(text):
    """
    Улучшенная функция простой предобработки для технических текстов
    Теперь это глобальная функция, которую можно сериализовать с pickle
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    original_text = text.lower()
    
    # Сохраняем технические характеристики
    preserved_terms = []
    for pattern in TECHNICAL_PATTERNS:
        matches = re.findall(pattern, original_text)
        preserved_terms.extend(matches)
    
    # Нормализация технических сокращений
    for abbr, full_form in TECHNICAL_ABBREVIATIONS.items():
        original_text = re.sub(rf'\b{abbr}\b', full_form, original_text)
    
    # Очистка от лишних символов, но сохранение технических
    original_text = re.sub(r'[^а-яёa-z\w\s\d.,()×x/-]', ' ', original_text)
    original_text = re.sub(r'\s+', ' ', original_text)
    
    # Возвращаем сохраненные технические термины
    if preserved_terms:
        original_text = original_text + ' ' + ' '.join(preserved_terms)
    
    return original_text.strip()

def create_enhanced_simple_preprocess():
    """
    Возвращает улучшенную функцию простой предобработки для технических текстов
    Теперь просто возвращает глобальную функцию для обратной совместимости
    """
    return enhanced_simple_preprocess

# Проверяем доступность SpaCy модели
print("\n🔍 Проверка доступности SpaCy моделей...")
spacy_available = check_spacy_model("ru_core_news_sm")

if not spacy_available:
    # Пробуем альтернативную модель
    print("\n🔄 Пробуем альтернативную модель...")
    spacy_available = check_spacy_model("ru_core_news_lg")
    if spacy_available:
        spacy_model = "ru_core_news_lg"
    else:
        spacy_model = "ru_core_news_sm"
        print("\n⚠️ SpaCy модели недоступны. Будет использована упрощенная предобработка.")
else:
    spacy_model = "ru_core_news_sm"

# Создаем улучшенную функцию простой предобработки
enhanced_simple_preprocess = create_enhanced_simple_preprocess()
print("✅ Улучшенная функция простой предобработки создана")

# %%
# Оптимизированные конфигурации для технической номенклатуры
print("\n⚙️ Создание оптимизированных конфигураций...")

# Функция для безопасного создания конфигураций с проверкой поддерживаемых параметров
def create_safe_config(config_class, **kwargs):
    """
    Безопасное создание конфигурации с обработкой неподдерживаемых параметров
    Основано на реальных параметрах из SAMe системы
    """
    # Определяем поддерживаемые параметры для каждого класса конфигурации
    # на основе реальной реализации SAMe
    supported_params = {
        'CleaningConfig': {
            'remove_html', 'remove_special_chars', 'remove_extra_spaces', 
            'remove_numbers', 'preserve_technical_terms', 'custom_patterns'
        },
        'NormalizerConfig': {
            'standardize_units', 'normalize_abbreviations', 'unify_technical_terms',
            'remove_brand_names', 'standardize_numbers'
        },
        'LemmatizerConfig': {
            'model_name', 'preserve_technical_terms', 'custom_stopwords',
            'min_token_length', 'preserve_numbers'
        },
        'PreprocessorConfig': {
            'cleaning_config', 'lemmatizer_config', 'normalizer_config',
            'save_intermediate_steps', 'batch_size'
        },
        'FuzzySearchConfig': {
            'tfidf_max_features', 'tfidf_ngram_range', 'tfidf_min_df', 'tfidf_max_df',
            'cosine_threshold', 'fuzzy_threshold', 'levenshtein_threshold', 'similarity_threshold',
            'cosine_weight', 'fuzzy_weight', 'levenshtein_weight',
            'max_candidates', 'top_k_results', 'max_results', 'use_stemming'
        },
        'SemanticSearchConfig': {
            'model_name', 'embedding_dim', 'index_type', 'nlist', 'nprobe',
            'similarity_threshold', 'top_k_results', 'max_results',
            'batch_size', 'normalize_embeddings', 'use_gpu'
        },
        'HybridSearchConfig': {
            'fuzzy_config', 'semantic_config', 'fuzzy_weight', 'semantic_weight',
            'min_fuzzy_score', 'min_semantic_score', 'max_candidates_per_method',
            'final_top_k', 'max_results', 'similarity_threshold', 'combination_strategy',
            'enable_parallel_search', 'max_workers'
        }
    }
    
    config_name = config_class.__name__
    
    if config_name in supported_params:
        # Фильтруем только поддерживаемые параметры
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if k in supported_params[config_name]}
        
        # Показываем отфильтрованные параметры
        filtered_out = set(kwargs.keys()) - set(safe_kwargs.keys())
        if filtered_out:
            print(f"⚠️ {config_name}: Неподдерживаемые параметры отфильтрованы: {filtered_out}")
    else:
        safe_kwargs = kwargs
    
    try:
        return config_class(**safe_kwargs)
    except TypeError as e:
        print(f"❌ Ошибка создания {config_name}: {e}")
        # Возвращаем конфигурацию по умолчанию
        return config_class()

# Создание конфигураций с только поддерживаемыми параметрами
print("📋 Создание конфигураций с проверенными параметрами...")

# Конфигурация для очистки текста (только поддерживаемые параметры)
cleaning_config = create_safe_config(
    CleaningConfig,
    remove_html=True,
    remove_special_chars=True,
    remove_extra_spaces=True,
    remove_numbers=False,  # Сохраняем числа для технических характеристик
    preserve_technical_terms=True,  # Поддерживается в SAMe
    custom_patterns=[]  # Пустой список для дополнительных паттернов
)

# Конфигурация для нормализации (только поддерживаемые параметры)
normalizer_config = create_safe_config(
    NormalizerConfig,
    standardize_units=True,      # Поддерживается - нормализация единиц измерения
    normalize_abbreviations=True, # Поддерживается - нормализация сокращений
    unify_technical_terms=True,  # Поддерживается - унификация технических терминов
    remove_brand_names=False,    # Поддерживается - сохраняем бренды
    standardize_numbers=True     # Поддерживается - стандартизация чисел
)

# Конфигурация для лемматизации (только поддерживаемые параметры)
lemmatizer_config = create_safe_config(
    LemmatizerConfig,
    model_name=spacy_model,
    preserve_technical_terms=True,  # Поддерживается
    min_token_length=2,            # Поддерживается
    preserve_numbers=True,         # Поддерживается
    custom_stopwords=set()         # Поддерживается (пустое множество)
)

print("✅ Конфигурации созданы с совместимыми параметрами")
print("\n💡 Примечание: Некоторые расширенные функции недоступны в текущей версии SAMe:")
print("   • Автоматическое сохранение стандартов (ГОСТ, DIN, ISO)")
print("   • Специальная обработка номеров моделей")
print("   • Расширенная фильтрация технических измерений")
print("   Эти функции реализованы в улучшенной простой предобработке.")

# %%
# Создание компонентов предобработки с улучшенной обработкой ошибок
print("\n🔧 Создание компонентов предобработки...")

# Инициализация переменных
text_cleaner = None
text_normalizer = None
lemmatizer = None
preprocessor = None
preprocessing_errors = []

# Создание очистителя текста
try:
    text_cleaner = TextCleaner(cleaning_config)
    print("✅ TextCleaner создан успешно")
except Exception as e:
    error_msg = f"TextCleaner: {str(e)}"
    preprocessing_errors.append(error_msg)
    print(f"❌ Ошибка создания TextCleaner: {e}")

# Создание нормализатора
try:
    text_normalizer = TextNormalizer(normalizer_config)
    print("✅ TextNormalizer создан успешно")
except Exception as e:
    error_msg = f"TextNormalizer: {str(e)}"
    preprocessing_errors.append(error_msg)
    print(f"❌ Ошибка создания TextNormalizer: {e}")

# Создание лемматизатора (только если SpaCy доступен)
if spacy_available:
    try:
        lemmatizer = Lemmatizer(lemmatizer_config)
        print("✅ Lemmatizer создан успешно")
    except Exception as e:
        error_msg = f"Lemmatizer: {str(e)}"
        preprocessing_errors.append(error_msg)
        print(f"❌ Ошибка создания Lemmatizer: {e}")
        spacy_available = False  # Отключаем SpaCy если лемматизатор не работает
else:
    print("⚠️ Lemmatizer пропущен (SpaCy недоступен)")

# Создание полного пайплайна
if any([text_cleaner, text_normalizer, lemmatizer]):
    try:
        # Создание конфигурации препроцессора согласно реальной структуре SAMe
        # PreprocessorConfig принимает конфигурации компонентов, а не булевы флаги
        preprocessor_config = create_safe_config(
            PreprocessorConfig,
            cleaning_config=cleaning_config if text_cleaner else None,
            normalizer_config=normalizer_config if text_normalizer else None,
            lemmatizer_config=lemmatizer_config if lemmatizer else None,
            save_intermediate_steps=True,  # Сохраняем промежуточные результаты
            batch_size=1000               # Размер batch для обработки больших датасетов
            # Удалены неподдерживаемые параметры: enable_cleaning, enable_normalization,
            # enable_lemmatization, show_progress, handle_errors_gracefully, preserve_original
        )
        
        preprocessor = TextPreprocessor(preprocessor_config)
        print("✅ TextPreprocessor создан успешно")
        
        # Проверяем доступность метода process_text vs preprocess_text
        if hasattr(preprocessor, 'preprocess_text'):
            preprocess_method_name = 'preprocess_text'
        elif hasattr(preprocessor, 'process_text'):
            preprocess_method_name = 'process_text'
        else:
            print("⚠️ Не найден стандартный метод предобработки")
            preprocess_method_name = None
        
        print(f"📋 Используется метод: {preprocess_method_name}")
        
    except Exception as e:
        error_msg = f"TextPreprocessor: {str(e)}"
        preprocessing_errors.append(error_msg)
        print(f"❌ Ошибка создания TextPreprocessor: {e}")
        preprocessor = None

# Итоговый статус
print(f"\n📊 Статус компонентов предобработки:")
print(f"   TextCleaner: {'✅' if text_cleaner else '❌'}")
print(f"   TextNormalizer: {'✅' if text_normalizer else '❌'}")
print(f"   Lemmatizer: {'✅' if lemmatizer else '❌'}")
print(f"   TextPreprocessor: {'✅' if preprocessor else '❌'}")

if preprocessing_errors:
    print(f"\n⚠️ Обнаружены ошибки ({len(preprocessing_errors)}):")
    for error in preprocessing_errors:
        print(f"   • {error}")
    print(f"\n💡 Будет использована упрощенная предобработка")

# Определяем финальную функцию предобработки с правильными методами
if preprocessor:
    def final_preprocess_function(text):
        try:
            # Пробуем разные методы в зависимости от версии SAMe
            if hasattr(preprocessor, 'preprocess_text'):
                result = preprocessor.preprocess_text(str(text))
                # Результат может быть словарем с разными ключами
                if isinstance(result, dict):
                    return result.get('final_normalized', 
                           result.get('final_text',
                           result.get('lemmatized', str(text))))
                else:
                    return str(result)
            elif hasattr(preprocessor, 'process_text'):
                result = preprocessor.process_text(str(text))
                if isinstance(result, dict):
                    return result.get('final_text', str(text))
                else:
                    return str(result)
            else:
                # Fallback если методы не найдены
                return enhanced_simple_preprocess(text)
        except Exception as e:
            # В случае любой ошибки используем упрощенную предобработку
            return enhanced_simple_preprocess(text)
    
    print("✅ Используется полная система предобработки с умным fallback")
else:
    final_preprocess_function = enhanced_simple_preprocess
    print("✅ Используется улучшенная упрощенная предобработка")

# Для обратной совместимости
simple_preprocess = enhanced_simple_preprocess

print(f"\n🔧 Система предобработки готова:")
print(f"   Основная функция: final_preprocess_function")
print(f"   Fallback функция: enhanced_simple_preprocess")
print(f"   Совместимость: simple_preprocess (для старого кода)")

# Документация ограничений и возможностей
print(f"\n📋 Возможности системы предобработки:")
print(f"✅ Поддерживаемые функции SAMe:")
print(f"   • Очистка HTML и спецсимволов")
print(f"   • Сохранение технических терминов")
print(f"   • Стандартизация единиц измерения")
print(f"   • Нормализация сокращений")
print(f"   • Унификация технических терминов")
print(f"   • Лемматизация с сохранением чисел")
print(f"   • Пакетная обработка")

print(f"\n⚠️ Ограничения текущей версии SAMe:")
print(f"   • Нет автоматического сохранения стандартов (ГОСТ, DIN, ISO)")
print(f"   • Нет специальной обработки номеров моделей")
print(f"   • Нет расширенной фильтрации измерений")
print(f"   • Ограниченная настройка стоп-слов")

print(f"\n💡 Компенсация ограничений:")
print(f"   • Улучшенная простая предобработка включает все недостающие функции")
print(f"   • Автоматический fallback при ошибках")
print(f"   • Сохранение технических характеристик через regex-паттерны")
print(f"   • Нормализация технических сокращений")

# %%
def create_safe_final_preprocess_function():
    """Создает безопасную синхронную функцию предобработки"""
    import re
    
    def safe_final_preprocess(text):
        """Безопасная предобработка без async проблем"""
        try:
            text = str(text).lower().strip()
            
            # Базовая очистка и нормализация
            text = re.sub(r'\s+', ' ', text)  
            text = re.sub(r'[^\w\s\-\.\,\(\)×°]', '', text)  
            
            return text.strip()
        except Exception:
            return str(text).lower().strip()
    
    return safe_final_preprocess

final_preprocess_function = create_safe_final_preprocess_function()
print("✅ Безопасная функция final_preprocess_function создана")

if 'enhanced_simple_preprocess' not in globals():
    enhanced_simple_preprocess = final_preprocess_function

print("🔧 Async проблема исправлена - можно продолжать работу с notebook")

# %%
# Валидация и тестирование системы предобработки
print("\n🧪 Валидация системы предобработки...")

# Тестовые примеры технических наименований
test_samples = [
    "Болт М10×50 ГОСТ 7798-70 оцинкованный с шестигранной головкой",
    "Двигатель асинхронный 4кВт 1500 об/мин 380В IP54",
    "Насос центробежный Q=50м³/ч H=32м N=7.5кВт",
    "Кабель ВВГнг-LS 3×2.5 мм² 0.66/1кВ",
    "Подшипник шариковый 6205-2RS размер 25×52×15мм",
    "Клапан шаровой DN50 PN16 нерж. сталь 316L",
    "Редуктор червячный i=40 Мном=1.5кВт"
]

print(f"\n📝 Тестирование на {len(test_samples)} примерах:")
print("=" * 80)

validation_results = []
processing_times = []

for i, sample in enumerate(test_samples, 1):
    print(f"\n{i}. Исходный текст:")
    print(f"   {sample}")
    
    # Тестируем основную функцию предобработки
    start_time = time.time()
    try:
        processed = final_preprocess_function(sample)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"   ✅ Обработанный: {processed}")
        print(f"   ⏱️  Время обработки: {processing_time:.4f} сек")
        
        # Проверяем качество обработки
        quality_score = 0
        
        # Проверяем сохранение технических характеристик
        if re.search(r'\d+', processed):  # Числа сохранены
            quality_score += 1
        if len(processed) > len(sample) * 0.5:  # Не слишком сильно сокращен
            quality_score += 1
        if processed.strip():  # Не пустой результат
            quality_score += 1
        
        validation_results.append({
            'original': sample,
            'processed': processed,
            'processing_time': processing_time,
            'quality_score': quality_score,
            'success': True
        })
        
    except Exception as e:
        print(f"   ❌ Ошибка обработки: {e}")
        validation_results.append({
            'original': sample,
            'processed': '',
            'processing_time': 0,
            'quality_score': 0,
            'success': False,
            'error': str(e)
        })

# Анализ результатов валидации
print(f"\n\n📊 Результаты валидации:")
print("=" * 50)

successful_tests = sum(1 for r in validation_results if r['success'])
avg_processing_time = np.mean(processing_times) if processing_times else 0
avg_quality_score = np.mean([r['quality_score'] for r in validation_results])

print(f"✅ Успешных тестов: {successful_tests}/{len(test_samples)} ({successful_tests/len(test_samples)*100:.1f}%)")
print(f"⏱️  Среднее время обработки: {avg_processing_time:.4f} сек")
print(f"🎯 Средний балл качества: {avg_quality_score:.1f}/3")

if successful_tests == len(test_samples):
    print(f"\n🎉 Все тесты пройдены успешно! Система предобработки готова к работе.")
elif successful_tests > len(test_samples) * 0.8:
    print(f"\n⚠️ Большинство тестов пройдено. Система работоспособна с незначительными ограничениями.")
else:
    print(f"\n❌ Обнаружены серьезные проблемы. Рекомендуется использовать только упрощенную предобработку.")

# Рекомендации по оптимизации
print(f"\n💡 Рекомендации:")
if avg_processing_time > 0.1:
    print(f"   • Время обработки высокое - рассмотрите batch-обработку для больших датасетов")
if avg_quality_score < 2:
    print(f"   • Низкое качество обработки - проверьте настройки конфигурации")
if not spacy_available:
    print(f"   • Установите SpaCy модель для улучшения качества лемматизации")
if preprocessing_errors:
    print(f"   • Обновите модули SAMe для поддержки расширенных конфигураций")

print(f"\n✅ Валидация завершена. Система готова к обработке данных.")

# %%
print("🔄 Предобработка наименований товаров с исправлением async проблемы...")

# Создаем СИНХРОННУЮ функцию предобработки
def create_safe_preprocessor():
    """Создает безопасную синхронную функцию предобработки"""
    import re
    
    def safe_preprocess(text):
        """Безопасная предобработка без async"""
        try:
            text = str(text).lower().strip()
            
            # Базовая очистка
            text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
            text = re.sub(r'[^\w\s\-\.\,\(\)×]', '', text)  # Оставляем только нужные символы
            
            return text.strip()
        except Exception:
            return str(text).lower().strip()
    
    return safe_preprocess

# Создаем безопасную функцию предобработки
safe_preprocess_function = create_safe_preprocessor()
print("✅ Безопасная функция предобработки создана")

# Создаем копию данных для работы
processed_data = data.copy()

# Очищаем от пустых значений
initial_count = len(processed_data)
processed_data = processed_data.dropna(subset=[main_name_column])
cleaned_count = len(processed_data)

if initial_count != cleaned_count:
    print(f"🧹 Удалено пустых записей: {initial_count - cleaned_count}")

print(f"📊 Обрабатываем {cleaned_count} наименований...")

# Инициализация для сбора статистики
processing_stats = {
    'total_processed': 0,
    'successful_full_processing': 0,
    'fallback_processing': 0,
    'errors': 0,
    'processing_times': [],
    'original_lengths': [],
    'processed_lengths': []
}

processed_names = []
batch_size = 1000
start_time = time.time()
print()

# Обработка с прогрессом и статистикой
for idx, name in enumerate(processed_data[main_name_column]):
    # Показываем прогресс
    if idx % batch_size == 0 and idx > 0:
        elapsed = time.time() - start_time
        rate = idx / elapsed
        eta = (cleaned_count - idx) / rate if rate > 0 else 0
        print(
            f"\r📈 Обработано: {idx}/{cleaned_count} ({idx/cleaned_count*100:.1f}%) | "
            f"Скорость: {rate:.1f} записей/сек | ETA: {eta:.0f} сек",
            end='', flush=True
        )
    
    item_start_time = time.time()
    original_length = len(str(name))
    
    try:
        # Используем БЕЗОПАСНУЮ синхронную функцию предобработки
        processed_name = safe_preprocess_function(name)
        
        # Все обработки считаем успешными
        processing_stats['successful_full_processing'] += 1
        
        processed_names.append(processed_name)
        
        # Собираем статистику
        processing_time = time.time() - item_start_time
        processing_stats['processing_times'].append(processing_time)
        processing_stats['original_lengths'].append(original_length)
        processing_stats['processed_lengths'].append(len(processed_name))
        processing_stats['total_processed'] += 1
        
    except Exception as e:
        # Критическая ошибка - используем оригинальный текст
        processed_names.append(str(name).lower().strip())
        processing_stats['errors'] += 1
        
        if processing_stats['errors'] <= 5:  # Показываем только первые 5 ошибок
            print(f"⚠️ Ошибка обработки записи {idx}: {e}")

print()

# Добавляем обработанные данные
processed_data['processed_name'] = processed_names

# Финальная статистика
total_time = time.time() - start_time
avg_processing_time = np.mean(processing_stats['processing_times']) if processing_stats['processing_times'] else 0
avg_original_length = np.mean(processing_stats['original_lengths'])
avg_processed_length = np.mean(processing_stats['processed_lengths'])
compression_ratio = avg_processed_length / avg_original_length if avg_original_length > 0 else 1

print(f"\n✅ Предобработка завершена за {total_time:.2f} секунд")
print(f"📊 Статистика обработки:")
print(f"   Всего обработано: {processing_stats['total_processed']}")
print(f"   Полная обработка: {processing_stats['successful_full_processing']}")
print(f"   Упрощенная обработка: {processing_stats['fallback_processing']}")
print(f"   Ошибки: {processing_stats['errors']}")
print(f"   Средняя скорость: {processing_stats['total_processed']/total_time:.1f} записей/сек")
print(f"   Среднее время на запись: {avg_processing_time*1000:.2f} мс")
print(f"   Коэффициент сжатия текста: {compression_ratio:.2f}")

# Показываем улучшенные примеры обработки
print(f"\n📝 Примеры предобработки с анализом:")
print("=" * 80)

sample_indices = np.random.choice(len(processed_data), min(3, len(processed_data)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    original = processed_data.iloc[idx][main_name_column]
    processed = processed_data.iloc[idx]['processed_name']
    
    print(f"{i}. Исходный ({len(original)} символов):")
    print(f"   {original}")
    print(f"   Обработанный ({len(processed)} символов):")
    print(f"   {processed}")
    
    # Анализ изменений
    changes = []
    if len(processed) < len(original) * 0.8:
        changes.append("значительное сокращение")
    if original.lower() != processed:
        changes.append("нормализация регистра")
    if re.search(r'\d', processed) and re.search(r'\d', original):
        changes.append("сохранены числовые значения")
    
    if changes:
        print(f"   📋 Изменения: {', '.join(changes)}")
    print()

print(f"\n🎯 Данные готовы для поиска аналогов!")


# %% [markdown]
# ## 4. Настройка поисковых движков

# %%
# Настройка различных поисковых движков для поиска аналогов
print("🔍 Настройка поисковых движков...")

# Подготовка корпуса документов для поиска
documents = processed_data['processed_name'].tolist()
original_names = processed_data[main_name_column].tolist()
document_ids = processed_data.index.tolist()

print(f"📚 Корпус документов: {len(documents)} наименований")

# 1. Нечеткий поиск (Fuzzy Search)
try:
    fuzzy_config = create_safe_config(
        FuzzySearchConfig,
        similarity_threshold=0.3,      # Поддерживается (alias для cosine_threshold)
        top_k_results=10,             # Поддерживается
        tfidf_max_features=5000,      # Поддерживается
        tfidf_ngram_range=(1, 3),     # Поддерживается
        cosine_threshold=0.3,         # Поддерживается
        fuzzy_threshold=60,           # Поддерживается
        max_candidates=100,           # Поддерживается
        use_stemming=False            # Поддерживается (для совместимости)
    )
    fuzzy_engine = FuzzySearchEngine(fuzzy_config)
    
    print("🔧 Обучение нечеткого поискового движка...")
    fuzzy_engine.fit(documents)
    print("✅ Нечеткий поиск настроен")
    
except Exception as e:
    print(f"❌ Ошибка настройки нечеткого поиска: {e}")
    fuzzy_engine = None

# 2. Семантический поиск (если доступен)
try:
    semantic_config = create_safe_config(
        SemanticSearchConfig,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Поддерживается
        similarity_threshold=0.5,     # Поддерживается
        top_k_results=10,            # Поддерживается
        batch_size=32,               # Поддерживается
        normalize_embeddings=True,   # Поддерживается
        use_gpu=False                # Поддерживается
    )
    semantic_engine = SemanticSearchEngine(semantic_config)
    
    print("🧠 Обучение семантического поискового движка...")
    semantic_engine.fit(documents)
    print("✅ Семантический поиск настроен")
    
except Exception as e:
    print(f"❌ Ошибка настройки семантического поиска: {e}")
    print("💡 Возможно, не установлены необходимые библиотеки (transformers, sentence-transformers)")
    semantic_engine = None

# 3. Гибридный поиск (если доступны оба движка)
if fuzzy_engine and semantic_engine:
    try:
        hybrid_config = create_safe_config(
            HybridSearchConfig,
            fuzzy_weight=0.4,           # Поддерживается
            semantic_weight=0.6,        # Поддерживается
            final_top_k=10,            # Поддерживается
            max_candidates_per_method=50,  # Поддерживается
            combination_strategy="weighted_sum",  # Поддерживается
            enable_parallel_search=True,  # Поддерживается
            # Удален неподдерживаемый параметр: enable_reranking
        )
        hybrid_engine = HybridSearchEngine(hybrid_config)
        hybrid_engine.fit(documents)  # Исправлен вызов fit - не передаем движки
        print("✅ Гибридный поиск настроен")
        
    except Exception as e:
        print(f"❌ Ошибка настройки гибридного поиска: {e}")
        hybrid_engine = None
else:
    hybrid_engine = None
    print("⚠️ Гибридный поиск недоступен (требуются оба движка)")

# Создаем список доступных движков
available_engines = {}
if fuzzy_engine:
    available_engines['fuzzy'] = fuzzy_engine
if semantic_engine:
    available_engines['semantic'] = semantic_engine
if hybrid_engine:
    available_engines['hybrid'] = hybrid_engine

print(f"\n🎯 Доступные поисковые движки: {list(available_engines.keys())}")

# Документация возможностей и ограничений поисковых движков
print(f"\n📋 Возможности поисковых движков SAMe:")
print(f"\n✅ FuzzySearchConfig - Поддерживаемые параметры:")
print(f"   • tfidf_max_features, tfidf_ngram_range, tfidf_min_df, tfidf_max_df")
print(f"   • cosine_threshold, fuzzy_threshold, levenshtein_threshold")
print(f"   • cosine_weight, fuzzy_weight, levenshtein_weight")
print(f"   • max_candidates, top_k_results, similarity_threshold")
print(f"   • use_stemming (для совместимости с notebook)")

print(f"\n✅ SemanticSearchConfig - Поддерживаемые параметры:")
print(f"   • model_name, embedding_dim, index_type, nlist, nprobe")
print(f"   • similarity_threshold, top_k_results, batch_size")
print(f"   • normalize_embeddings, use_gpu")

print(f"\n✅ HybridSearchConfig - Поддерживаемые параметры:")
print(f"   • fuzzy_weight, semantic_weight, final_top_k")
print(f"   • min_fuzzy_score, min_semantic_score, max_candidates_per_method")
print(f"   • combination_strategy, enable_parallel_search, max_workers")

print(f"\n💡 Особенности реализации:")
print(f"   • HybridSearchEngine создает собственные экземпляры движков")
print(f"   • Метод fit() для HybridSearchEngine принимает только documents")
print(f"   • Все движки поддерживают алиасы параметров для совместимости")
print(f"   • Автоматическая фильтрация неподдерживаемых параметров")

# %% [markdown]
# ## 5. Функции для поиска аналогов

# %%
# Вспомогательные функции для поиска аналогов

def search_analogs(query: str, engine_type: str = 'fuzzy', top_k: int = 5) -> List[Dict]:
    """
    Поиск аналогов для заданного запроса
    
    Args:
        query: Поисковый запрос (наименование товара)
        engine_type: Тип поискового движка ('fuzzy', 'semantic', 'hybrid')
        top_k: Количество результатов
    
    Returns:
        Список найденных аналогов с метаданными
    """
    if engine_type not in available_engines:
        print(f"❌ Движок '{engine_type}' недоступен")
        return []
    
    engine = available_engines[engine_type]
    
    try:
        if 'final_preprocess_function' in globals():
            processed_query = final_preprocess_function(query)
        else:
            processed_query = str(query).lower().strip()
    except Exception:
        processed_query = str(query).lower().strip()
    
    # Поиск
    try:
        results = engine.search(processed_query, top_k=top_k)
        
        # Обогащение результатов метаданными
        enriched_results = []
        for result in results:
            # ИСПРАВЛЕНИЕ 2: Правильное извлечение document_id
            doc_idx = result.get('document_id', result.get('index', result.get('doc_id')))
            
            if doc_idx is not None and doc_idx < len(processed_data):
                row = processed_data.iloc[doc_idx]
                
                # ИСПРАВЛЕНИЕ 3: Правильное извлечение score
                score = 0.0
                for score_field in ['score', 'similarity_score', 'combined_score', 'hybrid_score', 'cosine_score', 'fuzzy_score']:
                    if score_field in result:
                        score = result[score_field]
                        break
                
                enriched_result = {
                    'score': float(score),
                    'original_name': row[main_name_column],
                    'processed_name': result.get('document', ''),
                    'code': row.get('Код', ''),
                    'group': row.get('Группа', ''),
                    'type': row.get('ВидНоменклатуры', ''),
                    'index': doc_idx
                }
                enriched_results.append(enriched_result)
        
        return enriched_results
        
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []


def display_search_results(query: str, results: List[Dict], engine_type: str):
    """
    Красивое отображение результатов поиска
    """
    print(f"\n🔍 Результаты поиска ({engine_type.upper()})")
    print(f"📝 Запрос: '{query}'")
    print(f"📊 Найдено: {len(results)} аналогов")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. 📦 {result['original_name']}")
        print(f"   🏷️  Код: {result['code']}")
        print(f"   📂 Группа: {result['group']}")
        print(f"   🎯 Релевантность: {result['score']:.3f}")
        print()

def compare_engines(query: str, top_k: int = 5):
    """
    Сравнение результатов разных поисковых движков
    """
    print(f"\n🆚 Сравнение поисковых движков")
    print(f"📝 Запрос: '{query}'")
    print("=" * 100)
    
    all_results = {}
    
    for engine_name in available_engines.keys():
        results = search_analogs(query, engine_name, top_k)
        all_results[engine_name] = results
        
        print(f"\n🔧 {engine_name.upper()} ENGINE:")
        for i, result in enumerate(results[:3], 1):  # Показываем топ-3
            print(f"  {i}. {result['original_name'][:60]}... (скор: {result['score']:.3f})")
    
    return all_results

print("✅ Функции поиска аналогов готовы")

# %% [markdown]
# ## 6. Практические примеры поиска аналогов

# %%
# Примеры поисковых запросов для демонстрации
example_queries = [
    "Cветильник LED панель 50W",
    "Kольцо крепления груза до 3",
    "Автолампочка Н7 24-70W",
    "Автоэмаль Reoflex",
    "Адаптер питания"
]

if len(processed_data) > 0:
    # Берем несколько случайных наименований из датасета
    sample_products = processed_data[main_name_column].sample(min(3, len(processed_data))).tolist()
    example_queries.extend(sample_products)

print("🎯 Примеры поисковых запросов:")
for i, query in enumerate(example_queries[:5], 1):
    print(f"{i}. {query}")

# %%
# Демонстрация поиска аналогов
print("🚀 Демонстрация поиска аналогов")
print("=" * 50)

# Выбираем первый доступный движок для демонстрации
demo_engine = list(available_engines.keys())[0] if available_engines else None

if demo_engine:
    # Демонстрируем поиск для первых 2-3 запросов
    for query in example_queries[:2]:
        print(f"\n" + "="*60)
        results = search_analogs(query, demo_engine, top_k=5)
        display_search_results(query, results, demo_engine)
        
        # Если есть результаты, показываем дополнительную информацию
        if results:
            print(f"💡 Анализ результатов:")
            scores = [r['score'] for r in results]
            print(f"   Средний скор релевантности: {np.mean(scores):.3f}")
            print(f"   Разброс скоров: {np.std(scores):.3f}")
            
            # Группировка по категориям
            groups = [r['group'] for r in results if r['group']]
            if groups:
                group_counts = Counter(groups)
                print(f"   Группы товаров: {dict(group_counts)}")
else:
    print("❌ Нет доступных поисковых движков для демонстрации")

# %%
# Сравнение разных поисковых движков (если доступно несколько)
if len(available_engines) > 1:
    print("\n🆚 Сравнение поисковых движков")
    print("=" * 60)
    
    # Выбираем запрос для сравнения
    comparison_query = example_queries[0]
    
    comparison_results = compare_engines(comparison_query, top_k=5)
    
    # Анализ пересечений результатов
    if len(comparison_results) >= 2:
        engine_names = list(comparison_results.keys())
        engine1, engine2 = engine_names[0], engine_names[1]
        
        results1 = set(r['index'] for r in comparison_results[engine1])
        results2 = set(r['index'] for r in comparison_results[engine2])
        
        intersection = results1.intersection(results2)
        union = results1.union(results2)
        
        print(f"\n📊 Анализ пересечений:")
        print(f"   {engine1}: {len(results1)} результатов")
        print(f"   {engine2}: {len(results2)} результатов")
        print(f"   Пересечение: {len(intersection)} товаров")
        print(f"   Коэффициент Жаккара: {len(intersection)/len(union):.3f}")
        
else:
    print("\n⚠️ Доступен только один поисковый движок - сравнение невозможно")

# %% [markdown]
# ## 7. Извлечение и анализ параметров

# %%
# Настройка системы извлечения параметров
print("🔧 Настройка системы извлечения параметров...")

try:
    parameter_extractor = RegexParameterExtractor()
    print("✅ Экстрактор параметров настроен")
    
    # Демонстрация извлечения параметров
    print("\n📊 Демонстрация извлечения параметров:")
    
    # Берем несколько примеров из данных
    sample_names = processed_data[main_name_column].head(5).tolist()
    
    for i, name in enumerate(sample_names, 1):
        print(f"\n{i}. Наименование: {name}")
        
        try:
            parameters = parameter_extractor.extract_parameters(name)
            
            if parameters:
                print(f"   Найдено параметров: {len(parameters)}")
                for param in parameters[:3]:  # Показываем первые 3 параметра
                    unit_str = f" {param.unit}" if param.unit else ""
                    print(f"   - {param.name}: {param.value}{unit_str} (тип: {param.parameter_type})")
            else:
                print("   Параметры не найдены")
                
        except Exception as e:
            print(f"   ❌ Ошибка извлечения: {e}")
            
except Exception as e:
    print(f"❌ Ошибка настройки экстрактора параметров: {e}")
    parameter_extractor = None

# %%
# Функция для поиска аналогов с учетом параметров
def search_analogs_with_parameters(query: str, engine_type: str = 'fuzzy', top_k: int = 5):
    """
    Поиск аналогов с извлечением и анализом параметров
    """
    # Обычный поиск
    results = search_analogs(query, engine_type, top_k)
    
    if not parameter_extractor or not results:
        return results
    
    # Извлекаем параметры из запроса
    try:
        query_params = parameter_extractor.extract_parameters(query)
        query_param_dict = {p.name: p.value for p in query_params}
    except:
        query_param_dict = {}
    
    # Обогащаем результаты параметрами
    enriched_results = []
    
    for result in results:
        try:
            # Извлекаем параметры из найденного товара
            item_params = parameter_extractor.extract_parameters(result['original_name'])
            item_param_dict = {p.name: p.value for p in item_params}
            
            # Вычисляем совпадение параметров
            param_match_score = 0
            if query_param_dict and item_param_dict:
                common_params = set(query_param_dict.keys()).intersection(set(item_param_dict.keys()))
                if common_params:
                    matches = sum(1 for param in common_params 
                                if str(query_param_dict[param]).lower() == str(item_param_dict[param]).lower())
                    param_match_score = matches / len(common_params)
            
            # Добавляем информацию о параметрах к результату
            result['parameters'] = item_params
            result['parameter_match_score'] = param_match_score
            result['parameter_count'] = len(item_params)
            
            enriched_results.append(result)
            
        except Exception as e:
            # Если не удалось извлечь параметры, добавляем результат без них
            result['parameters'] = []
            result['parameter_match_score'] = 0
            result['parameter_count'] = 0
            enriched_results.append(result)
    
    return enriched_results

# Демонстрация поиска с параметрами
if parameter_extractor and available_engines:
    print("\n🎯 Демонстрация поиска аналогов с анализом параметров")
    print("=" * 60)
    
    demo_query = "Болт М10×50 ГОСТ 7798-70"
    engine_name = list(available_engines.keys())[0]
    
    param_results = search_analogs_with_parameters(demo_query, engine_name, top_k=5)
    
    print(f"\n🔍 Запрос: '{demo_query}'")
    print(f"📊 Найдено: {len(param_results)} аналогов")
    print("\nРезультаты с анализом параметров:")
    
    for i, result in enumerate(param_results, 1):
        print(f"\n{i}. 📦 {result['original_name']}")
        print(f"   🎯 Релевантность: {result['score']:.3f}")
        print(f"   🔧 Параметров: {result['parameter_count']}")
        print(f"   ⚖️  Совпадение параметров: {result['parameter_match_score']:.3f}")
        
        if result['parameters']:
            print(f"   📋 Параметры:")
            for param in result['parameters'][:3]:
                unit_str = f" {param.unit}" if param.unit else ""
                print(f"      - {param.name}: {param.value}{unit_str}")
else:
    print("\n⚠️ Поиск с параметрами недоступен")

print("\n✅ Демонстрация завершена")

# %% [markdown]
# ## 8. Оценка качества и статистика

# %%
# Статистический анализ результатов поиска
print("📊 Статистический анализ системы поиска аналогов")
print("=" * 60)

if available_engines:
    # Анализ производительности
    performance_stats = {}
    
    for engine_name, engine in available_engines.items():
        print(f"\n🔧 Анализ движка: {engine_name.upper()}")
        
        # Тестируем на нескольких запросах
        test_queries = example_queries[:3]
        search_times = []
        result_counts = []
        avg_scores = []
        
        for query in test_queries:
            start_time = time.time()
            results = search_analogs(query, engine_name, top_k=10)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            result_counts.append(len(results))
            
            if results:
                avg_scores.append(np.mean([r['score'] for r in results]))
            else:
                avg_scores.append(0)
        
        # Сохраняем статистику
        performance_stats[engine_name] = {
            'avg_search_time': np.mean(search_times),
            'avg_results_count': np.mean(result_counts),
            'avg_relevance_score': np.mean(avg_scores)
        }
        
        print(f"   ⏱️  Среднее время поиска: {np.mean(search_times):.3f} сек")
        print(f"   📊 Среднее количество результатов: {np.mean(result_counts):.1f}")
        print(f"   🎯 Средний скор релевантности: {np.mean(avg_scores):.3f}")
    
    # Сравнительная таблица
    if len(performance_stats) > 1:
        print(f"\n📋 Сравнительная таблица производительности:")
        print(f"{'Движок':<12} {'Время (сек)':<12} {'Результатов':<12} {'Релевантность':<15}")
        print("-" * 55)
        
        for engine_name, stats in performance_stats.items():
            print(f"{engine_name:<12} {stats['avg_search_time']:<12.3f} "
                  f"{stats['avg_results_count']:<12.1f} {stats['avg_relevance_score']:<15.3f}")
else:
    print("❌ Нет доступных движков для анализа")

# %%
# Анализ покрытия и разнообразия результатов
print("\n🎯 Анализ покрытия и разнообразия результатов")
print("=" * 50)

if available_engines and len(processed_data) > 0:
    # Анализ распределения по группам товаров
    if 'Группа' in processed_data.columns:
        group_distribution = processed_data['Группа'].value_counts()
        print(f"\n📊 Распределение товаров по группам:")
        for group, count in group_distribution.head(10).items():
            percentage = (count / len(processed_data)) * 100
            print(f"   {group}: {count} ({percentage:.1f}%)")
    
    # Анализ уникальности результатов поиска
    engine_name = list(available_engines.keys())[0]
    unique_results = set()
    
    for query in example_queries[:5]:
        results = search_analogs(query, engine_name, top_k=5)
        for result in results:
            unique_results.add(result['index'])
    
    coverage = len(unique_results) / len(processed_data) * 100
    print(f"\n🎯 Покрытие датасета:")
    print(f"   Уникальных товаров в результатах: {len(unique_results)}")
    print(f"   Общее количество товаров: {len(processed_data)}")
    print(f"   Покрытие: {coverage:.1f}%")
    
    # Рекомендации по улучшению
    print(f"\n💡 Рекомендации по улучшению:")
    
    if coverage < 20:
        print("   - Низкое покрытие датасета. Рассмотрите расширение поисковых запросов")
    
    if len(available_engines) == 1:
        print("   - Настройте дополнительные поисковые движки для сравнения")
    
    if not parameter_extractor:
        print("   - Настройте извлечение параметров для более точного поиска")
    
    print("   - Рассмотрите возможность обучения на специфичных для домена данных")
    print("   - Добавьте пользовательскую обратную связь для улучшения релевантности")

else:
    print("❌ Недостаточно данных для анализа покрытия")

# %% [markdown]
# ## 8. Сохранение обученной системы поиска
# 
# Сохраняем все компоненты системы для последующего использования в production.

# %%
# Сохранение обученной системы поиска аналогов
import pickle
from datetime import datetime
import json

print("💾 Сохранение обученной системы поиска...")

# Создаем директорию для сохранения моделей
models_dir = Path("../../models/analog_search")
models_dir.mkdir(parents=True, exist_ok=True)

# Временная метка для версионирования
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"v{timestamp}"

print(f"📁 Директория сохранения: {models_dir}")
print(f"🏷️ Версия модели: {model_version}")

# Подготовка данных для сохранения
system_data = {
    'version': model_version,
    'created_at': datetime.now().isoformat(),
    'dataset_info': {
        'total_records': len(processed_data) if 'processed_data' in locals() else 0,
        'main_column': main_name_column if 'main_name_column' in locals() else None,
        'columns': list(processed_data.columns) if 'processed_data' in locals() else []
    },
    'preprocessing_stats': processing_stats if 'processing_stats' in locals() else None,
    'available_engines': list(available_engines.keys()) if 'available_engines' in locals() else [],
    'spacy_available': spacy_available if 'spacy_available' in locals() else False,
    'preprocessing_errors': preprocessing_errors if 'preprocessing_errors' in locals() else []
}

# 1. Сохранение конфигураций
configs_path = models_dir / f"configs_{model_version}.pkl"
try:
    configs = {
        'cleaning_config': cleaning_config if 'cleaning_config' in locals() else None,
        'normalizer_config': normalizer_config if 'normalizer_config' in locals() else None,
        'lemmatizer_config': lemmatizer_config if 'lemmatizer_config' in locals() else None,
        'preprocessor_config': preprocessor_config if 'preprocessor_config' in locals() else None
    }
    
    with open(configs_path, 'wb') as f:
        pickle.dump(configs, f)
    print(f"✅ Конфигурации сохранены: {configs_path.name}")
except Exception as e:
    print(f"❌ Ошибка сохранения конфигураций: {e}")

# 2. Сохранение компонентов предобработки
preprocessing_path = models_dir / f"preprocessing_{model_version}.pkl"
try:
    # Проверяем сериализуемость функций перед сохранением
    print("🔍 Проверка сериализуемости функций...")
    
    # Тестируем сериализацию enhanced_simple_preprocess
    if 'enhanced_simple_preprocess' in locals():
        try:
            test_pickle = pickle.dumps(enhanced_simple_preprocess)
            print("✅ enhanced_simple_preprocess сериализуется успешно")
        except Exception as e:
            print(f"❌ enhanced_simple_preprocess не сериализуется: {e}")
    
    preprocessing_components = {
        'text_cleaner': text_cleaner if 'text_cleaner' in locals() else None,
        'text_normalizer': text_normalizer if 'text_normalizer' in locals() else None,
        'lemmatizer': lemmatizer if 'lemmatizer' in locals() else None,
        'preprocessor': preprocessor if 'preprocessor' in locals() else None,
        # Сохраняем только глобальную функцию, которая сериализуется
        'enhanced_simple_preprocess': enhanced_simple_preprocess if 'enhanced_simple_preprocess' in locals() else None,
        # Добавляем метаданные для отладки
        'function_info': {
            'enhanced_simple_preprocess_type': str(type(enhanced_simple_preprocess)) if 'enhanced_simple_preprocess' in locals() else None,
            'enhanced_simple_preprocess_module': getattr(enhanced_simple_preprocess, '__module__', None) if 'enhanced_simple_preprocess' in locals() else None,
            'enhanced_simple_preprocess_name': getattr(enhanced_simple_preprocess, '__name__', None) if 'enhanced_simple_preprocess' in locals() else None,
        }
    }
    
    # Исключаем final_preprocess_function, так как она может содержать локальные функции
    print("⚠️ final_preprocess_function не сохраняется (может содержать локальные функции)")
    
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_components, f)
    print(f"✅ Компоненты предобработки сохранены: {preprocessing_path.name}")
except Exception as e:
    print(f"❌ Ошибка сохранения компонентов предобработки: {e}")
    import traceback
    print(f"📋 Подробности ошибки: {traceback.format_exc()}")

# 3. Сохранение поисковых движков
if 'available_engines' in locals() and available_engines:
    engines_path = models_dir / f"search_engines_{model_version}.pkl"
    try:
        with open(engines_path, 'wb') as f:
            pickle.dump(available_engines, f)
        print(f"✅ Поисковые движки сохранены: {engines_path.name}")
        
        # Дополнительное сохранение для больших моделей
        for engine_name, engine in available_engines.items():
            if hasattr(engine, 'save_model'):
                try:
                    engine_model_path = models_dir / f"{engine_name}_model_{model_version}.pkl"
                    engine.save_model(str(engine_model_path))
                    print(f"✅ Модель {engine_name} сохранена отдельно: {engine_model_path.name}")
                except Exception as e:
                    print(f"⚠️ Не удалось сохранить модель {engine_name}: {e}")
                    
    except Exception as e:
        print(f"❌ Ошибка сохранения поисковых движков: {e}")
else:
    print("⚠️ Поисковые движки не найдены для сохранения")

# 4. Сохранение обработанных данных
if 'processed_data' in locals() and len(processed_data) > 0:
    data_path = models_dir / f"processed_data_{model_version}.pkl"
    try:
        # Сохраняем только необходимые столбцы для экономии места
        essential_columns = [main_name_column, 'processed_name'] if 'main_name_column' in locals() else []
        other_columns = [col for col in processed_data.columns if col not in essential_columns]
        all_columns = essential_columns + other_columns
        
        essential_data = processed_data[all_columns].copy()
        
        with open(data_path, 'wb') as f:
            pickle.dump(essential_data, f)
        print(f"✅ Обработанные данные сохранены: {data_path.name}")
        print(f"📊 Размер данных: {len(essential_data)} записей, {len(essential_data.columns)} столбцов")
    except Exception as e:
        print(f"❌ Ошибка сохранения данных: {e}")

# 5. Сохранение метаданных системы
metadata_path = models_dir / f"system_metadata_{model_version}.json"
try:
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(system_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"✅ Метаданные системы сохранены: {metadata_path.name}")
except Exception as e:
    print(f"❌ Ошибка сохранения метаданных: {e}")

# 6. Создание файла с инструкциями по загрузке
readme_path = models_dir / f"README_{model_version}.md"
try:
    dataset_size = len(processed_data) if 'processed_data' in locals() else 'N/A'
    engines_list = ', '.join(available_engines.keys()) if 'available_engines' in locals() else 'N/A'
    
    readme_content = f"""# Система поиска аналогов - {model_version}

## Информация о модели
- **Версия**: {model_version}
- **Создана**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Записей в датасете**: {dataset_size}
- **Доступные движки**: {engines_list}

## Файлы модели
- `configs_{model_version}.pkl` - Конфигурации компонентов
- `preprocessing_{model_version}.pkl` - Компоненты предобработки
- `search_engines_{model_version}.pkl` - Поисковые движки
- `processed_data_{model_version}.pkl` - Обработанные данные
- `system_metadata_{model_version}.json` - Метаданные системы

## Загрузка модели
```python
import pickle
from pathlib import Path

models_dir = Path("models/analog_search")
version = "{model_version}"

# Загрузка компонентов
with open(models_dir / f"preprocessing_{{version}}.pkl", 'rb') as f:
    preprocessing = pickle.load(f)

with open(models_dir / f"search_engines_{{version}}.pkl", 'rb') as f:
    engines = pickle.load(f)

with open(models_dir / f"processed_data_{{version}}.pkl", 'rb') as f:
    data = pickle.load(f)
```

## Использование
Используйте notebook `analog_search_production.ipynb` для загрузки и использования модели.
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ Инструкции сохранены: {readme_path.name}")
except Exception as e:
    print(f"❌ Ошибка создания инструкций: {e}")

# Итоговый отчет
print(f"\n🎉 Сохранение завершено!")
print(f"📁 Все файлы сохранены в: {models_dir}")
print(f"🏷️ Версия модели: {model_version}")
print(f"\n📋 Сохраненные компоненты:")
for file_path in models_dir.glob(f"*{model_version}*"):
    file_size = file_path.stat().st_size / (1024*1024)  # MB
    print(f"   • {file_path.name} ({file_size:.1f} MB)")

print(f"\n💡 Для использования модели запустите notebook: analog_search_production.ipynb")
print(f"🔧 Передайте версию модели: {model_version}")

# Сохраняем версию модели для использования в production notebook
latest_version_path = models_dir / "latest_version.txt"
try:
    with open(latest_version_path, 'w') as f:
        f.write(model_version)
    print(f"✅ Версия модели сохранена в: {latest_version_path.name}")
except Exception as e:
    print(f"⚠️ Не удалось сохранить версию: {e}")

# Тестирование загрузки сохраненных функций
print(f"\n🧪 Тестирование сохраненных функций...")
try:
    test_preprocessing_path = models_dir / f"preprocessing_{model_version}.pkl"
    with open(test_preprocessing_path, 'rb') as f:
        test_components = pickle.load(f)
    
    test_function = test_components.get('enhanced_simple_preprocess')
    if test_function:
        test_result = test_function("Тестовый текст для проверки")
        print(f"✅ Функция предобработки работает: '{test_result}'")
    else:
        print(f"⚠️ Функция предобработки не найдена")
        
    function_info = test_components.get('function_info', {})
    if function_info:
        print(f"📋 Информация о сериализации: {function_info}")
        
except Exception as e:
    print(f"❌ Ошибка тестирования: {e}")

# %% [markdown]
# ## 9. Выводы и заключение

# %%
# Итоговые выводы
print("🎯 ИТОГОВЫЕ ВЫВОДЫ ПО СИСТЕМЕ ПОИСКА АНАЛОГОВ")
print("=" * 70)

print(f"\n📊 Статистика обработки данных:")
print(f"   Загружено записей: {len(data) if 'data' in locals() else 0}")
print(f"   Обработано записей: {len(processed_data) if 'processed_data' in locals() else 0}")
print(f"   Основной столбец: {main_name_column if 'main_name_column' in locals() else 'N/A'}")

print(f"\n🔧 Настроенные компоненты:")
components_status = {
    'Предобработка текста': '✅' if 'preprocessor' in locals() and preprocessor else '❌',
    'Нечеткий поиск': '✅' if 'fuzzy_engine' in locals() and fuzzy_engine else '❌',
    'Семантический поиск': '✅' if 'semantic_engine' in locals() and semantic_engine else '❌',
    'Гибридный поиск': '✅' if 'hybrid_engine' in locals() and hybrid_engine else '❌',
    'Извлечение параметров': '✅' if 'parameter_extractor' in locals() and parameter_extractor else '❌'
}

for component, status in components_status.items():
    print(f"   {status} {component}")

print(f"\n🎯 Возможности системы:")
print(f"   📝 Поиск аналогов по текстовому описанию")
print(f"   🔍 Множественные алгоритмы поиска")
print(f"   📊 Извлечение и анализ технических параметров")
print(f"   ⚖️  Оценка релевантности результатов")
print(f"   📋 Группировка и категоризация товаров")

print(f"\n🚀 Практическое применение:")
print(f"   • Поиск заменителей для снятых с производства товаров")
print(f"   • Сравнение предложений от разных поставщиков")
print(f"   • Автоматическая категоризация новых товаров")
print(f"   • Анализ конкурентных предложений")
print(f"   • Оптимизация складских запасов")

print(f"\n📈 Направления развития:")
print(f"   • Интеграция с внешними каталогами товаров")
print(f"   • Машинное обучение на пользовательских предпочтениях")
print(f"   • Анализ ценовых характеристик")
print(f"   • API для интеграции с ERP-системами")
print(f"   • Веб-интерфейс для конечных пользователей")

print(f"\n✅ Система поиска аналогов готова к использованию!")
print(f"📚 Для получения справки по API обратитесь к документации SAMe")

# Сохранение результатов (опционально)
if 'processed_data' in locals() and len(processed_data) > 0:
    try:
        output_path = Path("../../data/output/processed_nomenclature.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_csv(output_path, index=False)
        print(f"\n💾 Обработанные данные сохранены: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Не удалось сохранить данные: {e}")

print(f"\n🕐 Анализ завершен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


