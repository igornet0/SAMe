#!/usr/bin/env python3
"""
Обработчик поиска аналогов для проекта SAMe

Этот скрипт:
1. Загружает данные из data/input/main_dataset.xlsx
2. Обрабатывает их с помощью всех методов SAMe
3. Находит дубликаты и аналоги
4. Создает Excel отчет в формате analog_analysis_report

"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import gc
import psutil
import time
import glob
import re

# Добавляем src в путь для импортов
sys.path.append('src')

# Импорты модулей SAMe
try:
    from same.text_processing.text_cleaner import TextCleaner, CleaningConfig
    from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
    from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
    from same.text_processing.preprocessor import TextPreprocessor, PreprocessorConfig
    from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
    from same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
    from same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
    from same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig
    from same.data_manager.DataManager import DataManager
    from same.data_manager import data_helper
    print("✅ Модули SAMe успешно загружены")
except ImportError as e:
    print(f"❌ Ошибка импорта модулей SAMe: {e}")
    print("💡 Убедитесь что модули созданы в директории src/same/")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)-45s:%(lineno)-3d - %(levelname)-7s - %(message)s',
    handlers=[
        logging.FileHandler('analog_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Статистика обработки"""
    total_records: int = 0
    processed_records: int = 0
    duplicates_found: int = 0
    analogs_found: int = 0
    processing_errors: int = 0
    start_time: datetime = None
    end_time: datetime = None
    memory_usage_mb: float = 0.0
    batch_count: int = 0
    # Статистика предзагруженных результатов
    preloaded_duplicates: int = 0
    preloaded_analogs: int = 0
    preloaded_processed_names: int = 0
    existing_report_path: str = None


class AnalogSearchProcessor:
    """Главный класс для обработки поиска аналогов"""
    
    def __init__(self):
        self.data_input_path = Path("src/data/input/main_dataset.xlsx")
        self.data_output_dir = Path("src/data/output")
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.preprocessor = None
        self.search_engines = {}
        self.data = None
        self.processed_data = None
        self.stats = ProcessingStats()
        
        # Настройки поиска
        self.similarity_thresholds = {
            'duplicate': 0.95,  # Порог для дубликатов
            'close_analog': 0.85,  # Близкий аналог
            'analog': 0.70,  # Аналог
            'possible_analog': 0.60,  # Возможный аналог
            'similar': 0.50  # Похожий товар
        }

        # Настройки оптимизации
        self.batch_size = 1000  # Размер батча для обработки
        self.memory_limit_mb = 4000  # Лимит памяти в MB
        self.use_processed_cache = True  # Использовать кэш обработанных данных

        # Индексы для оптимизации поиска
        self.analog_index = {}  # Индекс найденных аналогов
        self.processed_pairs = set()  # Обработанные пары аналогов

        # Предзагруженные результаты
        self.preloaded_results = []  # Ранее найденные результаты
        self.preloaded_processed_names = set()  # Уже обработанные наименования
        self.preloaded_analog_pairs = set()  # Предзагруженные пары аналогов
        self.load_existing_results = False  # Флаг загрузки существующих результатов

    def _get_memory_usage(self) -> float:
        """Получение текущего использования памяти в MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _check_memory_limit(self) -> bool:
        """Проверка превышения лимита памяти"""
        current_memory = self._get_memory_usage()
        self.stats.memory_usage_mb = current_memory
        return current_memory > self.memory_limit_mb

    def _cleanup_memory(self):
        """Очистка памяти"""
        gc.collect()
        logger.info(f"🧹 Очистка памяти: {self._get_memory_usage():.1f} MB")

    def _load_processed_data_if_exists(self) -> bool:
        """Загрузка обработанных данных из кэша если существуют"""
        if not self.use_processed_cache:
            return False

        try:
            # Ищем последний файл обработанных данных
            processed_files = list(self.data_output_dir.parent.glob("processed/processed_data_*.parquet"))
            processed_files.extend(list(self.data_output_dir.parent.glob("processed/processed_data_*.csv")))

            if processed_files:
                # Сортируем по времени модификации
                processed_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_file = processed_files[0]

                # Проверяем, что файл не пустой
                if latest_file.stat().st_size > 0:
                    logger.info(f"📂 Загрузка кэшированных данных: {latest_file}")

                    if latest_file.suffix == '.parquet':
                        self.processed_data = pd.read_parquet(latest_file)
                    else:
                        self.processed_data = pd.read_csv(latest_file)

                    logger.info(f"✅ Загружено {len(self.processed_data)} обработанных записей из кэша")
                    return True
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить кэшированные данные: {e}")

        return False

    def _save_processed_data(self):
        """Сохранение обработанных данных в кэш"""
        if not self.use_processed_cache or self.processed_data is None:
            return

        try:
            # Создаем директорию для кэша
            cache_dir = self.data_output_dir.parent / "processed"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Сохраняем с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = cache_dir / f"processed_data_{timestamp}.parquet"

            self.processed_data.to_parquet(cache_file, index=False)
            logger.info(f"💾 Обработанные данные сохранены в кэш: {cache_file}")

        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить данные в кэш: {e}")

    def _find_latest_report(self) -> Optional[Path]:
        """Поиск последнего созданного отчета в директории output"""
        try:
            # Ищем файлы отчетов по паттерну
            pattern = str(self.data_output_dir / "analog_analysis_report_*.xlsx")
            report_files = glob.glob(pattern)

            if not report_files:
                logger.info("📂 Существующие отчеты не найдены")
                return None

            # Сортируем по времени модификации (последний - первый)
            report_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_report = Path(report_files[0])

            logger.info(f"📂 Найден последний отчет: {latest_report.name}")
            return latest_report

        except Exception as e:
            logger.warning(f"⚠️ Ошибка поиска существующих отчетов: {e}")
            return None

    def _load_existing_results(self) -> bool:
        """Загрузка существующих результатов из последнего отчета"""
        if not self.load_existing_results:
            return False

        latest_report = self._find_latest_report()
        if not latest_report or not latest_report.exists():
            logger.info("📂 Нет существующих отчетов для загрузки")
            return False

        try:
            logger.info(f"📂 Загрузка существующих результатов из {latest_report.name}")

            # Загружаем Excel файл
            existing_df = pd.read_excel(latest_report)

            # Проверяем структуру файла
            required_columns = ['Raw_Name', 'Candidate_Name', 'Relation_Type', 'Similarity_Score']
            missing_columns = [col for col in required_columns if col not in existing_df.columns]

            if missing_columns:
                logger.warning(f"⚠️ Отсутствуют необходимые колонки в отчете: {missing_columns}")
                return False

            # Извлекаем информацию о уже обработанных наименованиях
            processed_names = set()
            analog_pairs = set()

            for _, row in existing_df.iterrows():
                raw_name = str(row['Raw_Name']).strip()
                candidate_name = str(row['Candidate_Name']).strip()
                relation_type = str(row['Relation_Type']).strip()

                # Добавляем в индекс обработанных наименований
                processed_names.add(raw_name)

                # Для аналогов добавляем пары (исключая дубликаты)
                if relation_type != 'дубль' and not candidate_name.startswith('ДУБЛИКАТ:'):
                    processed_names.add(candidate_name)
                    analog_pairs.add((raw_name, candidate_name))
                    analog_pairs.add((candidate_name, raw_name))  # Обратная связь

            # Оптимизируем предзагруженные результаты (дедупликация старых дубликатов)
            optimized_results = self._optimize_preloaded_duplicates(existing_df)

            # Сохраняем предзагруженные данные
            self.preloaded_results = optimized_results
            self.preloaded_processed_names = processed_names
            self.preloaded_analog_pairs = analog_pairs

            # Обновляем статистику
            self.stats.preloaded_duplicates = len(existing_df[existing_df['Relation_Type'] == 'дубль'])
            self.stats.preloaded_analogs = len(existing_df[existing_df['Relation_Type'] != 'дубль'])
            self.stats.preloaded_processed_names = len(processed_names)
            self.stats.existing_report_path = str(latest_report)

            logger.info(f"✅ Загружено из существующего отчета:")
            logger.info(f"   📊 Всего записей: {len(existing_df)}")
            logger.info(f"   🔄 Дубликатов: {self.stats.preloaded_duplicates}")
            logger.info(f"   🔍 Аналогов: {self.stats.preloaded_analogs}")
            logger.info(f"   📝 Обработанных наименований: {self.stats.preloaded_processed_names}")
            logger.info(f"   🔗 Пар аналогов: {len(analog_pairs)//2}")  # Делим на 2, т.к. пары двунаправленные

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки существующих результатов: {e}")
            return False

    def _optimize_preloaded_duplicates(self, existing_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Оптимизация предзагруженных дубликатов - объединение повторяющихся записей"""
        logger.info("🔧 Оптимизация предзагруженных дубликатов...")

        optimized_results = []
        processed_duplicate_names = set()  # Для отслеживания уже обработанных Raw_Name дубликатов

        # Сначала обрабатываем все НЕ дубликаты
        non_duplicates = existing_df[existing_df['Relation_Type'] != 'дубль']
        for _, row in non_duplicates.iterrows():
            optimized_results.append(row.to_dict())

        # Затем обрабатываем дубликаты с полной дедупликацией по Raw_Name
        duplicates = existing_df[existing_df['Relation_Type'] == 'дубль']

        # Группируем дубликаты по Raw_Name для полной дедупликации
        duplicate_groups = duplicates.groupby('Raw_Name')

        for raw_name, group in duplicate_groups:
            # Подсчитываем общее количество дубликатов в группе
            group_size = len(group)

            # Проверяем, есть ли в группе записи с разными candidate_name
            unique_candidates = group['Candidate_Name'].unique()

            # Если есть записи с "ДУБЛИКАТ: X записи", извлекаем максимальное число
            max_count = group_size
            for candidate in unique_candidates:
                if candidate.startswith('ДУБЛИКАТ:'):
                    try:
                        # Извлекаем число из строки "ДУБЛИКАТ: X записи"
                        import re
                        match = re.search(r'ДУБЛИКАТ: (\d+)', candidate)
                        if match:
                            count = int(match.group(1))
                            max_count = max(max_count, count)
                    except:
                        pass

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обрабатываем только настоящие дубликаты
            if max_count > 1:
                # Берем первую запись из группы как представителя
                representative_row = group.iloc[0]
                row_dict = representative_row.to_dict()

                # Обновляем запись в оптимизированном формате
                row_dict['Candidate_Name'] = f"ДУБЛИКАТ: {max_count} записи"
                row_dict['Comment'] = f'Точный дубликат, найдено {max_count} записи'
                row_dict['Similarity_Score'] = 1.0

                optimized_results.append(row_dict)
                processed_duplicate_names.add(raw_name)

                # Логируем дедупликацию
                if group_size > 1:
                    logger.debug(f"🔧 Дедуплицирована группа: {raw_name} ({group_size} -> 1 запись, размер группы: {max_count})")
            else:
                # ЛОЖНЫЙ ДУБЛИКАТ: Это одиночная запись, которая была неправильно помечена как дубликат
                # Исключаем её из результатов дубликатов
                logger.debug(f"⏭️ Исключен ложный дубликат: {raw_name} (размер группы: {max_count})")
                # НЕ добавляем в optimized_results - эта запись должна быть обработана как обычная запись для поиска аналогов

        original_count = len(existing_df)
        optimized_count = len(optimized_results)
        deduplicated_count = original_count - optimized_count

        # Дополнительная статистика
        original_duplicates = len(duplicates)
        optimized_duplicates = len([r for r in optimized_results if r['Relation_Type'] == 'дубль'])
        false_duplicates_removed = original_duplicates - optimized_duplicates

        if deduplicated_count > 0 or false_duplicates_removed > 0:
            logger.info(f"✅ Оптимизация завершена:")
            logger.info(f"   📊 Исходных записей: {original_count}")
            logger.info(f"   📊 Оптимизированных записей: {optimized_count}")
            logger.info(f"   🔧 Дедуплицировано повторов: {deduplicated_count}")
            logger.info(f"   🔄 Дубликатов до: {original_duplicates} -> после: {optimized_duplicates}")
            if false_duplicates_removed > 0:
                logger.info(f"   ❌ Исключено ложных дубликатов: {false_duplicates_removed}")
        else:
            logger.info(f"✅ Дедупликация не требуется - данные уже оптимизированы")

        return optimized_results

    def _configure_engine_for_small_dataset(self, search_engine, num_documents: int):
        """Адаптивная настройка поискового движка для малых наборов данных"""
        try:
            # Настройка для нечеткого поиска
            if hasattr(search_engine, 'fuzzy_engine'):
                fuzzy_engine = search_engine.fuzzy_engine
                if hasattr(fuzzy_engine, 'config'):
                    # Адаптивные настройки TF-IDF для малых данных
                    fuzzy_engine.config.tfidf_min_df = 1
                    fuzzy_engine.config.tfidf_max_df = 1.0  # Убираем ограничение max_df
                    fuzzy_engine.config.cosine_threshold = 0.1  # Снижаем порог
                    fuzzy_engine.config.tfidf_max_features = min(1000, num_documents * 50)
                    logger.info(f"🔧 Настроен нечеткий поиск для {num_documents} документов")

            # Настройка для гибридного поиска
            if hasattr(search_engine, 'config'):
                if hasattr(search_engine.config, 'min_fuzzy_score'):
                    search_engine.config.min_fuzzy_score = 0.1
                if hasattr(search_engine.config, 'min_semantic_score'):
                    search_engine.config.min_semantic_score = 0.1

            # Настройка для семантического поиска
            if hasattr(search_engine, 'semantic_engine'):
                semantic_engine = search_engine.semantic_engine
                if hasattr(semantic_engine, 'config'):
                    semantic_engine.config.similarity_threshold = 0.1
                    logger.info(f"🔧 Настроен семантический поиск для {num_documents} документов")

            return search_engine

        except Exception as e:
            logger.warning(f"⚠️ Не удалось настроить движок для малых данных: {e}")
            return search_engine

    def _simple_search_fallback(self, unique_data, processed_names: set, analog_pairs: set) -> List[Dict[str, Any]]:
        """Простой fallback поиск для очень малых наборов данных"""
        logger.info("🔄 Использование простого поиска на основе строкового сходства...")

        analogs = []
        documents = unique_data['processed_name'].tolist()

        # Простой поиск на основе строкового сходства
        for idx, row in unique_data.iterrows():
            query = row['processed_name']

            if query in processed_names:
                continue

            best_match = None
            best_score = 0.0

            # Ищем лучшее совпадение среди других документов
            for other_idx, other_row in unique_data.iterrows():
                if idx == other_idx:
                    continue

                candidate = other_row['processed_name']
                if candidate in processed_names:
                    continue

                # Простое строковое сходство
                from difflib import SequenceMatcher
                score = SequenceMatcher(None, query, candidate).ratio()

                if score > best_score and score > 0.6:  # Порог для простого поиска
                    best_score = score
                    best_match = other_row

            if best_match is not None:
                analog_info = {
                    'Raw_Name': row['Raw_Name'],
                    'Cleaned_Name': row['Cleaned_Name'],
                    'Lemmatized_Name': row['Lemmatized_Name'],
                    'Normalized_Name': row['Normalized_Name'],
                    'Candidate_Name': best_match['Raw_Name'],
                    'Similarity_Score': f"{best_score:.3f}",
                    'Relation_Type': 'возможный аналог',
                    'Suggested_Category': best_match.get('Группа', ''),
                    'Final_Decision': 'Требует проверки',
                    'Comment': f'Найдено простым поиском (малый набор данных), схожесть: {best_score:.3f}',
                    'Original_Category': row.get('Группа', ''),
                    'Original_Code': row.get('Код', ''),
                    'Search_Engine': 'simple_fallback'
                }
                analogs.append(analog_info)
                processed_names.add(query)
                processed_names.add(best_match['processed_name'])

        logger.info(f"✅ Простой поиск нашел {len(analogs)} аналогов")
        return analogs
        
    def initialize_components(self):
        """Инициализация компонентов обработки"""
        logger.info("🔧 Инициализация компонентов SAMe...")
        
        # Конфигурация предобработчика
        preprocessor_config = PreprocessorConfig(
            save_intermediate_steps=True,
            enable_parallel_processing=True,
            max_workers=4
        )
        
        self.preprocessor = TextPreprocessor(preprocessor_config)
        
        # Инициализация поисковых движков
        try:
            # Fuzzy Search
            fuzzy_config = FuzzySearchConfig(
                similarity_threshold=0.6,
                fuzzy_threshold=60,
                max_candidates=100
            )
            self.search_engines['fuzzy'] = FuzzySearchEngine(fuzzy_config)

            # Semantic Search (если доступен)
            try:
                semantic_config = SemanticSearchConfig(
                    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    use_gpu=False,
                    similarity_threshold=0.5
                )
                self.search_engines['semantic'] = SemanticSearchEngine(semantic_config)
            except Exception as e:
                logger.warning(f"Semantic search недоступен: {e}")

            # Hybrid Search
            if 'semantic' in self.search_engines:
                hybrid_config = HybridSearchConfig(
                    fuzzy_weight=0.4,
                    semantic_weight=0.6,
                    fuzzy_config=fuzzy_config,
                    semantic_config=semantic_config
                )
                self.search_engines['hybrid'] = HybridSearchEngine(hybrid_config)
            
            logger.info(f"✅ Инициализированы движки: {list(self.search_engines.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации поисковых движков: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Загрузка данных из Excel файла"""
        logger.info(f"📂 Загрузка данных из {self.data_input_path}")
        
        if not self.data_input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_input_path}")
        
        try:
            self.data = pd.read_excel(self.data_input_path)
            self.stats.total_records = len(self.data)
            
            logger.info(f"✅ Загружено {len(self.data)} записей")
            logger.info(f"📊 Колонки: {list(self.data.columns)}")
            
            # Определяем основную колонку с наименованиями
            name_columns = [col for col in self.data.columns 
                          if 'наименование' in col.lower() or 'название' in col.lower()]
            
            if name_columns:
                self.main_name_column = name_columns[0]
            else:
                self.main_name_column = 'Наименование'  # По умолчанию
            
            logger.info(f"📝 Основная колонка: '{self.main_name_column}'")
            
            return self.data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """Предобработка данных с оптимизацией памяти"""
        logger.info("🔄 Предобработка данных...")
        self.stats.start_time = datetime.now()

        if self.data is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")

        # Проверяем кэш обработанных данных
        if self._load_processed_data_if_exists():
            self.stats.processed_records = len(self.processed_data)
            return self.processed_data

        try:
            # Обработка данных батчами для экономии памяти
            total_records = len(self.data)
            processed_chunks = []

            logger.info(f"📊 Обработка {total_records} записей батчами по {self.batch_size}")

            for i in range(0, total_records, self.batch_size):
                batch_end = min(i + self.batch_size, total_records)
                batch_data = self.data.iloc[i:batch_end].copy()

                logger.info(f"🔄 Обработка батча {i//self.batch_size + 1}/{(total_records + self.batch_size - 1)//self.batch_size} ({i+1}-{batch_end})")

                # Предобработка батча
                processed_batch = self.preprocessor.preprocess_dataframe(
                    batch_data,
                    self.main_name_column,
                    output_columns={
                        'cleaned': 'Cleaned_Name',
                        'normalized': 'Normalized_Name',
                        'lemmatized': 'Lemmatized_Name',
                        'final': 'processed_name'
                    }
                )

                # Добавляем исходное наименование как Raw_Name
                processed_batch['Raw_Name'] = processed_batch[self.main_name_column]

                processed_chunks.append(processed_batch)
                self.stats.batch_count += 1

                # Проверяем использование памяти
                if self._check_memory_limit():
                    logger.warning(f"⚠️ Превышен лимит памяти: {self.stats.memory_usage_mb:.1f} MB")
                    self._cleanup_memory()

                # Освобождаем память от батча
                del batch_data

            # Объединяем все батчи
            logger.info("🔄 Объединение обработанных батчей...")
            self.processed_data = pd.concat(processed_chunks, ignore_index=True)

            # Очищаем промежуточные данные
            del processed_chunks
            self._cleanup_memory()

            self.stats.processed_records = len(self.processed_data)

            # Сохраняем в кэш
            self._save_processed_data()

            logger.info(f"✅ Предобработка завершена: {len(self.processed_data)} записей")
            logger.info(f"💾 Использование памяти: {self._get_memory_usage():.1f} MB")

            return self.processed_data

        except Exception as e:
            logger.error(f"❌ Ошибка предобработки: {e}")
            self.stats.processing_errors += 1
            raise
    
    def find_duplicates(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """Поиск дубликатов в данных с оптимизированным представлением"""
        logger.info("🔍 Поиск дубликатов...")

        duplicates = []
        duplicate_groups = {}  # Для отслеживания групп дубликатов

        # Группируем по обработанным наименованиям
        grouped = self.processed_data.groupby('processed_name')

        for processed_name, group in grouped:
            if len(group) > 1:
                # Сохраняем группу дубликатов для исключения из поиска аналогов
                duplicate_groups[processed_name] = group.index.tolist()

                # Создаем ОДНУ запись для всей группы дубликатов
                # Берем первую запись как представителя группы
                representative_row = group.iloc[0]

                # Собираем все наименования в группе
                all_names = group['Raw_Name'].tolist()

                duplicate_info = {
                    'index': representative_row.name,
                    'Raw_Name': representative_row['Raw_Name'],
                    'Cleaned_Name': representative_row['Cleaned_Name'],
                    'Lemmatized_Name': representative_row['Lemmatized_Name'],
                    'Normalized_Name': representative_row['Normalized_Name'],
                    'Candidate_Name': f"ДУБЛИКАТ: {len(group)} записи",  # Исправлено: "записи" вместо "записей"
                    'Similarity_Score': 1.000,
                    'Relation_Type': 'дубль',
                    'Suggested_Category': representative_row.get('Группа', ''),
                    'Final_Decision': 'Дубликат - объединить',
                    'Comment': f'Точный дубликат, найдено {len(group)} записи',  # Исправлено: "записи" вместо "записей"
                    'Original_Category': representative_row.get('Группа', ''),
                    'Original_Code': representative_row.get('Код', ''),
                    'Search_Engine': 'exact_match',
                    'Duplicate_Count': len(group),
                    'All_Duplicate_Names': '; '.join(all_names[:5]) + (f' и еще {len(all_names)-5}' if len(all_names) > 5 else '')
                }
                duplicates.append(duplicate_info)

        self.stats.duplicates_found = len(duplicates)
        total_duplicate_records = sum(len(group) for group in duplicate_groups.values())
        logger.info(f"✅ Найдено {len(duplicates)} групп дубликатов ({total_duplicate_records} записей)")

        return duplicates, duplicate_groups

    def search_analogs(self, exclude_duplicates: bool = True, duplicate_groups: Dict[str, List[int]] = None) -> List[Dict[str, Any]]:
        """Поиск аналогов с помощью поисковых движков с оптимизацией памяти и исключением дубликатов"""
        logger.info("🔍 Поиск аналогов...")

        analogs = []
        processed_items = set()  # Для контроля обработанных элементов

        # Инициализируем с предзагруженными данными
        processed_names = set(self.preloaded_processed_names)  # Предзагруженные обработанные наименования
        analog_pairs = set(self.preloaded_analog_pairs)  # Предзагруженные пары аналогов

        # Логируем информацию о предзагруженных данных
        if self.preloaded_processed_names:
            logger.info(f"📂 Используются предзагруженные результаты:")
            logger.info(f"   📝 Исключено из поиска: {len(self.preloaded_processed_names)} наименований")
            logger.info(f"   🔗 Предзагружено пар аналогов: {len(self.preloaded_analog_pairs)//2}")  # Делим на 2, т.к. пары двунаправленные

        # Диагностическая информация о фильтрации данных
        initial_data_size = len(self.processed_data)
        logger.info(f"📊 Диагностика фильтрации данных:")
        logger.info(f"   📋 Исходных записей в processed_data: {initial_data_size}")

        # Получаем уникальные записи (исключаем дубликаты если нужно)
        if exclude_duplicates:
            unique_data = self.processed_data.drop_duplicates(subset=['processed_name'])
            logger.info(f"   📋 После удаления дубликатов processed_name: {len(unique_data)}")
        else:
            unique_data = self.processed_data
            logger.info(f"   📋 Без удаления дубликатов: {len(unique_data)}")

        # Исключаем записи, которые являются дубликатами
        if duplicate_groups:
            duplicate_indices = set()
            for group_indices in duplicate_groups.values():
                duplicate_indices.update(group_indices)

            before_duplicate_filter = len(unique_data)
            # Оставляем только записи, которые НЕ являются дубликатами
            unique_data = unique_data[~unique_data.index.isin(duplicate_indices)]
            logger.info(f"   📋 После исключения групп дубликатов: {len(unique_data)} (исключено {before_duplicate_filter - len(unique_data)})")
            logger.info(f"🔄 Исключено {len(duplicate_indices)} записей-дубликатов из поиска аналогов")

        # Исключаем записи, которые уже обработаны в предыдущих запусках
        if self.preloaded_processed_names:
            # Фильтруем данные по Raw_Name, исключая уже обработанные
            before_filter = len(unique_data)
            unique_data = unique_data[~unique_data['Raw_Name'].isin(self.preloaded_processed_names)]
            after_filter = len(unique_data)
            excluded_count = before_filter - after_filter

            logger.info(f"   📋 После исключения предзагруженных: {after_filter} (исключено {excluded_count})")

            if excluded_count > 0:
                logger.info(f"🔄 Исключено {excluded_count} записей, уже обработанных в предыдущих запусках")

            # Дополнительная диагностика
            if excluded_count > before_filter * 0.9:  # Если исключено более 90%
                logger.warning(f"⚠️ ВНИМАНИЕ: Исключено {excluded_count}/{before_filter} ({excluded_count/before_filter*100:.1f}%) записей!")
                logger.warning(f"   Это может указывать на проблему с инкрементальной обработкой")

        # КРИТИЧЕСКАЯ ПРОВЕРКА: Убеждаемся, что осталось достаточно данных для поиска
        min_documents_required = 10  # Минимальное количество документов для эффективного поиска

        if len(unique_data) < min_documents_required:
            logger.warning(f"⚠️ Недостаточно данных для поиска аналогов: {len(unique_data)} записей")
            logger.warning(f"   Минимум требуется: {min_documents_required} записей")

            if len(unique_data) == 0:
                logger.info("✅ Все записи уже обработаны в предыдущих запусках")
                return []

            # Если данных мало, но они есть - продолжаем с предупреждением
            logger.info(f"🔄 Продолжаем обработку {len(unique_data)} записей с ограниченной эффективностью")

        # Выбираем лучший доступный движок
        search_engine_name = self._get_best_search_engine()
        search_engine = self.search_engines[search_engine_name]

        logger.info(f"🔧 Используется движок: {search_engine_name}")

        # Подготавливаем данные для поиска
        documents = unique_data['processed_name'].tolist()

        # Адаптивная настройка поискового движка для малых наборов данных
        if len(documents) < 50:
            logger.info(f"🔧 Малый набор данных ({len(documents)} документов) - применяем адаптивные настройки")
            search_engine = self._configure_engine_for_small_dataset(search_engine, len(documents))

        # Инициализируем поисковый движок
        try:
            logger.info("🔄 Инициализация поискового движка...")
            if hasattr(search_engine, 'fit'):
                search_engine.fit(documents)
            elif hasattr(search_engine, 'build_index'):
                search_engine.build_index(documents)
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации движка: {e}")
            logger.error(f"   Количество документов: {len(documents)}")
            logger.error(f"   Тип движка: {type(search_engine).__name__}")

            # Попытка использовать fallback движок для малых данных
            if len(documents) < 10:
                logger.info("🔄 Попытка использования простого поиска для малого набора данных...")
                return self._simple_search_fallback(unique_data, processed_names, analog_pairs)

            return []

        # Обработка данных батчами для экономии памяти
        total_records = len(unique_data)
        batch_size = min(self.batch_size, 500)  # Меньший батч для поиска

        logger.info(f"📊 Поиск аналогов для {total_records} записей батчами по {batch_size}")

        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            batch_data = unique_data.iloc[batch_start:batch_end]

            logger.info(f"🔄 Обработка батча поиска {batch_start//batch_size + 1}/{(total_records + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end})")

            # Поиск аналогов для батча
            for idx, row in batch_data.iterrows():
                if idx in processed_items:
                    continue

                query = row['processed_name']

                # Пропускаем если это наименование уже обработано
                if query in processed_names:
                    continue

                try:
                    # Выполняем поиск
                    results = search_engine.search(query, top_k=10)

                    # ИСПРАВЛЕНИЕ: Добавляем валидацию результатов
                    if not results:
                        logger.debug(f"Пустые результаты поиска для '{query[:30]}...'")
                        continue

                    # Валидируем и нормализуем результаты
                    validated_results = self._validate_search_results(results, query)
                    if not validated_results:
                        logger.debug(f"Все результаты не прошли валидацию для '{query[:30]}...'")
                        continue

                    # Обрабатываем результаты с учетом уже найденных аналогов
                    best_analog = self._process_search_results_optimized(
                        row, validated_results, unique_data, search_engine_name, processed_names, analog_pairs
                    )

                    if best_analog:
                        analogs.append(best_analog)
                        processed_items.add(idx)
                        processed_names.add(query)

                        # Добавляем пару аналогов для предотвращения обратного поиска
                        candidate_name = best_analog.get('processed_candidate_name')
                        if candidate_name:
                            analog_pairs.add((query, candidate_name))
                            analog_pairs.add((candidate_name, query))  # Обратная связь
                            processed_names.add(candidate_name)  # Исключаем кандидата из дальнейшего поиска

                except KeyError as e:
                    # Специальная обработка для KeyError (включая 'hybrid_score')
                    logger.warning(f"Отсутствует ключ {e} в результатах поиска для '{query[:50]}...'. Движок: {search_engine_name}")
                    self.stats.processing_errors += 1
                    continue
                except Exception as e:
                    logger.warning(f"Ошибка поиска для '{query[:50]}...': {type(e).__name__}: {e}")
                    self.stats.processing_errors += 1
                    continue

            # Проверяем использование памяти
            if self._check_memory_limit():
                logger.warning(f"⚠️ Превышен лимит памяти: {self.stats.memory_usage_mb:.1f} MB")
                self._cleanup_memory()

        self.stats.analogs_found = len(analogs)
        logger.info(f"✅ Найдено {len(analogs)} аналогов")
        logger.info(f"💾 Использование памяти: {self._get_memory_usage():.1f} MB")

        return analogs

    def _get_best_search_engine(self) -> str:
        """Выбор лучшего доступного поискового движка"""
        if 'hybrid' in self.search_engines:
            return 'hybrid'
        elif 'semantic' in self.search_engines:
            return 'semantic'
        elif 'fuzzy' in self.search_engines:
            return 'fuzzy'
        else:
            raise ValueError("Нет доступных поисковых движков")

    def _validate_search_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        ИСПРАВЛЕНИЕ: Валидация и нормализация результатов поиска

        Args:
            results: Список результатов поиска
            query: Поисковый запрос (для логирования)

        Returns:
            Список валидированных результатов
        """
        if not results:
            return []

        validated_results = []
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                logger.warning(f"Результат {i} не является словарем для запроса '{query[:30]}...'")
                continue

            # Проверяем наличие обязательных ключей
            required_keys = ['document_id', 'document']
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                logger.warning(f"Отсутствуют обязательные ключи {missing_keys} в результате {i} для '{query[:30]}...'")
                continue

            # Нормализуем скоры - добавляем отсутствующие
            result_copy = result.copy()

            # Убеждаемся что есть хотя бы один скор
            score_keys = ['hybrid_score', 'similarity_score', 'combined_score', 'score']
            has_score = any(key in result_copy for key in score_keys)

            if not has_score:
                logger.warning(f"Отсутствуют все скоры в результате {i} для '{query[:30]}...'. Добавляем default_score=0.5")
                result_copy['hybrid_score'] = 0.5  # Добавляем дефолтный скор

            validated_results.append(result_copy)

        return validated_results

    def _process_search_results_optimized(self, query_row: pd.Series, results: List[Dict],
                              data: pd.DataFrame, engine_name: str, processed_names: set, analog_pairs: set) -> Optional[Dict[str, Any]]:
        """Оптимизированная обработка результатов поиска с предотвращением повторного поиска"""

        if not results:
            return None

        query_text = query_row['processed_name']

        # Фильтруем результаты: исключаем саму запись и уже обработанные наименования
        filtered_results = []
        for r in results:
            candidate_text = r.get('document', '')

            # Исключаем саму запись
            if candidate_text == query_text:
                continue

            # Исключаем уже обработанные наименования
            if candidate_text in processed_names:
                continue

            # Исключаем если эта пара уже найдена в обратном направлении
            if (candidate_text, query_text) in analog_pairs:
                continue

            filtered_results.append(r)

        if not filtered_results:
            return None

        # Берем лучший результат
        best_result = filtered_results[0]

        # ИСПРАВЛЕНИЕ: Более надежное извлечение скора с детальным логированием
        similarity_score = 0.0
        score_source = "default"

        # Приоритетный порядок извлечения скора
        score_keys = ['hybrid_score', 'similarity_score', 'combined_score', 'score']
        for key in score_keys:
            if key in best_result and best_result[key] is not None:
                try:
                    similarity_score = float(best_result[key])
                    score_source = key
                    break
                except (ValueError, TypeError):
                    continue

        # Логируем если не удалось найти валидный скор
        if similarity_score == 0.0:
            logger.debug(f"Не удалось извлечь скор для результата. Доступные ключи: {list(best_result.keys())}")
            # Попытка извлечь любое числовое значение
            for key, value in best_result.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    similarity_score = float(value)
                    score_source = f"{key}_fallback"
                    break

        # Определяем тип связи
        relation_type = self._determine_relation_type(similarity_score)

        if relation_type == 'нет аналогов':
            return None

        # Находим соответствующую запись в данных
        candidate_text = best_result.get('document', '')
        candidate_rows = data[data['processed_name'] == candidate_text]

        if len(candidate_rows) == 0:
            return None

        candidate_row = candidate_rows.iloc[0]

        # Формируем результат
        analog_info = {
            'Raw_Name': query_row['Raw_Name'],
            'Cleaned_Name': query_row['Cleaned_Name'],
            'Lemmatized_Name': query_row['Lemmatized_Name'],
            'Normalized_Name': query_row['Normalized_Name'],
            'Candidate_Name': candidate_row['Raw_Name'],
            'Similarity_Score': f"{similarity_score:.3f}",
            'Relation_Type': relation_type,
            'Suggested_Category': candidate_row.get('Группа', ''),
            'Final_Decision': self._get_decision(relation_type, similarity_score),
            'Comment': self._generate_comment(query_row, candidate_row, similarity_score),
            'Original_Category': query_row.get('Группа', ''),
            'Original_Code': query_row.get('Код', ''),
            'Search_Engine': engine_name,
            'processed_candidate_name': candidate_text  # Для внутреннего использования
        }

        return analog_info

    def _process_search_results(self, query_row: pd.Series, results: List[Dict],
                              data: pd.DataFrame, engine_name: str) -> Optional[Dict[str, Any]]:
        """Обработка результатов поиска для одной записи"""

        if not results:
            return None

        # Исключаем саму запись из результатов
        query_text = query_row['processed_name']
        filtered_results = [r for r in results if r.get('document', '') != query_text]

        if not filtered_results:
            return None

        # Берем лучший результат
        best_result = filtered_results[0]

        # ИСПРАВЛЕНИЕ: Более надежное извлечение скора (аналогично первому методу)
        similarity_score = 0.0

        # Приоритетный порядок извлечения скора
        score_keys = ['hybrid_score', 'similarity_score', 'combined_score', 'score']
        for key in score_keys:
            if key in best_result and best_result[key] is not None:
                try:
                    similarity_score = float(best_result[key])
                    break
                except (ValueError, TypeError):
                    continue

        # Fallback: попытка извлечь любое числовое значение
        if similarity_score == 0.0:
            for key, value in best_result.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    similarity_score = float(value)
                    break

        # Определяем тип связи
        relation_type = self._determine_relation_type(similarity_score)

        if relation_type == 'нет аналогов':
            return None

        # Находим соответствующую запись в данных
        candidate_text = best_result.get('document', '')
        candidate_row = data[data['processed_name'] == candidate_text].iloc[0] if len(data[data['processed_name'] == candidate_text]) > 0 else None

        if candidate_row is None:
            return None

        # Формируем результат
        analog_info = {
            'Raw_Name': query_row['Raw_Name'],
            'Cleaned_Name': query_row['Cleaned_Name'],
            'Lemmatized_Name': query_row['Lemmatized_Name'],
            'Normalized_Name': query_row['Normalized_Name'],
            'Candidate_Name': candidate_row['Raw_Name'],
            'Similarity_Score': f"{similarity_score:.3f}",
            'Relation_Type': relation_type,
            'Suggested_Category': candidate_row.get('Группа', ''),
            'Final_Decision': self._get_decision(relation_type, similarity_score),
            'Comment': self._generate_comment(query_row, candidate_row, similarity_score),
            'Original_Category': query_row.get('Группа', ''),
            'Original_Code': query_row.get('Код', ''),
            'Search_Engine': engine_name
        }

        return analog_info

    def _determine_relation_type(self, similarity_score: float) -> str:
        """Определение типа связи на основе оценки схожести"""
        if similarity_score >= self.similarity_thresholds['duplicate']:
            return 'дубль'
        elif similarity_score >= self.similarity_thresholds['close_analog']:
            return 'близкий аналог'
        elif similarity_score >= self.similarity_thresholds['analog']:
            return 'аналог'
        elif similarity_score >= self.similarity_thresholds['possible_analog']:
            return 'возможный аналог'
        elif similarity_score >= self.similarity_thresholds['similar']:
            return 'похожий товар'
        else:
            return 'нет аналогов'

    def _get_decision(self, relation_type: str, similarity_score: float) -> str:
        """Получение рекомендации по принятию решения"""
        decisions = {
            'дубль': 'Дубликат - объединить',
            'близкий аналог': 'Близкий аналог - проверить',
            'аналог': 'Аналог - рассмотреть замену',
            'возможный аналог': 'Возможный аналог - требует анализа',
            'похожий товар': 'Похожий товар - низкая вероятность замены',
            'нет аналогов': 'Аналоги не найдены'
        }
        return decisions.get(relation_type, 'Требует ручной проверки')

    def _generate_comment(self, query_row: pd.Series, candidate_row: pd.Series,
                         similarity_score: float) -> str:
        """Генерация комментария для результата"""
        comments = []

        # Оценка схожести
        if similarity_score >= 0.9:
            comments.append("Высокая схожесть")
        elif similarity_score >= 0.7:
            comments.append("Средняя схожесть")
        else:
            comments.append("Низкая схожесть")

        # Сравнение категорий
        query_category = query_row.get('Группа', '')
        candidate_category = candidate_row.get('Группа', '')

        if query_category and candidate_category:
            if query_category != candidate_category:
                comments.append(f"Разные категории: {query_category} vs {candidate_category}")
            else:
                comments.append("Одинаковые категории")

        # Дополнительная информация
        comments.append(f"Поисковый скор: {similarity_score:.3f}")

        return "; ".join(comments)

    def create_excel_report(self, duplicates: List[Dict], analogs: List[Dict]) -> str:
        """Создание Excel отчета в формате analog_analysis_report с оптимизацией памяти"""
        logger.info("📊 Создание Excel отчета...")

        # Объединяем новые дубликаты и аналоги
        new_results = duplicates + analogs

        # Объединяем с предзагруженными результатами
        all_results = self.preloaded_results + new_results

        if not all_results:
            logger.warning("⚠️ Нет данных для создания отчета")
            return None

        # Логируем статистику объединения
        if self.preloaded_results:
            logger.info(f"📊 Объединение результатов:")
            logger.info(f"   📂 Предзагруженных: {len(self.preloaded_results)}")
            logger.info(f"   🆕 Новых: {len(new_results)}")
            logger.info(f"   📊 Итого: {len(all_results)}")

        logger.info(f"📊 Создание отчета для {len(all_results)} записей")

        try:
            # Создаем DataFrame батчами для больших данных
            if len(all_results) > 10000:
                logger.info("📊 Большой объем данных, создание DataFrame батчами...")
                batch_size = 5000
                df_chunks = []

                for i in range(0, len(all_results), batch_size):
                    batch_end = min(i + batch_size, len(all_results))
                    batch_results = all_results[i:batch_end]
                    df_chunk = pd.DataFrame(batch_results)
                    df_chunks.append(df_chunk)

                    # Проверяем память
                    if self._check_memory_limit():
                        self._cleanup_memory()

                results_df = pd.concat(df_chunks, ignore_index=True)
                del df_chunks  # Освобождаем память
            else:
                results_df = pd.DataFrame(all_results)

            # Сортируем по типу связи и схожести
            type_order = ['дубль', 'близкий аналог', 'аналог', 'возможный аналог', 'похожий товар', 'нет аналогов']
            results_df['type_order'] = results_df['Relation_Type'].apply(
                lambda x: type_order.index(x) if x in type_order else len(type_order)
            )
            results_df = results_df.sort_values(['type_order', 'Similarity_Score'], ascending=[True, False])
            results_df = results_df.drop('type_order', axis=1)

            # Создаем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.data_output_dir / f"analog_analysis_report_{timestamp}.xlsx"

            logger.info(f"💾 Сохранение отчета: {output_path}")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Основной лист с результатами (сохраняем батчами для больших файлов)
                if len(results_df) > 50000:
                    logger.info("📊 Большой отчет, сохранение батчами...")
                    # Для очень больших отчетов сохраняем только основные данные
                    main_columns = ['Raw_Name', 'Cleaned_Name', 'Lemmatized_Name', 'Normalized_Name',
                                  'Candidate_Name', 'Similarity_Score', 'Relation_Type',
                                  'Final_Decision', 'Comment']
                    available_columns = [col for col in main_columns if col in results_df.columns]
                    results_df[available_columns].to_excel(writer, sheet_name='Analog_Analysis', index=False)
                else:
                    results_df.to_excel(writer, sheet_name='Analog_Analysis', index=False)

                # Лист со статистикой по типам связей
                relation_stats = results_df['Relation_Type'].value_counts().reset_index()
                relation_stats.columns = ['Relation_Type', 'Count']
                relation_stats['Percentage'] = (relation_stats['Count'] / len(results_df) * 100).round(2)
                relation_stats.to_excel(writer, sheet_name='Relation_Statistics', index=False)

                # Лист со статистикой по категориям
                if 'Original_Category' in results_df.columns:
                    category_stats = results_df['Original_Category'].value_counts().head(20).reset_index()
                    category_stats.columns = ['Category', 'Count']
                    category_stats.to_excel(writer, sheet_name='Category_Statistics', index=False)

                # Лист с общей статистикой
                general_stats = pd.DataFrame([
                    ['Всего записей обработано', self.stats.total_records],
                    ['Найдено дубликатов', self.stats.duplicates_found],
                    ['Найдено аналогов', self.stats.analogs_found],
                    ['Батчей обработано', self.stats.batch_count],
                    ['Максимальное использование памяти (MB)', f"{self.stats.memory_usage_mb:.1f}"],
                    ['Время обработки (мин)', self._get_processing_time()],
                    ['Дата создания отчета', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                ], columns=['Параметр', 'Значение'])
                general_stats.to_excel(writer, sheet_name='General_Statistics', index=False)

            # Очищаем память
            del results_df
            self._cleanup_memory()

            logger.info(f"✅ Excel отчет создан: {output_path}")
            logger.info(f"💾 Размер файла: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Ошибка создания Excel отчета: {e}")
            raise

    def _get_processing_time(self) -> float:
        """Получение времени обработки в минутах"""
        if self.stats.start_time and self.stats.end_time:
            delta = self.stats.end_time - self.stats.start_time
            return round(delta.total_seconds() / 60, 2)
        return 0.0

    def run_full_analysis(self, sample_size: Optional[int] = None) -> str:
        """Запуск полного анализа поиска аналогов"""
        logger.info("🚀 Запуск полного анализа поиска аналогов...")

        try:
            # 1. Инициализация компонентов
            self.initialize_components()

            # 2. Загрузка существующих результатов (если включено)
            if self.load_existing_results:
                self._load_existing_results()

            # 3. Загрузка данных
            self.load_data()

            # 4. Ограничение выборки если указано
            if sample_size and sample_size < len(self.data):
                logger.info(f"📋 Используется выборка: {sample_size} из {len(self.data)} записей")
                self.data = self.data.sample(n=sample_size, random_state=42)
                self.stats.total_records = len(self.data)

            # 5. Предобработка данных
            self.preprocess_data()

            # 6. Поиск дубликатов
            duplicates, duplicate_groups = self.find_duplicates()

            # 7. Поиск аналогов (исключая дубликаты)
            analogs = self.search_analogs(exclude_duplicates=True, duplicate_groups=duplicate_groups)

            # 8. Создание отчета
            self.stats.end_time = datetime.now()
            report_path = self.create_excel_report(duplicates, analogs)

            # 9. Вывод итоговой статистики
            self._print_final_statistics()

            return report_path

        except Exception as e:
            logger.error(f"❌ Ошибка в процессе анализа: {e}")
            raise

    def _print_final_statistics(self):
        """Вывод итоговой статистики"""
        logger.info("📈 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"   📊 Всего записей: {self.stats.total_records}")
        logger.info(f"   ✅ Обработано записей: {self.stats.processed_records}")

        # Статистика по результатам
        total_duplicates = self.stats.duplicates_found + self.stats.preloaded_duplicates
        total_analogs = self.stats.analogs_found + self.stats.preloaded_analogs

        logger.info(f"   🔄 Найдено дубликатов: {self.stats.duplicates_found} (новых) + {self.stats.preloaded_duplicates} (предзагруженных) = {total_duplicates}")
        logger.info(f"   🔍 Найдено аналогов: {self.stats.analogs_found} (новых) + {self.stats.preloaded_analogs} (предзагруженных) = {total_analogs}")

        # Техническая статистика
        logger.info(f"   📦 Батчей обработано: {self.stats.batch_count}")
        logger.info(f"   💾 Максимальное использование памяти: {self.stats.memory_usage_mb:.1f} MB")
        logger.info(f"   ⏱️ Время обработки: {self._get_processing_time()} мин")
        logger.info(f"   ❌ Ошибок обработки: {self.stats.processing_errors}")

        # Информация о предзагруженных результатах
        if self.stats.preloaded_processed_names > 0:
            logger.info(f"   📂 Предзагружено обработанных наименований: {self.stats.preloaded_processed_names}")
            logger.info(f"   📄 Источник предзагруженных данных: {Path(self.stats.existing_report_path).name if self.stats.existing_report_path else 'N/A'}")

        # Рекомендации по оптимизации
        if self.stats.memory_usage_mb > self.memory_limit_mb:
            logger.warning(f"⚠️ Превышен лимит памяти. Рекомендуется увеличить batch_size или memory_limit_mb")

        if self.stats.processing_errors > 0:
            logger.warning(f"⚠️ Обнаружены ошибки обработки. Проверьте логи для деталей")

        # Рекомендации по инкрементальной обработке
        if not self.load_existing_results and self._find_latest_report():
            logger.info(f"💡 Совет: Используйте --load-existing-results для инкрементальной обработки")


def main():
    """Главная функция"""
    import argparse

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='SAMe Analog Search Processor')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Размер выборки для анализа (по умолчанию: весь датасет)')
    parser.add_argument('--input-file', type=str, default='src/data/input/main_dataset.xlsx',
                       help='Путь к входному файлу Excel')
    parser.add_argument('--output-dir', type=str, default='src/data/output',
                       help='Директория для сохранения отчета')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Размер батча для обработки (по умолчанию: 1000)')
    parser.add_argument('--memory-limit', type=int, default=4000,
                       help='Лимит памяти в MB (по умолчанию: 4000)')
    parser.add_argument('--disable-cache', action='store_true',
                       help='Отключить кэширование обработанных данных')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Принудительно переобработать данные (игнорировать кэш)')
    parser.add_argument('--load-existing-results', action='store_true',
                       help='Загрузить существующие результаты для инкрементальной обработки')

    args = parser.parse_args()

    print("🚀 SAMe Analog Search Processor")
    print("=" * 50)
    print(f"📂 Входной файл: {args.input_file}")
    print(f"📁 Выходная директория: {args.output_dir}")
    if args.sample_size:
        print(f"📊 Размер выборки: {args.sample_size}")
    else:
        print(f"📊 Обработка полного датасета")
    print()

    # Настройки анализа
    SAMPLE_SIZE = args.sample_size

    try:
        # Создаем процессор
        processor = AnalogSearchProcessor()

        # Обновляем пути если указаны
        if args.input_file != 'src/data/input/main_dataset.xlsx':
            processor.data_input_path = Path(args.input_file)
        if args.output_dir != 'src/data/output':
            processor.data_output_dir = Path(args.output_dir)
            processor.data_output_dir.mkdir(parents=True, exist_ok=True)

        # Применяем настройки оптимизации
        processor.batch_size = args.batch_size
        processor.memory_limit_mb = args.memory_limit
        processor.use_processed_cache = not args.disable_cache
        processor.load_existing_results = args.load_existing_results

        # Если принудительная переобработка, очищаем кэш
        if args.force_reprocess:
            processor.use_processed_cache = False
            logger.info("🔄 Принудительная переобработка данных (кэш отключен)")

        logger.info(f"⚙️ Настройки оптимизации:")
        logger.info(f"   📦 Размер батча: {processor.batch_size}")
        logger.info(f"   💾 Лимит памяти: {processor.memory_limit_mb} MB")
        logger.info(f"   🗄️ Кэширование: {'включено' if processor.use_processed_cache else 'отключено'}")
        logger.info(f"   📂 Загрузка существующих результатов: {'включено' if processor.load_existing_results else 'отключено'}")

        # Запускаем анализ
        report_path = processor.run_full_analysis(sample_size=SAMPLE_SIZE)

        if report_path:
            print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print(f"📄 Отчет сохранен: {report_path}")
            print(f"\n💡 Структура отчета:")
            print(f"   • Raw_Name - Оригинальное наименование")
            print(f"   • Cleaned_Name - Очищенное наименование")
            print(f"   • Lemmatized_Name - Лемматизированное наименование")
            print(f"   • Normalized_Name - Нормализованное наименование")
            print(f"   • Candidate_Name - Найденный аналог/дубликат")
            print(f"   • Similarity_Score - Оценка схожести (0-1)")
            print(f"   • Relation_Type - Тип связи (дубль/аналог/похожий)")
            print(f"   • Suggested_Category - Рекомендованная категория")
            print(f"   • Final_Decision - Финальное решение")
            print(f"   • Comment - Комментарий системы")
        else:
            print("❌ Не удалось создать отчет")

    except KeyboardInterrupt:
        print("\n👋 Анализ прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
