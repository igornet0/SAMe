#!/usr/bin/env python3
"""
Модуль обработки данных каталога
Создает обработанный набор данных со всеми полями и новыми индексами в папке data

УЛУЧШЕНИЯ v2.0:
- Интеграция специализированных модулей из same_clear
- Улучшенная нормализация цветов и технических терминов  
- Обработка синонимов и единиц измерения
- ML-подход для извлечения параметров
- Анализ качества извлеченных данных
- Замена самописной логики на специализированные модули
- Новые колонки: ml_parameters, parameter_analysis, final_name
- Гибкая конфигурация через аргументы командной строки

Автор: SAMe Data Processing System
Дата: 2025-01-27
Версия: 2.0 (улучшенная)
"""

import asyncio
import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
import gc
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import time
from datetime import datetime
from dataclasses import dataclass

# Импорт tqdm с fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc="Processing", **kwargs):
        """Fallback для tqdm если не установлен"""
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:  # Показываем прогресс каждые 5%
                print(f"{desc}: {i}/{total} ({i*100//total}%)")
            yield item

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

# Импорт модулей с обработкой ошибок
try:
    from data_manager import DataManager
    from same_clear.text_processing.enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessorConfig
    from same_clear.text_processing.text_cleaner import TextCleaner, CleaningConfig
    from same_clear.text_processing.normalizer import TextNormalizer, NormalizerConfig
    from same_clear.text_processing.color_normalizer import ColorNormalizer, ColorNormalizerConfig
    from same_clear.text_processing.technical_terms_normalizer import TechnicalTermsNormalizer, TechnicalTermsNormalizerConfig
    from same_clear.text_processing.synonyms_processor import SynonymsProcessor, SynonymsConfig
    from same_clear.text_processing.units_processor import UnitsProcessor, UnitsConfig
    from same_clear.text_processing.tech_codes_processor import TechCodesProcessor, TechCodesConfig
    from same_clear.parameter_extraction.regex_extractor import RegexParameterExtractor
    from same_clear.parameter_extraction.enhanced_parameter_extractor import EnhancedParameterExtractor
    from same_clear.parameter_extraction.ml_extractor import MLParameterExtractor, MLExtractorConfig
    from same_clear.parameter_extraction.parameter_utils import ParameterFormatter, ParameterAnalyzer, ParameterDataFrameUtils
    from same_search.categorization.category_classifier import CategoryClassifier, CategoryClassifierConfig
    from same_api.export.excel_exporter import ExcelExporter
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some SAMe modules not available: {e}")
    print("Running in basic mode without advanced features")
    MODULES_AVAILABLE = False
    
    # Заглушки для недоступных модулей
    class DataManager:
        def __init__(self): pass
    
    class EnhancedPreprocessor:
        def __init__(self, config): pass
        def preprocess_text(self, text): return {"normalized": text.lower(), "cleaned": text.lower()}
    
    class EnhancedPreprocessorConfig:
        def __init__(self): pass
    
    class TextCleaner:
        def __init__(self, config=None): pass
        def clean_text(self, text): return {"normalized": text}
    
    class CleaningConfig:
        def __init__(self): pass
    
    class TextNormalizer:
        def __init__(self, config=None): pass
        def normalize_text(self, text): return {"final_normalized": text.lower()}
    
    class NormalizerConfig:
        def __init__(self): pass
    
    class ColorNormalizer:
        def __init__(self, config=None): pass
        def normalize_colors(self, text): return {"normalized": text}
    
    class ColorNormalizerConfig:
        def __init__(self): pass
    
    class TechnicalTermsNormalizer:
        def __init__(self, config=None): pass
        def normalize_technical_terms(self, text): return {"normalized": text}
    
    class TechnicalTermsNormalizerConfig:
        def __init__(self): pass
    
    class SynonymsProcessor:
        def __init__(self, config=None): pass
        def process_synonyms(self, text): return {"processed": text}
    
    class SynonymsConfig:
        def __init__(self): pass
    
    class UnitsProcessor:
        def __init__(self, config=None): pass
        def process_units(self, text): return {"processed": text}
    
    class UnitsConfig:
        def __init__(self): pass
    
    class TechCodesProcessor:
        def __init__(self, config=None): pass
        def process_codes(self, text): return {"processed": text}
    
    class TechCodesConfig:
        def __init__(self): pass
    
    class RegexParameterExtractor:
        def __init__(self): pass
        def extract_parameters(self, text): return []
    
    class EnhancedParameterExtractor:
        def __init__(self): pass
        def extract_parameters(self, text): return []
    
    class MLParameterExtractor:
        def __init__(self, config=None): pass
        def extract_parameters(self, text): return []
    
    class MLExtractorConfig:
        def __init__(self): pass
    
    class ParameterFormatter:
        def __init__(self): pass
        @staticmethod
        def format_parameters_list(params): return ""
    
    class ParameterAnalyzer:
        def __init__(self): pass
        def analyze_parameters(self, params): return {}
    
    class ParameterDataFrameUtils:
        def __init__(self): pass
    
    class CategoryClassifier:
        def __init__(self, config): pass
        def classify(self, text): return ("общие_товары", 0.5)
    
    class CategoryClassifierConfig:
        def __init__(self): pass
    
    class ExcelExporter:
        def __init__(self): pass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Список брендов для извлечения
MODEL_BRANDS = [
    'neox', 'osairous', 'yealink', 'sanfor', 'санфор', 'биолан', 'нэфис',
    'персил', 'dallas', 'премиум', 'маяк', 'chint', 'andeli', 'grass',
    'kraft', 'reoflex', 'керхер', 'huawei', 'honor', 'ВЫСОТА', 'ugreen',
    'alisafox', 'маякавто', 'техноавиа', 'восток-сервис', 'attache', 'камаз',
    'зубр', 'hp', 'ekf', 'dexp', 'matrix', 'siemens', 'комус', 'gigant',
    'hyundai', 'iveco', 'stayer', 'brauberg', 'makita', 'bentec', 'сибртех',
    'bosch', 'rexant', 'sampa', 'kyocera', 'avrora', 'derrick', 'cummins',
    'economy', 'samsung', 'ofite', 'professional', 'caterpillar', 'intel',
    'proxima', 'core', 'shantui', 'king', 'office', 'петролеум', 'трейл',
    'skf', 'форвелд', 'скаймастер', 'tony', 'kentek', 'ресанта', 'dexter',
    'electric', 'оттм'
]

@dataclass
class ProcessingConfig:
    """Конфигурация обработки данных"""
    batch_size: int = 1000
    max_workers: int = 4
    save_format: str = "csv"  # csv, excel, parquet
    include_statistics: bool = True
    include_metadata: bool = True
    
    # Новые опции для улучшенной обработки
    enable_advanced_text_processing: bool = True
    enable_color_normalization: bool = True
    enable_technical_terms_normalization: bool = True
    enable_synonyms_processing: bool = True
    enable_units_processing: bool = True
    enable_tech_codes_processing: bool = True
    enable_ml_parameter_extraction: bool = True
    enable_parameter_analysis: bool = True
    
    # Пороги уверенности
    min_parameter_confidence: float = 0.3
    min_classification_confidence: float = 0.5

class DataProcessor:
    """Основной класс для обработки данных каталога"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.modules_available = MODULES_AVAILABLE
        
        # Инициализация компонентов обработки
        try:
            self.data_manager = DataManager()
            
            # Основные компоненты
            self.preprocessor = EnhancedPreprocessor(EnhancedPreprocessorConfig())
            self.parameter_extractor = RegexParameterExtractor()
            self.enhanced_parameter_extractor = EnhancedParameterExtractor()
            self.category_classifier = CategoryClassifier(CategoryClassifierConfig())
            self.excel_exporter = ExcelExporter()
            
            # Дополнительные компоненты для улучшенной обработки
            if self.config.enable_advanced_text_processing and self.modules_available:
                self.text_cleaner = TextCleaner(CleaningConfig())
                self.text_normalizer = TextNormalizer(NormalizerConfig())
                
                # Специализированные нормализаторы
                if self.config.enable_color_normalization:
                    self.color_normalizer = ColorNormalizer(ColorNormalizerConfig())
                else:
                    self.color_normalizer = None
                    
                if self.config.enable_technical_terms_normalization:
                    self.technical_terms_normalizer = TechnicalTermsNormalizer(TechnicalTermsNormalizerConfig())
                else:
                    self.technical_terms_normalizer = None
                    
                if self.config.enable_synonyms_processing:
                    self.synonyms_processor = SynonymsProcessor(SynonymsConfig())
                else:
                    self.synonyms_processor = None
                    
                if self.config.enable_units_processing:
                    self.units_processor = UnitsProcessor(UnitsConfig())
                else:
                    self.units_processor = None
                    
                if self.config.enable_tech_codes_processing:
                    self.tech_codes_processor = TechCodesProcessor(TechCodesConfig())
                else:
                    self.tech_codes_processor = None
                    
                # ML-экстрактор параметров
                if self.config.enable_ml_parameter_extraction:
                    try:
                        self.ml_parameter_extractor = MLParameterExtractor(MLExtractorConfig())
                    except Exception as e:
                        logger.warning(f"Failed to initialize ML parameter extractor: {e}")
                        self.ml_parameter_extractor = None
                else:
                    self.ml_parameter_extractor = None
                    
                # Анализатор параметров
                if self.config.enable_parameter_analysis:
                    self.parameter_analyzer = ParameterAnalyzer()
                    self.parameter_formatter = ParameterFormatter()
                else:
                    self.parameter_analyzer = None
                    self.parameter_formatter = None
            else:
                # Заглушки для базового режима
                self.text_cleaner = None
                self.text_normalizer = None
                self.color_normalizer = None
                self.technical_terms_normalizer = None
                self.synonyms_processor = None
                self.units_processor = None
                self.tech_codes_processor = None
                self.ml_parameter_extractor = None
                self.parameter_analyzer = None
                self.parameter_formatter = None
            
            if self.modules_available:
                logger.info("DataProcessor initialized successfully with full features")
            else:
                logger.info("DataProcessor initialized in basic mode")
        except Exception as e:
            logger.warning(f"Error initializing some components: {e}")
            self.modules_available = False
        
        # Данные
        self.original_df = None
        self.processed_df = None
    
    def extract_model_brand(self, text: str) -> Optional[str]:
        """Извлечение модели/бренда из текста"""
        if not text:
            return None
        
        text_lower = text.lower()
        
        for brand in MODEL_BRANDS:
            if brand.lower() in text_lower:
                return brand
        
        return None
    
    def extract_final_name_advanced(self, text: str) -> str:
        """Улучшенная очистка названий с использованием специализированных модулей"""
        if not text:
            return ""
        
        # Если доступны специализированные модули, используем их
        if (self.config.enable_advanced_text_processing and 
            self.modules_available and 
            self.text_cleaner and 
            self.text_normalizer):
            
            try:
                # Этап 1: Базовая очистка
                cleaned_result = self.text_cleaner.clean_text(text)
                cleaned_text = cleaned_result.get('normalized', text)
                
                # Этап 2: Обработка единиц измерения (убираем числовые значения с единицами)
                if self.units_processor:
                    units_result = self.units_processor.process_units(cleaned_text)
                    cleaned_text = units_result.get('processed', cleaned_text)
                
                # Этап 3: Обработка технических кодов
                if self.tech_codes_processor:
                    codes_result = self.tech_codes_processor.process_codes(cleaned_text)
                    cleaned_text = codes_result.get('processed', cleaned_text)
                
                # Этап 4: Нормализация цветов
                if self.color_normalizer:
                    color_result = self.color_normalizer.normalize_colors(cleaned_text)
                    cleaned_text = color_result.get('normalized', cleaned_text)
                
                # Этап 5: Нормализация технических терминов
                if self.technical_terms_normalizer:
                    tech_result = self.technical_terms_normalizer.normalize_technical_terms(cleaned_text)
                    cleaned_text = tech_result.get('normalized', cleaned_text)
                
                # Этап 6: Финальная нормализация
                final_result = self.text_normalizer.normalize_text(cleaned_text)
                final_text = final_result.get('final_normalized', cleaned_text)
                
                # Базовая проверка на качество результата
                if len(final_text.strip()) < 3:
                    return self.extract_final_name_fallback(text)
                
                return final_text.strip()
                
            except Exception as e:
                logger.warning(f"Advanced text processing failed: {e}, falling back to basic method")
                return self.extract_final_name_fallback(text)
        else:
            # Fallback к старому методу
            return self.extract_final_name_fallback(text)
    
    def extract_final_name_fallback(self, text: str) -> str:
        """Извлечение чистого наименования без параметров (старый метод)"""
        if not text:
            return ""
        
        import re
        
        # Создаем копию текста для обработки
        final_name = text.strip()
        
        # Сначала проверяем, не является ли это специальным форматом (с подчеркиваниями или кодами)
        # Если да, то применяем более щадящую очистку
        is_coded_format = bool(re.search(r'[_]{2,}|[А-Я]{2,}_|_[А-Я]{2,}', final_name))
        
        if is_coded_format:
            # Для кодированных форматов (типа Бейсболка_РН_ХБ_То_54) применяем минимальную очистку
            # Убираем только очевидные размеры в конце
            final_name = re.sub(r'_?\d+-\d+$', '', final_name)  # убираем размеры типа _55-57
            final_name = re.sub(r'_\d+$', '', final_name)       # убираем размеры типа _54
            final_name = re.sub(r'\s+\d+-\d+\s+\d+-\d+$', '', final_name)  # размеры через пробел
            
            # Убираем объемы в скобках
            final_name = re.sub(r'\(\s*\d+[.,]?\d*\s*(мл|л|г|кг)\s*\)', '', final_name)
            
        else:
            # Для обычных названий применяем полную очистку
            patterns_to_remove = [
                # Размеры и объемы (более точные паттерны)
                r'\b\d+[.,]?\d*\s*(мм|см|м|л|мл|г|кг|т)\b',
                r'\b\d+[xх×]\d+[xх×]?\d*\s*(мм|см|м)?\b',  # размеры типа 100x200
                
                # Мощность и электрические характеристики
                r'\b\d+[.,]?\d*\s*(Вт|W|в|V|А|A|кВт|кВ)\b',
                r'\b\d+[.,]?\d*\s*(ватт|вольт|ампер)\b',
                
                # Температура и цвета
                r'\b\d+[.,]?\d*\s*k\b',  # цветовая температура
                r'\b\d+[.,]?\d*°?[CС]\b',  # температура
                
                # IP рейтинги
                r'\bIP\d{2,3}\b',
                
                # Артикулы и коды (только четкие паттерны)
                r'\b[А-Я]{2,}-\d+\b',  # ДПО-108
                
                # Объемы в скобках
                r'\(\s*\d+[.,]?\d*\s*(мл|л|г|кг)\s*\)',
                
                # Лишние скобки с брендами
                r'\([^)]*[Сс]анфор[^)]*\)',
                r'\([^)]*[Нн]эфис[^)]*\)',
                
                # Технические спецификации
                r'\bEVRO-\d+\b',
                r'\bЕВРО-\d+\b',
                r'\bLED\b',
                r'\bSDS-plus\b',
                r'\bECAM\s+[\d.]+\b',
                r'\bGSB\s+[\dV-]+\b',
            ]
            
            # Применяем паттерны удаления только для обычных названий
            for pattern in patterns_to_remove:
                final_name = re.sub(pattern, ' ', final_name, flags=re.IGNORECASE)
        
        # Общая очистка для всех типов
        # Убираем лишние пробелы и знаки препинания, но более аккуратно
        final_name = re.sub(r'[,;.]{2,}', ' ', final_name)  # множественные знаки
        final_name = re.sub(r'\s*[,;.]\s*$', '', final_name)  # знаки в конце
        final_name = re.sub(r'^\s*[,;.]\s*', '', final_name)  # знаки в начале
        final_name = re.sub(r'\s+', ' ', final_name)
        final_name = final_name.strip()
        
        # Убираем слова-характеристики только для обычных форматов
        if not is_coded_format:
            words_to_remove = [
                'professional', 'профессиональный', 'premium', 'премиум',
                'standard', 'стандартный', 'basic', 'базовый',
                'ultra', 'ультра', 'super', 'супер',
                'plus', 'плюс', 'max', 'макс',
                'панель', 'светодиодный', 'светодиодная', 'светодиодное',
                'потолочный', 'потолочная', 'потолочное',
                'беспроводной', 'беспроводная', 'беспроводное',
            ]
            
            words = final_name.split()
            if len(words) > 2:  # Оставляем слова-характеристики только если название очень короткое
                filtered_words = []
                for word in words:
                    if word.lower() not in words_to_remove:
                        filtered_words.append(word)
                if len(filtered_words) >= 2:  # Убеждаемся что остается достаточно слов
                    final_name = ' '.join(filtered_words)
        
        # Финальная очистка от остаточных артефактов
        final_name = re.sub(r'\s*[-/]\s*[-/]\s*', '', final_name)  # убираем "- / -"
        final_name = re.sub(r'\s*[-/]\s*$', '', final_name)       # убираем "- /" в конце
        final_name = re.sub(r'^\s*[-/]\s*', '', final_name)       # убираем "- /" в начале
        final_name = re.sub(r'\s*,\s*,\s*', ' ', final_name)      # убираем двойные запятые
        final_name = re.sub(r'\s*,\s*$', '', final_name)          # убираем запятые в конце
        final_name = re.sub(r'^\s*,\s*', '', final_name)          # убираем запятые в начале
        final_name = re.sub(r'\s+', ' ', final_name)
        final_name = final_name.strip()
        
        # Дополнительная очистка размеров для всех форматов
        # Убираем размеры в формате "104-108 158-164", "р.104-108", "80-84 146-152"
        final_name = re.sub(r'\s+\d+-\d+\s+\d+-\d+$', '', final_name)
        final_name = re.sub(r'\s+р\.\d+-\d+\s+\d+-\d+$', '', final_name)
        final_name = re.sub(r'\s+\d+-\d+$', '', final_name)  # одинарные размеры в конце
        final_name = final_name.strip()
        
        # Проверка качества результата
        # Если результат состоит только из знаков препинания или слишком короткий
        if (not final_name or 
            len(final_name) < 3 or 
            re.match(r'^[-_/\s\d.]*$', final_name) or
            final_name.count('/') > len(final_name) // 3):
            # Возвращаем исходное название, но с минимальной очисткой
            final_name = text.strip()
            # Убираем только очевидные размеры в конце
            final_name = re.sub(r'\s+\d+-\d+\s+\d+-\d+$', '', final_name)
            final_name = re.sub(r'\s+р\.\d+.*$', '', final_name)  # убираем "р.61" и далее
            final_name = re.sub(r'\(\s*\d+[.,]?\d*\s*(мл|л|г|кг)\s*\)$', '', final_name)
            final_name = final_name.strip()
        
        return final_name
    
    def extract_parameters_advanced(self, text: str) -> Tuple[List, List, List, Dict]:
        """Улучшенное извлечение параметров с использованием ML и анализом качества"""
        basic_parameters = []
        enhanced_parameters = []
        ml_parameters = []
        analysis_result = {}
        
        try:
            # 1. Базовое извлечение через regex
            basic_parameters = self.parameter_extractor.extract_parameters(text)
            
            # 2. Расширенное извлечение
            try:
                enhanced_parameters = self.enhanced_parameter_extractor.extract_parameters(text)
            except Exception as e:
                logger.warning(f"Enhanced parameter extraction failed: {e}")
                enhanced_parameters = []
            
            # 3. ML-извлечение (если доступно)
            if self.ml_parameter_extractor and self.config.enable_ml_parameter_extraction:
                try:
                    ml_parameters = self.ml_parameter_extractor.extract_parameters(text)
                except Exception as e:
                    logger.warning(f"ML parameter extraction failed: {e}")
                    ml_parameters = []
            
            # 4. Анализ качества параметров (если доступно)
            if self.parameter_analyzer and self.config.enable_parameter_analysis:
                try:
                    all_params = basic_parameters + enhanced_parameters + ml_parameters
                    analysis_result = self.parameter_analyzer.analyze_parameters(all_params)
                except Exception as e:
                    logger.warning(f"Parameter analysis failed: {e}")
                    analysis_result = {}
            
            # 5. Фильтрация по уверенности
            filtered_basic = [p for p in basic_parameters if getattr(p, 'confidence', 1.0) >= self.config.min_parameter_confidence]
            filtered_enhanced = [p for p in enhanced_parameters if getattr(p, 'confidence', 1.0) >= self.config.min_parameter_confidence]
            filtered_ml = [p for p in ml_parameters if getattr(p, 'confidence', 1.0) >= self.config.min_parameter_confidence]
            
            return filtered_basic, filtered_enhanced, filtered_ml, analysis_result
            
        except Exception as e:
            logger.error(f"Advanced parameter extraction failed: {e}")
            return [], [], [], {}
    
    def format_parameters_advanced(self, basic_params: List, enhanced_params: List, ml_params: List = None) -> str:
        """Улучшенное форматирование параметров"""
        if self.parameter_formatter and self.config.enable_parameter_analysis:
            try:
                # Объединяем все параметры
                all_params = basic_params + enhanced_params
                if ml_params:
                    all_params.extend(ml_params)
                
                return self.parameter_formatter.format_parameters_list(all_params)
            except Exception as e:
                logger.warning(f"Advanced parameter formatting failed: {e}")
        
        # Fallback к базовому форматированию
        basic_str = '; '.join([f"{p.name}: {p.value}" for p in basic_params])
        enhanced_str = '; '.join([f"{p.name}: {p.value}" for p in enhanced_params])
        
        combined = []
        if basic_str:
            combined.append(basic_str)
        if enhanced_str:
            combined.append(enhanced_str)
        
        return '; '.join(combined)
    
    def find_name_column(self, df: pd.DataFrame) -> str:
        """Поиск колонки с названиями товаров"""
        possible_names = ['наименование', 'название', 'name', 'product_name', 'товар', 'item']
        
        for col in df.columns:
            col_lower = col.lower()
            for name_variant in possible_names:
                if name_variant in col_lower:
                    return col
        
        # Если не найдена специальная колонка, используем первую
        return df.columns[0]
    
    async def load_data(self, input_file: str) -> pd.DataFrame:
        """Загрузка исходных данных"""
        logger.info(f"Loading data from {input_file}")
        
        try:
            # Определение формата файла и загрузка
            if input_file.endswith('.csv'):
                self.original_df = pd.read_csv(input_file)
            elif input_file.endswith(('.xlsx', '.xls')):
                self.original_df = pd.read_excel(input_file)
            elif input_file.endswith('.parquet'):
                self.original_df = pd.read_parquet(input_file)
            else:
                raise ValueError(f"Unsupported file format: {input_file}")
            
            logger.info(f"Successfully loaded {len(self.original_df)} records")
            logger.info(f"Columns: {list(self.original_df.columns)}")
            
            return self.original_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    async def process_data(self) -> pd.DataFrame:
        """Основная обработка данных"""
        if self.original_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data processing...")
        
        # Создание копии для обработки
        self.processed_df = self.original_df.copy()
        
        # Добавление новых колонок с индексами и обработанными данными
        new_columns = {
            'original_index': range(len(self.processed_df)),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processed_name': '',      # Базовая очистка (исходный текст с минимальными изменениями)
            'normalized_name': '',     # Полная нормализация (лемматизация, приведение к нижнему регистру)
            'cleaned_name': '',        # Промежуточная очистка
            'final_name': '',          # Чистое наименование без параметров, только суть товара
            'model_brand': '',
            'extracted_parameters': '',
            'enhanced_parameters': '',
            'ml_parameters': '',       # ML-извлеченные параметры
            'parameter_analysis': '',  # Анализ качества параметров
            'category': '',
            'category_confidence': 0.0,
            'processing_status': 'pending'
        }
        
        for col_name, values in new_columns.items():
            self.processed_df[col_name] = values
        
        # Определение колонки с названиями
        name_column = self.find_name_column(self.original_df)
        logger.info(f"Using '{name_column}' as name column")
        
        # Обработка записей с прогресс-баром
        logger.info("Processing records...")
        
        total_records = len(self.processed_df)
        processed_count = 0
        error_count = 0
        
        for idx in tqdm(self.processed_df.index, desc="Processing records"):
            try:
                original_name = str(self.processed_df.loc[idx, name_column])
                
                # 1. Предобработка текста
                if self.modules_available:
                    try:
                        processed_result = self.preprocessor.preprocess_text(original_name)
                        
                        # Правильное извлечение результатов обработки
                        if isinstance(processed_result, dict):
                            # Получаем различные стадии обработки
                            original_text = processed_result.get('original', original_name)
                            final_text = processed_result.get('final_text', original_name.lower())
                            
                            # Пробуем извлечь промежуточные результаты
                            processing_stages = processed_result.get('processing_stages', {})
                            
                            # processed_name - сохраняем исходный текст с минимальной очисткой
                            processed_text = original_text.strip()
                            
                            # cleaned_name - промежуточная очистка
                            if 'cleaning' in processing_stages:
                                cleaned_text = processing_stages['cleaning'].get('cleaned_text', original_name.lower())
                            else:
                                cleaned_text = original_name.lower().strip()
                            
                            # normalized_name - финальная нормализация (лемматизация, приведение к нижнему регистру)
                            normalized_text = final_text
                            
                            self.processed_df.loc[idx, 'processed_name'] = processed_text
                            self.processed_df.loc[idx, 'cleaned_name'] = cleaned_text
                            self.processed_df.loc[idx, 'normalized_name'] = normalized_text
                        else:
                            # Fallback если результат не словарь
                            processed_text = original_name.strip()
                            cleaned_text = original_name.lower().strip()
                            normalized_text = str(processed_result).lower()
                            
                            self.processed_df.loc[idx, 'processed_name'] = processed_text
                            self.processed_df.loc[idx, 'cleaned_name'] = cleaned_text
                            self.processed_df.loc[idx, 'normalized_name'] = normalized_text
                    except Exception as e:
                        logger.warning(f"Text preprocessing failed for row {idx}: {e}")
                        # Базовая обработка в случае ошибки
                        processed_text = original_name.strip()
                        cleaned_text = original_name.lower().strip()
                        normalized_text = cleaned_text
                        
                        self.processed_df.loc[idx, 'processed_name'] = processed_text
                        self.processed_df.loc[idx, 'cleaned_name'] = cleaned_text
                        self.processed_df.loc[idx, 'normalized_name'] = normalized_text
                else:
                    # Простая обработка без модулей
                    processed_text = original_name.strip()
                    cleaned_text = original_name.lower().strip()
                    # Убираем лишние пробелы и знаки препинания
                    import re
                    normalized_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
                    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
                    
                    self.processed_df.loc[idx, 'processed_name'] = processed_text
                    self.processed_df.loc[idx, 'cleaned_name'] = cleaned_text
                    self.processed_df.loc[idx, 'normalized_name'] = normalized_text
                
                # 2. Извлечение чистого наименования товара (без параметров)
                final_name = self.extract_final_name_advanced(original_name)
                self.processed_df.loc[idx, 'final_name'] = final_name
                
                # 3. Извлечение модели/бренда
                model_brand = self.extract_model_brand(original_name)
                self.processed_df.loc[idx, 'model_brand'] = model_brand if model_brand else ''
                
                # 4. Улучшенное извлечение параметров
                if self.config.enable_advanced_text_processing and self.modules_available:
                    try:
                        basic_params, enhanced_params, ml_params, analysis = self.extract_parameters_advanced(original_name)
                        
                        # Форматирование параметров
                        basic_param_str = self.format_parameters_advanced(basic_params, [])
                        enhanced_param_str = self.format_parameters_advanced([], enhanced_params)
                        
                        self.processed_df.loc[idx, 'extracted_parameters'] = basic_param_str
                        self.processed_df.loc[idx, 'enhanced_parameters'] = enhanced_param_str
                        
                        # Добавляем ML параметры если доступны
                        if ml_params and self.config.enable_ml_parameter_extraction:
                            ml_param_str = self.format_parameters_advanced([], [], ml_params)
                            self.processed_df.loc[idx, 'ml_parameters'] = ml_param_str
                        else:
                            self.processed_df.loc[idx, 'ml_parameters'] = ''
                        
                        # Добавляем анализ качества если доступен
                        if analysis and self.config.enable_parameter_analysis:
                            self.processed_df.loc[idx, 'parameter_analysis'] = str(analysis)
                        else:
                            self.processed_df.loc[idx, 'parameter_analysis'] = ''
                            
                    except Exception as e:
                        logger.warning(f"Advanced parameter extraction failed for row {idx}: {e}")
                        # Fallback к базовому методу
                        basic_parameters = self.parameter_extractor.extract_parameters(original_name)
                        param_str = '; '.join([f"{p.name}: {p.value}" for p in basic_parameters])
                        self.processed_df.loc[idx, 'extracted_parameters'] = param_str
                        self.processed_df.loc[idx, 'enhanced_parameters'] = ''
                        self.processed_df.loc[idx, 'ml_parameters'] = ''
                        self.processed_df.loc[idx, 'parameter_analysis'] = ''
                else:
                    # Базовое извлечение параметров
                    basic_parameters = self.parameter_extractor.extract_parameters(original_name)
                    param_str = '; '.join([f"{p.name}: {p.value}" for p in basic_parameters])
                    self.processed_df.loc[idx, 'extracted_parameters'] = param_str
                    
                    # Расширенное извлечение параметров
                    try:
                        enhanced_parameters = self.enhanced_parameter_extractor.extract_parameters(original_name)
                        enhanced_param_str = '; '.join([f"{p.name}: {p.value}" for p in enhanced_parameters])
                        self.processed_df.loc[idx, 'enhanced_parameters'] = enhanced_param_str
                    except Exception as e:
                        logger.warning(f"Enhanced parameter extraction failed for row {idx}: {e}")
                        self.processed_df.loc[idx, 'enhanced_parameters'] = ''
                    
                    self.processed_df.loc[idx, 'ml_parameters'] = ''
                    self.processed_df.loc[idx, 'parameter_analysis'] = ''
                
                # 5. Классификация категории с фильтрацией по уверенности
                try:
                    category, confidence = self.category_classifier.classify(original_name)
                    
                    # Проверяем минимальную уверенность
                    if confidence >= self.config.min_classification_confidence:
                        self.processed_df.loc[idx, 'category'] = category
                        self.processed_df.loc[idx, 'category_confidence'] = confidence
                    else:
                        self.processed_df.loc[idx, 'category'] = 'общие_товары'
                        self.processed_df.loc[idx, 'category_confidence'] = confidence
                        
                except Exception as e:
                    logger.warning(f"Category classification failed for row {idx}: {e}")
                    self.processed_df.loc[idx, 'category'] = 'общие_товары'
                    self.processed_df.loc[idx, 'category_confidence'] = 0.0
                
                # Отметка успешной обработки
                self.processed_df.loc[idx, 'processing_status'] = 'completed'
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.processed_df.loc[idx, 'processing_status'] = f'error: {str(e)[:100]}'
                error_count += 1
                continue
        
        # Статистика обработки
        logger.info(f"Processing completed:")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Successfully processed: {processed_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Success rate: {(processed_count/total_records)*100:.1f}%")
        
        return self.processed_df
    
    async def save_processed_data(self, output_dir: Optional[Path] = None) -> Path:
        """Сохранение обработанных данных"""
        if self.processed_df is None:
            raise ValueError("No processed data. Call process_data() first.")
        
        # Определение папки для сохранения
        if output_dir is None:
            output_dir = Path("src/data/processed")
        
        # Создание папки с датой
        date_folder = output_dir / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        date_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {date_folder}")
        
        # 1. Основной файл с обработанными данными
        if self.config.save_format.lower() == "excel":
            main_file = date_folder / "processed_dataset.xlsx"
            self.processed_df.to_excel(main_file, index=False)
        elif self.config.save_format.lower() == "parquet":
            main_file = date_folder / "processed_dataset.parquet"
            self.processed_df.to_parquet(main_file, index=False)
        else:  # CSV по умолчанию
            main_file = date_folder / "processed_dataset.csv"
            self.processed_df.to_csv(main_file, index=False, encoding='utf-8')
        
        logger.info(f"Main dataset saved to {main_file}")
        
        # 2. Сводка по обработке
        if self.config.include_statistics:
            await self._save_processing_statistics(date_folder)
        
        # 3. Метаданные
        if self.config.include_metadata:
            await self._save_metadata(date_folder)
        
        # 4. Отдельные файлы по категориям (если есть категории)
        await self._save_category_files(date_folder)
        
        # 5. Файл с извлеченными параметрами
        await self._save_parameters_summary(date_folder)
        
        logger.info(f"All files saved successfully to {date_folder}")
        return date_folder
    
    async def _save_processing_statistics(self, output_dir: Path):
        """Сохранение статистики обработки"""
        stats_file = output_dir / "processing_statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("СТАТИСТИКА ОБРАБОТКИ ДАННЫХ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего записей: {len(self.processed_df)}\n\n")
            
            # Статистика по статусам
            status_counts = self.processed_df['processing_status'].value_counts()
            f.write("Статистика по статусам обработки:\n")
            for status, count in status_counts.items():
                f.write(f"  {status}: {count}\n")
            f.write("\n")
            
            # Статистика по категориям
            if 'category' in self.processed_df.columns:
                category_counts = self.processed_df['category'].value_counts()
                f.write("Статистика по категориям:\n")
                for category, count in category_counts.head(10).items():
                    f.write(f"  {category}: {count}\n")
                f.write("\n")
            
            # Статистика по брендам
            brand_counts = self.processed_df['model_brand'].value_counts()
            f.write("Топ-10 найденных брендов:\n")
            for brand, count in brand_counts.head(10).items():
                if brand:  # Пропускаем пустые значения
                    f.write(f"  {brand}: {count}\n")
        
        logger.info(f"Statistics saved to {stats_file}")
    
    async def _save_metadata(self, output_dir: Path):
        """Сохранение метаданных"""
        metadata_file = output_dir / "metadata.json"
        
        metadata = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "processor_version": "1.0.0",
                "total_records": len(self.processed_df),
                "original_columns": list(self.original_df.columns),
                "new_columns": [
                    "original_index", "processing_date", "processed_name", 
                    "normalized_name", "cleaned_name", "final_name", "model_brand", 
                    "extracted_parameters", "enhanced_parameters", "ml_parameters",
                    "parameter_analysis", "category", "category_confidence", 
                    "processing_status"
                ]
            },
            "configuration": {
                "batch_size": self.config.batch_size,
                "save_format": self.config.save_format,
                "include_statistics": self.config.include_statistics,
                "include_metadata": self.config.include_metadata
            }
        }
        
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")
    
    async def _save_category_files(self, output_dir: Path):
        """Сохранение отдельных файлов по категориям"""
        if 'category' in self.processed_df.columns:
            categories_dir = output_dir / "by_categories"
            categories_dir.mkdir(exist_ok=True)
            
            categories = self.processed_df['category'].unique()
            
            for category in categories:
                if category and category != 'общие_товары':
                    category_df = self.processed_df[self.processed_df['category'] == category]
                    
                    # Безопасное имя файла
                    safe_category = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    category_file = categories_dir / f"{safe_category}.csv"
                    
                    category_df.to_csv(category_file, index=False, encoding='utf-8')
                    
            logger.info(f"Category files saved to {categories_dir}")
    
    async def _save_parameters_summary(self, output_dir: Path):
        """Сохранение сводки по параметрам"""
        params_file = output_dir / "parameters_summary.txt"
        
        with open(params_file, 'w', encoding='utf-8') as f:
            f.write("СВОДКА ПО ИЗВЛЕЧЕННЫМ ПАРАМЕТРАМ\n")
            f.write("=" * 50 + "\n\n")
            
            # Подсчет записей с параметрами
            basic_params_count = len(self.processed_df[self.processed_df['extracted_parameters'] != ''])
            enhanced_params_count = len(self.processed_df[self.processed_df['enhanced_parameters'] != ''])
            
            f.write(f"Записи с базовыми параметрами: {basic_params_count}\n")
            f.write(f"Записи с расширенными параметрами: {enhanced_params_count}\n\n")
            
            # Примеры извлеченных параметров
            f.write("ПРИМЕРЫ ИЗВЛЕЧЕННЫХ ПАРАМЕТРОВ:\n")
            f.write("-" * 30 + "\n")
            
            sample_data = self.processed_df[
                (self.processed_df['extracted_parameters'] != '') | 
                (self.processed_df['enhanced_parameters'] != '')
            ].head(10)
            
            name_column = self.find_name_column(self.original_df)
            
            for idx, row in sample_data.iterrows():
                f.write(f"\nНазвание: {row[name_column]}\n")
                if row['extracted_parameters']:
                    f.write(f"Базовые параметры: {row['extracted_parameters']}\n")
                if row['enhanced_parameters']:
                    f.write(f"Расширенные параметры: {row['enhanced_parameters']}\n")
                f.write("-" * 30 + "\n")
        
        logger.info(f"Parameters summary saved to {params_file}")

async def process_catalog_data(input_file: str, config: ProcessingConfig) -> bool:
    """Главная функция обработки каталога"""
    try:
        logger.info("Starting catalog data processing...")
        
        # Инициализация процессора
        processor = DataProcessor(config)
        
        # Загрузка данных
        await processor.load_data(input_file)
        
        # Обработка данных
        processed_df = await processor.process_data()
        
        # Сохранение результатов
        result_folder = await processor.save_processed_data()
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Results saved to: {result_folder}")
        logger.info(f"Processed {len(processed_df)} records")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in process_catalog_data: {e}")
        return False

def main():
    """Главная функция с аргументами командной строки"""
    parser = argparse.ArgumentParser(
        description='Обработчик данных каталога - создает обработанный набор данных со всеми полями',
        epilog='''
Примеры использования:
  %(prog)s catalog.xlsx                           # Полная обработка Excel файла
  %(prog)s data.csv --format parquet             # Сохранение в Parquet
  %(prog)s big_file.xlsx --batch-size 500        # Для больших файлов
  %(prog)s catalog.xlsx --disable-ml             # Без ML-извлечения
  %(prog)s data.csv --disable-advanced           # Базовая обработка
  %(prog)s catalog.xlsx --min-param-confidence 0.5  # Высокая точность параметров
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', 
                       help='Входной файл с каталогом (CSV, Excel, Parquet)')
    parser.add_argument('--format', choices=['csv', 'excel', 'parquet'], default='csv',
                       help='Формат выходного файла (по умолчанию: csv)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Размер пакета для обработки (по умолчанию: 1000)')
    parser.add_argument('--no-stats', action='store_true',
                       help='Не создавать файлы статистики')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Не создавать файл метаданных')
    parser.add_argument('--workers', type=int, default=4,
                       help='Количество рабочих процессов (по умолчанию: 4)')
    
    # Новые опции для улучшенной обработки
    parser.add_argument('--disable-advanced', action='store_true',
                       help='Отключить улучшенную обработку текста')
    parser.add_argument('--disable-color-norm', action='store_true',
                       help='Отключить нормализацию цветов')
    parser.add_argument('--disable-tech-terms', action='store_true',
                       help='Отключить нормализацию технических терминов')
    parser.add_argument('--disable-synonyms', action='store_true',
                       help='Отключить обработку синонимов')
    parser.add_argument('--disable-units', action='store_true',
                       help='Отключить обработку единиц измерения')
    parser.add_argument('--disable-tech-codes', action='store_true',
                       help='Отключить обработку технических кодов')
    parser.add_argument('--disable-ml', action='store_true',
                       help='Отключить ML-извлечение параметров')
    parser.add_argument('--disable-analysis', action='store_true',
                       help='Отключить анализ качества параметров')
    parser.add_argument('--min-param-confidence', type=float, default=0.3,
                       help='Минимальная уверенность для параметров (по умолчанию: 0.3)')
    parser.add_argument('--min-class-confidence', type=float, default=0.5,
                       help='Минимальная уверенность для классификации (по умолчанию: 0.5)')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = ProcessingConfig(
        batch_size=args.batch_size,
        max_workers=args.workers,
        save_format=args.format,
        include_statistics=not args.no_stats,
        include_metadata=not args.no_metadata,
        
        # Новые настройки улучшенной обработки
        enable_advanced_text_processing=not args.disable_advanced,
        enable_color_normalization=not args.disable_color_norm,
        enable_technical_terms_normalization=not args.disable_tech_terms,
        enable_synonyms_processing=not args.disable_synonyms,
        enable_units_processing=not args.disable_units,
        enable_tech_codes_processing=not args.disable_tech_codes,
        enable_ml_parameter_extraction=not args.disable_ml,
        enable_parameter_analysis=not args.disable_analysis,
        min_parameter_confidence=args.min_param_confidence,
        min_classification_confidence=args.min_class_confidence
    )
    
    logger.info(f"Starting data processing with parameters:")
    logger.info(f"  Input file: {args.input_file}")
    logger.info(f"  Output format: {args.format}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Include statistics: {config.include_statistics}")
    logger.info(f"  Include metadata: {config.include_metadata}")
    logger.info(f"  Advanced text processing: {config.enable_advanced_text_processing}")
    logger.info(f"  Color normalization: {config.enable_color_normalization}")
    logger.info(f"  Technical terms normalization: {config.enable_technical_terms_normalization}")
    logger.info(f"  Synonyms processing: {config.enable_synonyms_processing}")
    logger.info(f"  Units processing: {config.enable_units_processing}")
    logger.info(f"  Tech codes processing: {config.enable_tech_codes_processing}")
    logger.info(f"  ML parameter extraction: {config.enable_ml_parameter_extraction}")
    logger.info(f"  Parameter analysis: {config.enable_parameter_analysis}")
    logger.info(f"  Min parameter confidence: {config.min_parameter_confidence}")
    logger.info(f"  Min classification confidence: {config.min_classification_confidence}")
    
    # Запуск обработки
    result = asyncio.run(process_catalog_data(args.input_file, config))
    
    if result:
        logger.info("✅ Data processing completed successfully!")
    else:
        logger.error("❌ Data processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
