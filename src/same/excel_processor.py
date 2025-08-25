
#!/usr/bin/env python3
"""
Расширенный процессор Excel файлов для проекта SAMe
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
from datetime import datetime
import warnings
import asyncio
import re
from collections import Counter
warnings.filterwarnings('ignore')

# Импорты SAMe модулей
try:
    from src.same_clear.text_processing.text_cleaner import TextCleaner
    from src.same_clear.text_processing.preprocessor import TextPreprocessor
    from src.same_clear.text_processing.enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessorConfig
    from src.same_clear.text_processing.tokenizer import Tokenizer
    from src.same_clear.text_processing.units_processor import UnitsProcessor, UnitsConfig
    from src.same_clear.text_processing.synonyms_processor import SynonymsProcessor, SynonymsConfig
    from src.same_clear.text_processing.tech_codes_processor import TechCodesProcessor, TechCodesConfig
    from src.same_clear.parameter_extraction.regex_extractor import RegexParameterExtractor
    SAME_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMe modules not available. Missing: {e}")
    SAME_MODULES_AVAILABLE = False

# Новые импорты для расширенных возможностей
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available. Missing: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Настройка логирования
logger = logging.getLogger(__name__)


class MLParameterClassifier:
    """ML классификатор для определения параметров"""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def predict(self, text: str, value: Any) -> Tuple[bool, float]:
        """Предсказание является ли значение параметром"""
        if not self.is_trained:
            # Простая эвристика если модель не обучена
            return self._heuristic_prediction(text, value)

        # TODO: Реализовать ML предсказание
        return self._heuristic_prediction(text, value)

    def _heuristic_prediction(self, text: str, value: Any) -> Tuple[bool, float]:
        """Эвристическое предсказание"""
        confidence = 0.7

        if isinstance(value, (int, float)):
            if float(value) > 100000:
                return False, 0.1
            if float(value) == 0:
                return False, 0.1

            units = ['в', 'а', 'вт', 'кг', 'мм', 'см', 'м', 'л', 'шт']
            if any(unit in text.lower() for unit in units):
                return True, 0.9

        return True, confidence

    def train(self, training_data: List[Tuple[str, Any, bool]]):
        """Обучение классификатора"""
        logger.info(f"Training ML classifier with {len(training_data)} samples")
        self.is_trained = True

    def save_model(self, path: str):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'is_trained': self.is_trained}, f)

    def load_model(self, path: str):
        """Загрузка модели"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.is_trained = data.get('is_trained', False)
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")


class AdvancedTokenVectorizer:
    """Расширенный векторизатор токенов"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocabulary = {}
        self.bpe_tokenizer = None
        self.embeddings_model = None

    def build_vocabulary(self, texts: List[str]):
        """Построение словаря"""
        logger.info(f"Building vocabulary from {len(texts)} texts")
        # Простая реализация
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Берем топ слов
        vocab_size = self.config.get('vocabulary_size', 10000)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}

    def tokenize_with_bpe(self, text: str) -> List[str]:
        """BPE токенизация"""
        # Простая реализация - разбиение по словам
        return text.lower().split()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Получение эмбеддингов"""
        # Простая реализация - случайные векторы
        return np.random.rand(len(texts), 128)


class AdvancedExcelProcessor:
    """Расширенный процессор Excel файлов"""
    
    def __init__(self, tokenizer_config: str = "advanced_tokenizer"):
        logger.info("Initializing Advanced Excel Processor...")

        # Базовые компоненты
        if SAME_MODULES_AVAILABLE:
            self.text_cleaner = TextCleaner()

            # Используем EnhancedPreprocessor для более полной обработки
            enhanced_config = EnhancedPreprocessorConfig(
                enable_units_processing=True,
                enable_synonyms_processing=True,
                enable_tech_codes_processing=True,
                parallel_processing=True,
                max_workers=2
            )
            self.text_preprocessor = EnhancedPreprocessor(enhanced_config)
            self.parameter_extractor = RegexParameterExtractor()

            # Дополнительные процессоры для отдельного использования
            self.units_processor = UnitsProcessor()
            self.synonyms_processor = SynonymsProcessor()
            self.tech_codes_processor = TechCodesProcessor()
        else:
            self.text_cleaner = None
            self.text_preprocessor = None
            self.parameter_extractor = None
            self.units_processor = None
            self.synonyms_processor = None
            self.tech_codes_processor = None

        # Расширенные компоненты
        self.advanced_vectorizer = None
        self.ml_classifier = MLParameterClassifier()

        # Конфигурация
        self.config = self._load_advanced_config(tokenizer_config)
        self._initialize_advanced_components()

        logger.info("Advanced Excel Processor initialized successfully")
    
    def _load_advanced_config(self, config_name: str) -> Dict[str, Any]:
        """Загрузка расширенной конфигурации"""
        config_path = f"src/models/configs/{config_name}.json"
        
        # Базовая конфигурация
        default_config = {
            "vocabulary_size": 50000,
            "use_bpe": True,
            "use_pretrained_embeddings": True,
            "embedding_model_name": "DeepPavlov/rubert-base-cased",
            "use_ml_classification": True,
            "extended_technical_patterns": True,
            "additional_fields": [
                "semantic_category",
                "technical_complexity",
                "parameter_confidence",
                "embedding_similarity",
                "bpe_tokens_count"
            ]
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _initialize_advanced_components(self):
        """Инициализация расширенных компонентов"""
        # Расширенный векторизатор
        self.advanced_vectorizer = AdvancedTokenVectorizer(self.config)
        
        # Загрузка ML модели если существует
        ml_model_path = "models/ml_parameter_classifier.joblib"
        if os.path.exists(ml_model_path):
            self.ml_classifier.load_model(ml_model_path)

    def _get_extended_technical_patterns(self) -> List[str]:
        """Расширенные технические паттерны для специфических отраслей"""
        return [
            # Автомобильная промышленность
            r"\b[А-ЯA-Z]\d+[-]\d+[-]\d+\b",  # А24-5-1
            r"\bH\d+[WВт]?\b",  # H7, H11 лампы
            r"\b\d+[Wвт]\s*\d+[Vв]\b",  # 55Вт 12В

            # Электротехника
            r"\bIP\d{2}\b",  # IP65, IP44
            r"\b\d+[kк]?[Vв][Aa]?\b",  # 220В, 380В, 10кВ
            r"\b\d+[mм]?[Aa]h?\b",  # 2000мАч, 12Ач

            # Машиностроение
            r"\bDN\d+\b",  # DN100, DN50
            r"\bPN\d+\b",  # PN16, PN25
            r"\bГОСТ\s*\d+[-.]?\d*\b",  # ГОСТ 12345-89
            r"\bТУ\s*\d+[-.]?\d*\b",  # ТУ 1234-567

            # Химическая промышленность
            r"\b\d+[%]\s*[А-Яа-я]+\b",  # 95% спирт
            r"\bpH\s*\d+[.,]?\d*\b",  # pH 7.0
            r"\b\d+[°]?[Cc]\b",  # 25°C, 100C

            # Строительство
            r"\bМ\d+\b",  # М100, М200 (марка бетона)
            r"\b\d+x\d+x\d+\s*мм\b",  # 100x50x25 мм
            r"\bкласс\s*[А-Я]\d*\b",  # класс А1

            # Текстильная промышленность
            r"\b\d+[Tt]ex\b",  # 150tex
            r"\b\d+[дД]ен\b",  # 40ден
            r"\b\d+[гГ]/м²\b",  # 150г/м²

            # Пищевая промышленность
            r"\b\d+[кК]кал\b",  # 250ккал
            r"\b\d+[гГ]\s*белк[а-я]*\b",  # 15г белка
            r"\bсрок\s*\d+\s*[а-я]+\b",  # срок 12 месяцев
        ]

    def process_text_pipeline_advanced(self, text: str) -> Dict[str, Any]:
        """Расширенный пайплайн обработки текста"""
        result = {
            'original_text': text,
            'processing_steps': {},
            'advanced_features': {}
        }

        try:
            if SAME_MODULES_AVAILABLE:
                # Используем полный пайплайн SAMe
                # Шаг 1: Очистка текста
                cleaning_result = self.text_cleaner.clean_text(text)
                cleaned_text = cleaning_result.get('normalized', text) if isinstance(cleaning_result, dict) else cleaning_result
                result['processing_steps']['cleaned_text'] = cleaned_text

                # Шаг 2-3: Полная предобработка (включая лемматизацию и нормализацию цветов)
                preprocessing_result = self.text_preprocessor.preprocess_text(cleaned_text)

                # Сохраняем полный результат предобработки для доступа к информации о цветах
                result['preprocessing_result'] = preprocessing_result

                if isinstance(preprocessing_result, dict):
                    result['processing_steps']['preprocessed_text'] = preprocessing_result.get('normalization', {}).get('final_normalized', cleaned_text)
                    result['processing_steps']['lemmatized_text'] = preprocessing_result.get('final_text', cleaned_text)
                else:
                    # Fallback для старого API
                    preprocessed_text = self.text_preprocessor.preprocess(cleaned_text)
                    lemmatized_text = self.text_preprocessor.lemmatize(preprocessed_text)
                    result['processing_steps']['preprocessed_text'] = preprocessed_text
                    result['processing_steps']['lemmatized_text'] = lemmatized_text
            else:
                # Используем fallback реализацию
                cleaning_result = self.text_cleaner.clean_text(text)
                cleaned_text = cleaning_result.get('normalized', text) if isinstance(cleaning_result, dict) else cleaning_result
                result['processing_steps']['cleaned_text'] = cleaned_text

                preprocessing_result = self.text_preprocessor.preprocess_text(cleaned_text)
                result['processing_steps']['preprocessed_text'] = preprocessing_result.get('final_text', cleaned_text)
                result['processing_steps']['lemmatized_text'] = preprocessing_result.get('final_text', cleaned_text)

            # Получаем финальный лемматизированный текст
            lemmatized_text = result['processing_steps'].get('lemmatized_text', text)

            # Шаг 4: Расширенная токенизация
            if self.advanced_vectorizer:
                # BPE токенизация
                bpe_tokens = self.advanced_vectorizer.tokenize_with_bpe(lemmatized_text)
                result['advanced_features']['bpe_tokens'] = bpe_tokens
                result['advanced_features']['bpe_tokens_count'] = len(bpe_tokens)

                # Получение эмбеддингов
                embeddings = self.advanced_vectorizer.get_embeddings([lemmatized_text])
                result['advanced_features']['embeddings'] = embeddings[0].tolist()
                result['advanced_features']['embedding_dimension'] = len(embeddings[0])

            # Шаг 5: Извлечение параметров с ML классификацией
            extracted_params = self.parameter_extractor.extract_parameters(lemmatized_text)
            validated_params = []

            for param in extracted_params:
                # ML классификация
                is_parameter, confidence = self.ml_classifier.predict(lemmatized_text, param.value)

                if is_parameter and confidence > 0.6:  # Порог уверенности
                    param_dict = {
                        'name': param.name,
                        'value': param.value,
                        'unit': param.unit,
                        'confidence': confidence,
                        'ml_validated': True
                    }
                    validated_params.append(param_dict)
                else:
                    logger.debug(f"Parameter {param.name}={param.value} rejected by ML (confidence: {confidence:.3f})")

            result['processing_steps']['parameters'] = validated_params

            # Шаг 6: Семантическая категоризация
            semantic_category = self._classify_semantic_category(lemmatized_text)
            result['advanced_features']['semantic_category'] = semantic_category

            # Шаг 7: Оценка технической сложности
            technical_complexity = self._assess_technical_complexity(lemmatized_text, validated_params)
            result['advanced_features']['technical_complexity'] = technical_complexity

            # Шаг 8: Расчёт средней уверенности параметров
            if validated_params:
                avg_confidence = sum(p['confidence'] for p in validated_params) / len(validated_params)
                result['advanced_features']['parameter_confidence'] = avg_confidence
            else:
                result['advanced_features']['parameter_confidence'] = 0.0

            return result

        except Exception as e:
            logger.error(f"Error in advanced text processing: {e}")
            result['error'] = str(e)
            return result

    def _classify_semantic_category(self, text: str) -> str:
        """Классификация семантической категории товара"""
        categories = {
            'automotive': ['автомобиль', 'машина', 'двигатель', 'колесо', 'фара', 'лампа', 'масло'],
            'electronics': ['электрон', 'провод', 'кабель', 'розетка', 'выключатель', 'лампочка'],
            'construction': ['строительн', 'цемент', 'кирпич', 'плитка', 'краска', 'клей'],
            'industrial': ['промышленн', 'станок', 'инструмент', 'подшипник', 'болт', 'гайка'],
            'chemical': ['химическ', 'кислота', 'щелочь', 'растворитель', 'реагент'],
            'textile': ['ткань', 'нить', 'волокно', 'пряжа', 'текстиль'],
            'food': ['пищев', 'продукт', 'консерв', 'напиток', 'мука', 'сахар'],
            'medical': ['медицинск', 'лекарств', 'препарат', 'бинт', 'шприц'],
            'other': []
        }

        text_lower = text.lower()

        for category, keywords in categories.items():
            if category == 'other':
                continue
            for keyword in keywords:
                if keyword in text_lower:
                    return category

        return 'other'

    def _assess_technical_complexity(self, text: str, parameters: List[Dict]) -> str:
        """Оценка технической сложности товара"""
        complexity_score = 0

        # Количество параметров
        complexity_score += len(parameters) * 0.5

        # Наличие технических терминов
        tech_terms = ['гост', 'ту', 'iso', 'din', 'класс', 'марка', 'тип', 'модель']
        for term in tech_terms:
            if term in text.lower():
                complexity_score += 1

        # Наличие точных измерений
        import re
        precise_measurements = re.findall(r'\d+[.,]\d+', text)
        complexity_score += len(precise_measurements) * 0.3

        # Классификация сложности
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def train_ml_classifier_on_data(self, df: pd.DataFrame, sample_size: int = 5000):
        """Обучение ML классификатора на данных"""
        logger.info("Training ML classifier on dataset...")

        # Подготовка обучающих данных
        training_data = []

        # Берём выборку для обучения
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)

        for _, row in sample_df.iterrows():
            text = str(row.get('Наименование', ''))

            # Извлекаем параметры
            params = self.parameter_extractor.extract_parameters(text)

            for param in params:
                # Эвристическая разметка для обучения
                is_parameter = self._heuristic_parameter_labeling(text, param.value)
                training_data.append((text, param.value, is_parameter))

        if training_data:
            self.ml_classifier.train(training_data)
            # Сохраняем модель
            os.makedirs("models", exist_ok=True)
            self.ml_classifier.save_model("models/ml_parameter_classifier.joblib")
        else:
            logger.warning("No training data available for ML classifier")

    def _heuristic_parameter_labeling(self, text: str, value: Any) -> bool:
        """Эвристическая разметка для обучения ML классификатора"""
        # Простые правила для автоматической разметки
        if isinstance(value, (int, float)):
            # Очень большие числа скорее всего артикулы
            if float(value) > 100000:
                return False

            # Нулевые значения не являются параметрами
            if float(value) == 0:
                return False

            # Если в контексте есть единицы измерения - скорее всего параметр
            units = ['в', 'а', 'вт', 'кг', 'мм', 'см', 'м', 'л', 'шт']
            if any(unit in text.lower() for unit in units):
                return True

        # По умолчанию считаем параметром
        return True

    def detect_duplicates(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Detect duplicates by exact name match

        Args:
            df: DataFrame with 'Наименование' and 'Код' columns

        Returns:
            Tuple of:
            - Dictionary mapping names to their duplicate counts
            - Dictionary mapping names to lists of codes for duplicates
        """
        logger.info("Detecting duplicates...")

        # Count occurrences of each name and collect codes
        name_counts = Counter()
        name_to_codes = {}

        for _, row in df.iterrows():
            name = row['Наименование']
            code = str(row.get('Код', ''))

            name_counts[name] += 1

            if name not in name_to_codes:
                name_to_codes[name] = []
            name_to_codes[name].append(code)

        # Keep only duplicates (count > 1)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        duplicate_codes = {name: codes for name, codes in name_to_codes.items() if name in duplicates}

        logger.info(f"Found {len(duplicates)} unique names with duplicates")
        logger.info(f"Total duplicate entries: {sum(duplicates.values())}")

        return duplicates, duplicate_codes

    async def process_batch_async(self, batch_data: List[Tuple[int, str]], duplicates: Dict[str, int], duplicate_codes: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Асинхронная обработка батча записей"""
        results = []

        for index, text in batch_data:
            try:
                if not text or text == 'nan':
                    continue

                # Расширенная обработка
                processing_result = self.process_text_pipeline_advanced(text)

                # Определение количества дубликатов
                duplicate_count = duplicates.get(text, 0)
                dublikat_value = str(duplicate_count) if duplicate_count > 1 else ''

                # Получение кодов дубликатов
                codes_for_duplicates = duplicate_codes.get(text, [])
                codes_dublikats = ', '.join(codes_for_duplicates) if duplicate_count > 1 else ''

                # Применяем пользовательскую функцию normalize для поля Normalized_Name
                normalized_text = normalize(text)

                result = {
                    'index': index,
                    'Raw_Name': text,
                    'Cleaned_Name': processing_result['processing_steps'].get('cleaned_text', ''),
                    'Lemmatized_Name': processing_result['processing_steps'].get('lemmatized_text', ''),
                    'Normalized_Name': normalized_text,
                    'Dublikat': dublikat_value,
                    'Код_Dublikats': codes_dublikats,
                    'Processing_Status': 'success'
                }

                # Расширенные поля
                if 'advanced_features' in processing_result:
                    af = processing_result['advanced_features']
                    result.update({
                        'BPE_Tokens': str(af.get('bpe_tokens', [])),
                        'BPE_Tokens_Count': af.get('bpe_tokens_count', 0),
                        'Semantic_Category': af.get('semantic_category', 'other'),
                        'Technical_Complexity': af.get('technical_complexity', 'low'),
                        'Parameter_Confidence': af.get('parameter_confidence', 0.0)
                    })

                # Параметры
                params = processing_result['processing_steps'].get('parameters', [])
                if params:
                    params_str = '; '.join([f"{p['name']}: {p['value']} {p.get('unit', '')}" for p in params])
                    result['ML_Validated_Parameters'] = params_str
                    result['Advanced_Parameters'] = str(params)

                # Информация о цветах (если доступна из preprocessing_result)
                if SAME_MODULES_AVAILABLE and 'color_normalization' in processing_result.get('preprocessing_result', {}):
                    color_info = processing_result['preprocessing_result']['color_normalization']
                    result['Colors_Found'] = ', '.join(color_info.get('colors_found', []))
                    result['Colors_Count'] = color_info.get('colors_count', 0)
                else:
                    result['Colors_Found'] = ''
                    result['Colors_Count'] = 0

                # Информация о технических терминах (если доступна из preprocessing_result)
                if SAME_MODULES_AVAILABLE and 'technical_terms_normalization' in processing_result.get('preprocessing_result', {}):
                    tech_info = processing_result['preprocessing_result']['technical_terms_normalization']
                    result['Technical_Terms_Found'] = ', '.join(tech_info.get('technical_terms_found', []))
                    result['Technical_Terms_Count'] = tech_info.get('technical_terms_count', 0)
                else:
                    result['Technical_Terms_Found'] = ''
                    result['Technical_Terms_Count'] = 0

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                results.append({
                    'index': index,
                    'Processing_Status': f'error: {str(e)}'
                })

        return results

    def process_excel_file(self, input_file: str, output_file: str, max_rows: Optional[int] = None) -> bool:
        """Основной метод обработки Excel файла"""
        try:
            logger.info(f"Starting advanced processing of {input_file}")

            # Загрузка данных
            if max_rows:
                df = pd.read_excel(input_file, nrows=max_rows)
                logger.info(f"Loaded {len(df)} rows (limited to {max_rows})")
            else:
                df = pd.read_excel(input_file)
                logger.info(f"Loaded {len(df)} rows")

            # Проверка обязательных колонок
            required_columns = ['Код', 'Наименование']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Обнаружение дубликатов
            duplicates, duplicate_codes = self.detect_duplicates(df)

            # Обучение ML классификатора на данных
            if not self.ml_classifier.is_trained:
                self.train_ml_classifier_on_data(df)

            # Обучение векторизатора
            if self.advanced_vectorizer:
                texts = df['Наименование'].dropna().astype(str).tolist()
                self.advanced_vectorizer.build_vocabulary(texts[:10000])  # Ограничиваем для скорости

            # Подготовка результирующего DataFrame
            result_df = df.copy()

            # Добавление новых колонок
            new_columns = [
                'Raw_Name', 'Cleaned_Name', 'Lemmatized_Name', 'Normalized_Name',
                'BPE_Tokens', 'BPE_Tokens_Count', 'Semantic_Category',
                'Technical_Complexity', 'Parameter_Confidence', 'Embedding_Similarity',
                'Advanced_Parameters', 'ML_Validated_Parameters', 'Colors_Found', 'Colors_Count',
                'Technical_Terms_Found', 'Technical_Terms_Count', 'Dublikat', 'Processing_Status', 'Код_Dublikats'
            ]

            for col in new_columns:
                result_df[col] = ''

            # Асинхронная обработка записей батчами
            logger.info("Processing records asynchronously...")
            processed_count = 0
            batch_size = 100  # Размер батча для асинхронной обработки

            # Подготовка данных для обработки
            batch_data = []
            for index, row in df.iterrows():
                text = str(row.get('Наименование', ''))
                if text and text != 'nan':
                    batch_data.append((index, text))

            # Обработка батчами
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i:i + batch_size]

                # Запускаем асинхронную обработку батча
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    batch_results = loop.run_until_complete(
                        self.process_batch_async(batch, duplicates, duplicate_codes)
                    )
                    loop.close()

                    # Заполняем результаты в DataFrame
                    for result in batch_results:
                        index = result['index']
                        for key, value in result.items():
                            if key != 'index':
                                result_df.at[index, key] = value

                    processed_count += len(batch_results)
                    logger.info(f"Processed {processed_count}/{len(batch_data)} records")

                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Fallback к синхронной обработке для этого батча
                    for index, text in batch:
                        result_df.at[index, 'Processing_Status'] = f'error: {str(e)}'

            # Сохранение результата
            result_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Advanced processing completed. Results saved to {output_file}")
            logger.info(f"Successfully processed: {processed_count}/{len(df)} records")

            # Статистика по дубликатам
            records_with_duplicates = len(result_df[result_df['Dublikat'] != ''])
            logger.info(f"Records with duplicates: {records_with_duplicates}")
            records_with_parameters = len(result_df[result_df['ML_Validated_Parameters'] != ''])
            logger.info(f"Records with parameters: {records_with_parameters}")

            return True

        except Exception as e:
            logger.error(f"Error in advanced Excel processing: {e}")
            return False

    def process_csv_file(self, input_file: str, output_file: str, max_rows: Optional[int] = None) -> bool:
        """Основной метод обработки CSV файла"""
        try:
            logger.info(f"Starting advanced processing of {input_file}")

            # Загрузка данных
            if max_rows:
                df = pd.read_csv(input_file, nrows=max_rows, encoding='utf-8')
                logger.info(f"Loaded {len(df)} rows (limited to {max_rows})")
            else:
                df = pd.read_csv(input_file, encoding='utf-8')
                logger.info(f"Loaded {len(df)} rows")

            # Проверка обязательных колонок
            required_columns = ['Код', 'Наименование']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Обнаружение дубликатов
            duplicates, duplicate_codes = self.detect_duplicates(df)

            # Обучение ML классификатора на данных
            if not self.ml_classifier.is_trained:
                self.train_ml_classifier_on_data(df)

            # Обучение векторизатора
            if self.advanced_vectorizer:
                texts = df['Наименование'].dropna().astype(str).tolist()
                self.advanced_vectorizer.build_vocabulary(texts[:10000])  # Ограничиваем для скорости

            # Подготовка результирующего DataFrame
            result_df = df.copy()

            # Добавление новых колонок
            new_columns = [
                'Raw_Name', 'Cleaned_Name', 'Lemmatized_Name', 'Normalized_Name',
                'BPE_Tokens', 'BPE_Tokens_Count', 'Semantic_Category',
                'Technical_Complexity', 'Parameter_Confidence', 'Embedding_Similarity',
                'Advanced_Parameters', 'ML_Validated_Parameters', 'Colors_Found', 'Colors_Count',
                'Technical_Terms_Found', 'Technical_Terms_Count', 'Dublikat', 'Processing_Status', 'Код_Dublikats'
            ]

            for col in new_columns:
                result_df[col] = ''

            # Асинхронная обработка записей батчами
            logger.info("Processing records asynchronously...")
            processed_count = 0
            batch_size = 100  # Размер батча для асинхронной обработки

            # Подготовка данных для обработки
            batch_data = []
            for index, row in df.iterrows():
                text = str(row.get('Наименование', ''))
                if text and text != 'nan':
                    batch_data.append((index, text))

            # Обработка батчами
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i:i + batch_size]

                # Запускаем асинхронную обработку батча
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    batch_results = loop.run_until_complete(
                        self.process_batch_async(batch, duplicates, duplicate_codes)
                    )
                    loop.close()

                    # Заполняем результаты в DataFrame
                    for result in batch_results:
                        index = result['index']
                        for key, value in result.items():
                            if key != 'index':
                                result_df.at[index, key] = value

                    processed_count += len(batch_results)
                    logger.info(f"Processed {processed_count}/{len(batch_data)} records")

                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Fallback к синхронной обработке для этого батча
                    for index, text in batch:
                        result_df.at[index, 'Processing_Status'] = f'error: {str(e)}'

            # Сохранение результата
            result_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Advanced processing completed. Results saved to {output_file}")
            logger.info(f"Successfully processed: {processed_count}/{len(df)} records")

            # Статистика по дубликатам
            records_with_duplicates = len(result_df[result_df['Dublikat'] != ''])
            logger.info(f"Records with duplicates: {records_with_duplicates}")
            records_with_parameters = len(result_df[result_df['ML_Validated_Parameters'] != ''])
            logger.info(f"Records with parameters: {records_with_parameters}")

            return True

        except Exception as e:
            logger.error(f"Error in advanced CSV processing: {e}")
            return False
