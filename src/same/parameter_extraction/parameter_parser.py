"""
Главный модуль парсинга параметров, объединяющий regex и ML подходы
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

from .regex_extractor import RegexParameterExtractor, ExtractedParameter, ParameterType
from .ml_extractor import MLParameterExtractor, MLExtractorConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterParserConfig:
    """Конфигурация парсера параметров"""
    # Методы извлечения
    use_regex: bool = True
    use_ml: bool = False  # ML требует обучения
    
    # Стратегия комбинирования
    combination_strategy: str = "union"  # union, intersection, ml_priority, regex_priority
    
    # Фильтрация результатов
    min_confidence: float = 0.5
    remove_duplicates: bool = True
    max_parameters_per_text: int = 50
    
    # Конфигурации компонентов
    ml_config: MLExtractorConfig = None
    
    # Производительность
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 100


class ParameterParser:
    """Главный класс для парсинга параметров"""
    
    def __init__(self, config: ParameterParserConfig = None):
        self.config = config or ParameterParserConfig()
        
        # Инициализация экстракторов
        self.regex_extractor = None
        self.ml_extractor = None
        
        if self.config.use_regex:
            self.regex_extractor = RegexParameterExtractor()
            logger.info("Regex extractor initialized")
        
        if self.config.use_ml:
            self.ml_extractor = MLParameterExtractor(self.config.ml_config)
            logger.info("ML extractor initialized")
        
        # Статистика
        self.stats = {
            'total_processed': 0,
            'total_parameters_extracted': 0,
            'processing_time': 0,
            'last_update': None
        }
        
        logger.info("ParameterParser initialized")
    
    def parse_parameters(self, text: str) -> List[ExtractedParameter]:
        """
        Извлечение параметров из текста
        
        Args:
            text: Входной текст
            
        Returns:
            Список извлеченных параметров
        """
        if not text or not isinstance(text, str):
            return []
        
        start_time = time.time()
        
        # Результаты от разных методов
        regex_results = []
        ml_results = []
        
        # Regex извлечение
        if self.config.use_regex and self.regex_extractor:
            try:
                regex_results = self.regex_extractor.extract_parameters(text)
            except Exception as e:
                logger.error(f"Error in regex extraction: {e}")
        
        # ML извлечение
        if self.config.use_ml and self.ml_extractor and self.ml_extractor.is_trained:
            try:
                ml_results = self.ml_extractor.extract_parameters(text)
            except Exception as e:
                logger.error(f"Error in ML extraction: {e}")
        
        # Комбинирование результатов
        combined_results = self._combine_results(regex_results, ml_results, text)
        
        # Фильтрация и постобработка
        filtered_results = self._filter_results(combined_results)
        
        # Обновление статистики
        processing_time = time.time() - start_time
        self.stats['total_processed'] += 1
        self.stats['total_parameters_extracted'] += len(filtered_results)
        self.stats['processing_time'] += processing_time
        self.stats['last_update'] = time.time()
        
        return filtered_results
    
    def _combine_results(self, 
                        regex_results: List[ExtractedParameter], 
                        ml_results: List[ExtractedParameter],
                        text: str) -> List[ExtractedParameter]:
        """Комбинирование результатов разных методов"""
        
        if self.config.combination_strategy == "union":
            return self._union_combination(regex_results, ml_results)
        
        elif self.config.combination_strategy == "intersection":
            return self._intersection_combination(regex_results, ml_results)
        
        elif self.config.combination_strategy == "ml_priority":
            return self._priority_combination(ml_results, regex_results)
        
        elif self.config.combination_strategy == "regex_priority":
            return self._priority_combination(regex_results, ml_results)
        
        else:
            logger.warning(f"Unknown combination strategy: {self.config.combination_strategy}")
            return regex_results + ml_results
    
    def _union_combination(self, 
                          results1: List[ExtractedParameter], 
                          results2: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Объединение результатов (все уникальные параметры)"""
        combined = list(results1)
        
        # Добавляем результаты из второго списка, которых нет в первом
        existing_params = {(p.name, p.parameter_type, p.position) for p in results1}
        
        for param in results2:
            param_key = (param.name, param.parameter_type, param.position)
            if param_key not in existing_params:
                combined.append(param)
        
        return combined
    
    def _intersection_combination(self, 
                                 results1: List[ExtractedParameter], 
                                 results2: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Пересечение результатов (только совпадающие параметры)"""
        if not results1 or not results2:
            return []
        
        # Создаем индекс для быстрого поиска
        index2 = {(p.name, p.parameter_type): p for p in results2}
        
        intersection = []
        for param1 in results1:
            key = (param1.name, param1.parameter_type)
            if key in index2:
                param2 = index2[key]
                # Берем параметр с большей уверенностью
                if param1.confidence >= param2.confidence:
                    intersection.append(param1)
                else:
                    intersection.append(param2)
        
        return intersection
    
    def _priority_combination(self, 
                             primary_results: List[ExtractedParameter], 
                             secondary_results: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Комбинирование с приоритетом (сначала основные, затем дополнительные)"""
        combined = list(primary_results)
        
        # Создаем индекс существующих параметров
        existing_params = {(p.name, p.parameter_type) for p in primary_results}
        
        # Добавляем вторичные результаты, которых нет в основных
        for param in secondary_results:
            param_key = (param.name, param.parameter_type)
            if param_key not in existing_params:
                combined.append(param)
        
        return combined
    
    def _filter_results(self, results: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Фильтрация и постобработка результатов"""
        if not results:
            return []
        
        # Фильтрация по уверенности
        filtered = [p for p in results if p.confidence >= self.config.min_confidence]
        
        # Удаление дубликатов
        if self.config.remove_duplicates:
            filtered = self._remove_duplicates(filtered)
        
        # Ограничение количества параметров
        if len(filtered) > self.config.max_parameters_per_text:
            # Сортируем по уверенности и берем топ-N
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:self.config.max_parameters_per_text]
        
        # Валидация параметров
        filtered = self._validate_parameters(filtered)
        
        return filtered
    
    def _remove_duplicates(self, parameters: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Удаление дублирующихся параметров"""
        seen = {}
        unique_params = []
        
        for param in parameters:
            # Ключ для определения дубликатов
            key = (param.name, param.parameter_type, str(param.value))
            
            if key not in seen or param.confidence > seen[key].confidence:
                seen[key] = param
        
        return list(seen.values())
    
    def _validate_parameters(self, parameters: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Валидация извлеченных параметров"""
        valid_params = []
        
        for param in parameters:
            if self._is_valid_parameter(param):
                valid_params.append(param)
            else:
                logger.debug(f"Invalid parameter filtered out: {param.name} = {param.value}")
        
        return valid_params
    
    def _is_valid_parameter(self, param: ExtractedParameter) -> bool:
        """Проверка валидности параметра"""
        # Базовые проверки
        if not param.name or not param.value:
            return False
        
        # Проверка длины значения
        if len(str(param.value)) > 100:  # Слишком длинное значение
            return False
        
        # Специфичные проверки по типу параметра
        if param.parameter_type == ParameterType.DIMENSION:
            return self._validate_dimension_parameter(param)
        elif param.parameter_type == ParameterType.ELECTRICAL:
            return self._validate_electrical_parameter(param)
        elif param.parameter_type == ParameterType.WEIGHT:
            return self._validate_weight_parameter(param)
        
        return True
    
    def _validate_dimension_parameter(self, param: ExtractedParameter) -> bool:
        """Валидация размерного параметра"""
        try:
            if isinstance(param.value, (int, float)):
                return param.value > 0 and param.value < 10000  # Разумные пределы в мм
            elif isinstance(param.value, str):
                # Попытка извлечь число из строки
                import re
                numbers = re.findall(r'\d+(?:[.,]\d+)?', param.value)
                if numbers:
                    value = float(numbers[0].replace(',', '.'))
                    return value > 0 and value < 10000
        except:
            pass
        
        return True  # Если не можем проверить, считаем валидным
    
    def _validate_electrical_parameter(self, param: ExtractedParameter) -> bool:
        """Валидация электрического параметра"""
        try:
            if isinstance(param.value, (int, float)):
                # Разумные пределы для электрических параметров
                if param.name == 'voltage':
                    return 0 < param.value < 100000  # Вольты
                elif param.name == 'current':
                    return 0 < param.value < 10000   # Амперы
                elif param.name == 'power':
                    return 0 < param.value < 1000000 # Ватты
        except:
            pass
        
        return True
    
    def _validate_weight_parameter(self, param: ExtractedParameter) -> bool:
        """Валидация параметра веса"""
        try:
            if isinstance(param.value, (int, float)):
                return param.value > 0 and param.value < 100000  # Разумные пределы в кг
        except:
            pass
        
        return True
    
    def parse_batch(self, texts: List[str]) -> List[List[ExtractedParameter]]:
        """
        Пакетное извлечение параметров
        
        Args:
            texts: Список текстов
            
        Returns:
            Список результатов для каждого текста
        """
        if not texts:
            return []
        
        if self.config.enable_parallel_processing and len(texts) > self.config.batch_size:
            return self._parallel_parse_batch(texts)
        else:
            return self._sequential_parse_batch(texts)
    
    def _parallel_parse_batch(self, texts: List[str]) -> List[List[ExtractedParameter]]:
        """Параллельная обработка пакета"""
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Создаем задачи
            future_to_index = {
                executor.submit(self.parse_parameters, text): i 
                for i, text in enumerate(texts)
            }
            
            # Собираем результаты
            for future in future_to_index:
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing text {index}: {e}")
                    results[index] = []
        
        return results
    
    def _sequential_parse_batch(self, texts: List[str]) -> List[List[ExtractedParameter]]:
        """Последовательная обработка пакета"""
        results = []
        for text in texts:
            try:
                result = self.parse_parameters(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                results.append([])
        
        return results
    
    def parse_dataframe(self, 
                       df: pd.DataFrame, 
                       text_column: str,
                       output_column: str = 'extracted_parameters') -> pd.DataFrame:
        """
        Извлечение параметров из DataFrame
        
        Args:
            df: DataFrame с данными
            text_column: Название колонки с текстом
            output_column: Название выходной колонки
            
        Returns:
            DataFrame с добавленной колонкой параметров
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Получаем тексты
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Извлекаем параметры
        results = self.parse_batch(texts)
        
        # Добавляем результаты в DataFrame
        df_result = df.copy()
        df_result[output_column] = results
        
        return df_result
    
    def train_ml_extractor(self, 
                          texts: List[str], 
                          annotations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Обучение ML-экстрактора
        
        Args:
            texts: Тексты для обучения
            annotations: Аннотации параметров
            
        Returns:
            Метрики обучения
        """
        if not self.config.use_ml:
            raise ValueError("ML extraction is disabled in config")
        
        if not self.ml_extractor:
            self.ml_extractor = MLParameterExtractor(self.config.ml_config)
        
        # Подготовка данных
        training_data = self.ml_extractor.prepare_training_data(texts, annotations)
        
        # Обучение
        metrics = self.ml_extractor.train(training_data)
        
        logger.info("ML extractor training completed")
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики парсера"""
        stats = dict(self.stats)
        
        # Добавляем статистику компонентов
        if self.regex_extractor:
            stats['regex_extractor'] = self.regex_extractor.get_statistics()
        
        if self.ml_extractor:
            stats['ml_extractor'] = self.ml_extractor.get_statistics()
        
        # Вычисляем производные метрики
        if self.stats['total_processed'] > 0:
            stats['avg_parameters_per_text'] = (
                self.stats['total_parameters_extracted'] / self.stats['total_processed']
            )
            stats['avg_processing_time'] = (
                self.stats['processing_time'] / self.stats['total_processed']
            )
        
        return stats
    
    def save_models(self, directory: str):
        """Сохранение всех моделей"""
        from pathlib import Path
        
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем regex паттерны
        if self.regex_extractor:
            regex_path = save_dir / "regex_patterns.json"
            self.regex_extractor.save_patterns(str(regex_path))
        
        # Сохраняем ML модель
        if self.ml_extractor and self.ml_extractor.is_trained:
            ml_path = save_dir / "ml_extractor.pkl"
            self.ml_extractor.save_model(str(ml_path))
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Загрузка всех моделей"""
        from pathlib import Path
        
        load_dir = Path(directory)
        
        # Загружаем regex паттерны
        if self.regex_extractor:
            regex_path = load_dir / "regex_patterns.json"
            if regex_path.exists():
                self.regex_extractor.load_patterns(str(regex_path))
        
        # Загружаем ML модель
        if self.config.use_ml:
            ml_path = load_dir / "ml_extractor.pkl"
            if ml_path.exists():
                if not self.ml_extractor:
                    self.ml_extractor = MLParameterExtractor(self.config.ml_config)
                self.ml_extractor.load_model(str(ml_path))
        
        logger.info(f"Models loaded from {directory}")
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.stats = {
            'total_processed': 0,
            'total_parameters_extracted': 0,
            'processing_time': 0,
            'last_update': None
        }
        logger.info("Statistics reset")
