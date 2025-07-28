"""
Улучшенный предобработчик с интеграцией всех новых модулей
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .preprocessor import TextPreprocessor, PreprocessorConfig
from .units_processor import UnitsProcessor, UnitsConfig
from .synonyms_processor import SynonymsProcessor, SynonymsConfig
from .tech_codes_processor import TechCodesProcessor, TechCodesConfig
from .interfaces import TextPreprocessorInterface

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPreprocessorConfig:
    """Конфигурация улучшенного предобработчика"""
    # Базовая конфигурация
    base_config: Optional[PreprocessorConfig] = None
    
    # Конфигурации новых модулей
    units_config: Optional[UnitsConfig] = None
    synonyms_config: Optional[SynonymsConfig] = None
    tech_codes_config: Optional[TechCodesConfig] = None
    
    # Включение/выключение модулей
    enable_units_processing: bool = True
    enable_synonyms_processing: bool = True
    enable_tech_codes_processing: bool = True
    
    # Порядок обработки (ИСПРАВЛЕНО: units_processing перед base_preprocessing)
    processing_order: List[str] = field(default_factory=lambda: [
        'cleaning_only',         # Только очистка без нормализации чисел
        'units_processing',      # Обработка единиц измерения (до токенизации чисел)
        'tech_codes_processing', # Обработка технических кодов
        'base_normalization',    # Нормализация и лемматизация (после извлечения параметров)
        'synonyms_processing'    # Нормализация синонимов (в конце)
    ])
    
    # Настройки производительности
    parallel_processing: bool = True
    max_workers: int = 2


class EnhancedPreprocessor(TextPreprocessorInterface):
    """Улучшенный предобработчик товарных наименований"""
    
    def __init__(self, config: EnhancedPreprocessorConfig = None):
        self.config = config or EnhancedPreprocessorConfig()
        
        # Инициализация базового предобработчика
        self.base_preprocessor = TextPreprocessor(self.config.base_config)
        
        # Инициализация новых модулей
        if self.config.enable_units_processing:
            self.units_processor = UnitsProcessor(self.config.units_config)
        else:
            self.units_processor = None
            
        if self.config.enable_synonyms_processing:
            self.synonyms_processor = SynonymsProcessor(self.config.synonyms_config)
        else:
            self.synonyms_processor = None
            
        if self.config.enable_tech_codes_processing:
            self.tech_codes_processor = TechCodesProcessor(self.config.tech_codes_config)
        else:
            self.tech_codes_processor = None
        
        # ThreadPoolExecutor для параллельной обработки
        if self.config.parallel_processing:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="enhanced_preprocessor"
            )
        else:
            self._executor = None
        
        logger.info("EnhancedPreprocessor initialized")
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Полная предобработка одного текста
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с результатами всех этапов обработки
        """
        if not text or not isinstance(text, str):
            return self._empty_result(text)
        
        result = {
            'original': text,
            'final_text': text,
            'processing_successful': True,
            'extracted_parameters': [],
            'extracted_codes': [],
            'synonym_replacements': [],
            'processing_stages': {}
        }
        
        current_text = text
        
        try:
            # Выполняем обработку в заданном порядке
            for stage in self.config.processing_order:
                if stage == 'cleaning_only':
                    # Только очистка без нормализации чисел
                    stage_result = self._process_cleaning_only(current_text)
                    result['processing_stages']['cleaning'] = stage_result
                    current_text = stage_result['cleaned_text']

                elif stage == 'units_processing' and self.units_processor:
                    stage_result = self.units_processor.process_text(current_text)
                    result['processing_stages']['units'] = stage_result
                    result['extracted_parameters'].extend(stage_result['extracted_parameters'])
                    current_text = stage_result['processed']

                    # Отладочная информация
                    logger.debug(f"Units processing: '{text}' -> '{current_text}'")
                    logger.debug(f"Extracted parameters: {len(stage_result['extracted_parameters'])}")

                elif stage == 'tech_codes_processing' and self.tech_codes_processor:
                    stage_result = self.tech_codes_processor.process_text(current_text)
                    result['processing_stages']['tech_codes'] = stage_result
                    result['extracted_codes'].extend(stage_result['extracted_codes'])
                    current_text = stage_result['processed']

                elif stage == 'base_normalization':
                    # Нормализация и лемматизация после извлечения параметров
                    stage_result = self._process_normalization_and_lemmatization(current_text)
                    result['processing_stages']['normalization'] = stage_result
                    current_text = stage_result['final_text']

                elif stage == 'synonyms_processing' and self.synonyms_processor:
                    stage_result = self.synonyms_processor.process_text(current_text)
                    result['processing_stages']['synonyms'] = stage_result
                    result['synonym_replacements'].extend(stage_result['replacements'])
                    current_text = stage_result['normalized']
            
            result['final_text'] = current_text
            
            # Добавляем статистику
            result['stats'] = self._calculate_enhanced_stats(result)
            
        except Exception as e:
            logger.error(f"Error in enhanced preprocessing: {e}")
            result['processing_successful'] = False
            result['error'] = str(e)
            result['final_text'] = text  # Возвращаем исходный текст при ошибке
        
        return result
    
    def _process_cleaning_only(self, text: str) -> Dict[str, Any]:
        """Только очистка текста без нормализации чисел"""
        # Используем только очистку из базового предобработчика
        cleaning_result = self.base_preprocessor.cleaner.clean_text(text)
        return {
            'original': text,
            'cleaned_text': cleaning_result['normalized'],
            'processing_successful': True
        }

    def _process_normalization_and_lemmatization(self, text: str) -> Dict[str, Any]:
        """Нормализация и лемматизация после извлечения параметров"""
        # Защищаем типизированные префиксы от разбиения нормализатором
        protected_text, protected_prefixes = self._protect_typed_prefixes(text)

        # Создаем временную конфигурацию нормализатора без токенизации чисел
        from .normalizer import TextNormalizer, NormalizerConfig

        # Конфигурация без агрессивной токенизации чисел
        temp_normalizer_config = NormalizerConfig(
            standardize_units=True,
            normalize_abbreviations=True,
            unify_technical_terms=True,
            remove_brand_names=False,
            standardize_numbers=False,  # ОТКЛЮЧАЕМ токенизацию чисел
            reduce_numeric_weight=False,  # ОТКЛЮЧАЕМ замену на <NUM>
            preserve_units_with_numbers=True,
            normalize_ranges=False  # Диапазоны уже обработаны
        )

        temp_normalizer = TextNormalizer(temp_normalizer_config)
        normalization_result = temp_normalizer.normalize_text(protected_text)
        normalized_text = normalization_result['final_normalized']

        # Восстанавливаем типизированные префиксы
        normalized_text = self._restore_typed_prefixes(normalized_text, protected_prefixes)

        lemmatization_result = self.base_preprocessor.lemmatizer.lemmatize_text(normalized_text)

        return {
            'original': text,
            'normalized': normalized_text,
            'final_text': lemmatization_result['lemmatized'],
            'processing_successful': True,
            'normalization_result': normalization_result,
            'lemmatization_result': lemmatization_result
        }

    def _protect_typed_prefixes(self, text: str) -> tuple:
        """Защищает типизированные префиксы от разбиения нормализатором"""
        import re

        protected_prefixes = {}
        protected_text = text

        # Паттерны для типизированных префиксов
        prefix_patterns = [
            (r'frac_\d+-\d+', 'FRAC'),
            (r'ratio_\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?', 'RATIO'),
            (r'range_\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?-[a-zA-Zа-яА-Я°³]+', 'RANGE'),
            (r'dims=\[[^\]]+\]\s*unit="[^"]*"', 'DIMS')
        ]

        for pattern, prefix_type in prefix_patterns:
            matches = re.finditer(pattern, protected_text, re.IGNORECASE)
            for i, match in enumerate(matches):
                placeholder = f"__TYPED_{prefix_type}_{i}__"
                protected_prefixes[placeholder] = match.group(0)
                protected_text = protected_text.replace(match.group(0), placeholder, 1)

        return protected_text, protected_prefixes

    def _restore_typed_prefixes(self, text: str, protected_prefixes: dict) -> str:
        """Восстанавливает типизированные префиксы"""
        restored_text = text
        for placeholder, original in protected_prefixes.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def _process_base(self, text: str) -> Dict[str, Any]:
        """Базовая обработка через стандартный предобработчик (для обратной совместимости)"""
        return self.base_preprocessor.preprocess_text(text)
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Пакетная предобработка текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Список результатов обработки
        """
        if not texts:
            return []
        
        if self.config.parallel_processing and self._executor and len(texts) > 10:
            # Параллельная обработка для больших батчей
            return self._process_batch_parallel(texts)
        else:
            # Последовательная обработка
            return [self.preprocess_text(text) for text in texts]
    
    def _process_batch_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Параллельная пакетная обработка"""
        results = [None] * len(texts)
        
        # Разбиваем на чанки для параллельной обработки
        chunk_size = max(1, len(texts) // self.config.max_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Обрабатываем чанки параллельно
        futures = []
        for i, chunk in enumerate(chunks):
            future = self._executor.submit(self._process_chunk, chunk)
            futures.append((i * chunk_size, future))
        
        # Собираем результаты
        for start_idx, future in futures:
            try:
                chunk_results = future.result(timeout=300)  # 5 минут таймаут
                for j, result in enumerate(chunk_results):
                    results[start_idx + j] = result
            except Exception as e:
                logger.error(f"Error in parallel processing chunk: {e}")
                # Заполняем ошибочные результаты
                chunk_size = len(chunks[start_idx // chunk_size])
                for j in range(chunk_size):
                    if start_idx + j < len(results):
                        results[start_idx + j] = self._empty_result(texts[start_idx + j])
        
        return results
    
    def _process_chunk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Обработка чанка текстов"""
        return [self.preprocess_text(text) for text in texts]
    
    async def preprocess_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Асинхронная пакетная предобработка
        
        Args:
            texts: Список текстов
            
        Returns:
            Список результатов обработки
        """
        if not texts:
            return []
        
        # Используем ThreadPoolExecutor для CPU-интенсивных задач
        loop = asyncio.get_event_loop()
        
        if self.config.parallel_processing and len(texts) > 10:
            return await loop.run_in_executor(self._executor, self._process_batch_parallel, texts)
        else:
            return await loop.run_in_executor(None, self.preprocess_batch, texts)
    
    def _empty_result(self, text: str = '') -> Dict[str, Any]:
        """Создание пустого результата при ошибке"""
        return {
            'original': text,
            'final_text': text,
            'processing_successful': False,
            'extracted_parameters': [],
            'extracted_codes': [],
            'synonym_replacements': [],
            'processing_stages': {}
        }
    
    def _calculate_enhanced_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет расширенной статистики"""
        stats = {
            'original_length': len(result['original']),
            'final_length': len(result['final_text']),
            'parameters_extracted': len(result['extracted_parameters']),
            'codes_extracted': len(result['extracted_codes']),
            'synonyms_replaced': len(result['synonym_replacements'])
        }
        
        # Добавляем статистику по типам параметров
        param_types = {}
        for param in result['extracted_parameters']:
            param_type = param.get('type', 'unknown')
            param_types[param_type] = param_types.get(param_type, 0) + 1
        stats['parameter_types'] = param_types
        
        # Добавляем статистику по типам кодов
        code_types = {}
        for code in result['extracted_codes']:
            code_type = code.get('type', 'unknown')
            code_types[code_type] = code_types.get(code_type, 0) + 1
        stats['code_types'] = code_types
        
        return stats
    
    # Методы интерфейса TextPreprocessorInterface
    def set_cleaner(self, cleaner) -> None:
        """Установка компонента очистки"""
        if hasattr(self.base_preprocessor, 'cleaner'):
            self.base_preprocessor.cleaner = cleaner
    
    def set_lemmatizer(self, lemmatizer) -> None:
        """Установка компонента лемматизации"""
        if hasattr(self.base_preprocessor, 'lemmatizer'):
            self.base_preprocessor.lemmatizer = lemmatizer
    
    def set_normalizer(self, normalizer) -> None:
        """Установка компонента нормализации"""
        if hasattr(self.base_preprocessor, 'normalizer'):
            self.base_preprocessor.normalizer = normalizer
    
    def get_processing_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Получение общей статистики обработки"""
        if not results:
            return {}
        
        total_texts = len(results)
        successful = sum(1 for r in results if r.get('processing_successful', False))
        
        total_params = sum(len(r.get('extracted_parameters', [])) for r in results)
        total_codes = sum(len(r.get('extracted_codes', [])) for r in results)
        total_replacements = sum(len(r.get('synonym_replacements', [])) for r in results)
        
        return {
            'total_texts': total_texts,
            'successful_processing': successful,
            'success_rate': successful / total_texts if total_texts > 0 else 0,
            'total_parameters_extracted': total_params,
            'total_codes_extracted': total_codes,
            'total_synonym_replacements': total_replacements,
            'avg_parameters_per_text': total_params / total_texts if total_texts > 0 else 0,
            'avg_codes_per_text': total_codes / total_texts if total_texts > 0 else 0
        }
    
    def __del__(self):
        """Корректное закрытие ThreadPoolExecutor"""
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
