"""
Главный модуль предобработки, объединяющий все этапы
"""

import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd

from .text_cleaner import TextCleaner, CleaningConfig
from .lemmatizer import Lemmatizer, LemmatizerConfig
from .normalizer import TextNormalizer, NormalizerConfig
from .color_normalizer import ColorNormalizer, ColorNormalizerConfig
from .technical_terms_normalizer import TechnicalTermsNormalizer, TechnicalTermsNormalizerConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Общая конфигурация предобработчика"""
    cleaning_config: Optional[CleaningConfig] = None
    lemmatizer_config: Optional[LemmatizerConfig] = None
    normalizer_config: Optional[NormalizerConfig] = None
    color_normalizer_config: Optional[ColorNormalizerConfig] = None
    technical_terms_normalizer_config: Optional[TechnicalTermsNormalizerConfig] = None
    enable_color_normalization: bool = True  # Включить нормализацию цветов
    enable_technical_terms_normalization: bool = True  # Включить нормализацию технических терминов
    save_intermediate_steps: bool = True
    batch_size: int = 1000
    # Параметры параллельной обработки
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = auto-detect
    parallel_threshold: int = 100  # Минимальное количество текстов для параллельной обработки
    chunk_size: int = 50  # Размер чанка для каждого процесса


class TextPreprocessor:
    """Главный класс для предобработки текста"""
    
    def __init__(self, config: PreprocessorConfig = None):
        self.config = config or PreprocessorConfig()

        # Инициализация компонентов
        self.cleaner = TextCleaner(self.config.cleaning_config)
        self.lemmatizer = Lemmatizer(self.config.lemmatizer_config)
        self.normalizer = TextNormalizer(self.config.normalizer_config)

        # Инициализация нормализатора цветов
        self.color_normalizer = None
        if self.config.enable_color_normalization:
            self.color_normalizer = ColorNormalizer(self.config.color_normalizer_config)

        # Инициализация нормализатора технических терминов
        self.technical_terms_normalizer = None
        if self.config.enable_technical_terms_normalization:
            self.technical_terms_normalizer = TechnicalTermsNormalizer(self.config.technical_terms_normalizer_config)

        # Настройка параллельной обработки
        if self.config.max_workers is None:
            self.config.max_workers = min(cpu_count(), 4)  # Ограничиваем максимум 4 процессами

        self._process_executor = None
        self._thread_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preprocessor")

        logger.info(f"TextPreprocessor initialized with max_workers={self.config.max_workers}")
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Полная предобработка одного текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Dict с результатами всех этапов обработки
        """
        if not text or not isinstance(text, str):
            return self._empty_result(text)
        
        result = {'original': text}
        
        try:
            # Этап 1: Очистка
            cleaning_result = self.cleaner.clean_text(text)
            result['cleaning'] = cleaning_result
            cleaned_text = cleaning_result['normalized']
            
            # Этап 2: Нормализация
            normalization_result = self.normalizer.normalize_text(cleaned_text)
            result['normalization'] = normalization_result
            normalized_text = normalization_result['final_normalized']

            # Этап 2.3: Нормализация технических терминов (если включена)
            if self.technical_terms_normalizer:
                technical_normalization_result = self.technical_terms_normalizer.normalize_technical_terms(normalized_text)
                result['technical_terms_normalization'] = technical_normalization_result
                normalized_text = technical_normalization_result['normalized']

            # Этап 2.5: Нормализация цветов (если включена)
            if self.color_normalizer:
                color_normalization_result = self.color_normalizer.normalize_colors(normalized_text)
                result['color_normalization'] = color_normalization_result
                normalized_text = color_normalization_result['normalized']

            # Этап 3: Лемматизация
            lemmatization_result = self.lemmatizer.lemmatize_text(normalized_text)
            result['lemmatization'] = lemmatization_result

            # Финальный результат
            result['final_text'] = lemmatization_result['lemmatized']
            result['processing_successful'] = True
            
            # Статистика
            result['stats'] = self._calculate_stats(result)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            result['processing_successful'] = False
            result['error'] = str(e)
            result['final_text'] = text
        
        return result
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Пакетная предобработка текстов (синхронная версия для обратной совместимости)

        Args:
            texts: Список текстов для обработки

        Returns:
            Список результатов обработки
        """
        if not texts:
            return []

        # Используем параллельную обработку для больших датасетов
        if (self.config.enable_parallel_processing and
            len(texts) >= self.config.parallel_threshold):
            return self._preprocess_batch_parallel(texts)
        else:
            # Простая синхронная реализация для малых датасетов
            results = []
            for text in texts:
                result = self.preprocess_text(text)
                results.append(result)
            return results
        
    async def preprocess_batch_async(self, texts: Union[List[str], Dict[Any, str]]) -> List[Dict[str, Any]]:
        """
        Асинхронная пакетная предобработка текстов с оптимизацией производительности

        Args:
            texts: Список текстов для обработки

        Returns:
            Список результатов обработки
        """

        if not texts:
            return []

        # Поддержка входа как списка и как словаря id -> text
        input_is_dict = isinstance(texts, dict)
        if input_is_dict:
            # Сохраняем детерминированный порядок
            keys_list = list(texts.keys())
            values_list = [texts[k] for k in keys_list]
        else:
            values_list = list(texts)

        # Для больших датасетов используем параллельную обработку в ThreadPoolExecutor
        if (self.config.enable_parallel_processing and
            len(values_list) >= self.config.parallel_threshold):

            logger.info(f"Starting async parallel processing: {len(values_list)} texts")
            loop = asyncio.get_event_loop()
            logger.info(f"Thread executor: {self._thread_executor}")
            results = await loop.run_in_executor(self._thread_executor, self._preprocess_batch_parallel, values_list)
        else:
            results = []
            batch_size = self.config.batch_size
            total_batches = (len(values_list) - 1) // batch_size + 1

            logger.info(f"Starting async batch processing: {len(values_list)} texts in {total_batches} batches")

            # Обрабатываем батчами для оптимизации памяти
            for i in range(0, len(values_list), batch_size):
                batch = values_list[i:i + batch_size]
                batch_num = i // batch_size + 1

                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

                try:
                    # Этап 1: Очистка батча 
                    cleaning_results = self.cleaner.clean_batch(batch)
                    cleaned_texts = [r['normalized'] for r in cleaning_results]

                    # Этап 2: Нормализация батча 
                    normalization_results = self.normalizer.normalize_batch(cleaned_texts)
                    normalized_texts = [r['final_normalized'] for r in normalization_results]

                    # Этап 3: Лемматизация батча 
                    lemmatization_results = await self.lemmatizer.lemmatize_batch_async(normalized_texts)

                    # Объединяем результаты
                    batch_results = []
                    for j, original_text in enumerate(batch):
                        result = {
                            'original': original_text,
                            'cleaning': cleaning_results[j],
                            'normalization': normalization_results[j],
                            'lemmatization': lemmatization_results[j],
                            'final_text': lemmatization_results[j]['lemmatized'],
                            'processing_successful': True
                        }

                        # Добавляем статистику только если требуется
                        if self.config.save_intermediate_steps:
                            result['stats'] = self._calculate_stats(result)

                        batch_results.append(result)

                    results.extend(batch_results)
                    logger.debug(f"Batch {batch_num} completed successfully")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    # Добавляем результаты с ошибками для каждого текста в батче
                    for text in batch:
                        results.append({
                            'original': text,
                            'final_text': text,
                            'processing_successful': False,
                            'error': str(e)
                        })

            logger.info(f"Async batch processing completed: {len(results)} results")

        # Если вход был словарем, нам важно сохранить соответствие порядку ключей
        # Результаты уже выровнены по values_list, поэтому просто возвращаем их списком
        return results

    async def preprocess_batch_async_temp(self, texts: Union[List[str], Dict[Any, str]]) -> List[Dict[str, Any]]:
        """
        Асинхронная пакетная предобработка текстов с оптимизацией производительности

        Args:
            texts: Список текстов для обработки

        Returns:
            Список результатов обработки
        """
        if not texts:
            return []

        # Для больших датасетов используем параллельную обработку в ThreadPoolExecutor
        if (self.config.enable_parallel_processing and
            len(texts) >= self.config.parallel_threshold):

            logger.info(f"Starting async parallel processing: {len(texts)} texts")
            loop = asyncio.get_event_loop()
            logger.info(f"Thread executor: {self._thread_executor}")
            return await loop.run_in_executor(self._thread_executor, self._preprocess_batch_parallel, texts)

        results = []
        batch_size = self.config.batch_size
        total_batches = (len(texts) - 1) // batch_size + 1

        logger.info(f"Starting async batch processing: {len(texts)} texts in {total_batches} batches")

        # Обрабатываем батчами для оптимизации памяти
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                # Этап 1: Очистка батча 
                cleaning_results = self.cleaner.clean_batch(batch)
                cleaned_texts = [r['normalized'] for r in cleaning_results]

                # Этап 2: Нормализация батча 
                normalization_results = self.normalizer.normalize_batch(cleaned_texts)
                normalized_texts = [r['final_normalized'] for r in normalization_results]

                # Этап 3: Лемматизация батча 
                lemmatization_results = await self.lemmatizer.lemmatize_batch_async(normalized_texts)

                # Объединяем результаты
                batch_results = []
                for j, original_text in enumerate(batch):
                    result = {
                        'original': original_text,
                        'cleaning': cleaning_results[j],
                        'normalization': normalization_results[j],
                        'lemmatization': lemmatization_results[j],
                        'final_text': lemmatization_results[j]['lemmatized'],
                        'processing_successful': True
                    }

                    # Добавляем статистику только если требуется
                    if self.config.save_intermediate_steps:
                        result['stats'] = self._calculate_stats(result)

                    batch_results.append(result)

                results.extend(batch_results)
                logger.debug(f"Batch {batch_num} completed successfully")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Добавляем результаты с ошибками
                for text in batch:
                    results.append({
                        'original': text,
                        'final_text': text,
                        'processing_successful': False,
                        'error': str(e)
                    })

        logger.info(f"Async batch processing completed: {len(results)} results")
        return results
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                           output_columns: Dict[str, str] = None) -> pd.DataFrame:
        """
        Предобработка DataFrame с текстами
        
        Args:
            df: DataFrame с данными
            text_column: Название колонки с текстом
            output_columns: Маппинг названий выходных колонок
            
        Returns:
            DataFrame с добавленными колонками обработанного текста
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Настройки выходных колонок по умолчанию
        default_columns = {
            'cleaned': f'{text_column}_cleaned',
            'normalized': f'{text_column}_normalized', 
            'lemmatized': f'{text_column}_lemmatized',
            'final': f'{text_column}_processed'
        }
        
        if output_columns:
            default_columns.update(output_columns)
        
        # Получаем тексты для обработки
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Обрабатываем тексты
        results = self.preprocess_batch(texts)
        
        # Добавляем результаты в DataFrame
        df_result = df.copy()
        
        if self.config.save_intermediate_steps:
            df_result[default_columns['cleaned']] = [r['cleaning']['normalized'] for r in results]
            df_result[default_columns['normalized']] = [r['normalization']['final_normalized'] for r in results]
            df_result[default_columns['lemmatized']] = [r['lemmatization']['lemmatized'] for r in results]
        
        df_result[default_columns['final']] = [r['final_text'] for r in results]
        
        # Добавляем статистику
        df_result[f'{text_column}_processing_success'] = [r['processing_successful'] for r in results]
        
        return df_result
    
    def _empty_result(self, text: str) -> Dict[str, Any]:
        """Создание пустого результата для некорректного входа"""
        return {
            'original': text or '',
            'final_text': '',
            'processing_successful': False,
            'error': 'Empty or invalid input'
        }
    
    def _calculate_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет статистики обработки"""
        original = result['original']
        final = result['final_text']

        stats = {
            'original_length': len(original),
            'final_length': len(final),
            'compression_ratio': round((len(original) - len(final)) / len(original) * 100, 2) if original else 0,
            'word_count_original': len(original.split()) if original else 0,
            'word_count_final': len(final.split()) if final else 0
        }

        # Добавляем статистику по цветам если доступна
        if 'color_normalization' in result:
            color_result = result['color_normalization']
            stats.update({
                'colors_found': len(color_result.get('colors_found', [])),
                'colors_normalized': color_result.get('colors_count', 0)
            })

        # Добавляем статистику по техническим терминам если доступна
        if 'technical_terms_normalization' in result:
            tech_result = result['technical_terms_normalization']
            stats.update({
                'technical_terms_found': len(tech_result.get('technical_terms_found', [])),
                'technical_terms_normalized': tech_result.get('technical_terms_count', 0)
            })

        # Добавляем статистику лемматизации если доступна
        if 'lemmatization' in result:
            try:
                lemma_stats = self.lemmatizer.get_lemmatization_stats(result['lemmatization'])
                stats.update(lemma_stats)
            except AttributeError:
                # Метод get_lemmatization_stats может не существовать
                pass

        return stats

    def _preprocess_batch_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Параллельная обработка текстов с использованием ProcessPoolExecutor"""
        logger.info(f"Starting parallel processing of {len(texts)} texts")

        # Разбиваем тексты на чанки
        chunks = self._split_into_chunks(texts, self.config.chunk_size)

        try:
            # Создаем ProcessPoolExecutor только когда нужно
            if self._process_executor is None:
                self._process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)

            # Отправляем чанки на обработку
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                future = self._process_executor.submit(_process_chunk_worker, chunk, self.config)
                future_to_chunk[future] = i

            # Собираем результаты в правильном порядке
            chunk_results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_results[chunk_index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    # Fallback к синхронной обработке для этого чанка
                    chunk = chunks[chunk_index]
                    chunk_results[chunk_index] = [self.preprocess_text(text) for text in chunk]

            # Объединяем результаты всех чанков
            results = []
            for chunk_result in chunk_results:
                if chunk_result:
                    results.extend(chunk_result)

            logger.info(f"Parallel processing completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}, falling back to sequential")
            # Fallback к синхронной обработке
            results = []
            for text in texts:
                try:
                    result = self.preprocess_text(text)
                    results.append(result)
                except Exception as text_error:
                    logger.warning(f"Error processing text '{text[:50]}...': {text_error}")
                    results.append({
                        'original': text,
                        'final_text': text,
                        'processing_successful': False,
                        'error': str(text_error)
                    })
            return results

    def _split_into_chunks(self, texts: List[str], chunk_size: int) -> List[List[str]]:
        """Разбивает список текстов на чанки заданного размера"""
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunks.append(texts[i:i + chunk_size])
        return chunks

    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Получение сводной статистики обработки"""
        total_texts = len(results)
        successful = sum(1 for r in results if r.get('processing_successful', False))
        
        if successful == 0:
            return {
                'total_texts': total_texts,
                'successful_processing': 0,
                'success_rate': 0,
                'average_compression': 0
            }
        
        # Средняя степень сжатия
        compressions = [r['stats']['compression_ratio'] for r in results 
                       if r.get('processing_successful') and 'stats' in r]
        avg_compression = sum(compressions) / len(compressions) if compressions else 0
        
        return {
            'total_texts': total_texts,
            'successful_processing': successful,
            'success_rate': round(successful / total_texts * 100, 2),
            'average_compression': round(avg_compression, 2),
            'failed_processing': total_texts - successful
        }

    def __del__(self):
        """Корректное закрытие пулов процессов и потоков при удалении объекта"""
        if hasattr(self, '_process_executor') and self._process_executor:
            try:
                self._process_executor.shutdown(wait=False)
            except Exception:
                pass

        if hasattr(self, '_thread_executor') and self._thread_executor:
            try:
                self._thread_executor.shutdown(wait=False)
            except Exception:
                pass


def _process_chunk_worker(texts_chunk: List[str], config: PreprocessorConfig) -> List[Dict[str, Any]]:
    """
    Функция-воркер для обработки чанка текстов в отдельном процессе

    Args:
        texts_chunk: Чанк текстов для обработки
        config: Конфигурация предобработчика

    Returns:
        Список результатов обработки
    """
    try:
        # Создаем новый экземпляр предобработчика в процессе
        # Отключаем параллельную обработку чтобы избежать рекурсии
        worker_config = PreprocessorConfig(
            cleaning_config=config.cleaning_config,
            lemmatizer_config=config.lemmatizer_config,
            normalizer_config=config.normalizer_config,
            save_intermediate_steps=config.save_intermediate_steps,
            batch_size=config.batch_size,
            enable_parallel_processing=False  # Важно: отключаем параллельную обработку
        )

        preprocessor = TextPreprocessor(worker_config)

        # Обрабатываем тексты последовательно в этом процессе
        results = []
        for text in texts_chunk:
            try:
                result = preprocessor.preprocess_text(text)
                results.append(result)
            except Exception as e:
                # Добавляем результат с ошибкой
                results.append({
                    'original_text': text,
                    'processing_successful': False,
                    'error': str(e),
                    'cleaned_text': text,
                    'normalized_text': text,
                    'lemmatization': {
                        'original': text,
                        'lemmatized': text,
                        'tokens': [],
                        'lemmas': [],
                        'pos_tags': [],
                        'filtered_lemmas': []
                    }
                })

        return results

    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        # Возвращаем результаты с ошибками для всех текстов
        return [{
            'original_text': text,
            'processing_successful': False,
            'error': str(e),
            'cleaned_text': text,
            'normalized_text': text,
            'lemmatization': {
                'original': text,
                'lemmatized': text,
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }
        } for text in texts_chunk]
