"""
Главный модуль предобработки, объединяющий все этапы
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd

from .text_cleaner import TextCleaner, CleaningConfig
from .lemmatizer import Lemmatizer, LemmatizerConfig
from .normalizer import TextNormalizer, NormalizerConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Общая конфигурация предобработчика"""
    cleaning_config: Optional[CleaningConfig] = None
    lemmatizer_config: Optional[LemmatizerConfig] = None
    normalizer_config: Optional[NormalizerConfig] = None
    save_intermediate_steps: bool = True
    batch_size: int = 1000


class TextPreprocessor:
    """Главный класс для предобработки текста"""
    
    def __init__(self, config: PreprocessorConfig = None):
        self.config = config or PreprocessorConfig()
        
        # Инициализация компонентов
        self.cleaner = TextCleaner(self.config.cleaning_config)
        self.lemmatizer = Lemmatizer(self.config.lemmatizer_config)
        self.normalizer = TextNormalizer(self.config.normalizer_config)
        
        logger.info("TextPreprocessor initialized")
    
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
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            # Если есть активный event loop, создаем задачу
            return asyncio.create_task(self.preprocess_batch_async(texts))
        except RuntimeError:
            # Если нет активного loop, создаем новый
            return asyncio.run(self.preprocess_batch_async(texts))

    async def preprocess_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Асинхронная пакетная предобработка текстов с оптимизацией производительности

        Args:
            texts: Список текстов для обработки

        Returns:
            Список результатов обработки
        """
        if not texts:
            return []

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
                # Параллельная обработка этапов где возможно
                import asyncio

                # Этап 1: Очистка батча (синхронная, но быстрая)
                cleaning_results = self.cleaner.clean_batch(batch)
                cleaned_texts = [r['normalized'] for r in cleaning_results]

                # Этап 2: Нормализация батча (синхронная, но быстрая)
                normalization_results = self.normalizer.normalize_batch(cleaned_texts)
                normalized_texts = [r['final_normalized'] for r in normalization_results]

                # Этап 3: Лемматизация батча (асинхронная, самая медленная)
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
        
        # Добавляем статистику лемматизации если доступна
        if 'lemmatization' in result:
            lemma_stats = self.lemmatizer.get_lemmatization_stats(result['lemmatization'])
            stats.update(lemma_stats)
        
        return stats
    
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
