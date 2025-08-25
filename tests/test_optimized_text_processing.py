"""
Тесты для оптимизированных модулей обработки текста
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

try:
    from same_clear.text_processing import TextPreprocessor, PreprocessorConfig
    from same_clear.text_processing import Lemmatizer, LemmatizerConfig
    from same_core.interfaces import TextProcessorInterface as TextPreprocessorInterface
except ImportError:
    # Fallback imports
    from same.text_processing import TextPreprocessor, PreprocessorConfig
    from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
    from same.text_processing.interfaces import TextPreprocessorInterface


class TestOptimizedLemmatizer:
    """Тесты для оптимизированного лемматизатора"""
    
    @pytest.fixture
    def lemmatizer_config(self):
        return LemmatizerConfig(
            enable_caching=True,
            cache_max_size=100,
            cache_ttl_seconds=60,
            enable_fallback=True,
            graceful_degradation=True
        )
    
    @pytest.fixture
    def lemmatizer(self, lemmatizer_config):
        return Lemmatizer(lemmatizer_config)
    
    def test_caching_functionality(self, lemmatizer):
        """Тест функциональности кэширования"""
        text = "Тестовый текст для проверки кэширования"
        
        # Первый вызов - должен кэшироваться
        result1 = lemmatizer.lemmatize_text(text)
        
        # Второй вызов - должен браться из кэша
        start_time = time.time()
        result2 = lemmatizer.lemmatize_text(text)
        cache_time = time.time() - start_time
        
        # Результаты должны быть идентичными
        assert result1 == result2
        
        # Второй вызов должен быть быстрее (из кэша)
        assert cache_time < 0.1  # Менее 100мс для кэшированного результата
        
        # Проверяем статистику кэша
        cache_stats = lemmatizer.get_cache_stats()
        assert cache_stats['caching_enabled'] is True
        assert cache_stats['cache_size'] > 0
    
    def test_cache_size_management(self, lemmatizer):
        """Тест управления размером кэша"""
        # Заполняем кэш до предела
        for i in range(lemmatizer.config.cache_max_size + 10):
            text = f"Тестовый текст номер {i}"
            lemmatizer.lemmatize_text(text)
        
        cache_stats = lemmatizer.get_cache_stats()
        # Размер кэша не должен превышать максимальный
        assert cache_stats['cache_size'] <= lemmatizer.config.cache_max_size
    
    def test_fallback_mode(self):
        """Тест fallback режима при недоступности SpaCy"""
        config = LemmatizerConfig(
            model_name="non_existent_model",
            enable_fallback=True,
            graceful_degradation=True
        )
        
        lemmatizer = Lemmatizer(config)
        
        # Должен работать даже с несуществующей моделью
        result = lemmatizer.lemmatize_text("Тестовый текст")
        
        assert result is not None
        assert 'original' in result
        assert 'lemmatized' in result
    
    @pytest.mark.asyncio
    async def test_async_performance(self, lemmatizer):
        """Тест производительности асинхронной обработки"""
        texts = [f"Тестовый текст номер {i}" for i in range(50)]
        
        start_time = time.time()
        results = await lemmatizer.lemmatize_batch_async(texts)
        async_time = time.time() - start_time
        
        assert len(results) == len(texts)
        assert async_time < 30  # Должно выполняться менее чем за 30 секунд
        
        # Проверяем, что все результаты корректны
        for result in results:
            assert 'original' in result
            assert 'lemmatized' in result


class TestOptimizedPreprocessor:
    """Тесты для оптимизированного предобработчика"""
    
    @pytest.fixture
    def preprocessor_config(self):
        return PreprocessorConfig(
            enable_parallel_processing=True,
            max_workers=2,
            parallel_threshold=10,
            chunk_size=5
        )
    
    @pytest.fixture
    def preprocessor(self, preprocessor_config):
        return TextPreprocessor(preprocessor_config)
    
    def test_parallel_processing_performance(self, preprocessor):
        """Тест производительности параллельной обработки"""
        texts = [f"Тестовый текст для обработки номер {i}" for i in range(20)]
        
        # Последовательная обработка
        start_time = time.time()
        sequential_results = []
        for text in texts:
            result = preprocessor.preprocess_text(text)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Параллельная обработка
        start_time = time.time()
        parallel_results = preprocessor.preprocess_batch(texts)
        parallel_time = time.time() - start_time
        
        # Результаты должны быть одинаковыми
        assert len(sequential_results) == len(parallel_results)
        
        # Параллельная обработка должна быть быстрее для больших объемов
        if len(texts) >= preprocessor.config.parallel_threshold:
            assert parallel_time < sequential_time * 0.8  # Минимум 20% ускорение
    
    def test_small_batch_optimization(self, preprocessor):
        """Тест оптимизации для малых батчей"""
        small_texts = ["Короткий текст", "Еще один текст"]
        
        # Для малых батчей должна использоваться последовательная обработка
        results = preprocessor.preprocess_batch(small_texts)
        
        assert len(results) == len(small_texts)
        for result in results:
            assert 'processing_successful' in result
    
    @pytest.mark.asyncio
    async def test_async_parallel_processing(self, preprocessor):
        """Тест асинхронной параллельной обработки"""
        texts = [f"Асинхронный тест {i}" for i in range(15)]
        
        start_time = time.time()
        results = await preprocessor.preprocess_batch_async(texts)
        async_time = time.time() - start_time
        
        assert len(results) == len(texts)
        assert async_time < 60  # Разумное время выполнения
        
        # Проверяем качество результатов
        successful_count = sum(1 for r in results if r.get('processing_successful', False))
        assert successful_count >= len(texts) * 0.8  # Минимум 80% успешных обработок
    
    def test_dependency_injection(self):
        """Тест dependency injection"""
        # Создаем mock компоненты
        mock_cleaner = Mock()
        mock_lemmatizer = Mock()
        mock_normalizer = Mock()
        
        preprocessor = TextPreprocessor()
        
        # Проверяем, что можем заменить компоненты
        assert hasattr(preprocessor, '_preprocessor')  # Проверяем наличие внутреннего препроцессора
        
        # Тест интерфейса
        assert isinstance(preprocessor, TextPreprocessorInterface)


class TestPerformanceBenchmarks:
    """Бенчмарки производительности"""
    
    def test_memory_usage_optimization(self):
        """Тест оптимизации использования памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Создаем большой объем данных для обработки
        large_texts = [f"Большой текст для тестирования памяти номер {i} " * 100 
                      for i in range(100)]
        
        preprocessor = TextPreprocessor(PreprocessorConfig(
            enable_parallel_processing=True,
            chunk_size=10
        ))
        
        # Обрабатываем данные
        results = preprocessor.preprocess_batch(large_texts)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Проверяем, что увеличение памяти разумное
        assert memory_increase < 500  # Менее 500MB увеличения
        assert len(results) == len(large_texts)
    
    def test_processing_speed_benchmark(self):
        """Бенчмарк скорости обработки"""
        texts = [f"Текст для бенчмарка скорости {i}" for i in range(100)]
        
        preprocessor = TextPreprocessor(PreprocessorConfig(
            enable_parallel_processing=True
        ))
        
        start_time = time.time()
        results = preprocessor.preprocess_batch(texts)
        processing_time = time.time() - start_time
        
        # Скорость обработки должна быть разумной
        texts_per_second = len(texts) / processing_time
        assert texts_per_second > 5  # Минимум 5 текстов в секунду
        
        # Качество обработки
        successful_count = sum(1 for r in results if r.get('processing_successful', False))
        success_rate = successful_count / len(texts)
        assert success_rate > 0.9  # Минимум 90% успешных обработок


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
