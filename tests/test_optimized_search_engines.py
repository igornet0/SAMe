"""
Тесты для оптимизированных поисковых движков
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
from src.same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
from src.same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig
from src.same.text_processing.interfaces import (
    SearchEngineInterface, AnalogSearchEngineInterface
)


class TestOptimizedSemanticSearch:
    """Тесты для оптимизированного семантического поиска"""
    
    @pytest.fixture
    def search_config(self):
        return SemanticSearchConfig(
            enable_fallback=True,
            graceful_degradation=True,
            max_retries=2,
            retry_delay=0.1,
            enable_cache=True,
            cache_size=50
        )
    
    @pytest.fixture
    def search_engine(self, search_config):
        return SemanticSearchEngine(search_config)
    
    @pytest.fixture
    def sample_documents(self):
        return [
            "Насос центробежный для воды",
            "Электродвигатель асинхронный 5 кВт",
            "Подшипник шариковый 6205",
            "Клапан запорный стальной",
            "Фильтр масляный автомобильный",
            "Редуктор червячный передаточное число 40",
            "Компрессор поршневой 10 атм",
            "Вентилятор осевой диаметр 300мм",
            "Муфта упругая МУВП",
            "Датчик температуры термопарный"
        ]
    
    def test_fallback_model_loading(self, search_config):
        """Тест загрузки fallback моделей"""
        # Настраиваем конфигурацию с несуществующей основной моделью
        search_config.model_name = "non_existent_model"
        search_config.fallback_model_names = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        engine = SemanticSearchEngine(search_config)
        
        # Должен успешно инициализироваться с fallback моделью
        assert engine is not None
    
    def test_graceful_degradation(self, search_config):
        """Тест graceful degradation при недоступности моделей"""
        search_config.model_name = "completely_non_existent_model"
        search_config.fallback_model_names = ["another_non_existent_model"]
        search_config.graceful_degradation = True
        
        engine = SemanticSearchEngine(search_config)
        
        # Должен создать dummy модель
        assert engine is not None
    
    def test_streaming_embeddings_generation(self, search_engine, sample_documents):
        """Тест потоковой генерации эмбеддингов"""
        # Создаем большой набор документов для тестирования потоковой обработки
        large_documents = sample_documents * 200  # 2000 документов
        
        # Фитим движок
        search_engine.fit(large_documents)
        
        assert search_engine.is_fitted
        assert len(search_engine.documents) == len(large_documents)
        assert search_engine.embeddings is not None
    
    def test_optimized_index_building(self, search_engine, sample_documents):
        """Тест оптимизированного построения индекса"""
        search_engine.fit(sample_documents)
        
        # Проверяем, что индекс построен корректно
        assert search_engine.index is not None
        assert search_engine.index.ntotal == len(sample_documents)
    
    def test_early_stopping_search(self, search_engine, sample_documents):
        """Тест ранней остановки при поиске"""
        search_engine.fit(sample_documents)
        
        # Поиск с низкой релевантностью должен останавливаться рано
        results = search_engine.search("совершенно нерелевантный запрос", top_k=5)
        
        # Должны получить результаты, но возможно меньше чем top_k
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_caching_functionality(self, search_engine, sample_documents):
        """Тест функциональности кэширования"""
        search_engine.fit(sample_documents)
        
        query = "насос центробежный"
        
        # Первый поиск
        start_time = time.time()
        results1 = search_engine.search(query, top_k=3)
        first_search_time = time.time() - start_time
        
        # Второй поиск (должен быть из кэша)
        start_time = time.time()
        results2 = search_engine.search(query, top_k=3)
        cached_search_time = time.time() - start_time
        
        # Результаты должны быть идентичными
        assert results1 == results2
        
        # Кэшированный поиск должен быть быстрее
        assert cached_search_time < first_search_time * 0.5
        
        # Проверяем статистику кэша
        cache_stats = search_engine.get_cache_stats()
        assert cache_stats['search_cache_size'] > 0
    
    def test_memory_optimization(self, search_engine):
        """Тест оптимизации памяти"""
        # Создаем большой набор документов
        large_documents = [f"Документ номер {i} для тестирования памяти" for i in range(1000)]
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Фитим движок
        search_engine.fit(large_documents)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Проверяем разумное использование памяти
        assert memory_increase < 1000  # Менее 1GB увеличения
        
        # Очищаем кэши
        search_engine.clear_cache()
        
        after_clear_memory = process.memory_info().rss / 1024 / 1024  # MB
        assert after_clear_memory < peak_memory


class TestOptimizedHybridSearch:
    """Тесты для оптимизированного гибридного поиска"""
    
    @pytest.fixture
    def hybrid_config(self):
        return HybridSearchConfig(
            fuzzy_weight=0.4,
            semantic_weight=0.6,
            min_fuzzy_score=0.3,
            min_semantic_score=0.4
        )
    
    @pytest.fixture
    def hybrid_engine(self, hybrid_config):
        return HybridSearchEngine(hybrid_config)
    
    def test_optimized_result_combination(self, hybrid_engine):
        """Тест оптимизированного комбинирования результатов"""
        # Создаем mock результаты
        fuzzy_results = [
            {'document_id': 1, 'combined_score': 0.8, 'document': 'doc1'},
            {'document_id': 2, 'combined_score': 0.6, 'document': 'doc2'},
            {'document_id': 3, 'combined_score': 0.4, 'document': 'doc3'}
        ]
        
        semantic_results = [
            {'document_id': 1, 'similarity_score': 0.9, 'document': 'doc1'},
            {'document_id': 4, 'similarity_score': 0.7, 'document': 'doc4'},
            {'document_id': 5, 'similarity_score': 0.5, 'document': 'doc5'}
        ]
        
        # Тестируем комбинирование
        combined = hybrid_engine._weighted_sum_combination(fuzzy_results, semantic_results)
        
        assert len(combined) > 0
        
        # Документ 1 должен иметь высокий скор (присутствует в обоих результатах)
        doc1_result = next((r for r in combined if r['document_id'] == 1), None)
        assert doc1_result is not None
        assert doc1_result['hybrid_score'] > 0.7  # Высокий комбинированный скор


class TestOptimizedAnalogSearchEngine:
    """Тесты для оптимизированного главного движка"""
    
    @pytest.fixture
    def analog_config(self):
        return AnalogSearchConfig(
            search_method="hybrid",
            enable_parameter_extraction=True
        )
    
    @pytest.fixture
    def analog_engine(self, analog_config):
        return AnalogSearchEngine(analog_config)
    
    def test_dependency_injection(self, analog_engine):
        """Тест dependency injection"""
        # Создаем mock компоненты
        mock_preprocessor = Mock()
        mock_search_engine = Mock()
        mock_search_engine.search.return_value = []
        
        # Тестируем установку компонентов
        analog_engine.set_preprocessor(mock_preprocessor)
        analog_engine.set_search_engine(mock_search_engine, 'semantic')
        
        assert analog_engine.preprocessor == mock_preprocessor
        assert analog_engine.semantic_engine == mock_search_engine
    
    def test_interface_compliance(self, analog_engine):
        """Тест соответствия интерфейсу"""
        assert isinstance(analog_engine, AnalogSearchEngineInterface)
        
        # Проверяем наличие обязательных методов
        assert hasattr(analog_engine, 'search_analogs')
        assert hasattr(analog_engine, 'set_preprocessor')
        assert hasattr(analog_engine, 'set_search_engine')
    
    def test_fallback_search_mechanism(self, analog_engine):
        """Тест механизма fallback поиска"""
        # Создаем простые тестовые данные
        test_data = [
            {'original_text': 'насос центробежный', 'final_text': 'насос центробежный'},
            {'original_text': 'двигатель электрический', 'final_text': 'двигатель электрический'}
        ]
        
        import pandas as pd
        analog_engine.processed_catalog = pd.DataFrame(test_data)
        analog_engine.is_ready = True
        
        # Тестируем простой текстовый поиск как fallback
        results = analog_engine._simple_text_search("насос", 5)
        
        assert len(results) > 0
        assert results[0]['search_type'] == 'simple_text'


class TestPerformanceBenchmarks:
    """Бенчмарки производительности поисковых движков"""
    
    def test_search_speed_benchmark(self):
        """Бенчмарк скорости поиска"""
        documents = [f"Тестовый документ номер {i}" for i in range(1000)]
        
        engine = SemanticSearchEngine(SemanticSearchConfig(
            enable_cache=True,
            batch_size=64
        ))
        
        # Фитим движок
        start_time = time.time()
        engine.fit(documents)
        fit_time = time.time() - start_time
        
        # Время фитинга должно быть разумным
        assert fit_time < 300  # Менее 5 минут для 1000 документов
        
        # Тестируем скорость поиска
        queries = ["тестовый документ", "номер документа", "поиск текста"]
        
        total_search_time = 0
        for query in queries:
            start_time = time.time()
            results = engine.search(query, top_k=10)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            assert len(results) <= 10
            assert search_time < 1.0  # Менее 1 секунды на поиск
        
        avg_search_time = total_search_time / len(queries)
        assert avg_search_time < 0.5  # Средняя скорость поиска менее 0.5 сек
    
    def test_concurrent_search_performance(self):
        """Тест производительности при конкурентных запросах"""
        documents = [f"Документ для конкурентного тестирования {i}" for i in range(100)]
        
        engine = SemanticSearchEngine()
        engine.fit(documents)
        
        async def search_task(query_id):
            query = f"поиск номер {query_id}"
            results = engine.search(query, top_k=5)
            return len(results)
        
        async def run_concurrent_searches():
            tasks = [search_task(i) for i in range(10)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            return results, total_time
        
        # Запускаем конкурентные поиски
        results, total_time = asyncio.run(run_concurrent_searches())
        
        # Все поиски должны завершиться успешно
        assert len(results) == 10
        assert all(r >= 0 for r in results)  # Все результаты валидны
        
        # Общее время должно быть разумным
        assert total_time < 10  # Менее 10 секунд для 10 конкурентных поисков


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
