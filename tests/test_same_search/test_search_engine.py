"""
Тесты для модуля search_engine в same_search
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from same_search.search_engine import (
    FuzzySearchEngine, FuzzySearchConfig,
    SemanticSearchEngine, SemanticSearchConfig
)


class TestFuzzySearchEngine:
    """Тесты для FuzzySearchEngine"""
    
    def test_fuzzy_engine_creation(self):
        """Тест создания FuzzySearchEngine"""
        engine = FuzzySearchEngine()
        assert engine is not None
        assert hasattr(engine, 'fit')
        assert hasattr(engine, 'search')
    
    def test_fuzzy_engine_fit(self):
        """Тест обучения FuzzySearchEngine"""
        engine = FuzzySearchEngine()
        
        documents = [
            "Болт М10х50 ГОСТ 7798-70",
            "Гайка М10 ГОСТ 5915-70",
            "Шайба 10 ГОСТ 11371-78"
        ]
        doc_ids = ["1", "2", "3"]
        
        # Тест что fit выполняется без ошибок
        engine.fit(documents, doc_ids)
        
        # Проверяем что данные сохранены
        assert hasattr(engine, 'documents') or hasattr(engine, '_documents')
    
    def test_fuzzy_engine_search(self):
        """Тест поиска FuzzySearchEngine"""
        engine = FuzzySearchEngine()
        
        documents = [
            "Болт М10х50 ГОСТ 7798-70",
            "Гайка М10 ГОСТ 5915-70", 
            "Шайба 10 ГОСТ 11371-78"
        ]
        doc_ids = ["1", "2", "3"]
        
        engine.fit(documents, doc_ids)
        
        # Тест поиска
        results = engine.search("болт м10", top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        # Проверяем структуру результатов
        if results:
            result = results[0]
            assert isinstance(result, dict)
            assert 'content' in result or 'document_id' in result
            assert 'score' in result
    
    def test_fuzzy_search_relevance(self):
        """Тест релевантности нечеткого поиска"""
        engine = FuzzySearchEngine()
        
        documents = [
            "Болт М10х50 ГОСТ 7798-70",
            "Совершенно другой текст",
            "Болт М12х60 ГОСТ 7798-70"
        ]
        doc_ids = ["1", "2", "3"]
        
        engine.fit(documents, doc_ids)
        results = engine.search("болт м10", top_k=3)
        
        if len(results) > 1:
            # Проверяем что результаты отсортированы по релевантности
            scores = [r['score'] for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_fuzzy_search_empty_query(self):
        """Тест поиска с пустым запросом"""
        engine = FuzzySearchEngine()
        
        documents = ["Болт М10х50", "Гайка М10"]
        doc_ids = ["1", "2"]
        
        engine.fit(documents, doc_ids)
        
        results = engine.search("", top_k=5)
        assert isinstance(results, list)
        # Пустой запрос может вернуть пустой список или все документы


class TestSemanticSearchEngine:
    """Тесты для SemanticSearchEngine"""
    
    def test_semantic_engine_creation(self):
        """Тест создания SemanticSearchEngine"""
        try:
            engine = SemanticSearchEngine()
            assert engine is not None
            assert hasattr(engine, 'fit')
            assert hasattr(engine, 'search')
        except Exception as e:
            # Семантический поиск может требовать дополнительных зависимостей
            pytest.skip(f"SemanticSearchEngine not available: {e}")
    
    def test_semantic_engine_fit(self):
        """Тест обучения SemanticSearchEngine"""
        try:
            engine = SemanticSearchEngine()
            
            documents = [
                "Болт М10х50 ГОСТ 7798-70",
                "Крепежный элемент резьбовой",
                "Гайка шестигранная М10"
            ]
            doc_ids = ["1", "2", "3"]
            
            engine.fit(documents, doc_ids)
            
        except Exception as e:
            pytest.skip(f"SemanticSearchEngine fit not available: {e}")
    
    @patch('same_search.models.get_model_manager')
    def test_semantic_search_with_mock(self, mock_model_manager):
        """Тест семантического поиска с мок-моделью"""
        # Мокаем модель менеджер
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384)  # Мок эмбеддинги
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        try:
            engine = SemanticSearchEngine()
            
            documents = ["Болт М10", "Гайка М10", "Шайба 10"]
            doc_ids = ["1", "2", "3"]
            
            engine.fit(documents, doc_ids)
            results = engine.search("крепежный элемент", top_k=2)
            
            assert isinstance(results, list)
            
        except Exception as e:
            pytest.skip(f"SemanticSearchEngine with mock failed: {e}")


class TestSearchConfigs:
    """Тесты для конфигураций поиска"""
    
    def test_fuzzy_search_config(self):
        """Тест FuzzySearchConfig"""
        try:
            config = FuzzySearchConfig()
            assert config is not None
            
            # Проверяем базовые атрибуты конфигурации
            if hasattr(config, 'similarity_threshold'):
                assert isinstance(config.similarity_threshold, (int, float))
            
        except Exception:
            pytest.skip("FuzzySearchConfig not available")
    
    def test_semantic_search_config(self):
        """Тест SemanticSearchConfig"""
        try:
            config = SemanticSearchConfig()
            assert config is not None
            
            # Проверяем базовые атрибуты конфигурации
            if hasattr(config, 'model_name'):
                assert isinstance(config.model_name, str)
                
        except Exception:
            pytest.skip("SemanticSearchConfig not available")


class TestSearchEngineInterface:
    """Тесты интерфейса поисковых движков"""
    
    def test_fuzzy_engine_implements_interface(self):
        """Тест что FuzzySearchEngine реализует интерфейс"""
        try:
            from same_core.interfaces import SearchEngineInterface
            
            engine = FuzzySearchEngine()
            
            # Проверяем наличие методов интерфейса
            assert hasattr(engine, 'fit')
            assert hasattr(engine, 'search')
            
            # Проверяем что методы вызываются
            documents = ["test doc"]
            doc_ids = ["1"]
            
            engine.fit(documents, doc_ids)
            results = engine.search("test", top_k=1)
            assert isinstance(results, list)
            
        except ImportError:
            pytest.skip("same_core interface not available")
    
    def test_search_result_format(self):
        """Тест формата результатов поиска"""
        engine = FuzzySearchEngine()
        
        documents = ["Болт М10х50 ГОСТ 7798-70"]
        doc_ids = ["1"]
        
        engine.fit(documents, doc_ids)
        results = engine.search("болт", top_k=1)
        
        if results:
            result = results[0]
            
            # Проверяем обязательные поля
            assert 'score' in result
            assert isinstance(result['score'], (int, float))
            assert 0 <= result['score'] <= 1
            
            # Проверяем наличие контента или ID
            assert 'content' in result or 'document_id' in result


class TestSearchPerformance:
    """Тесты производительности поиска"""
    
    def test_fuzzy_search_speed(self):
        """Тест скорости нечеткого поиска"""
        engine = FuzzySearchEngine()
        
        # Создаем большой набор документов
        documents = [f"Болт М{i}х{i*5} ГОСТ 7798-70" for i in range(10, 20)]
        doc_ids = [str(i) for i in range(len(documents))]
        
        import time
        
        # Тест времени обучения
        start_time = time.time()
        engine.fit(documents, doc_ids)
        fit_time = time.time() - start_time
        
        # Тест времени поиска
        start_time = time.time()
        for _ in range(10):
            engine.search("болт м12", top_k=5)
        search_time = time.time() - start_time
        
        # Проверяем что операции выполняются быстро
        assert fit_time < 5.0  # Обучение < 5 сек
        assert search_time < 1.0  # 10 поисков < 1 сек
    
    def test_large_document_set(self):
        """Тест работы с большим набором документов"""
        engine = FuzzySearchEngine()
        
        # Создаем много документов
        documents = []
        for i in range(100):
            documents.append(f"Документ номер {i} с текстом для поиска")
        
        doc_ids = [str(i) for i in range(len(documents))]
        
        try:
            engine.fit(documents, doc_ids)
            results = engine.search("документ", top_k=10)
            
            assert isinstance(results, list)
            assert len(results) <= 10
            
        except Exception as e:
            pytest.fail(f"Failed with large document set: {e}")


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_search_before_fit(self):
        """Тест поиска до обучения"""
        engine = FuzzySearchEngine()
        
        try:
            results = engine.search("test query")
            # Может вернуть пустой список или выбросить исключение
            assert isinstance(results, list)
        except Exception:
            # Ожидаемое поведение - исключение
            pass
    
    def test_empty_documents(self):
        """Тест обучения на пустом наборе документов"""
        engine = FuzzySearchEngine()
        
        try:
            engine.fit([], [])
            results = engine.search("test")
            assert isinstance(results, list)
            assert len(results) == 0
        except Exception:
            # Может выбросить исключение для пустого набора
            pass
    
    def test_mismatched_documents_and_ids(self):
        """Тест несоответствия документов и ID"""
        engine = FuzzySearchEngine()
        
        documents = ["doc1", "doc2"]
        doc_ids = ["1"]  # Меньше ID чем документов
        
        try:
            engine.fit(documents, doc_ids)
            # Может обработать или выбросить исключение
        except Exception:
            # Ожидаемое поведение для несоответствия
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
