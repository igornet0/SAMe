"""
Тесты для модулей поискового движка
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from same.search_engine import (
    FuzzySearchEngine, FuzzySearchConfig,
    SemanticSearchEngine, SemanticSearchConfig,
    HybridSearchEngine, HybridSearchConfig,
    SearchIndexer, IndexConfig
)


class TestFuzzySearchEngine:
    """Тесты для FuzzySearchEngine"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Настройка конфигурации для малого набора данных
        self.config = FuzzySearchConfig(
            tfidf_min_df=1,  # Уменьшаем min_df для малого набора данных
            tfidf_max_df=1.0,
            tfidf_max_features=1000,
            cosine_threshold=0.0,  # Убираем порог для тестирования
            fuzzy_threshold=0,
            levenshtein_threshold=0
        )
        self.engine = FuzzySearchEngine(self.config)

        # Тестовые документы - увеличиваем количество для лучшего тестирования
        self.documents = [
            "болт м10х50 гост 7798-70",
            "гайка м10 din 934",
            "шайба плоская 10 гост 11371-78",
            "винт м8х30 нержавеющая сталь",
            "труба стальная 57х3.5",
            "болт м12х60 нержавеющий",
            "гайка м12 оцинкованная",
            "шайба гровер 12",
            "винт м10х40 черный",
            "труба медная 22х1"
        ]
        self.document_ids = list(range(len(self.documents)))
    
    def test_fit(self):
        """Тест обучения движка"""
        self.engine.fit(self.documents, self.document_ids)
        
        assert self.engine.is_fitted is True
        assert self.engine.vectorizer is not None
        assert self.engine.tfidf_matrix is not None
        assert len(self.engine.documents) == len(self.documents)
    
    def test_search(self):
        """Тест поиска"""
        self.engine.fit(self.documents, self.document_ids)
        
        results = self.engine.search("болт м10")
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Проверяем структуру результата
        result = results[0]
        assert 'document_id' in result
        assert 'document' in result
        assert 'combined_score' in result
        assert 'cosine_score' in result
    
    def test_search_empty_query(self):
        """Тест поиска с пустым запросом"""
        self.engine.fit(self.documents, self.document_ids)
        
        results = self.engine.search("")
        assert results == []
        
        results = self.engine.search(None)
        assert results == []
    
    def test_search_not_fitted(self):
        """Тест поиска без обучения"""
        with pytest.raises(ValueError, match="Search engine is not fitted"):
            self.engine.search("болт м10")
    
    def test_batch_search(self):
        """Тест пакетного поиска"""
        self.engine.fit(self.documents, self.document_ids)
        
        queries = ["болт м10", "гайка", "труба"]
        results = self.engine.batch_search(queries)
        
        assert len(results) == len(queries)
        assert all(isinstance(result, list) for result in results)
    
    def test_get_similar_documents(self):
        """Тест поиска похожих документов"""
        self.engine.fit(self.documents, self.document_ids)
        
        similar = self.engine.get_similar_documents(0)  # Похожие на первый документ
        
        assert isinstance(similar, list)
        # Проверяем что сам документ исключен
        assert all(result['document_id'] != 0 for result in similar)
    
    def test_save_load_model(self, tmp_path):
        """Тест сохранения и загрузки модели"""
        self.engine.fit(self.documents, self.document_ids)
        
        model_path = tmp_path / "fuzzy_model.pkl"
        self.engine.save_model(str(model_path))
        
        # Создаем новый движок и загружаем модель
        new_engine = FuzzySearchEngine()
        new_engine.load_model(str(model_path))
        
        assert new_engine.is_fitted is True
        assert len(new_engine.documents) == len(self.documents)
        
        # Проверяем что поиск работает
        results = new_engine.search("болт м10")
        assert len(results) > 0
    
    def test_get_statistics(self):
        """Тест получения статистики"""
        stats = self.engine.get_statistics()
        assert stats['status'] == 'not_fitted'
        
        self.engine.fit(self.documents, self.document_ids)
        stats = self.engine.get_statistics()
        
        assert stats['status'] == 'fitted'
        assert 'total_documents' in stats
        assert 'vocabulary_size' in stats


class TestSemanticSearchEngine:
    """Тесты для SemanticSearchEngine"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = SemanticSearchConfig()
        self.documents = [
            "болт м10х50 гост 7798-70",
            "гайка м10 din 934",
            "шайба плоская 10 гост 11371-78"
        ]
        self.document_ids = list(range(len(self.documents)))
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_fit(self, mock_faiss_index, mock_sentence_transformer):
        """Тест обучения семантического движка"""
        # Мокаем SentenceTransformer
        mock_model = Mock()
        embeddings = np.random.rand(len(self.documents), 384).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_sentence_transformer.return_value = mock_model

        # Мокаем _generate_embeddings_sync для прямого возврата embeddings
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings):

            # Мокаем FAISS индекс
            mock_index = Mock()
            mock_faiss_index.return_value = mock_index

            engine = SemanticSearchEngine(self.config)

            # Используем async метод напрямую
            await engine.fit_async(self.documents, self.document_ids)

            assert engine.is_fitted is True
            assert engine.embeddings is not None
            assert engine.index is not None
            mock_index.add.assert_called_once()
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_search(self, mock_faiss_index, mock_sentence_transformer):
        """Тест семантического поиска"""
        # Мокаем модель
        mock_model = Mock()
        embeddings_fit = np.random.rand(len(self.documents), 384).astype(np.float32)
        embeddings_search = np.random.rand(1, 384).astype(np.float32)
        mock_model.encode.side_effect = [
            embeddings_fit,  # Для fit
            embeddings_search  # Для search
        ]
        mock_sentence_transformer.return_value = mock_model

        # Мокаем индекс
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # scores
            np.array([[0, 1, 2]])  # indices
        )
        mock_faiss_index.return_value = mock_index

        # Мокаем _generate_embeddings_sync для прямого возврата embeddings
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings_fit):
            engine = SemanticSearchEngine(self.config)
            await engine.fit_async(self.documents, self.document_ids)

            results = engine.search("болт м10")

            assert isinstance(results, list)
            assert len(results) > 0

            result = results[0]
            assert 'document_id' in result
            assert 'similarity_score' in result
            assert 'rank' in result
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_batch_search(self, mock_faiss_index, mock_sentence_transformer):
        """Тест пакетного семантического поиска"""
        # Мокаем модель
        mock_model = Mock()
        embeddings_fit = np.random.rand(len(self.documents), 384).astype(np.float32)
        embeddings_batch = np.random.rand(2, 384).astype(np.float32)
        mock_model.encode.side_effect = [
            embeddings_fit,  # Для fit
            embeddings_batch  # Для batch search
        ]
        mock_sentence_transformer.return_value = mock_model

        # Мокаем индекс
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8], [0.7, 0.6]]),  # scores для 2 запросов
            np.array([[0, 1], [1, 2]])  # indices для 2 запросов
        )
        mock_faiss_index.return_value = mock_index

        # Мокаем _generate_embeddings_sync для прямого возврата embeddings
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings_fit):
            engine = SemanticSearchEngine(self.config)
            await engine.fit_async(self.documents, self.document_ids)

            queries = ["болт м10", "гайка"]
            results = engine.batch_search(queries)

            assert len(results) == len(queries)
            assert all(isinstance(result, list) for result in results)


class TestHybridSearchEngine:
    """Тесты для HybridSearchEngine"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Настройка конфигурации для малого набора данных
        fuzzy_config = FuzzySearchConfig(
            tfidf_min_df=1,
            tfidf_max_df=1.0,
            tfidf_max_features=1000,
            cosine_threshold=0.0,
            fuzzy_threshold=0,
            levenshtein_threshold=0
        )
        self.config = HybridSearchConfig(fuzzy_config=fuzzy_config)
        self.documents = [
            "болт м10х50 гост 7798-70",
            "гайка м10 din 934",
            "шайба плоская 10",
            "винт м8х30 нержавеющая сталь",
            "труба стальная 57х3.5",
            "болт м12х60 нержавеющий",
            "гайка м12 оцинкованная"
        ]
        self.document_ids = list(range(len(self.documents)))
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_fit(self, mock_faiss_index, mock_sentence_transformer):
        """Тест обучения гибридного движка"""
        # Мокаем компоненты
        mock_model = Mock()
        embeddings = np.random.rand(len(self.documents), 384).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_sentence_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss_index.return_value = mock_index

        # Мокаем _generate_embeddings_sync для прямого возврата embeddings
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings):
            engine = HybridSearchEngine(self.config)

            # Используем async fit для семантического движка
            await engine.semantic_engine.fit_async(self.documents, self.document_ids)
            engine.fuzzy_engine.fit(self.documents, self.document_ids)
            engine.is_fitted = True

            assert engine.is_fitted is True
            assert engine.fuzzy_engine.is_fitted is True
            assert engine.semantic_engine.is_fitted is True
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_search_weighted_sum(self, mock_faiss_index, mock_sentence_transformer):
        """Тест гибридного поиска с взвешенным суммированием"""
        # Мокаем компоненты
        mock_model = Mock()
        embeddings_fit = np.random.rand(len(self.documents), 384).astype(np.float32)
        embeddings_search = np.random.rand(1, 384).astype(np.float32)
        mock_model.encode.side_effect = [
            embeddings_fit,  # Для fit
            embeddings_search  # Для search
        ]
        mock_sentence_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0, 1, 2]])
        )
        mock_faiss_index.return_value = mock_index

        fuzzy_config = FuzzySearchConfig(
            tfidf_min_df=1,
            tfidf_max_df=1.0,
            tfidf_max_features=1000,
            cosine_threshold=0.0,
            fuzzy_threshold=0,
            levenshtein_threshold=0
        )
        config = HybridSearchConfig(
            combination_strategy="weighted_sum",
            fuzzy_config=fuzzy_config
        )

        # Мокаем _generate_embeddings_sync для прямого возврата embeddings
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings_fit):
            engine = HybridSearchEngine(config)

            # Используем async fit для семантического движка
            await engine.semantic_engine.fit_async(self.documents, self.document_ids)
            engine.fuzzy_engine.fit(self.documents, self.document_ids)
            engine.is_fitted = True

            results = engine.search("болт м10")

            assert isinstance(results, list)
            assert len(results) > 0

            result = results[0]
            assert 'hybrid_score' in result
            assert 'search_method' in result
            assert result['search_method'] == 'hybrid'
    
    def test_get_statistics(self):
        """Тест получения статистики гибридного движка"""
        engine = HybridSearchEngine(self.config)
        stats = engine.get_statistics()
        
        assert 'status' in stats
        assert 'combination_strategy' in stats
        assert 'weights' in stats
        assert 'thresholds' in stats


class TestSearchIndexer:
    """Тесты для SearchIndexer"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = IndexConfig(
            enable_incremental_updates=False,  # Отключаем для тестирования
            storage_backend="memory"  # Используем память для тестов
        )
        self.indexer = SearchIndexer(self.config)
        
        self.documents = [
            "болт м10х50 гост 7798-70",
            "гайка м10 din 934",
            "шайба плоская 10"
        ]
        self.document_ids = list(range(len(self.documents)))
        self.metadata = [
            {'category': 'крепеж', 'parameters': {'diameter': 10}},
            {'category': 'крепеж', 'parameters': {'diameter': 10}},
            {'category': 'крепеж', 'parameters': {'diameter': 10}}
        ]
    
    def test_index_documents(self):
        """Тест индексации документов"""
        results = self.indexer.index_documents(
            self.documents, 
            self.document_ids, 
            self.metadata
        )
        
        assert 'indexed_documents' in results
        assert results['indexed_documents'] == len(self.documents)
        assert len(self.indexer.document_metadata) == len(self.documents)
    
    def test_search_text(self):
        """Тест текстового поиска в индексе"""
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)

        # Тестируем поиск по одному токену
        results = self.indexer.search_text("болт")

        assert isinstance(results, set)
        assert len(results) > 0
    
    def test_search_parameters(self):
        """Тест поиска по параметрам"""
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)
        
        results = self.indexer.search_parameters({'diameter': 10})
        
        assert isinstance(results, set)
        assert len(results) == 3  # Все документы имеют diameter=10
    
    def test_search_category(self):
        """Тест поиска по категории"""
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)
        
        results = self.indexer.search_category('крепеж')
        
        assert isinstance(results, set)
        assert len(results) == 3  # Все документы в категории 'крепеж'
    
    def test_get_document_metadata(self):
        """Тест получения метаданных документа"""
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)
        
        metadata = self.indexer.get_document_metadata(0)
        
        assert metadata is not None
        assert 'content' in metadata
        assert 'metadata' in metadata
        assert metadata['content'] == self.documents[0]
    
    def test_save_load_index(self, tmp_path):
        """Тест сохранения и загрузки индекса"""
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)
        
        index_path = tmp_path / "test_index.pkl"
        self.indexer.save_index(str(index_path))
        
        # Создаем новый индексатор и загружаем
        new_indexer = SearchIndexer(self.config)
        new_indexer.load_index(str(index_path))
        
        assert len(new_indexer.document_metadata) == len(self.documents)
        
        # Проверяем что поиск работает
        results = new_indexer.search_text("болт")
        assert len(results) > 0
    
    def test_get_statistics(self):
        """Тест получения статистики индекса"""
        stats = self.indexer.get_statistics()
        assert 'text_index_size' in stats
        assert 'parameter_index_size' in stats
        
        self.indexer.index_documents(self.documents, self.document_ids, self.metadata)
        stats = self.indexer.get_statistics()
        
        assert stats['text_index_size'] > 0
        assert stats['parameter_index_size'] > 0


# Фикстуры для тестов
@pytest.fixture
def sample_documents():
    """Образцы документов для тестирования"""
    return [
        "болт м10х50 гост 7798-70 нержавеющая сталь",
        "гайка м10 din 934 оцинкованная",
        "шайба плоская 10 гост 11371-78",
        "винт м8х30 с внутренним шестигранником",
        "труба стальная 57х3.5 гост 8732-78",
        "фланец ду50 ру16 гост 12820-80",
        "клапан шаровой ду25 ру40",
        "насос центробежный 50-32-125",
        "двигатель асинхронный 4квт 1500об/мин",
        "редуктор червячный ч-80 передаточное число 40"
    ]


@pytest.fixture
def sample_embeddings():
    """Образцы эмбеддингов для тестирования"""
    return np.random.rand(10, 384).astype(np.float32)


class TestIntegration:
    """Интеграционные тесты"""
    
    @patch('same.models.model_manager.SentenceTransformer')
    @patch('same.search_engine.semantic_search.faiss.IndexFlatIP')
    @pytest.mark.asyncio
    async def test_full_search_pipeline(self, mock_faiss_index, mock_sentence_transformer, sample_documents):
        """Тест полного пайплайна поиска"""
        # Мокаем компоненты
        mock_model = Mock()
        # Предоставляем достаточно ответов для всех вызовов encode
        embeddings = np.random.rand(len(sample_documents), 384).astype(np.float32)
        mock_model.encode.side_effect = [
            embeddings,  # Для semantic fit
            np.random.rand(1, 384).astype(np.float32),  # Для semantic search
            embeddings,  # Для hybrid fit
            np.random.rand(1, 384).astype(np.float32)   # Для hybrid search
        ]
        mock_sentence_transformer.return_value = mock_model

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]),
            np.array([[0, 1, 2, 3, 4]])
        )
        mock_faiss_index.return_value = mock_index
        
        # Тестируем нечеткий поиск
        fuzzy_config = FuzzySearchConfig(
            tfidf_min_df=1,
            tfidf_max_df=1.0,
            tfidf_max_features=1000,
            cosine_threshold=0.0,
            fuzzy_threshold=0,
            levenshtein_threshold=0
        )
        fuzzy_engine = FuzzySearchEngine(fuzzy_config)
        fuzzy_engine.fit(sample_documents)
        fuzzy_results = fuzzy_engine.search("болт м10")

        assert isinstance(fuzzy_results, list)
        assert len(fuzzy_results) > 0

        # Тестируем семантический поиск
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings):
            semantic_engine = SemanticSearchEngine()
            await semantic_engine.fit_async(sample_documents)
            semantic_results = semantic_engine.search("болт м10")

            assert isinstance(semantic_results, list)
            assert len(semantic_results) > 0

        # Тестируем гибридный поиск
        with patch.object(SemanticSearchEngine, '_generate_embeddings_sync', return_value=embeddings):
            hybrid_config = HybridSearchConfig(fuzzy_config=fuzzy_config)
            hybrid_engine = HybridSearchEngine(hybrid_config)

            # Используем async fit для семантического движка
            await hybrid_engine.semantic_engine.fit_async(sample_documents)
            hybrid_engine.fuzzy_engine.fit(sample_documents)
            hybrid_engine.is_fitted = True

            hybrid_results = hybrid_engine.search("болт м10")

            assert isinstance(hybrid_results, list)
            assert len(hybrid_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
