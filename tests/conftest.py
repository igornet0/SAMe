"""
Pytest configuration and shared fixtures for SAMe semantic search tests
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import Mock
import numpy as np

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_russian_products():
    """Sample Russian product names for testing"""
    return [
        # Safety equipment (средства защиты)
        "ледоход проф 10 х10 противоскользящий",
        "ледоход профессиональный 10 шипов",
        "ледоходы проф 10 размер универсальный",
        "каска защитная белая пластиковая",
        "перчатки рабочие хлопчатобумажные",
        "респиратор противопылевой",
        "очки защитные прозрачные",
        
        # Chemicals (химия)
        "сольвент 10 литров технический",
        "растворитель 646 универсальный",
        "краска водоэмульсионная белая",
        "ацетон технический чистый",
        "спирт этиловый медицинский",
        
        # Metal products (металлопрокат)
        "швеллер 10 мм стальной горячекатаный",
        "уголок 50x50 равнополочный",
        "лист стальной 2мм горячекатаный",
        "арматура 12 мм рифленая",
        "проволока стальная оцинкованная",
        
        # Fasteners (крепеж)
        "болт 10 мм оцинкованный с гайкой",
        "винт самонарезающий по металлу",
        "гайка М10 шестигранная",
        "шуруп по дереву желтый",
        "заклепка алюминиевая",
        
        # Textiles (текстиль)
        "полог 10 х 10 метров брезентовый",
        "тент защитный водостойкий",
        "ткань хлопчатобумажная суровая",
        "брезент огнестойкий",
        "мешок полипропиленовый"
    ]


@pytest.fixture
def sample_product_metadata(sample_russian_products):
    """Generate metadata for sample products"""
    metadata = []
    
    for product in sample_russian_products:
        meta = {"original_name": product}
        product_lower = product.lower()
        
        # Categorize based on keywords
        if any(word in product_lower for word in ['ледоход', 'каска', 'перчатки', 'респиратор', 'очки']):
            meta['category'] = 'средства_защиты'
        elif any(word in product_lower for word in ['сольвент', 'растворитель', 'краска', 'ацетон', 'спирт']):
            meta['category'] = 'химия'
        elif any(word in product_lower for word in ['швеллер', 'уголок', 'лист', 'арматура', 'проволока']):
            meta['category'] = 'металлопрокат'
        elif any(word in product_lower for word in ['болт', 'винт', 'гайка', 'шуруп', 'заклепка']):
            meta['category'] = 'крепеж'
        elif any(word in product_lower for word in ['полог', 'тент', 'ткань', 'брезент', 'мешок']):
            meta['category'] = 'текстиль'
        else:
            meta['category'] = 'общие_товары'
        
        metadata.append(meta)
    
    return metadata


@pytest.fixture
def problematic_test_cases():
    """Test cases that demonstrate the original similarity scoring problems"""
    return {
        "query": "ледоход проф 10",
        "correct_results": [
            "ледоход проф 10 х10 противоскользящий",
            "ледоход профессиональный 10 шипов",
            "ледоходы проф 10 размер универсальный",
            "шипы ледоход 10 штук комплект"
        ],
        "wrong_results": [
            "сольвент 10 литров технический",
            "швеллер 10 мм стальной горячекатаный",
            "полог 10 х 10 метров брезентовый",
            "растворитель 10 универсальный"
        ],
        "expected_improvements": {
            "category_filtering": "Should limit search to 'средства_защиты' category",
            "numeric_penalty": "Should penalize matches based only on '10'",
            "morphological": "Should normalize 'профессиональный' to 'проф'",
            "multi_metric": "Should use semantic + lexical + key term scoring"
        }
    }


@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings for testing"""
    def _generate_mock_embeddings(texts, dim=384):
        """Generate consistent mock embeddings for given texts"""
        np.random.seed(42)  # For reproducible results
        return np.random.rand(len(texts), dim).astype(np.float32)
    
    return _generate_mock_embeddings


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing"""
    class MockFAISSIndex:
        def __init__(self, embeddings):
            self.embeddings = embeddings
            self.ntotal = len(embeddings) if embeddings is not None else 0
        
        def search(self, query_embedding, k):
            if self.embeddings is None:
                return np.array([[0.0] * k]), np.array([[-1] * k])
            
            # Simple mock search - return indices in order with decreasing scores
            n_docs = len(self.embeddings)
            k = min(k, n_docs)
            
            scores = np.array([[0.9 - i * 0.1 for i in range(k)]])
            indices = np.array([list(range(k))])
            
            return scores, indices
        
        def add(self, embeddings):
            self.embeddings = embeddings
            self.ntotal = len(embeddings)
    
    return MockFAISSIndex


@pytest.fixture
def performance_test_data():
    """Generate larger dataset for performance testing"""
    products = []
    categories = ['средства_защиты', 'химия', 'металлопрокат', 'крепеж', 'текстиль']
    
    for i in range(100):
        category = categories[i % len(categories)]
        if category == 'средства_защиты':
            products.append(f"ледоход проф {i % 20} модель {i}")
        elif category == 'химия':
            products.append(f"сольвент {i % 20} литров марка {i}")
        elif category == 'металлопрокат':
            products.append(f"швеллер {i % 20} мм сталь {i}")
        elif category == 'крепеж':
            products.append(f"болт {i % 20} мм класс {i}")
        else:
            products.append(f"полог {i % 20} метров тип {i}")
    
    metadata = [{"category": categories[i % len(categories)]} for i in range(len(products))]
    
    return products, metadata


@pytest.fixture
def benchmark_queries():
    """Standard benchmark queries for testing"""
    return [
        {
            "query": "ледоход проф 10",
            "expected_category": "средства_защиты",
            "expected_keywords": ["ледоход", "проф"]
        },
        {
            "query": "сольвент технический",
            "expected_category": "химия",
            "expected_keywords": ["сольвент", "технический"]
        },
        {
            "query": "швеллер стальной",
            "expected_category": "металлопрокат",
            "expected_keywords": ["швеллер", "стальной"]
        },
        {
            "query": "болт оцинкованный",
            "expected_category": "крепеж",
            "expected_keywords": ["болт", "оцинкованный"]
        }
    ]


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Custom assertion helpers
class SearchTestHelpers:
    """Helper methods for search testing"""
    
    @staticmethod
    def assert_search_quality(results, query, expected_relevant_terms):
        """Assert search result quality"""
        assert len(results) > 0, "Should return results"
        
        # Check that results contain expected terms
        top_result = results[0]['document'].lower()
        relevant_found = any(term.lower() in top_result for term in expected_relevant_terms)
        assert relevant_found, f"Top result should contain relevant terms: {expected_relevant_terms}"
    
    @staticmethod
    def assert_ranking_improvement(results, correct_terms, wrong_terms):
        """Assert that correct results rank higher than wrong results"""
        correct_ranks = []
        wrong_ranks = []
        
        for i, result in enumerate(results):
            doc = result['document'].lower()
            if any(term.lower() in doc for term in correct_terms):
                correct_ranks.append(i + 1)
            elif any(term.lower() in doc for term in wrong_terms):
                wrong_ranks.append(i + 1)
        
        if correct_ranks and wrong_ranks:
            avg_correct_rank = sum(correct_ranks) / len(correct_ranks)
            avg_wrong_rank = sum(wrong_ranks) / len(wrong_ranks)
            
            assert avg_correct_rank < avg_wrong_rank, \
                f"Correct results should rank higher: {avg_correct_rank:.1f} vs {avg_wrong_rank:.1f}"
    
    @staticmethod
    def calculate_precision_at_k(results, relevant_terms, k=5):
        """Calculate precision@k metric"""
        top_k = results[:k]
        relevant_count = 0
        
        for result in top_k:
            doc = result['document'].lower()
            if any(term.lower() in doc for term in relevant_terms):
                relevant_count += 1
        
        return relevant_count / len(top_k) if top_k else 0.0


@pytest.fixture
def search_helpers():
    """Provide search test helper methods"""
    return SearchTestHelpers


# Skip tests that require heavy dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle optional dependencies"""
    skip_heavy = pytest.mark.skip(reason="Heavy dependencies not available")
    
    for item in items:
        # Skip tests that require actual ML models
        if "semantic" in item.name and "mock" not in item.name:
            # Check if we're in a CI environment or if models are not available
            try:
                import sentence_transformers
                import spacy
            except ImportError:
                item.add_marker(skip_heavy)
