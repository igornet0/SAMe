"""
Unit tests for Enhanced Semantic Search
Tests multi-metric scoring, category filtering, and numeric penalty system
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from same_search.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
except ImportError:
    # Fallback на старый импорт
    from same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig


class TestEnhancedSemanticSearch:
    """Test suite for enhanced semantic search functionality"""
    
    @pytest.fixture
    def enhanced_config(self):
        """Create enhanced semantic search configuration"""
        return SemanticSearchConfig(
            enable_category_filtering=True,
            enable_enhanced_scoring=True,
            numeric_token_weight=0.3,
            semantic_weight=0.6,
            lexical_weight=0.3,
            key_term_weight=0.1,
            similarity_threshold=0.2,
            top_k_results=10,
            category_similarity_threshold=0.7,
            max_category_candidates=100
        )
    
    @pytest.fixture
    def test_catalog(self):
        """Create realistic test catalog with problematic and correct results"""
        documents = [
            # PROBLEMATIC RESULTS (should rank low)
            "сольвент 10 литров технический",
            "швеллер 10 мм стальной горячекатаный", 
            "полог 10 х 10 метров брезентовый",
            "растворитель 10 универсальный",
            "уголок 10 х 10 мм стальной",
            "краска 10 кг белая водоэмульсионная",
            "болт 10 мм оцинкованный с гайкой",
            
            # CORRECT RESULTS (should rank high)
            "ледоход проф 10 х10 противоскользящий",
            "ледоход профессиональный 10 шипов",
            "ледоходы проф 10 размер универсальный",
            "шипы ледоход 10 штук комплект",
            "ледоход проф 10 х10 xl размер",
            "ледоход профессиональный размер 10",
            "противоскользящие ледоходы 10 шипов",
            
            # MEDIUM RESULTS (same category, different product)
            "каска защитная 10 класс белая",
            "перчатки рабочие 10 размер хлопок",
            "сапоги защитные 10 размер кожаные"
        ]
        
        metadata = []
        for i, doc in enumerate(documents):
            meta = {'original_name': doc}
            
            # Categorize based on content
            doc_lower = doc.lower()
            if any(word in doc_lower for word in ['ледоход', 'шип', 'противоскользящ', 'каска', 'перчатки', 'сапоги']):
                meta['category'] = 'средства_защиты'
            elif any(word in doc_lower for word in ['сольвент', 'растворитель', 'краска']):
                meta['category'] = 'химия'
            elif any(word in doc_lower for word in ['швеллер', 'уголок']):
                meta['category'] = 'металлопрокат'
            elif 'болт' in doc_lower:
                meta['category'] = 'крепеж'
            elif 'полог' in doc_lower:
                meta['category'] = 'текстиль'
            else:
                meta['category'] = 'общие_товары'
            
            metadata.append(meta)
        
        return documents, metadata
    
    @pytest.fixture
    def search_engine(self, enhanced_config):
        """Create search engine with mocked model for testing"""
        engine = SemanticSearchEngine(enhanced_config)
        
        # Mock the model to avoid loading heavy dependencies
        engine._model = Mock()
        engine._initialized = True
        
        return engine
    
    def test_category_filtering_functionality(self, search_engine, test_catalog):
        """Test that category filtering works correctly"""
        documents, metadata = test_catalog
        
        # Mock embeddings for testing
        with patch.object(search_engine, '_generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            
            # Fit the engine
            search_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Test category filtering
            candidates = search_engine._get_search_candidates("ледоход проф 10", "средства_защиты")
            
            if candidates is not None:
                # Should only include items from средства_защиты category
                for doc_id in candidates:
                    category = search_engine.document_categories.get(doc_id, 'unknown')
                    assert category == 'средства_защиты', f"Wrong category for doc_id {doc_id}: {category}"
    
    def test_enhanced_scoring_components(self, search_engine):
        """Test individual components of enhanced scoring"""
        query = "ледоход проф 10"
        
        # Test cases with expected scoring behavior
        test_cases = [
            {
                'document': 'ледоход проф 10 х10',
                'expected_high_semantic': True,
                'expected_high_lexical': True,
                'expected_high_key_terms': True,
                'expected_low_numeric_penalty': True
            },
            {
                'document': 'сольвент 10',
                'expected_high_semantic': False,
                'expected_high_lexical': False,
                'expected_high_key_terms': False,
                'expected_low_numeric_penalty': False  # Should have numeric penalty
            }
        ]
        
        for case in test_cases:
            # Test lexical similarity (Levenshtein)
            lexical_score = search_engine._levenshtein_similarity(query.lower(), case['document'].lower())
            
            if case['expected_high_lexical']:
                assert lexical_score > 0.3, f"Expected high lexical score for '{case['document']}', got {lexical_score}"
            
            # Test key term scoring
            key_term_score = search_engine._calculate_key_term_score(query, case['document'])
            
            if case['expected_high_key_terms']:
                assert key_term_score > 0.3, f"Expected high key term score for '{case['document']}', got {key_term_score}"
            
            # Test numeric penalty
            numeric_penalty = search_engine._calculate_numeric_penalty(query, case['document'])
            
            if case['expected_low_numeric_penalty']:
                assert numeric_penalty < 0.3, f"Expected low numeric penalty for '{case['document']}', got {numeric_penalty}"
            else:
                # For cases like "сольвент 10" - should have higher penalty
                assert numeric_penalty >= 0.0, f"Numeric penalty should be non-negative, got {numeric_penalty}"
    
    def test_problematic_query_ranking(self, search_engine, test_catalog):
        """Test that problematic query now ranks correctly"""
        documents, metadata = test_catalog
        query = "ледоход проф 10"
        
        # Mock embeddings and search functionality
        with patch.object(search_engine, '_generate_embeddings') as mock_embeddings, \
             patch.object(search_engine, 'index') as mock_index:
            
            # Setup mock embeddings
            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            
            # Setup mock FAISS index
            mock_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05] + [0.01] * (len(documents) - 10)]),
                np.array([list(range(len(documents)))])
            )
            
            # Fit the engine
            search_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Perform search
            results = search_engine.search(query, top_k=10)
            
            # Analyze results
            correct_results = []
            wrong_results = []
            
            for result in results:
                doc = result['document'].lower()
                if any(word in doc for word in ['ледоход', 'шип', 'противоскользящ']):
                    correct_results.append(result)
                elif any(word in doc for word in ['сольвент', 'швеллер', 'полог', 'растворитель']):
                    wrong_results.append(result)
            
            # Verify improvements
            if correct_results and wrong_results:
                avg_correct_score = np.mean([r['similarity_score'] for r in correct_results])
                avg_wrong_score = np.mean([r['similarity_score'] for r in wrong_results])
                
                assert avg_correct_score > avg_wrong_score, \
                    f"Correct results should score higher: {avg_correct_score:.3f} vs {avg_wrong_score:.3f}"
    
    def test_numeric_penalty_system(self, search_engine):
        """Test numeric penalty system for preventing false matches"""
        query = "ледоход проф 10"
        
        test_cases = [
            # Cases that should have LOW penalty (legitimate matches)
            {
                'document': 'ледоход проф 10 х10',
                'expected_penalty': 'low',
                'reason': 'Has matching text terms beyond numbers'
            },
            {
                'document': 'ледоход профессиональный 10',
                'expected_penalty': 'low',
                'reason': 'Has matching text terms beyond numbers'
            },
            
            # Cases that should have HIGH penalty (numeric-only matches)
            {
                'document': 'сольвент 10',
                'expected_penalty': 'high',
                'reason': 'Only number matches, no text similarity'
            },
            {
                'document': 'краска 10 кг',
                'expected_penalty': 'high',
                'reason': 'Only number matches, no text similarity'
            }
        ]
        
        for case in test_cases:
            penalty = search_engine._calculate_numeric_penalty(query, case['document'])
            
            if case['expected_penalty'] == 'low':
                assert penalty < 0.3, f"Expected low penalty for '{case['document']}', got {penalty:.3f}. {case['reason']}"
            else:  # high penalty
                assert penalty > 0.2, f"Expected high penalty for '{case['document']}', got {penalty:.3f}. {case['reason']}"
    
    def test_multi_metric_scoring_integration(self, search_engine):
        """Test integration of multiple scoring metrics"""
        query = "ледоход проф 10"
        
        # Test with a perfect match
        perfect_match = "ледоход проф 10 х10"
        enhanced_score = search_engine._calculate_enhanced_score(
            query, perfect_match, 0.9, 'doc_1'  # High semantic score
        )
        
        # Test with a poor match
        poor_match = "сольвент 10"
        poor_score = search_engine._calculate_enhanced_score(
            query, poor_match, 0.3, 'doc_2'  # Low semantic score
        )
        
        assert enhanced_score > poor_score, \
            f"Perfect match should score higher: {enhanced_score:.3f} vs {poor_score:.3f}"
        
        # Enhanced score should be different from pure semantic score
        assert abs(enhanced_score - 0.9) > 0.05, \
            "Enhanced score should differ from pure semantic score"
    
    def test_category_similarity_matching(self, search_engine):
        """Test category similarity matching for filtering"""
        # Test exact category match
        exact_candidates = search_engine._get_search_candidates("ледоход проф 10", "средства_защиты")
        
        # Test similar category matching
        similar_candidates = search_engine._find_similar_categories("защита")
        
        if similar_candidates:
            assert any("защит" in cat.lower() for cat in similar_candidates), \
                "Should find categories containing 'защит'"
    
    def test_search_performance_with_enhancements(self, search_engine, test_catalog):
        """Test that enhancements don't significantly impact performance"""
        import time
        
        documents, metadata = test_catalog
        query = "ледоход проф 10"
        
        with patch.object(search_engine, '_generate_embeddings') as mock_embeddings, \
             patch.object(search_engine, 'index') as mock_index:
            
            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            mock_index.search.return_value = (
                np.array([[0.9] * len(documents)]),
                np.array([list(range(len(documents)))])
            )
            
            # Fit the engine
            search_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Measure search time
            start_time = time.time()
            for _ in range(10):  # Multiple searches
                results = search_engine.search(query, top_k=5)
            end_time = time.time()
            
            avg_search_time = (end_time - start_time) / 10
            
            assert avg_search_time < 1.0, f"Search too slow: {avg_search_time:.3f}s per query"
            assert len(results) > 0, "Should return results"
    
    def test_backward_compatibility(self, test_catalog):
        """Test that enhanced features can be disabled for backward compatibility"""
        # Create config with enhancements disabled
        basic_config = SemanticSearchConfig(
            enable_category_filtering=False,
            enable_enhanced_scoring=False,
            similarity_threshold=0.5,
            top_k_results=10
        )
        
        engine = SemanticSearchEngine(basic_config)
        engine._model = Mock()
        engine._initialized = True
        
        documents, metadata = test_catalog
        
        with patch.object(engine, '_generate_embeddings') as mock_embeddings, \
             patch.object(engine, 'index') as mock_index:
            
            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            mock_index.search.return_value = (
                np.array([[0.9] * len(documents)]),
                np.array([list(range(len(documents)))])
            )
            
            # Should work without metadata
            engine.fit(documents, list(range(len(documents))))
            results = engine.search("ледоход проф 10", top_k=5)
            
            assert len(results) > 0, "Basic search should still work"
            
            # Results should have basic structure
            for result in results:
                assert 'document_id' in result
                assert 'document' in result
                assert 'similarity_score' in result
    
    def test_edge_cases_and_error_handling(self, search_engine, test_catalog):
        """Test edge cases and error handling"""
        documents, metadata = test_catalog
        
        with patch.object(search_engine, '_generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            
            search_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Test edge cases
            edge_cases = [
                "",                    # Empty query
                "   ",                 # Whitespace only
                "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",  # Long Russian text
                "1234567890",          # Numbers only
                "a" * 1000,            # Very long query
            ]
            
            for case in edge_cases:
                try:
                    results = search_engine.search(case, top_k=5)
                    assert isinstance(results, list), f"Should return list for '{case}'"
                except Exception as e:
                    # Should handle gracefully, not crash
                    assert "not fitted" not in str(e).lower(), f"Unexpected error for '{case}': {e}"


class TestSemanticSearchConfiguration:
    """Test configuration options for semantic search"""
    
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Valid configuration
        valid_config = SemanticSearchConfig(
            numeric_token_weight=0.3,
            semantic_weight=0.6,
            lexical_weight=0.3,
            key_term_weight=0.1
        )
        
        assert valid_config.numeric_token_weight == 0.3
        assert valid_config.semantic_weight == 0.6
        
        # Test weight constraints
        assert 0.0 <= valid_config.numeric_token_weight <= 1.0
        assert 0.0 <= valid_config.semantic_weight <= 1.0
    
    def test_different_scoring_weights(self):
        """Test different scoring weight configurations"""
        configs = [
            # Semantic-heavy
            SemanticSearchConfig(semantic_weight=0.8, lexical_weight=0.2, key_term_weight=0.0),
            # Lexical-heavy  
            SemanticSearchConfig(semantic_weight=0.3, lexical_weight=0.6, key_term_weight=0.1),
            # Balanced
            SemanticSearchConfig(semantic_weight=0.5, lexical_weight=0.3, key_term_weight=0.2)
        ]
        
        for config in configs:
            engine = SemanticSearchEngine(config)
            assert engine.config.semantic_weight + engine.config.lexical_weight + engine.config.key_term_weight <= 1.1
    
    def test_category_filtering_options(self):
        """Test category filtering configuration options"""
        config = SemanticSearchConfig(
            enable_category_filtering=True,
            category_similarity_threshold=0.8,
            max_category_candidates=500
        )
        
        engine = SemanticSearchEngine(config)
        
        assert engine.config.enable_category_filtering == True
        assert engine.config.category_similarity_threshold == 0.8
        assert engine.config.max_category_candidates == 500


class TestSemanticSearchBenchmarks:
    """Performance and quality benchmarks for semantic search"""

    def test_search_quality_metrics(self, enhanced_config, test_catalog):
        """Test search quality metrics and improvements"""
        documents, metadata = test_catalog
        query = "ледоход проф 10"

        engine = SemanticSearchEngine(enhanced_config)
        engine._model = Mock()
        engine._initialized = True

        with patch.object(engine, '_generate_embeddings') as mock_embeddings, \
             patch.object(engine, 'index') as mock_index:

            mock_embeddings.return_value = np.random.rand(len(documents), 384)
            mock_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05] + [0.01] * (len(documents) - 10)]),
                np.array([list(range(len(documents)))])
            )

            engine.fit(documents, list(range(len(documents))), metadata)
            results = engine.search(query, top_k=10)

            # Calculate precision@5
            top_5 = results[:5]
            relevant_in_top_5 = sum(1 for r in top_5 if 'ледоход' in r['document'].lower())
            precision_at_5 = relevant_in_top_5 / 5

            assert precision_at_5 >= 0.6, f"Precision@5 too low: {precision_at_5:.2f}"

            # Calculate MRR (Mean Reciprocal Rank)
            first_relevant_rank = None
            for i, result in enumerate(results, 1):
                if 'ледоход' in result['document'].lower():
                    first_relevant_rank = i
                    break

            if first_relevant_rank:
                mrr = 1.0 / first_relevant_rank
                assert mrr >= 0.5, f"MRR too low: {mrr:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
