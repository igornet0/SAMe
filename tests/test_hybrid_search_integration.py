"""
Integration tests for Hybrid Search Engine
Tests end-to-end search workflow with all enhancements and backward compatibility
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from same_search.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
except ImportError:
    # Fallback на старый импорт
    from same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
try:
    from same_search.search_engine.fuzzy_search import FuzzySearchConfig
except ImportError:
    # Fallback на старый импорт
    from same.search_engine.fuzzy_search import FuzzySearchConfig
try:
    from same_search.search_engine.semantic_search import SemanticSearchConfig
except ImportError:
    # Fallback на старый импорт
    from same.search_engine.semantic_search import SemanticSearchConfig
try:
    from same_search.categorization.category_classifier import CategoryClassifierConfig
except ImportError:
    from same.categorization import CategoryClassifierConfig


class TestHybridSearchIntegration:
    """Integration tests for the complete hybrid search system"""
    
    @pytest.fixture
    def enhanced_hybrid_config(self):
        """Create enhanced hybrid search configuration"""
        return HybridSearchConfig(
            # Enhanced features
            enable_category_filtering=True,
            combination_strategy="weighted_sum",
            
            # Search weights
            semantic_weight=0.6,
            fuzzy_weight=0.4,
            
            # Performance settings
            enable_parallel_search=True,
            max_workers=2,
            
            # Result limits
            max_candidates_per_method=50,
            final_top_k=10,
            
            # Component configs
            semantic_config=SemanticSearchConfig(
                enable_category_filtering=True,
                enable_enhanced_scoring=True,
                numeric_token_weight=0.3,
                semantic_weight=0.6,
                lexical_weight=0.3,
                key_term_weight=0.1
            ),
            fuzzy_config=FuzzySearchConfig(
                similarity_threshold=0.5
            ),
            category_config=CategoryClassifierConfig(
                min_confidence=0.6
            )
        )
    
    @pytest.fixture
    def realistic_catalog(self):
        """Create realistic product catalog for integration testing"""
        products = [
            # PROBLEMATIC RESULTS (should rank low after improvements)
            {"name": "сольвент 10 литров технический", "category": "химия"},
            {"name": "швеллер 10 мм стальной горячекатаный", "category": "металлопрокат"},
            {"name": "полог 10 х 10 метров брезентовый", "category": "текстиль"},
            {"name": "растворитель 10 универсальный", "category": "химия"},
            {"name": "уголок 10 х 10 мм стальной", "category": "металлопрокат"},
            {"name": "краска 10 кг белая водоэмульсионная", "category": "химия"},
            {"name": "болт 10 мм оцинкованный с гайкой", "category": "крепеж"},
            {"name": "труба 10 мм стальная водогазопроводная", "category": "трубы"},
            
            # CORRECT RESULTS (should rank high after improvements)
            {"name": "ледоход проф 10 х10 противоскользящий", "category": "средства_защиты"},
            {"name": "ледоход профессиональный 10 шипов", "category": "средства_защиты"},
            {"name": "ледоходы проф 10 размер универсальный", "category": "средства_защиты"},
            {"name": "шипы ледоход 10 штук комплект", "category": "средства_защиты"},
            {"name": "ледоход проф 10 х10 xl размер", "category": "средства_защиты"},
            {"name": "ледоход профессиональный размер 10", "category": "средства_защиты"},
            {"name": "противоскользящие ледоходы 10 шипов", "category": "средства_защиты"},
            {"name": "ледоход проф модель 10 усиленный", "category": "средства_защиты"},
            
            # MEDIUM RESULTS (same category, different product)
            {"name": "каска защитная 10 класс белая", "category": "средства_защиты"},
            {"name": "перчатки рабочие 10 размер хлопок", "category": "средства_защиты"},
            {"name": "сапоги защитные 10 размер кожаные", "category": "средства_защиты"},
            {"name": "жилет сигнальный 10 класс оранжевый", "category": "средства_защиты"},
        ]
        
        documents = [p["name"] for p in products]
        metadata = [{"category": p["category"], "original_name": p["name"]} for p in products]
        
        return documents, metadata
    
    @pytest.fixture
    def hybrid_engine(self, enhanced_hybrid_config):
        """Create hybrid search engine with mocked components"""
        engine = HybridSearchEngine(enhanced_hybrid_config)
        
        # Mock the underlying engines to avoid heavy dependencies
        engine.fuzzy_engine = Mock()
        engine.semantic_engine = Mock()
        engine.category_classifier = Mock()
        
        return engine
    
    def test_end_to_end_search_workflow(self, hybrid_engine, realistic_catalog):
        """Test complete end-to-end search workflow"""
        documents, metadata = realistic_catalog
        query = "ледоход проф 10"
        
        # Mock the fit process
        with patch.object(hybrid_engine.fuzzy_engine, 'fit'), \
             patch.object(hybrid_engine.semantic_engine, 'fit'):
            
            # Fit the engine
            hybrid_engine.fit(documents, list(range(len(documents))), metadata)
            assert hybrid_engine.is_fitted == True
            
            # Mock search results from individual engines
            fuzzy_results = [
                {'document_id': 8, 'document': documents[8], 'similarity_score': 0.85, 'rank': 1},
                {'document_id': 9, 'document': documents[9], 'similarity_score': 0.80, 'rank': 2},
                {'document_id': 0, 'document': documents[0], 'similarity_score': 0.75, 'rank': 3}
            ]
            
            semantic_results = [
                {'document_id': 8, 'document': documents[8], 'similarity_score': 0.90, 'rank': 1, 'category': 'средства_защиты'},
                {'document_id': 10, 'document': documents[10], 'similarity_score': 0.85, 'rank': 2, 'category': 'средства_защиты'},
                {'document_id': 0, 'document': documents[0], 'similarity_score': 0.60, 'rank': 3, 'category': 'химия'}
            ]
            
            hybrid_engine.fuzzy_engine.search.return_value = fuzzy_results
            hybrid_engine.semantic_engine.search.return_value = semantic_results
            
            # Mock category classification
            hybrid_engine.category_classifier.classify.return_value = ("средства_защиты", 0.8)
            
            # Perform search
            results = hybrid_engine.search(query, top_k=5)
            
            # Verify results structure
            assert isinstance(results, list), "Results should be a list"
            assert len(results) > 0, "Should return results"
            
            for result in results:
                assert 'document_id' in result
                assert 'document' in result
                assert 'hybrid_score' in result
                assert isinstance(result['hybrid_score'], (int, float))
    
    def test_category_integration(self, hybrid_engine, realistic_catalog):
        """Test integration with category classification"""
        documents, metadata = realistic_catalog
        query = "ледоход проф 10"
        
        # Mock category classifier
        hybrid_engine.category_classifier.classify.return_value = ("средства_защиты", 0.8)
        
        with patch.object(hybrid_engine.fuzzy_engine, 'fit'), \
             patch.object(hybrid_engine.semantic_engine, 'fit'):
            
            hybrid_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Verify that metadata was processed correctly
            assert hasattr(hybrid_engine.semantic_engine, 'fit')
            
            # Check that category classification was called during fit
            if hybrid_engine.config.enable_category_filtering:
                # Should have processed categories for documents without them
                call_args = hybrid_engine.semantic_engine.fit.call_args
                if call_args and len(call_args[0]) > 2:
                    fitted_metadata = call_args[0][2]
                    assert isinstance(fitted_metadata, list)
                    assert len(fitted_metadata) == len(documents)
    
    def test_parallel_vs_sequential_search(self, enhanced_hybrid_config, realistic_catalog):
        """Test parallel vs sequential search performance and consistency"""
        documents, metadata = realistic_catalog
        query = "ледоход проф 10"
        
        # Test parallel search
        parallel_config = enhanced_hybrid_config
        parallel_config.enable_parallel_search = True
        parallel_engine = HybridSearchEngine(parallel_config)
        parallel_engine.fuzzy_engine = Mock()
        parallel_engine.semantic_engine = Mock()
        
        # Test sequential search
        sequential_config = enhanced_hybrid_config
        sequential_config.enable_parallel_search = False
        sequential_engine = HybridSearchEngine(sequential_config)
        sequential_engine.fuzzy_engine = Mock()
        sequential_engine.semantic_engine = Mock()
        
        # Mock results
        mock_fuzzy_results = [{'document_id': 0, 'document': documents[0], 'similarity_score': 0.8}]
        mock_semantic_results = [{'document_id': 1, 'document': documents[1], 'similarity_score': 0.9}]
        
        for engine in [parallel_engine, sequential_engine]:
            engine.fuzzy_engine.search.return_value = mock_fuzzy_results
            engine.semantic_engine.search.return_value = mock_semantic_results
            
            with patch.object(engine.fuzzy_engine, 'fit'), \
                 patch.object(engine.semantic_engine, 'fit'):
                
                engine.fit(documents, list(range(len(documents))), metadata)
                
                # Measure search time
                start_time = time.time()
                results = engine.search(query, top_k=5)
                search_time = time.time() - start_time
                
                assert len(results) > 0, f"No results from {'parallel' if engine.config.enable_parallel_search else 'sequential'} search"
                assert search_time < 5.0, f"Search too slow: {search_time:.2f}s"
    
    def test_backward_compatibility(self, realistic_catalog):
        """Test backward compatibility with existing APIs"""
        documents, metadata = realistic_catalog
        
        # Create basic config without enhancements
        basic_config = HybridSearchConfig(
            enable_category_filtering=False,
            combination_strategy="weighted_sum",
            semantic_weight=0.6,
            fuzzy_weight=0.4
        )
        
        engine = HybridSearchEngine(basic_config)
        engine.fuzzy_engine = Mock()
        engine.semantic_engine = Mock()
        
        # Should work with old-style fit (no metadata)
        with patch.object(engine.fuzzy_engine, 'fit'), \
             patch.object(engine.semantic_engine, 'fit'):
            
            engine.fit(documents, list(range(len(documents))))
            assert engine.is_fitted == True
            
            # Mock search results
            engine.fuzzy_engine.search.return_value = [
                {'document_id': 0, 'document': documents[0], 'similarity_score': 0.8}
            ]
            engine.semantic_engine.search.return_value = [
                {'document_id': 1, 'document': documents[1], 'similarity_score': 0.9}
            ]
            
            # Should work with basic search
            results = engine.search("test query", top_k=5)
            assert isinstance(results, list)
    
    def test_metadata_handling(self, hybrid_engine, realistic_catalog):
        """Test proper handling of metadata throughout the pipeline"""
        documents, metadata = realistic_catalog
        
        with patch.object(hybrid_engine.fuzzy_engine, 'fit'), \
             patch.object(hybrid_engine.semantic_engine, 'fit'):
            
            # Test with complete metadata
            hybrid_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Verify semantic engine received metadata
            semantic_fit_calls = hybrid_engine.semantic_engine.fit.call_args_list
            assert len(semantic_fit_calls) > 0
            
            # Test with partial metadata
            partial_metadata = metadata[:len(metadata)//2]
            hybrid_engine.fit(documents, list(range(len(documents))), partial_metadata)
            
            # Should handle gracefully
            assert hybrid_engine.is_fitted == True
    
    def test_error_handling_and_robustness(self, hybrid_engine, realistic_catalog):
        """Test error handling and system robustness"""
        documents, metadata = realistic_catalog
        
        # Test with empty documents
        try:
            hybrid_engine.fit([], [])
            assert False, "Should raise error for empty documents"
        except (ValueError, AssertionError):
            pass  # Expected
        
        # Test with mismatched documents and IDs
        try:
            hybrid_engine.fit(documents, list(range(len(documents) - 1)))
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            assert "mismatch" in str(e).lower() or "length" in str(e).lower()
        
        # Test search before fitting
        try:
            results = hybrid_engine.search("test query")
            assert False, "Should raise error when not fitted"
        except (ValueError, RuntimeError):
            pass  # Expected
    
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test valid configuration
        valid_config = HybridSearchConfig(
            semantic_weight=0.6,
            fuzzy_weight=0.4,
            enable_category_filtering=True
        )
        
        engine = HybridSearchEngine(valid_config)
        assert engine.config.semantic_weight == 0.6
        assert engine.config.fuzzy_weight == 0.4
        
        # Test weight normalization
        assert abs(engine.config.semantic_weight + engine.config.fuzzy_weight - 1.0) < 0.01
    
    def test_search_result_quality_metrics(self, hybrid_engine, realistic_catalog):
        """Test search result quality and ranking improvements"""
        documents, metadata = realistic_catalog
        query = "ледоход проф 10"
        
        with patch.object(hybrid_engine.fuzzy_engine, 'fit'), \
             patch.object(hybrid_engine.semantic_engine, 'fit'):
            
            hybrid_engine.fit(documents, list(range(len(documents))), metadata)
            
            # Mock results that simulate the improvement
            # Correct results should have higher scores
            correct_results = [
                {'document_id': 8, 'document': documents[8], 'similarity_score': 0.95, 'category': 'средства_защиты'},
                {'document_id': 9, 'document': documents[9], 'similarity_score': 0.90, 'category': 'средства_защиты'},
                {'document_id': 10, 'document': documents[10], 'similarity_score': 0.85, 'category': 'средства_защиты'}
            ]
            
            # Wrong results should have lower scores
            wrong_results = [
                {'document_id': 0, 'document': documents[0], 'similarity_score': 0.40, 'category': 'химия'},
                {'document_id': 1, 'document': documents[1], 'similarity_score': 0.35, 'category': 'металлопрокат'}
            ]
            
            all_results = correct_results + wrong_results
            
            hybrid_engine.fuzzy_engine.search.return_value = all_results
            hybrid_engine.semantic_engine.search.return_value = all_results
            
            # Mock category classification
            hybrid_engine.category_classifier.classify.return_value = ("средства_защиты", 0.8)
            
            results = hybrid_engine.search(query, top_k=10)
            
            if results:
                # Check that results are properly ranked
                scores = [r.get('hybrid_score', r.get('similarity_score', 0)) for r in results]
                assert scores == sorted(scores, reverse=True), "Results should be sorted by score"
                
                # Check precision in top results
                top_3 = results[:3]
                relevant_in_top_3 = sum(1 for r in top_3 if 'ледоход' in r['document'].lower())
                precision_at_3 = relevant_in_top_3 / 3 if len(top_3) == 3 else 0
                
                # With improvements, should have good precision
                assert precision_at_3 >= 0.5, f"Precision@3 too low: {precision_at_3:.2f}"


class TestHybridSearchPerformance:
    """Performance tests for hybrid search system"""
    
    def test_search_latency(self, enhanced_hybrid_config):
        """Test search latency under load"""
        engine = HybridSearchEngine(enhanced_hybrid_config)
        engine.fuzzy_engine = Mock()
        engine.semantic_engine = Mock()
        
        # Mock fast responses
        engine.fuzzy_engine.search.return_value = []
        engine.semantic_engine.search.return_value = []
        
        with patch.object(engine.fuzzy_engine, 'fit'), \
             patch.object(engine.semantic_engine, 'fit'):
            
            # Fit with minimal data
            engine.fit(["test doc"], [0])
            
            # Measure search latency
            queries = ["test query"] * 100
            
            start_time = time.time()
            for query in queries:
                engine.search(query, top_k=5)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / len(queries)
            assert avg_latency < 0.1, f"Average search latency too high: {avg_latency:.3f}s"
    
    def test_memory_usage(self, enhanced_hybrid_config):
        """Test memory usage with large catalogs"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = HybridSearchEngine(enhanced_hybrid_config)
        engine.fuzzy_engine = Mock()
        engine.semantic_engine = Mock()
        
        # Simulate large catalog
        large_documents = [f"test document {i}" for i in range(1000)]
        large_metadata = [{"category": "test"} for _ in range(1000)]
        
        with patch.object(engine.fuzzy_engine, 'fit'), \
             patch.object(engine.semantic_engine, 'fit'):
            
            engine.fit(large_documents, list(range(len(large_documents))), large_metadata)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not use excessive memory
            assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
