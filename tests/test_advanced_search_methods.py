#!/usr/bin/env python3
"""
Тесты для продвинутых методов поиска в системе SAMe

Этот модуль содержит тесты для проверки корректности работы
новых алгоритмов поиска и бенчмарки производительности.
"""

import sys
import unittest
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from same_clear.search import TokenSearchEngine, SearchResult, SearchConfig
    from src.same.excel_processor import ExcelProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the SAMe modules are properly installed")
    sys.exit(1)


class TestAdvancedSearchMethods(unittest.TestCase):
    """Тесты для продвинутых методов поиска"""
    
    @classmethod
    def setUpClass(cls):
        """Настройка тестового окружения"""
        cls.test_data_file = "test_final_improvements_output.csv"
        cls.tokenizer_config = "vectorized_tokenizer"
        
        # Создаем процессор и поисковый движок
        cls.processor = ExcelProcessor(tokenizer_config_name=cls.tokenizer_config)
        
        # Конфигурация с включенными всеми методами
        cls.config = SearchConfig(
            enable_trie_search=True,
            enable_inverted_index=True,
            enable_tfidf_search=True,
            enable_lsh_search=True,
            enable_spatial_search=True,
            enable_graph_search=False,  # Отключаем для тестов (медленный)
            max_results=50
        )
        
        cls.search_engine = TokenSearchEngine(cls.config)
        
        # Загружаем данные
        if Path(cls.test_data_file).exists():
            # Обучаем векторизатор если нужно
            if not cls.processor.tokenizer._vectorizer.is_fitted:
                df = pd.read_csv(cls.test_data_file)
                cls.processor.train_vectorizer_on_data(df, sample_size=500)
            
            # Загружаем данные в поисковый движок
            success = cls.search_engine.load_data(
                cls.test_data_file,
                cls.processor.tokenizer._vectorizer,
                cls.processor.tokenizer
            )
            
            if not success:
                raise Exception("Failed to load test data")
        else:
            raise Exception(f"Test data file not found: {cls.test_data_file}")
    
    def test_trie_search(self):
        """Тест префиксного поиска"""
        results = self.search_engine.prefix_search("свет", top_k=10)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Trie search should return results")
        
        # Проверяем, что все результаты содержат префикс
        for result in results:
            self.assertIsInstance(result, SearchResult)
            raw_name = result.raw_name.lower()
            self.assertTrue(
                any("свет" in word for word in raw_name.split()),
                f"Result should contain prefix 'свет': {raw_name}"
            )
    
    def test_inverted_index_search(self):
        """Тест поиска по обратному индексу"""
        results = self.search_engine.inverted_index_search("светильник LED", top_k=10)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Inverted index search should return results")
        
        for result in results:
            self.assertIsInstance(result, SearchResult)
            self.assertGreater(result.score, 0, "Score should be positive")
    
    def test_tfidf_search(self):
        """Тест TF-IDF поиска"""
        results = self.search_engine.tfidf_search("автомат защиты", top_k=10)
        
        self.assertIsInstance(results, list)
        
        if len(results) > 0:
            for result in results:
                self.assertIsInstance(result, SearchResult)
                self.assertGreaterEqual(result.score, 0, "TF-IDF score should be non-negative")
                self.assertLessEqual(result.score, 1, "TF-IDF score should be <= 1")
    
    def test_lsh_search(self):
        """Тест LSH поиска"""
        results = self.search_engine.lsh_search("болт М10", top_k=10)
        
        self.assertIsInstance(results, list)
        
        if len(results) > 0:
            for result in results:
                self.assertIsInstance(result, SearchResult)
                self.assertGreaterEqual(result.score, 0, "LSH score should be non-negative")
                self.assertLessEqual(result.score, 1, "LSH score should be <= 1")
    
    def test_spatial_search(self):
        """Тест пространственного поиска"""
        if self.search_engine.embeddings_matrix is not None:
            # Используем первый эмбеддинг как запрос
            query_embedding = self.search_engine.embeddings_matrix[0]
            results = self.search_engine.spatial_search(query_embedding, top_k=10)
            
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0, "Spatial search should return results")
            
            for result in results:
                self.assertIsInstance(result, SearchResult)
                self.assertGreater(result.score, 0, "Spatial score should be positive")
    
    def test_advanced_hybrid_search(self):
        """Тест продвинутого гибридного поиска"""
        results = self.search_engine.advanced_hybrid_search("кабель силовой", top_k=10)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Advanced hybrid search should return results")
        
        for result in results:
            self.assertIsInstance(result, SearchResult)
            self.assertGreater(result.score, 0, "Score should be positive")
            self.assertEqual(result.match_type, "advanced_hybrid")
            
            # Проверяем наличие детальных оценок
            if hasattr(result, 'method_scores'):
                self.assertIsInstance(result.method_scores, dict)
    
    def test_search_by_tokens_new_methods(self):
        """Тест основного метода поиска с новыми методами"""
        query = "выключатель автоматический"
        
        methods_to_test = [
            'advanced_hybrid', 'prefix', 'inverted_index', 
            'tfidf', 'lsh', 'spatial'
        ]
        
        for method in methods_to_test:
            with self.subTest(method=method):
                try:
                    results = self.search_engine.search_by_tokens(query, method, top_k=5)
                    self.assertIsInstance(results, list)
                    # Некоторые методы могут не возвращать результаты для конкретного запроса
                    if len(results) > 0:
                        for result in results:
                            self.assertIsInstance(result, SearchResult)
                except Exception as e:
                    self.fail(f"Method {method} failed with error: {e}")


class SearchBenchmark:
    """Класс для бенчмарков производительности поиска"""
    
    def __init__(self, search_engine: TokenSearchEngine):
        self.search_engine = search_engine
        self.test_queries = [
            "светильник LED 50W",
            "автомат дифференциальный АВДТ",
            "болт М10х50 ГОСТ",
            "кабель силовой ВВГ",
            "выключатель автоматический",
            "розетка электрическая",
            "трансформатор понижающий",
            "провод медный ПВС"
        ]
    
    def run_benchmark(self, methods: List[str] = None, iterations: int = 3) -> Dict[str, Any]:
        """
        Запуск бенчмарка производительности
        
        Args:
            methods: Список методов для тестирования
            iterations: Количество итераций для каждого теста
            
        Returns:
            Результаты бенчмарка
        """
        if methods is None:
            methods = [
                'token_id', 'semantic', 'hybrid', 'extended_hybrid',
                'advanced_hybrid', 'prefix', 'inverted_index', 
                'tfidf', 'lsh', 'spatial'
            ]
        
        results = {}
        
        for method in methods:
            print(f"Benchmarking method: {method}")
            method_results = {
                'total_time': 0.0,
                'avg_time': 0.0,
                'total_results': 0,
                'avg_results': 0.0,
                'errors': 0,
                'query_results': []
            }
            
            for query in self.test_queries:
                query_times = []
                query_results_count = []
                
                for _ in range(iterations):
                    try:
                        start_time = time.time()
                        search_results = self.search_engine.search_by_tokens(query, method, top_k=10)
                        end_time = time.time()
                        
                        execution_time = end_time - start_time
                        query_times.append(execution_time)
                        query_results_count.append(len(search_results))
                        
                    except Exception as e:
                        method_results['errors'] += 1
                        print(f"Error in {method} for query '{query}': {e}")
                
                if query_times:
                    avg_time = sum(query_times) / len(query_times)
                    avg_results = sum(query_results_count) / len(query_results_count)
                    
                    method_results['query_results'].append({
                        'query': query,
                        'avg_time': avg_time,
                        'avg_results': avg_results
                    })
                    
                    method_results['total_time'] += avg_time
                    method_results['total_results'] += avg_results
            
            # Вычисляем средние значения
            if method_results['query_results']:
                method_results['avg_time'] = method_results['total_time'] / len(method_results['query_results'])
                method_results['avg_results'] = method_results['total_results'] / len(method_results['query_results'])
            
            results[method] = method_results
        
        return results
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """Вывод результатов бенчмарка"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Сортируем по средней скорости
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"{'Method':<20} {'Avg Time (s)':<15} {'Avg Results':<12} {'Errors':<8}")
        print("-" * 60)
        
        for method, result in sorted_methods:
            print(f"{method:<20} {result['avg_time']:<15.4f} {result['avg_results']:<12.1f} {result['errors']:<8}")
        
        print("\nDetailed results by query:")
        print("-" * 80)
        
        for method, result in results.items():
            if result['query_results']:
                print(f"\n{method}:")
                for query_result in result['query_results']:
                    print(f"  '{query_result['query'][:30]}...': {query_result['avg_time']:.4f}s, {query_result['avg_results']:.0f} results")


if __name__ == "__main__":
    # Запуск тестов
    print("Running advanced search methods tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Запуск бенчмарка
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE BENCHMARK")
    print("="*80)
    
    try:
        # Создаем тестовое окружение для бенчмарка
        processor = ExcelProcessor(tokenizer_config_name="vectorized_tokenizer")
        config = SearchConfig(
            enable_trie_search=True,
            enable_inverted_index=True,
            enable_tfidf_search=True,
            enable_lsh_search=True,
            enable_spatial_search=True,
            enable_graph_search=False
        )
        search_engine = TokenSearchEngine(config)
        
        # Загружаем данные
        test_data_file = "test_final_improvements_output.csv"
        if Path(test_data_file).exists():
            if not processor.tokenizer._vectorizer.is_fitted:
                df = pd.read_csv(test_data_file)
                processor.train_vectorizer_on_data(df, sample_size=500)
            
            success = search_engine.load_data(
                test_data_file,
                processor.tokenizer._vectorizer,
                processor.tokenizer
            )
            
            if success:
                # Запускаем бенчмарк
                benchmark = SearchBenchmark(search_engine)
                results = benchmark.run_benchmark(iterations=2)  # Меньше итераций для быстроты
                benchmark.print_benchmark_results(results)
            else:
                print("Failed to load data for benchmark")
        else:
            print(f"Test data file not found: {test_data_file}")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
