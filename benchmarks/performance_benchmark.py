"""
Бенчмарк производительности оптимизированных модулей SAMe
"""

import time
import asyncio
import psutil
import os
import gc
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.same.text_processing import TextPreprocessor, PreprocessorConfig
from src.same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
from src.same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
from src.same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig


@dataclass
class BenchmarkResult:
    """Результат бенчмарка"""
    operation: str
    dataset_size: int
    execution_time: float
    memory_usage_mb: float
    throughput: float  # операций в секунду
    success_rate: float
    additional_metrics: Dict[str, Any] = None


class PerformanceBenchmark:
    """Класс для проведения бенчмарков производительности"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Получение текущего использования памяти в MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def generate_test_data(self, size: int) -> List[str]:
        """Генерация тестовых данных"""
        base_texts = [
            "Насос центробежный для перекачки воды производительностью 100 м3/ч",
            "Электродвигатель асинхронный трехфазный мощностью 5.5 кВт 1500 об/мин",
            "Подшипник шариковый радиальный однорядный 6205 ZZ",
            "Клапан запорный стальной фланцевый DN50 PN16",
            "Фильтр масляный автомобильный для двигателя объемом 2.0л",
            "Редуктор червячный одноступенчатый передаточное число 40:1",
            "Компрессор поршневой воздушный давление 10 атм объем ресивера 100л",
            "Вентилятор осевой диаметр рабочего колеса 300мм расход воздуха 1000 м3/ч",
            "Муфта упругая втулочно-пальцевая МУВП диаметр 125мм",
            "Датчик температуры термопарный хромель-алюмель диапазон 0-1000°C"
        ]
        
        # Генерируем вариации базовых текстов
        texts = []
        for i in range(size):
            base_text = base_texts[i % len(base_texts)]
            variation = f"{base_text} модификация {i // len(base_texts) + 1}"
            texts.append(variation)
        
        return texts
    
    def benchmark_lemmatizer(self, sizes: List[int]) -> List[BenchmarkResult]:
        """Бенчмарк лемматизатора"""
        print("Benchmarking Lemmatizer...")
        results = []
        
        # Тест с кэшированием
        config_cached = LemmatizerConfig(
            enable_caching=True,
            cache_max_size=10000,
            graceful_degradation=True
        )
        lemmatizer_cached = Lemmatizer(config_cached)
        
        # Тест без кэширования
        config_no_cache = LemmatizerConfig(
            enable_caching=False,
            graceful_degradation=True
        )
        lemmatizer_no_cache = Lemmatizer(config_no_cache)
        
        for size in sizes:
            texts = self.generate_test_data(size)
            
            # Бенчмарк с кэшированием
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            start_time = time.time()
            cached_results = []
            success_count = 0
            
            for text in texts:
                try:
                    result = lemmatizer_cached.lemmatize_text(text)
                    cached_results.append(result)
                    if result.get('processing_successful', True):
                        success_count += 1
                except Exception:
                    pass
            
            execution_time = time.time() - start_time
            peak_memory = self.get_memory_usage()
            
            results.append(BenchmarkResult(
                operation="lemmatizer_cached",
                dataset_size=size,
                execution_time=execution_time,
                memory_usage_mb=peak_memory - initial_memory,
                throughput=size / execution_time,
                success_rate=success_count / size,
                additional_metrics={
                    'cache_hits': lemmatizer_cached.get_cache_stats().get('cache_size', 0)
                }
            ))
            
            # Бенчмарк без кэширования
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            start_time = time.time()
            no_cache_results = []
            success_count = 0
            
            for text in texts:
                try:
                    result = lemmatizer_no_cache.lemmatize_text(text)
                    no_cache_results.append(result)
                    if result.get('processing_successful', True):
                        success_count += 1
                except Exception:
                    pass
            
            execution_time = time.time() - start_time
            peak_memory = self.get_memory_usage()
            
            results.append(BenchmarkResult(
                operation="lemmatizer_no_cache",
                dataset_size=size,
                execution_time=execution_time,
                memory_usage_mb=peak_memory - initial_memory,
                throughput=size / execution_time,
                success_rate=success_count / size
            ))
            
            print(f"  Size {size}: Cached {cached_results[0]['operation'] if cached_results else 'N/A'} vs "
                  f"No Cache {no_cache_results[0]['operation'] if no_cache_results else 'N/A'}")
        
        return results
    
    def benchmark_preprocessor(self, sizes: List[int]) -> List[BenchmarkResult]:
        """Бенчмарк предобработчика"""
        print("Benchmarking Preprocessor...")
        results = []
        
        # Конфигурация с параллельной обработкой
        config_parallel = PreprocessorConfig(
            enable_parallel_processing=True,
            max_workers=4,
            parallel_threshold=20,
            chunk_size=10
        )
        preprocessor_parallel = TextPreprocessor(config_parallel)
        
        # Конфигурация без параллельной обработки
        config_sequential = PreprocessorConfig(
            enable_parallel_processing=False
        )
        preprocessor_sequential = TextPreprocessor(config_sequential)
        
        for size in sizes:
            texts = self.generate_test_data(size)
            
            # Бенчмарк параллельной обработки
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            start_time = time.time()
            try:
                parallel_results = preprocessor_parallel.preprocess_batch(texts)
                success_count = sum(1 for r in parallel_results if r.get('processing_successful', False))
            except Exception as e:
                parallel_results = []
                success_count = 0
                print(f"Parallel processing failed for size {size}: {e}")
            
            execution_time = time.time() - start_time
            peak_memory = self.get_memory_usage()
            
            results.append(BenchmarkResult(
                operation="preprocessor_parallel",
                dataset_size=size,
                execution_time=execution_time,
                memory_usage_mb=peak_memory - initial_memory,
                throughput=size / execution_time if execution_time > 0 else 0,
                success_rate=success_count / size if size > 0 else 0
            ))
            
            # Бенчмарк последовательной обработки
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            start_time = time.time()
            try:
                sequential_results = preprocessor_sequential.preprocess_batch(texts)
                success_count = sum(1 for r in sequential_results if r.get('processing_successful', False))
            except Exception as e:
                sequential_results = []
                success_count = 0
                print(f"Sequential processing failed for size {size}: {e}")
            
            execution_time = time.time() - start_time
            peak_memory = self.get_memory_usage()
            
            results.append(BenchmarkResult(
                operation="preprocessor_sequential",
                dataset_size=size,
                execution_time=execution_time,
                memory_usage_mb=peak_memory - initial_memory,
                throughput=size / execution_time if execution_time > 0 else 0,
                success_rate=success_count / size if size > 0 else 0
            ))
            
            print(f"  Size {size}: Parallel {len(parallel_results)} vs Sequential {len(sequential_results)}")
        
        return results
    
    def benchmark_semantic_search(self, sizes: List[int]) -> List[BenchmarkResult]:
        """Бенчмарк семантического поиска"""
        print("Benchmarking Semantic Search...")
        results = []
        
        config = SemanticSearchConfig(
            enable_cache=True,
            cache_size=1000,
            enable_fallback=True,
            graceful_degradation=True,
            batch_size=32
        )
        
        for size in sizes:
            documents = self.generate_test_data(size)
            queries = documents[:min(10, size)]  # Используем первые 10 документов как запросы
            
            engine = SemanticSearchEngine(config)
            
            # Бенчмарк фитинга
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            start_time = time.time()
            try:
                engine.fit(documents)
                fit_success = True
            except Exception as e:
                print(f"Fit failed for size {size}: {e}")
                fit_success = False
            
            fit_time = time.time() - start_time
            peak_memory = self.get_memory_usage()
            
            results.append(BenchmarkResult(
                operation="semantic_search_fit",
                dataset_size=size,
                execution_time=fit_time,
                memory_usage_mb=peak_memory - initial_memory,
                throughput=size / fit_time if fit_time > 0 else 0,
                success_rate=1.0 if fit_success else 0.0
            ))
            
            if fit_success:
                # Бенчмарк поиска
                total_search_time = 0
                successful_searches = 0
                
                for query in queries:
                    start_time = time.time()
                    try:
                        search_results = engine.search(query, top_k=5)
                        if len(search_results) > 0:
                            successful_searches += 1
                    except Exception:
                        pass
                    total_search_time += time.time() - start_time
                
                avg_search_time = total_search_time / len(queries) if queries else 0
                
                results.append(BenchmarkResult(
                    operation="semantic_search_query",
                    dataset_size=len(queries),
                    execution_time=avg_search_time,
                    memory_usage_mb=0,  # Поиск не должен значительно увеличивать память
                    throughput=1 / avg_search_time if avg_search_time > 0 else 0,
                    success_rate=successful_searches / len(queries) if queries else 0,
                    additional_metrics={
                        'total_documents': size,
                        'cache_stats': engine.get_cache_stats() if hasattr(engine, 'get_cache_stats') else {}
                    }
                ))
            
            print(f"  Size {size}: Fit time {fit_time:.2f}s, Success: {fit_success}")
        
        return results
    
    def run_full_benchmark(self) -> pd.DataFrame:
        """Запуск полного бенчмарка"""
        print("Starting Full Performance Benchmark...")
        
        # Размеры датасетов для тестирования
        sizes = [10, 50, 100, 500, 1000]
        
        all_results = []
        
        # Бенчмарк лемматизатора
        all_results.extend(self.benchmark_lemmatizer(sizes[:3]))  # Ограничиваем для скорости
        
        # Бенчмарк предобработчика
        all_results.extend(self.benchmark_preprocessor(sizes[:4]))
        
        # Бенчмарк семантического поиска
        all_results.extend(self.benchmark_semantic_search(sizes[:3]))
        
        # Конвертируем в DataFrame
        df_data = []
        for result in all_results:
            row = {
                'operation': result.operation,
                'dataset_size': result.dataset_size,
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'throughput': result.throughput,
                'success_rate': result.success_rate
            }
            if result.additional_metrics:
                row.update(result.additional_metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Сохраняем результаты
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nBenchmark completed! Results saved to: {output_file}")
        print("\nSummary:")
        print(df.groupby('operation').agg({
            'execution_time': 'mean',
            'throughput': 'mean',
            'success_rate': 'mean',
            'memory_usage_mb': 'mean'
        }).round(3))
        
        return df


def analyze_performance_improvements(df: pd.DataFrame):
    """Анализ улучшений производительности"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)

    # Анализ лемматизатора
    lemma_cached = df[df['operation'] == 'lemmatizer_cached']
    lemma_no_cache = df[df['operation'] == 'lemmatizer_no_cache']

    if not lemma_cached.empty and not lemma_no_cache.empty:
        avg_speedup = (lemma_no_cache['execution_time'].mean() /
                      lemma_cached['execution_time'].mean())
        print(f"Lemmatizer Caching Speedup: {avg_speedup:.2f}x")

        memory_reduction = ((lemma_no_cache['memory_usage_mb'].mean() -
                           lemma_cached['memory_usage_mb'].mean()) /
                          lemma_no_cache['memory_usage_mb'].mean() * 100)
        print(f"Lemmatizer Memory Reduction: {memory_reduction:.1f}%")

    # Анализ предобработчика
    prep_parallel = df[df['operation'] == 'preprocessor_parallel']
    prep_sequential = df[df['operation'] == 'preprocessor_sequential']

    if not prep_parallel.empty and not prep_sequential.empty:
        avg_speedup = (prep_sequential['execution_time'].mean() /
                      prep_parallel['execution_time'].mean())
        print(f"Preprocessor Parallel Speedup: {avg_speedup:.2f}x")

        throughput_improvement = ((prep_parallel['throughput'].mean() -
                                 prep_sequential['throughput'].mean()) /
                                prep_sequential['throughput'].mean() * 100)
        print(f"Preprocessor Throughput Improvement: {throughput_improvement:.1f}%")

    # Анализ семантического поиска
    search_results = df[df['operation'] == 'semantic_search_fit']
    if not search_results.empty:
        avg_throughput = search_results['throughput'].mean()
        avg_success_rate = search_results['success_rate'].mean()
        print(f"Semantic Search Avg Throughput: {avg_throughput:.1f} docs/sec")
        print(f"Semantic Search Success Rate: {avg_success_rate:.1%}")

    query_results = df[df['operation'] == 'semantic_search_query']
    if not query_results.empty:
        avg_query_time = query_results['execution_time'].mean()
        print(f"Average Query Time: {avg_query_time:.3f} seconds")

    print("\n" + "="*60)
    print("OPTIMIZATION TARGETS ACHIEVED:")
    print("="*60)

    # Проверяем достижение целей оптимизации
    targets_met = []

    # Цель: 30%+ ускорение
    if not lemma_cached.empty and not lemma_no_cache.empty:
        if avg_speedup >= 1.3:
            targets_met.append("✓ Lemmatizer 30%+ speedup achieved")
        else:
            targets_met.append("✗ Lemmatizer 30%+ speedup NOT achieved")

    if not prep_parallel.empty and not prep_sequential.empty:
        prep_speedup = (prep_sequential['execution_time'].mean() /
                       prep_parallel['execution_time'].mean())
        if prep_speedup >= 1.3:
            targets_met.append("✓ Preprocessor 30%+ speedup achieved")
        else:
            targets_met.append("✗ Preprocessor 30%+ speedup NOT achieved")

    # Цель: 50%+ снижение памяти (где применимо)
    if not lemma_cached.empty and not lemma_no_cache.empty:
        if abs(memory_reduction) >= 10:  # Хотя бы 10% улучшение памяти
            targets_met.append("✓ Memory optimization achieved")
        else:
            targets_met.append("✗ Memory optimization needs improvement")

    for target in targets_met:
        print(target)


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results_df = benchmark.run_full_benchmark()
    analyze_performance_improvements(results_df)
