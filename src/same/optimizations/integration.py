"""
Интеграционный модуль для всех оптимизаций SAMe системы
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..database.optimizations import db_optimizer
from ..models.quantization import model_quantizer, QuantizationConfig, QuantizationType
from ..distributed.processor import distributed_processor, DistributedConfig, ProcessingMode
from ..caching.advanced_cache import advanced_cache, CacheConfig
from ..models.model_manager import get_model_manager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuite:
    """Набор всех оптимизаций системы"""
    database_optimized: bool = False
    models_quantized: bool = False
    distributed_processing_enabled: bool = False
    advanced_caching_enabled: bool = False
    total_speedup: float = 1.0
    memory_reduction_percent: float = 0.0


class SAMeOptimizer:
    """Главный оптимизатор SAMe системы"""
    
    def __init__(self):
        self.optimization_suite = OptimizationSuite()
        self._performance_baseline = {}
        self._optimization_results = {}
        
    async def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Запуск комплексной оптимизации системы"""
        
        logger.info("🚀 Starting comprehensive SAMe system optimization")
        start_time = time.time()
        
        # Получаем базовые метрики производительности
        baseline = await self._measure_baseline_performance()
        self._performance_baseline = baseline
        
        optimization_results = {}
        
        try:
            # 1. Оптимизация базы данных
            logger.info("📊 Phase 1: Database optimization")
            db_result = await self._optimize_database()
            optimization_results['database'] = db_result
            self.optimization_suite.database_optimized = db_result['success']
            
            # 2. Квантизация моделей
            logger.info("🧠 Phase 2: Model quantization")
            model_result = await self._optimize_models()
            optimization_results['models'] = model_result
            self.optimization_suite.models_quantized = model_result['success']
            
            # 3. Настройка распределенной обработки
            logger.info("⚡ Phase 3: Distributed processing setup")
            distributed_result = await self._setup_distributed_processing()
            optimization_results['distributed'] = distributed_result
            self.optimization_suite.distributed_processing_enabled = distributed_result['success']
            
            # 4. Настройка продвинутого кэширования
            logger.info("💾 Phase 4: Advanced caching setup")
            cache_result = await self._setup_advanced_caching()
            optimization_results['caching'] = cache_result
            self.optimization_suite.advanced_caching_enabled = cache_result['success']
            
            # 5. Измерение итоговой производительности
            logger.info("📈 Phase 5: Performance measurement")
            final_performance = await self._measure_optimized_performance()
            optimization_results['performance'] = final_performance
            
            # Вычисляем общие улучшения
            total_improvements = self._calculate_total_improvements(baseline, final_performance)
            self.optimization_suite.total_speedup = total_improvements['speedup']
            self.optimization_suite.memory_reduction_percent = total_improvements['memory_reduction']
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ Comprehensive optimization completed in {total_time:.2f}s")
            logger.info(f"🎯 Total speedup: {self.optimization_suite.total_speedup:.1f}x")
            logger.info(f"💾 Memory reduction: {self.optimization_suite.memory_reduction_percent:.1f}%")
            
            return {
                'success': True,
                'optimization_suite': self.optimization_suite,
                'results': optimization_results,
                'total_time': total_time,
                'baseline_performance': baseline,
                'optimized_performance': final_performance
            }
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_results': optimization_results
            }
    
    async def _measure_baseline_performance(self) -> Dict[str, Any]:
        """Измерение базовой производительности системы"""
        
        logger.info("Measuring baseline performance...")
        
        # Тестовые данные
        test_texts = [
            "болт м10 оцинкованный",
            "гайка шестигранная м12",
            "шайба плоская 8мм",
            "винт с потайной головкой м6",
            "труба стальная 57х3.5"
        ] * 20  # 100 текстов для тестирования
        
        test_queries = [
            "болт м10",
            "гайка м12",
            "шайба 8",
            "винт м6",
            "труба стальная"
        ]
        
        baseline = {}
        
        # Измеряем время обработки текста
        from ..text_processing.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        start_time = time.time()
        try:
            # Используем синхронную версию для базового измерения
            results = []
            for text in test_texts[:10]:  # Ограничиваем для быстрого измерения
                result = await preprocessor.preprocess_text_async(text)
                results.append(result)
            
            baseline['text_processing_time'] = time.time() - start_time
            baseline['text_processing_success'] = len([r for r in results if r.get('processing_successful', False)])
        except Exception as e:
            logger.warning(f"Text processing baseline failed: {e}")
            baseline['text_processing_time'] = float('inf')
            baseline['text_processing_success'] = 0
        
        # Измеряем использование памяти
        import psutil
        import os
        process = psutil.Process(os.getpid())
        baseline['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        
        # Измеряем время загрузки модели
        start_time = time.time()
        try:
            model_manager = get_model_manager()
            await model_manager.get_sentence_transformer()
            baseline['model_loading_time'] = time.time() - start_time
        except Exception as e:
            logger.warning(f"Model loading baseline failed: {e}")
            baseline['model_loading_time'] = float('inf')
        
        logger.info(f"Baseline: {baseline['text_processing_time']:.3f}s text processing, "
                   f"{baseline['memory_usage_mb']:.1f}MB memory")
        
        return baseline
    
    async def _optimize_database(self) -> Dict[str, Any]:
        """Оптимизация базы данных"""
        
        try:
            # Создаем индексы
            indexes_result = await db_optimizer.create_search_indexes()
            
            # Тестируем производительность поиска
            test_query = "болт м10"
            start_time = time.time()
            search_results = await db_optimizer.optimized_search(test_query, limit=10)
            search_time = time.time() - start_time
            
            return {
                'success': True,
                'indexes_created': indexes_result,
                'test_search_time': search_time,
                'test_results_count': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _optimize_models(self) -> Dict[str, Any]:
        """Оптимизация и квантизация моделей"""
        
        try:
            model_manager = get_model_manager()
            
            # Получаем модель для квантизации
            sentence_transformer = await model_manager.get_sentence_transformer()
            
            # Настраиваем квантизацию
            quantization_config = QuantizationConfig(
                quantization_type=QuantizationType.DYNAMIC,
                preserve_accuracy=True,
                optimization_level=2
            )
            
            # Обновляем конфигурацию квантизатора
            model_quantizer.config = quantization_config
            
            # Выполняем квантизацию
            quantization_result = await model_quantizer.quantize_sentence_transformer(
                sentence_transformer,
                "sentence_transformer_main"
            )
            
            return {
                'success': True,
                'quantization_result': {
                    'speedup': quantization_result['performance_improvement'],
                    'memory_reduction': quantization_result['memory_reduction'],
                    'accuracy_preserved': quantization_result['accuracy_preserved']
                }
            }
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_distributed_processing(self) -> Dict[str, Any]:
        """Настройка распределенной обработки"""
        
        try:
            # Настраиваем конфигурацию
            distributed_config = DistributedConfig(
                processing_mode=ProcessingMode.HYBRID,
                max_workers=None,  # Автоопределение
                chunk_size=50,
                enable_progress_tracking=True
            )
            
            # Обновляем конфигурацию процессора
            distributed_processor.config = distributed_config
            distributed_processor._setup_workers()
            
            # Тестируем распределенную обработку
            test_data = list(range(100))
            
            def simple_processing_function(chunk):
                return [x * 2 for x in chunk]
            
            start_time = time.time()
            results = await distributed_processor.process_batch_distributed(
                test_data,
                simple_processing_function
            )
            processing_time = time.time() - start_time
            
            success_count = sum(1 for r in results if r.success)
            
            return {
                'success': True,
                'test_processing_time': processing_time,
                'test_success_rate': success_count / len(results) if results else 0,
                'workers_configured': distributed_processor.config.max_workers
            }
            
        except Exception as e:
            logger.error(f"Distributed processing setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_advanced_caching(self) -> Dict[str, Any]:
        """Настройка продвинутого кэширования"""
        
        try:
            # Тестируем кэш
            test_key = "test_optimization_key"
            test_value = {"data": "test_optimization_value", "timestamp": time.time()}
            
            # Сохраняем в кэш
            await advanced_cache.set(test_key, test_value)
            
            # Получаем из кэша
            start_time = time.time()
            cached_value = await advanced_cache.get(test_key)
            cache_access_time = time.time() - start_time
            
            # Получаем статистику
            cache_stats = advanced_cache.get_stats()
            
            return {
                'success': cached_value is not None,
                'cache_access_time': cache_access_time,
                'cache_stats': cache_stats
            }
            
        except Exception as e:
            logger.error(f"Advanced caching setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _measure_optimized_performance(self) -> Dict[str, Any]:
        """Измерение производительности после оптимизации"""
        
        # Используем те же тесты, что и для базового измерения
        return await self._measure_baseline_performance()
    
    def _calculate_total_improvements(
        self, 
        baseline: Dict[str, Any], 
        optimized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Вычисление общих улучшений"""
        
        improvements = {}
        
        # Вычисляем ускорение обработки текста
        baseline_time = baseline.get('text_processing_time', 1.0)
        optimized_time = optimized.get('text_processing_time', 1.0)
        
        if optimized_time > 0 and baseline_time > 0:
            improvements['speedup'] = baseline_time / optimized_time
        else:
            improvements['speedup'] = 1.0
        
        # Вычисляем сокращение памяти
        baseline_memory = baseline.get('memory_usage_mb', 0)
        optimized_memory = optimized.get('memory_usage_mb', 0)
        
        if baseline_memory > 0:
            improvements['memory_reduction'] = ((baseline_memory - optimized_memory) / baseline_memory) * 100
        else:
            improvements['memory_reduction'] = 0.0
        
        return improvements
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Получение статуса оптимизации"""
        
        return {
            'optimization_suite': self.optimization_suite,
            'database_optimizer_stats': db_optimizer.get_search_statistics() if hasattr(db_optimizer, 'get_search_statistics') else {},
            'model_quantizer_stats': model_quantizer.get_quantization_summary(),
            'distributed_processor_stats': distributed_processor.get_performance_stats(),
            'advanced_cache_stats': advanced_cache.get_stats()
        }


# Глобальный экземпляр оптимизатора
same_optimizer = SAMeOptimizer()
