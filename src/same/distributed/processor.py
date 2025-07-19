"""
Модуль распределенной обработки для масштабирования SAMe системы
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Режимы распределенной обработки"""
    MULTIPROCESSING = "multiprocessing"
    MULTITHREADING = "multithreading"
    ASYNC_CONCURRENT = "async_concurrent"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    """Конфигурация распределенной обработки"""
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    max_workers: Optional[int] = None
    chunk_size: int = 100
    enable_progress_tracking: bool = True
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 300
    retry_attempts: int = 3
    load_balancing: bool = True


class TaskResult:
    """Результат выполнения задачи"""
    
    def __init__(self, task_id: str, result: Any, execution_time: float, worker_id: str = None):
        self.task_id = task_id
        self.result = result
        self.execution_time = execution_time
        self.worker_id = worker_id
        self.timestamp = time.time()
        self.success = True
        self.error = None
    
    @classmethod
    def error_result(cls, task_id: str, error: Exception, execution_time: float):
        """Создание результата с ошибкой"""
        result = cls(task_id, None, execution_time)
        result.success = False
        result.error = str(error)
        return result


class DistributedProcessor:
    """Распределенный процессор для масштабирования обработки данных"""
    
    def __init__(self, config: DistributedConfig = None):
        self.config = config or DistributedConfig()
        self._setup_workers()
        self._task_queue = asyncio.Queue()
        self._results = {}
        self._performance_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0
        }
        
    def _setup_workers(self):
        """Настройка воркеров"""
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 8)
        
        logger.info(f"Setting up distributed processor with {self.config.max_workers} workers "
                   f"in {self.config.processing_mode.value} mode")
        
        if self.config.processing_mode in [ProcessingMode.MULTIPROCESSING, ProcessingMode.HYBRID]:
            self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        if self.config.processing_mode in [ProcessingMode.MULTITHREADING, ProcessingMode.HYBRID]:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_workers * 2)
    
    async def process_batch_distributed(
        self,
        data: List[Any],
        processing_function: Callable,
        function_args: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Распределенная пакетная обработка данных"""
        
        logger.info(f"Starting distributed processing of {len(data)} items")
        start_time = time.time()
        
        function_args = function_args or {}
        
        # Разбиваем данные на чанки
        chunks = self._create_chunks(data, self.config.chunk_size)
        
        # Создаем задачи
        tasks = []
        for i, chunk in enumerate(chunks):
            task_id = f"chunk_{i}"
            if self.config.processing_mode == ProcessingMode.ASYNC_CONCURRENT:
                task = self._process_chunk_async(task_id, chunk, processing_function, function_args)
            elif self.config.processing_mode == ProcessingMode.MULTIPROCESSING:
                task = self._process_chunk_multiprocessing(task_id, chunk, processing_function, function_args)
            elif self.config.processing_mode == ProcessingMode.MULTITHREADING:
                task = self._process_chunk_multithreading(task_id, chunk, processing_function, function_args)
            else:  # HYBRID
                task = self._process_chunk_hybrid(task_id, chunk, processing_function, function_args)
            
            tasks.append(task)
        
        # Выполняем задачи с отслеживанием прогресса
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress = completed / len(tasks)
                    await progress_callback(progress, completed, len(tasks))
                
                # Обновляем статистику
                self._update_stats(result)
                
            except Exception as e:
                logger.error(f"Task failed: {e}")
                error_result = TaskResult.error_result(f"task_{completed}", e, 0.0)
                results.append(error_result)
                self._update_stats(error_result)
        
        total_time = time.time() - start_time
        logger.info(f"Distributed processing completed in {total_time:.2f}s. "
                   f"Success rate: {self._calculate_success_rate():.1%}")
        
        return results
    
    async def _process_chunk_async(
        self,
        task_id: str,
        chunk: List[Any],
        processing_function: Callable,
        function_args: Dict[str, Any]
    ) -> TaskResult:
        """Асинхронная обработка чанка"""
        
        start_time = time.time()
        
        try:
            # Если функция асинхронная
            if asyncio.iscoroutinefunction(processing_function):
                result = await processing_function(chunk, **function_args)
            else:
                # Выполняем синхронную функцию в executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_executor, 
                    processing_function, 
                    chunk, 
                    **function_args
                )
            
            execution_time = time.time() - start_time
            return TaskResult(task_id, result, execution_time, "async_worker")
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult.error_result(task_id, e, execution_time)
    
    async def _process_chunk_multiprocessing(
        self,
        task_id: str,
        chunk: List[Any],
        processing_function: Callable,
        function_args: Dict[str, Any]
    ) -> TaskResult:
        """Обработка чанка в отдельном процессе"""
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_executor,
                _process_chunk_worker,
                chunk,
                processing_function,
                function_args
            )
            
            execution_time = time.time() - start_time
            return TaskResult(task_id, result, execution_time, "process_worker")
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult.error_result(task_id, e, execution_time)
    
    async def _process_chunk_multithreading(
        self,
        task_id: str,
        chunk: List[Any],
        processing_function: Callable,
        function_args: Dict[str, Any]
    ) -> TaskResult:
        """Обработка чанка в отдельном потоке"""
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_executor,
                processing_function,
                chunk,
                **function_args
            )
            
            execution_time = time.time() - start_time
            return TaskResult(task_id, result, execution_time, "thread_worker")
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult.error_result(task_id, e, execution_time)
    
    async def _process_chunk_hybrid(
        self,
        task_id: str,
        chunk: List[Any],
        processing_function: Callable,
        function_args: Dict[str, Any]
    ) -> TaskResult:
        """Гибридная обработка чанка (выбор оптимального метода)"""
        
        # Определяем оптимальный метод на основе характеристик задачи
        chunk_size = len(chunk)
        
        if chunk_size > 50:  # Большие чанки - в процессы
            return await self._process_chunk_multiprocessing(task_id, chunk, processing_function, function_args)
        elif asyncio.iscoroutinefunction(processing_function):  # Асинхронные функции
            return await self._process_chunk_async(task_id, chunk, processing_function, function_args)
        else:  # Маленькие чанки - в потоки
            return await self._process_chunk_multithreading(task_id, chunk, processing_function, function_args)
    
    def _create_chunks(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        """Разбиение данных на чанки"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    def _update_stats(self, result: TaskResult):
        """Обновление статистики производительности"""
        self._performance_stats['total_tasks'] += 1
        
        if result.success:
            self._performance_stats['completed_tasks'] += 1
        else:
            self._performance_stats['failed_tasks'] += 1
        
        self._performance_stats['total_processing_time'] += result.execution_time
        
        # Обновляем среднее время
        if self._performance_stats['total_tasks'] > 0:
            self._performance_stats['average_task_time'] = (
                self._performance_stats['total_processing_time'] / 
                self._performance_stats['total_tasks']
            )
    
    def _calculate_success_rate(self) -> float:
        """Вычисление коэффициента успешности"""
        total = self._performance_stats['total_tasks']
        if total == 0:
            return 0.0
        return self._performance_stats['completed_tasks'] / total
    
    async def process_text_batch_distributed(
        self,
        texts: List[str],
        preprocessor,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Специализированная распределенная обработка текстов"""
        
        async def text_processing_function(text_chunk: List[str]) -> List[Dict[str, Any]]:
            """Функция обработки чанка текстов"""
            return await preprocessor.preprocess_batch_async(text_chunk)
        
        results = await self.process_batch_distributed(
            texts,
            text_processing_function,
            progress_callback=progress_callback
        )
        
        # Объединяем результаты
        all_processed_texts = []
        for result in results:
            if result.success and result.result:
                all_processed_texts.extend(result.result)
        
        return all_processed_texts
    
    async def process_search_batch_distributed(
        self,
        queries: List[str],
        search_engine,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Специализированная распределенная обработка поисковых запросов"""
        
        async def search_function(query_chunk: List[str]) -> Dict[str, List[Dict[str, Any]]]:
            """Функция обработки чанка запросов"""
            results = {}
            for query in query_chunk:
                results[query] = search_engine.search(query)
            return results
        
        results = await self.process_batch_distributed(
            queries,
            search_function,
            progress_callback=progress_callback
        )
        
        # Объединяем результаты поиска
        all_search_results = {}
        for result in results:
            if result.success and result.result:
                all_search_results.update(result.result)
        
        return all_search_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        stats = self._performance_stats.copy()
        stats['success_rate'] = self._calculate_success_rate()
        stats['configuration'] = {
            'processing_mode': self.config.processing_mode.value,
            'max_workers': self.config.max_workers,
            'chunk_size': self.config.chunk_size
        }
        return stats
    
    async def shutdown(self):
        """Корректное завершение работы процессора"""
        logger.info("Shutting down distributed processor")
        
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=True)
        
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=True)
        
        logger.info("Distributed processor shutdown complete")


def _process_chunk_worker(chunk: List[Any], processing_function: Callable, function_args: Dict[str, Any]) -> Any:
    """Воркер функция для обработки чанка в отдельном процессе"""
    try:
        return processing_function(chunk, **function_args)
    except Exception as e:
        logger.error(f"Worker process error: {e}")
        raise


# Глобальный экземпляр распределенного процессора
distributed_processor = DistributedProcessor()
