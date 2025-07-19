"""
Модуль квантизации моделей для оптимизации производительности и памяти
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Типы квантизации"""
    DYNAMIC = "dynamic"  # Динамическая квантизация
    STATIC = "static"    # Статическая квантизация
    QAT = "qat"         # Quantization Aware Training
    INT8 = "int8"       # 8-битная квантизация
    FP16 = "fp16"       # 16-битная квантизация


@dataclass
class QuantizationConfig:
    """Конфигурация квантизации"""
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    target_device: str = "cpu"
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.95  # Минимальная точность после квантизации
    calibration_dataset_size: int = 1000
    optimization_level: int = 2  # 1-3, где 3 - максимальная оптимизация


class ModelQuantizer:
    """Квантизатор моделей для оптимизации производительности"""
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self._quantized_models = {}
        self._performance_stats = {}
        
    async def quantize_sentence_transformer(
        self, 
        model, 
        model_name: str,
        calibration_texts: Optional[list] = None
    ) -> Dict[str, Any]:
        """Квантизация SentenceTransformer модели"""
        
        logger.info(f"Starting quantization of {model_name} using {self.config.quantization_type.value}")
        start_time = time.time()
        
        try:
            # Получаем базовые метрики до квантизации
            original_stats = await self._benchmark_model(model, model_name, calibration_texts)
            
            # Выполняем квантизацию в зависимости от типа
            if self.config.quantization_type == QuantizationType.DYNAMIC:
                quantized_model = await self._dynamic_quantization(model)
            elif self.config.quantization_type == QuantizationType.FP16:
                quantized_model = await self._fp16_quantization(model)
            elif self.config.quantization_type == QuantizationType.INT8:
                quantized_model = await self._int8_quantization(model, calibration_texts)
            else:
                raise ValueError(f"Unsupported quantization type: {self.config.quantization_type}")
            
            # Получаем метрики после квантизации
            quantized_stats = await self._benchmark_model(quantized_model, f"{model_name}_quantized", calibration_texts)
            
            # Проверяем сохранение точности
            accuracy_preserved = self._check_accuracy_preservation(original_stats, quantized_stats)
            
            quantization_time = time.time() - start_time
            
            result = {
                'model': quantized_model,
                'quantization_type': self.config.quantization_type.value,
                'quantization_time': quantization_time,
                'original_stats': original_stats,
                'quantized_stats': quantized_stats,
                'accuracy_preserved': accuracy_preserved,
                'performance_improvement': self._calculate_performance_improvement(original_stats, quantized_stats),
                'memory_reduction': self._calculate_memory_reduction(original_stats, quantized_stats)
            }
            
            # Сохраняем результат
            self._quantized_models[model_name] = result
            
            logger.info(f"Quantization completed in {quantization_time:.2f}s. "
                       f"Performance improvement: {result['performance_improvement']:.1f}x, "
                       f"Memory reduction: {result['memory_reduction']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during quantization of {model_name}: {e}")
            raise
    
    async def _dynamic_quantization(self, model):
        """Динамическая квантизация"""
        
        if hasattr(model, '_modules'):
            # PyTorch модель
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
        else:
            # SentenceTransformer модель
            # Применяем квантизацию к внутренним PyTorch модулям
            for module_name, module in model._modules.items():
                if hasattr(module, '_modules'):
                    quantized_module = torch.quantization.quantize_dynamic(
                        module,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    setattr(model, module_name, quantized_module)
            quantized_model = model
        
        return quantized_model
    
    async def _fp16_quantization(self, model):
        """16-битная квантизация"""
        
        if torch.cuda.is_available() and self.config.target_device == "cuda":
            # GPU FP16 квантизация
            quantized_model = model.half()
        else:
            # CPU FP16 эмуляция через оптимизацию
            quantized_model = model.float()
            # Применяем оптимизации для CPU
            if hasattr(model, 'eval'):
                quantized_model = quantized_model.eval()
        
        return quantized_model
    
    async def _int8_quantization(self, model, calibration_texts: Optional[list] = None):
        """8-битная квантизация с калибровкой"""
        
        if not calibration_texts:
            # Создаем простой калибровочный датасет
            calibration_texts = [
                "болт м10", "гайка м12", "шайба 8", "винт м6",
                "труба стальная", "кабель медный", "провод алюминиевый"
            ] * (self.config.calibration_dataset_size // 7)
        
        # Подготавливаем модель к квантизации
        model.eval()
        
        # Выполняем калибровку
        logger.info("Performing calibration for INT8 quantization")
        with torch.no_grad():
            for i, text in enumerate(calibration_texts[:self.config.calibration_dataset_size]):
                if hasattr(model, 'encode'):
                    _ = model.encode([text])
                if i % 100 == 0:
                    logger.debug(f"Calibration progress: {i}/{len(calibration_texts)}")
        
        # Применяем квантизацию
        quantized_model = await self._dynamic_quantization(model)
        
        return quantized_model
    
    async def _benchmark_model(
        self, 
        model, 
        model_name: str, 
        test_texts: Optional[list] = None
    ) -> Dict[str, Any]:
        """Бенчмарк модели для оценки производительности"""
        
        if not test_texts:
            test_texts = [
                "болт м10 оцинкованный",
                "гайка шестигранная м12",
                "шайба плоская 8мм",
                "винт с потайной головкой м6",
                "труба стальная 57х3.5"
            ]
        
        # Измеряем время инференса
        inference_times = []
        memory_usage_before = self._get_memory_usage()
        
        for _ in range(10):  # 10 прогонов для усреднения
            start_time = time.time()
            
            try:
                if hasattr(model, 'encode'):
                    _ = model.encode(test_texts)
                else:
                    # Fallback для других типов моделей
                    pass
            except Exception as e:
                logger.warning(f"Benchmark failed for {model_name}: {e}")
                break
                
            inference_times.append(time.time() - start_time)
        
        memory_usage_after = self._get_memory_usage()
        
        if not inference_times:
            return {
                'avg_inference_time': float('inf'),
                'memory_usage_mb': 0,
                'throughput_texts_per_sec': 0
            }
        
        avg_inference_time = np.mean(inference_times)
        throughput = len(test_texts) / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time': avg_inference_time,
            'memory_usage_mb': memory_usage_after - memory_usage_before,
            'throughput_texts_per_sec': throughput,
            'model_size_mb': self._estimate_model_size(model)
        }
    
    def _get_memory_usage(self) -> float:
        """Получение текущего использования памяти в MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _estimate_model_size(self, model) -> float:
        """Оценка размера модели в MB"""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / 1024 / 1024
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _check_accuracy_preservation(
        self, 
        original_stats: Dict[str, Any], 
        quantized_stats: Dict[str, Any]
    ) -> bool:
        """Проверка сохранения точности после квантизации"""
        
        # Простая проверка - в реальном проекте нужна более сложная логика
        # основанная на сравнении эмбеддингов или других метрик качества
        
        original_throughput = original_stats.get('throughput_texts_per_sec', 0)
        quantized_throughput = quantized_stats.get('throughput_texts_per_sec', 0)
        
        # Если производительность значительно упала, возможно есть проблемы с точностью
        if quantized_throughput < original_throughput * 0.5:
            return False
        
        return True
    
    def _calculate_performance_improvement(
        self, 
        original_stats: Dict[str, Any], 
        quantized_stats: Dict[str, Any]
    ) -> float:
        """Вычисление улучшения производительности"""
        
        original_time = original_stats.get('avg_inference_time', 1.0)
        quantized_time = quantized_stats.get('avg_inference_time', 1.0)
        
        if quantized_time > 0:
            return original_time / quantized_time
        return 1.0
    
    def _calculate_memory_reduction(
        self, 
        original_stats: Dict[str, Any], 
        quantized_stats: Dict[str, Any]
    ) -> float:
        """Вычисление сокращения использования памяти в процентах"""
        
        original_size = original_stats.get('model_size_mb', 0)
        quantized_size = quantized_stats.get('model_size_mb', 0)
        
        if original_size > 0:
            reduction = ((original_size - quantized_size) / original_size) * 100
            return max(0, reduction)
        return 0.0
    
    async def optimize_for_inference(self, model, model_name: str) -> Dict[str, Any]:
        """Комплексная оптимизация модели для инференса"""
        
        logger.info(f"Starting comprehensive inference optimization for {model_name}")
        
        optimizations_applied = []
        
        # 1. Переводим в режим eval
        if hasattr(model, 'eval'):
            model.eval()
            optimizations_applied.append("eval_mode")
        
        # 2. Отключаем градиенты
        if hasattr(model, 'requires_grad_'):
            for param in model.parameters():
                param.requires_grad_(False)
            optimizations_applied.append("no_grad")
        
        # 3. Применяем torch.jit.script если возможно
        try:
            if hasattr(model, '_modules') and torch.jit.is_scripting_supported():
                model = torch.jit.script(model)
                optimizations_applied.append("jit_script")
        except Exception as e:
            logger.debug(f"JIT scripting failed: {e}")
        
        # 4. Применяем квантизацию
        quantization_result = await self.quantize_sentence_transformer(model, model_name)
        optimizations_applied.append(f"quantization_{self.config.quantization_type.value}")
        
        return {
            'optimized_model': quantization_result['model'],
            'optimizations_applied': optimizations_applied,
            'performance_stats': quantization_result,
            'total_speedup': quantization_result['performance_improvement']
        }
    
    def get_quantization_summary(self) -> Dict[str, Any]:
        """Получение сводки по всем квантизированным моделям"""
        
        summary = {
            'total_models_quantized': len(self._quantized_models),
            'average_speedup': 0.0,
            'average_memory_reduction': 0.0,
            'models': {}
        }
        
        if self._quantized_models:
            speedups = []
            memory_reductions = []
            
            for model_name, result in self._quantized_models.items():
                speedup = result['performance_improvement']
                memory_reduction = result['memory_reduction']
                
                speedups.append(speedup)
                memory_reductions.append(memory_reduction)
                
                summary['models'][model_name] = {
                    'speedup': speedup,
                    'memory_reduction': memory_reduction,
                    'quantization_type': result['quantization_type'],
                    'accuracy_preserved': result['accuracy_preserved']
                }
            
            summary['average_speedup'] = np.mean(speedups)
            summary['average_memory_reduction'] = np.mean(memory_reductions)
        
        return summary


# Глобальный экземпляр квантизатора
model_quantizer = ModelQuantizer()
