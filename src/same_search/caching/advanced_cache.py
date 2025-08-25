"""
Продвинутая многоуровневая система кэширования для SAMe
"""

import logging
import asyncio
import time
import hashlib
import pickle
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Уровни кэширования"""
    MEMORY = "memory"      # Оперативная память (L1)
    DISK = "disk"          # Локальный диск (L2)
    DISTRIBUTED = "distributed"  # Распределенный кэш (L3)


class CacheStrategy(Enum):
    """Стратегии кэширования"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Адаптивная стратегия


@dataclass
class CacheConfig:
    """Конфигурация кэша"""
    memory_limit_mb: int = 512
    disk_limit_gb: float = 2.0
    default_ttl_seconds: int = 3600
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_compression: bool = True
    enable_encryption: bool = False
    cache_dir: Path = Path("cache")
    distributed_nodes: List[str] = None


class CacheEntry:
    """Запись в кэше"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl = ttl
        self.size_bytes = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Вычисление размера значения в байтах"""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Приблизительная оценка
    
    def is_expired(self) -> bool:
        """Проверка истечения срока действия"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Обновление времени последнего доступа"""
        self.last_accessed = time.time()
        self.access_count += 1


class AdvancedCache:
    """Продвинутая многоуровневая система кэширования"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # Многоуровневые кэши
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._disk_cache_index: Dict[str, str] = {}  # key -> file_path
        
        # Статистика
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Настройка директории кэша
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем индекс дискового кэша
        self._load_disk_index()
        
        logger.info(f"Advanced cache initialized with {self.config.memory_limit_mb}MB memory limit")
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша"""
        self._stats['total_requests'] += 1
        
        # L1: Проверяем память
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                entry.touch()
                self._stats['memory_hits'] += 1
                logger.debug(f"Memory cache hit for key: {key[:50]}...")
                return entry.value
            else:
                # Удаляем истекшую запись
                del self._memory_cache[key]
        
        # L2: Проверяем диск
        disk_value = await self._get_from_disk(key)
        if disk_value is not None:
            # Перемещаем в память для быстрого доступа
            await self._set_memory(key, disk_value, self.config.default_ttl_seconds)
            self._stats['disk_hits'] += 1
            logger.debug(f"Disk cache hit for key: {key[:50]}...")
            return disk_value
        
        # L3: Распределенный кэш (если настроен)
        if self.config.distributed_nodes:
            distributed_value = await self._get_from_distributed(key)
            if distributed_value is not None:
                # Сохраняем в локальные кэши
                await self._set_memory(key, distributed_value, self.config.default_ttl_seconds)
                await self._set_disk(key, distributed_value)
                return distributed_value
        
        self._stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Сохранение значения в кэш"""
        ttl = ttl or self.config.default_ttl_seconds
        
        try:
            # Сохраняем во все уровни
            await self._set_memory(key, value, ttl)
            await self._set_disk(key, value)
            
            if self.config.distributed_nodes:
                await self._set_distributed(key, value, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def _set_memory(self, key: str, value: Any, ttl: int):
        """Сохранение в память"""
        entry = CacheEntry(key, value, ttl)
        
        # Проверяем лимит памяти
        current_size = self._calculate_memory_usage()
        if current_size + entry.size_bytes > self.config.memory_limit_mb * 1024 * 1024:
            await self._evict_memory()
        
        self._memory_cache[key] = entry
    
    async def _set_disk(self, key: str, value: Any):
        """Сохранение на диск"""
        try:
            file_path = self._get_disk_path(key)
            
            # Сериализуем и сохраняем
            data = {
                'value': value,
                'created_at': time.time(),
                'key': key
            }
            
            if self.config.enable_compression:
                import gzip
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Обновляем индекс
            self._disk_cache_index[key] = str(file_path)
            await self._save_disk_index()
            
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Получение с диска"""
        if key not in self._disk_cache_index:
            return None
        
        try:
            file_path = Path(self._disk_cache_index[key])
            if not file_path.exists():
                # Удаляем из индекса
                del self._disk_cache_index[key]
                return None
            
            # Загружаем данные
            if self.config.enable_compression:
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            return data['value']
            
        except Exception as e:
            logger.error(f"Error loading from disk cache: {e}")
            return None
    
    async def _get_from_distributed(self, key: str) -> Optional[Any]:
        """Получение из распределенного кэша (заглушка)"""
        # В реальной реализации здесь был бы код для работы с Redis/Memcached
        return None
    
    async def _set_distributed(self, key: str, value: Any, ttl: int):
        """Сохранение в распределенный кэш (заглушка)"""
        # В реальной реализации здесь был бы код для работы с Redis/Memcached
        pass
    
    def _get_disk_path(self, key: str) -> Path:
        """Получение пути к файлу на диске"""
        # Создаем хэш для имени файла
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.config.cache_dir / f"{hash_key}.cache"
    
    def _calculate_memory_usage(self) -> int:
        """Вычисление использования памяти в байтах"""
        return sum(entry.size_bytes for entry in self._memory_cache.values())
    
    async def _evict_memory(self):
        """Вытеснение записей из памяти"""
        if not self._memory_cache:
            return
        
        # Выбираем стратегию вытеснения
        if self.config.strategy == CacheStrategy.LRU:
            # Удаляем наименее недавно использованные
            to_remove = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )[:len(self._memory_cache) // 4]  # Удаляем 25%
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Удаляем наименее часто использованные
            to_remove = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].access_count
            )[:len(self._memory_cache) // 4]
        
        else:  # TTL или ADAPTIVE
            # Удаляем истекшие и старые записи
            now = time.time()
            to_remove = [
                (key, entry) for key, entry in self._memory_cache.items()
                if entry.is_expired() or (now - entry.created_at) > 1800  # 30 минут
            ]
        
        # Удаляем выбранные записи
        for key, _ in to_remove:
            del self._memory_cache[key]
            self._stats['evictions'] += 1
        
        logger.debug(f"Evicted {len(to_remove)} entries from memory cache")
    
    def _load_disk_index(self):
        """Загрузка индекса дискового кэша"""
        index_file = self.config.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._disk_cache_index = json.load(f)
                logger.info(f"Loaded disk cache index with {len(self._disk_cache_index)} entries")
            except Exception as e:
                logger.error(f"Error loading disk cache index: {e}")
                self._disk_cache_index = {}
    
    async def _save_disk_index(self):
        """Сохранение индекса дискового кэша"""
        index_file = self.config.cache_dir / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._disk_cache_index, f)
        except Exception as e:
            logger.error(f"Error saving disk cache index: {e}")
    
    async def invalidate(self, key: str) -> bool:
        """Инвалидация записи во всех уровнях кэша"""
        invalidated = False
        
        # Удаляем из памяти
        if key in self._memory_cache:
            del self._memory_cache[key]
            invalidated = True
        
        # Удаляем с диска
        if key in self._disk_cache_index:
            try:
                file_path = Path(self._disk_cache_index[key])
                if file_path.exists():
                    file_path.unlink()
                del self._disk_cache_index[key]
                await self._save_disk_index()
                invalidated = True
            except Exception as e:
                logger.error(f"Error removing disk cache file: {e}")
        
        return invalidated
    
    async def clear(self):
        """Очистка всех уровней кэша"""
        # Очищаем память
        self._memory_cache.clear()
        
        # Очищаем диск
        for file_path in self._disk_cache_index.values():
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error removing cache file {file_path}: {e}")
        
        self._disk_cache_index.clear()
        await self._save_disk_index()
        
        # Сбрасываем статистику
        self._stats = {key: 0 for key in self._stats}
        
        logger.info("All cache levels cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        total_requests = self._stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            hits = self._stats['memory_hits'] + self._stats['disk_hits']
            hit_rate = hits / total_requests
        
        return {
            'hit_rate': hit_rate,
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(self._disk_cache_index),
            'memory_usage_mb': self._calculate_memory_usage() / (1024 * 1024),
            'stats': self._stats.copy()
        }
    
    async def optimize(self):
        """Оптимизация кэша"""
        logger.info("Starting cache optimization")
        
        # Очищаем истекшие записи из памяти
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        # Проверяем дисковый кэш
        invalid_disk_keys = []
        for key, file_path in self._disk_cache_index.items():
            if not Path(file_path).exists():
                invalid_disk_keys.append(key)
        
        for key in invalid_disk_keys:
            del self._disk_cache_index[key]
        
        if invalid_disk_keys:
            await self._save_disk_index()
        
        logger.info(f"Cache optimization completed. Removed {len(expired_keys)} expired memory entries "
                   f"and {len(invalid_disk_keys)} invalid disk entries")


# Глобальный экземпляр продвинутого кэша
advanced_cache = AdvancedCache()
