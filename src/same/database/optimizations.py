"""
Модуль оптимизации базы данных для высокопроизводительного поиска
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy import text, Index, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select
import time

from .engine import get_db_helper
from .models.main_models import Item, ItemParameter

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Оптимизатор базы данных для поиска аналогов"""
    
    def __init__(self):
        self.session_factory = None  # Будет инициализирован асинхронно
        self._index_cache = {}
        self._query_cache = {}
        self._cache_ttl = 300  # 5 минут

    async def _ensure_session_factory(self):
        """Инициализация session_factory если еще не инициализирован"""
        if self.session_factory is None:
            db_helper = await get_db_helper()
            self.session_factory = db_helper.async_session

    async def create_search_indexes(self) -> Dict[str, bool]:
        """Создание оптимизированных индексов для поиска"""
        logger.info("Creating optimized search indexes")

        await self._ensure_session_factory()
        indexes_created = {}

        async with self.session_factory() as session:
            try:
                # 1. Полнотекстовые индексы для русского языка
                await self._create_fulltext_indexes(session)
                indexes_created['fulltext'] = True
                
                # 2. Композитные индексы для быстрого поиска
                await self._create_composite_indexes(session)
                indexes_created['composite'] = True
                
                # 3. Индексы для параметров
                await self._create_parameter_indexes(session)
                indexes_created['parameters'] = True
                
                # 4. Индексы для сортировки и фильтрации
                await self._create_sorting_indexes(session)
                indexes_created['sorting'] = True
                
                await session.commit()
                logger.info("All search indexes created successfully")
                
            except Exception as e:
                logger.error(f"Error creating indexes: {e}")
                await session.rollback()
                raise
                
        return indexes_created
    
    async def _create_fulltext_indexes(self, session: AsyncSession):
        """Создание полнотекстовых индексов"""
        
        # Полнотекстовый индекс для названий товаров
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_name_fulltext 
            ON items USING gin(to_tsvector('russian', name))
        """))
        
        # Полнотекстовый индекс для описаний
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_description_fulltext 
            ON items USING gin(to_tsvector('russian', description))
        """))
        
        # Комбинированный полнотекстовый индекс
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_combined_fulltext 
            ON items USING gin(to_tsvector('russian', name || ' ' || COALESCE(description, '')))
        """))
        
        logger.info("Fulltext indexes created")
    
    async def _create_composite_indexes(self, session: AsyncSession):
        """Создание композитных индексов"""
        
        # Индекс для быстрого поиска по ID и имени
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_id_name 
            ON items(id, name)
        """))
        
        # Индекс для поиска с сортировкой по времени создания
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_created_name 
            ON items(created DESC, name)
        """))
        
        logger.info("Composite indexes created")
    
    async def _create_parameter_indexes(self, session: AsyncSession):
        """Создание индексов для параметров"""
        
        # Индекс для быстрого поиска параметров по item_id
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_parameters_item_id 
            ON item_parameters(item_id)
        """))
        
        # Индекс для поиска по названию параметра
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_parameters_name 
            ON item_parameters(parameter_name)
        """))
        
        # Композитный индекс для поиска параметров
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_parameters_name_value 
            ON item_parameters(parameter_name, parameter_value)
        """))
        
        # Полнотекстовый индекс для значений параметров
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_parameters_value_fulltext 
            ON item_parameters USING gin(to_tsvector('russian', parameter_value))
        """))
        
        logger.info("Parameter indexes created")
    
    async def _create_sorting_indexes(self, session: AsyncSession):
        """Создание индексов для сортировки"""
        
        # Индекс для сортировки по времени обновления
        await session.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_updated_desc 
            ON items(updated DESC)
        """))
        
        logger.info("Sorting indexes created")
    
    async def optimized_search(
        self,
        query: str,
        limit: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Оптимизированный поиск с использованием полнотекстовых индексов"""

        await self._ensure_session_factory()

        # Проверяем кэш
        cache_key = f"search:{query}:{limit}"
        if use_cache and cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cache_entry['results']

        start_time = time.time()

        async with self.session_factory() as session:
            # Используем полнотекстовый поиск с ранжированием
            sql_query = text("""
                SELECT 
                    i.id,
                    i.name,
                    i.description,
                    i.created,
                    i.updated,
                    ts_rank(to_tsvector('russian', i.name || ' ' || COALESCE(i.description, '')), 
                           plainto_tsquery('russian', :query)) as rank
                FROM items i
                WHERE to_tsvector('russian', i.name || ' ' || COALESCE(i.description, '')) 
                      @@ plainto_tsquery('russian', :query)
                ORDER BY rank DESC, i.updated DESC
                LIMIT :limit
            """)
            
            result = await session.execute(sql_query, {"query": query, "limit": limit})
            rows = result.fetchall()
            
            # Преобразуем результаты
            results = []
            for row in rows:
                results.append({
                    'id': row.id,
                    'name': row.name,
                    'description': row.description,
                    'created': row.created.isoformat() if row.created else None,
                    'updated': row.updated.isoformat() if row.updated else None,
                    'relevance_score': float(row.rank)
                })
        
        search_time = time.time() - start_time
        logger.info(f"Optimized search completed in {search_time:.3f}s for query: {query[:50]}...")
        
        # Сохраняем в кэш
        if use_cache:
            self._query_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            
            # Управляем размером кэша
            if len(self._query_cache) > 1000:
                # Удаляем 20% старых записей
                old_keys = sorted(
                    self._query_cache.keys(),
                    key=lambda k: self._query_cache[k]['timestamp']
                )[:200]
                for key in old_keys:
                    del self._query_cache[key]
        
        return results
    
    async def optimized_search_with_parameters(
        self,
        query: str,
        parameter_filters: Optional[Dict[str, str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Оптимизированный поиск с фильтрацией по параметрам"""

        await self._ensure_session_factory()
        start_time = time.time()

        async with self.session_factory() as session:
            # Базовый запрос с JOIN для параметров
            base_query = select(Item).options(selectinload(Item.parameters))
            
            # Добавляем полнотекстовый поиск
            if query:
                base_query = base_query.where(
                    func.to_tsvector('russian', Item.name + ' ' + func.coalesce(Item.description, ''))
                    .match(func.plainto_tsquery('russian', query))
                )
            
            # Добавляем фильтры по параметрам
            if parameter_filters:
                for param_name, param_value in parameter_filters.items():
                    base_query = base_query.join(ItemParameter).where(
                        ItemParameter.parameter_name == param_name,
                        ItemParameter.parameter_value.ilike(f'%{param_value}%')
                    )
            
            # Добавляем сортировку и лимит
            base_query = base_query.order_by(Item.updated.desc()).limit(limit)
            
            result = await session.execute(base_query)
            items = result.scalars().all()
            
            # Преобразуем результаты
            results = []
            for item in items:
                item_data = {
                    'id': item.id,
                    'name': item.name,
                    'description': item.description,
                    'created': item.created.isoformat() if item.created else None,
                    'updated': item.updated.isoformat() if item.updated else None,
                    'parameters': [
                        {
                            'name': param.parameter_name,
                            'value': param.parameter_value
                        }
                        for param in item.parameters
                    ]
                }
                results.append(item_data)
        
        search_time = time.time() - start_time
        logger.info(f"Parameter search completed in {search_time:.3f}s")
        
        return results
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Получение статистики поиска"""

        await self._ensure_session_factory()
        async with self.session_factory() as session:
            # Общее количество товаров
            total_items = await session.scalar(select(func.count(Item.id)))
            
            # Количество товаров с параметрами
            items_with_params = await session.scalar(
                select(func.count(func.distinct(ItemParameter.item_id)))
            )
            
            # Общее количество параметров
            total_parameters = await session.scalar(select(func.count(ItemParameter.id)))
            
            # Статистика кэша
            cache_stats = {
                'cache_size': len(self._query_cache),
                'cache_hit_ratio': self._calculate_cache_hit_ratio()
            }
            
            return {
                'total_items': total_items,
                'items_with_parameters': items_with_params,
                'total_parameters': total_parameters,
                'cache_statistics': cache_stats,
                'indexes_status': await self._check_indexes_status(session)
            }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Вычисление коэффициента попаданий в кэш"""
        # Простая реализация - в реальном проекте нужна более сложная логика
        return 0.85  # Заглушка
    
    async def _check_indexes_status(self, session: AsyncSession) -> Dict[str, bool]:
        """Проверка статуса индексов"""
        
        indexes_to_check = [
            'idx_items_name_fulltext',
            'idx_items_description_fulltext',
            'idx_items_combined_fulltext',
            'idx_items_id_name',
            'idx_item_parameters_item_id',
            'idx_item_parameters_name_value'
        ]
        
        status = {}
        for index_name in indexes_to_check:
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = :index_name
                )
            """), {"index_name": index_name})
            
            status[index_name] = result.scalar()
        
        return status
    
    async def clear_cache(self):
        """Очистка кэша запросов"""
        self._query_cache.clear()
        logger.info("Query cache cleared")


# Глобальный экземпляр оптимизатора
db_optimizer = DatabaseOptimizer()
