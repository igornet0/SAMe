"""
Модуль лемматизации с использованием SpaCy
"""

import spacy
import logging
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from same_search.models import get_model_manager

logger = logging.getLogger(__name__)


@dataclass
class LemmatizerConfig:
    """Конфигурация лемматизатора"""
    model_name: str = "ru_core_news_sm"  # Изменено с ru_core_news_lg на ru_core_news_sm
    preserve_technical_terms: bool = True
    custom_stopwords: Set[str] = field(default_factory=set)
    min_token_length: int = 2
    preserve_numbers: bool = True

    normalize_product_variants: bool = True
    product_base_forms: Dict[str, str] = field(default_factory=dict)  
    preserve_professional_terms: bool = True  

    # Защищенные токены (Phase 1 improvement)
    enable_protected_tokens: bool = True
    protected_tokens: Set[str] = field(default_factory=lambda: {
        # Технические стандарты
        'ГОСТ', 'ТУ', 'ОСТ', 'СТО', 'СТП', 'ТИ', 'РД', 'DIN', 'ISO', 'EN', 'ANSI', 'ASTM',
        # Технические параметры
        'DN', 'PN', 'RPM', 'об/мин', 'м³/ч', 'л/мин', 'кВт', 'МВт', 'кПа', 'МПа', 'ГПа',
        # Единицы измерения
        '°C', '°F', 'мм', 'см', 'м', 'км', 'г', 'кг', 'т', 'л', 'мл', 'В', 'А', 'Вт', 'Гц', 'кГц', 'МГц', 'ГГц',
        'Па', 'бар', 'атм', 'м/с', 'км/ч',
        # Материалы и свойства
        'PVC', 'PTFE', 'EPDM', 'NBR', 'FKM', 'VITON', 'SS', 'CS', 'MS', 'Al', 'Cu', 'Zn',
        # Технические коды
        'IP', 'IEC', 'UL', 'CE', 'RoHS', 'REACH'
    })
    protected_token_patterns: Set[str] = field(default_factory=lambda: {
        # Паттерны для защищенных токенов с числами
        r'DN\d+', r'PN\d+', r'M\d+', r'G\d+', r'R\d+', r'IP\d+',
        r'\d+об/мин', r'\d+м³/ч', r'\d+л/мин', r'\d+°C', r'\d+°F',
        r'\d+кВт', r'\d+МВт', r'\d+В', r'\d+А', r'\d+Гц'
    })

    # Параметры кэширования
    enable_caching: bool = True
    cache_max_size: int = 10000  # Максимальный размер кэша лемматизации
    cache_ttl_seconds: int = 3600  # Время жизни кэша в секундах (1 час)

    # Обработка ошибок и fallback режимы
    enable_fallback: bool = True
    fallback_models: List[str] = field(default_factory=lambda: ["ru_core_news_sm", "ru_core_news_md"])
    max_retries: int = 3
    retry_delay: float = 0.5
    graceful_degradation: bool = True  # Продолжать работу при ошибках модели


class Lemmatizer:
    """Класс для лемматизации текста с использованием SpaCy"""

    def __init__(self, config: LemmatizerConfig = None):
        self.config = config or LemmatizerConfig()
        self.model_manager = get_model_manager()
        self._nlp = None
        self._stopwords = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lemmatizer")

        # Инициализация кэша лемматизации
        self._lemma_cache = {} if self.config.enable_caching else None
        self._cache_timestamps = {} if self.config.enable_caching else None

        # Инициализация защищенных токенов (Phase 1)
        self._protected_tokens_map = {}
        self._protected_patterns = []
        if self.config.enable_protected_tokens:
            self._init_protected_tokens()

    def _init_protected_tokens(self):
        """Инициализация защищенных токенов (Phase 1)"""
        import re

        # Создаем карту защищенных токенов (case-insensitive поиск -> canonical форма)
        for token in self.config.protected_tokens:
            self._protected_tokens_map[token.lower()] = token

        # Компилируем паттерны для защищенных токенов с числами
        self._protected_patterns = []
        for pattern in self.config.protected_token_patterns:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                self._protected_patterns.append((compiled_pattern, pattern))
            except re.error as e:
                logger.warning(f"Invalid protected token pattern '{pattern}': {e}")

        logger.info(f"Initialized {len(self._protected_tokens_map)} protected tokens and {len(self._protected_patterns)} patterns")

    def _is_protected_token(self, token: str) -> Optional[str]:
        """
        Проверяет, является ли токен защищенным

        Args:
            token: Токен для проверки

        Returns:
            Каноническая форма токена если защищен, иначе None
        """
        if not self.config.enable_protected_tokens:
            return None

        # Проверяем точное совпадение
        canonical = self._protected_tokens_map.get(token.lower())
        if canonical:
            return canonical

        # Проверяем паттерны
        for pattern, _ in self._protected_patterns:
            if pattern.match(token):
                # Для паттернов возвращаем токен в верхнем регистре
                return token.upper()

        return None

    async def _ensure_initialized(self):
        """Ленивая инициализация модели с fallback режимами"""
        if self._initialized:
            return

        # Список моделей для попытки загрузки
        models_to_try = [self.config.model_name]

        # Добавляем fallback модели если включен fallback режим
        if self.config.enable_fallback and self.config.fallback_models:
            models_to_try.extend(self.config.fallback_models)

        last_exception = None

        for attempt, model_name in enumerate(models_to_try):
            try:
                logger.info(f"Attempting to load SpaCy model: {model_name} (attempt {attempt + 1})")

                # Пытаемся загрузить через model_manager
                if hasattr(self.model_manager, 'get_spacy_model'):
                    self._nlp = await self.model_manager.get_spacy_model(model_name)
                else:
                    # Прямая загрузка как fallback
                    import spacy
                    self._nlp = spacy.load(model_name)

                self._setup_stopwords()
                self._setup_product_variants()
                self._initialized = True
                logger.info(f"Successfully loaded SpaCy model: {model_name}")
                return

            except Exception as e:
                last_exception = e
                logger.warning(f"Failed to load model {model_name}: {e}")

                # Пробуем с небольшой задержкой
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue

        # Если все модели не удалось загрузить
        if self.config.graceful_degradation:
            logger.error("All SpaCy models failed to load, creating simple fallback")
            self._nlp = self._create_simple_fallback()
            self._setup_simple_stopwords()
            self._initialized = True
        else:
            raise RuntimeError(f"Failed to load any SpaCy model. Last error: {last_exception}")

    def _setup_stopwords(self):
        """Настройка стоп-слов"""
        # Базовые стоп-слова из SpaCy
        self._stopwords = set(self._nlp.Defaults.stop_words)

        # Добавляем кастомные стоп-слова
        if self.config.custom_stopwords:
            self._stopwords.update(self.config.custom_stopwords)

        # Технические стоп-слова для МТР
        technical_stopwords = {
            'изделие', 'деталь', 'элемент', 'часть', 'компонент',
            'материал', 'оборудование', 'устройство', 'прибор',
            'система', 'блок', 'модуль', 'узел', 'механизм'
        }
        self._stopwords.update(technical_stopwords)

        logger.info(f"Stopwords configured: {len(self._stopwords)} words")

    def _setup_product_variants(self):
        """Настройка базовых форм для продуктовых вариантов"""
        if not self.config.normalize_product_variants:
            self.product_variants = {}
            return

        # Базовые формы для общих продуктовых терминов
        default_variants = {}

        # Объединяем с пользовательскими вариантами
        self.product_variants = {**default_variants, **self.config.product_base_forms}

        logger.info(f"Product variants configured: {len(self.product_variants)} mappings")

    @property
    def stopwords(self) -> Set[str]:
        """Получение стоп-слов (для обратной совместимости)"""
        if self._stopwords is None:
            # Синхронная инициализация для обратной совместимости
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._ensure_initialized())
            finally:
                loop.close()
        return self._stopwords
    
    async def lemmatize_text_async(self, text: str) -> Dict[str, any]:
        """
        Асинхронная лемматизация текста

        Args:
            text: Входной текст

        Returns:
            Dict с результатами лемматизации
        """
        await self._ensure_initialized()

        if text is None or not text or not isinstance(text, str):
            return {
                'original': text if text is not None else '',
                'lemmatized': '',
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }

        # Выполняем синхронную обработку в ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._lemmatize_text_sync, text)

    def lemmatize_text(self, text: str) -> Dict[str, any]:
        """
        Синхронная лемматизация текста (для обратной совместимости)

        Args:
            text: Входной текст

        Returns:
            Dict с результатами лемматизации
        """
        # Всегда используем синхронную версию для избежания проблем с event loop
        return self._lemmatize_text_sync(text)

    def _lemmatize_text_sync(self, text: str) -> Dict[str, any]:
        """
        Синхронная версия лемматизации без использования asyncio с кэшированием
        """
        if text is None or not text or not text.strip():
            return {
                'original': text if text is not None else '',
                'lemmatized': text if text is not None else '',
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }

        # Проверяем кэш
        cached_result = self._get_cached_lemmatization(text)
        if cached_result is not None:
            return cached_result

        try:
            # Инициализируем модель если нужно (синхронно)
            if self._nlp is None:
                import spacy
                try:
                    self._nlp = spacy.load(self.config.model_name)
                except OSError:
                    logger.warning(f"Model {self.config.model_name} not found, using default")
                    self._nlp = spacy.load("ru_core_news_lg")

                # Инициализируем стоп-слова
                if self._stopwords is None:
                    self._setup_stopwords()

            # Обрабатываем текст
            doc = self._nlp(text)
            result = self._process_doc(doc, text)

            # Кэшируем результат
            self._cache_lemmatization(text, result)

            return result

        except Exception as e:
            logger.error(f"Error in synchronous lemmatization: {e}")
            return {
                'original': text,
                'lemmatized': text,
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }


    def _should_include_token(self, token) -> bool:
        """Определяет, должен ли токен быть включен в результат"""
        # Пропускаем короткие токены
        if len(token.text) < self.config.min_token_length:
            return False

        # Проверяем, что lemma_ не None
        if token.lemma_ is None:
            return False

        # Пропускаем стоп-слова
        if self._stopwords is not None and token.lemma_.lower() in self._stopwords:
            return False
        
        # Сохраняем числа если нужно
        if token.like_num and not self.config.preserve_numbers:
            return False
        
        # Сохраняем технические термины
        if self.config.preserve_technical_terms and self._is_technical_term(token):
            return True
        
        # Пропускаем местоимения, предлоги, союзы
        if token.pos_ in ['PRON', 'ADP', 'CCONJ', 'SCONJ', 'PART']:
            return False
        
        return True
    
    def _is_technical_term(self, token) -> bool:
        """Проверяет, является ли токен техническим термином"""
        technical_pos = ['NOUN', 'ADJ', 'NUM']

        # Технические единицы измерения
        technical_units = {
            'мм', 'см', 'м', 'км', 'кг', 'г', 'т', 'л', 'мл',
            'в', 'а', 'вт', 'квт', 'мвт', 'гц', 'кгц', 'мгц', 'ггц',
            'па', 'кпа', 'мпа', 'гпа', 'бар', 'атм'
        }

        # Проверяем lemma_ на None
        if token.lemma_ is not None and token.lemma_.lower() in technical_units:
            return True
        
        # Технические аббревиатуры (заглавные буквы)
        if token.text.isupper() and len(token.text) >= 2:
            return True
        
        return token.pos_ in technical_pos
    
    async def lemmatize_batch_async(self, texts: List[str]) -> List[Dict[str, any]]:
        """Асинхронная пакетная лемматизация с оптимизацией производительности"""
        if not texts:
            return []

        await self._ensure_initialized()

        # Выполняем синхронную пакетную обработку в ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._lemmatize_batch_sync, texts)

    def lemmatize_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Синхронная пакетная лемматизация (для обратной совместимости)"""
        # Всегда используем синхронную версию для избежания проблем с event loop
        return self._lemmatize_batch_sync(texts)

    def _lemmatize_batch_sync(self, texts: List[str]) -> List[Dict[str, any]]:
        """Синхронная пакетная лемматизация с оптимизацией через pipe"""
        if not texts:
            return []

        # Убеждаемся, что модель инициализирована
        if not self._initialized:
            # Синхронная инициализация для случая, когда вызывается из ThreadPoolExecutor
            try:
                self._nlp = self.model_manager.get_spacy_model_sync(self.config.model_name)
                self._setup_stopwords()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize model synchronously: {e}")
                # Fallback к простой обработке
                return [self._lemmatize_text_sync(text) for text in texts]

        results = []
        batch_size = 100  # Оптимальный размер батча для SpaCy

        # Обрабатываем большие батчи частями
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Используем pipe для эффективной обработки
                docs = list(self._nlp.pipe(batch, disable=['parser', 'ner']))

                for j, doc in enumerate(docs):
                    original_text = batch[j]
                    result = self._process_doc(doc, original_text)
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in sync batch lemmatization: {e}")
                # Fallback к простой обработке для этого батча
                for text in batch:
                    try:
                        result = self._lemmatize_text_sync(text)
                        results.append(result)
                    except Exception:
                        # Последний fallback
                        results.append({
                            'original': text,
                            'lemmatized': text,
                            'tokens': [],
                            'lemmas': [],
                            'pos_tags': [],
                            'filtered_lemmas': []
                        })

        return results

    def _process_doc(self, doc, original_text: str) -> Dict[str, any]:
        """Обработка SpaCy документа"""
        tokens = []
        lemmas = []
        pos_tags = []
        filtered_lemmas = []
        
        for token in doc:
            if token.is_punct or token.is_space:
                continue

            tokens.append(token.text)

            # Phase 1: Проверяем защищенные токены перед лемматизацией
            protected_form = self._is_protected_token(token.text)
            if protected_form:
                # Используем каноническую форму защищенного токена
                lemmas.append(protected_form)
                pos_tags.append(token.pos_)

                if self._should_include_token(token):
                    filtered_lemmas.append(protected_form.lower())
            else:
                # Обычная лемматизация
                lemmas.append(token.lemma_ if token.lemma_ is not None else token.text)
                pos_tags.append(token.pos_)

                if self._should_include_token(token):
                    lemma_text = token.lemma_ if token.lemma_ is not None else token.text
                    lemma_lower = lemma_text.lower()

                    # Применяем нормализацию продуктовых вариантов
                    if self.config.normalize_product_variants and hasattr(self, 'product_variants'):
                        lemma_lower = self.product_variants.get(lemma_lower, lemma_lower)

                    filtered_lemmas.append(lemma_lower)
        
        return {
            'original': original_text,
            'lemmatized': ' '.join(filtered_lemmas),
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'filtered_lemmas': filtered_lemmas
        }
    
    def _get_text_hash(self, text: str) -> str:
        """Получение хэша текста для кэширования"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_cache_valid(self, text_hash: str) -> bool:
        """Проверка валидности кэша по времени"""
        if not self.config.enable_caching or text_hash not in self._cache_timestamps:
            return False

        import time
        current_time = time.time()
        cache_time = self._cache_timestamps[text_hash]
        return (current_time - cache_time) < self.config.cache_ttl_seconds

    def _get_cached_lemmatization(self, text: str) -> Optional[Dict[str, any]]:
        """Получение результата лемматизации из кэша"""
        if not self.config.enable_caching:
            return None

        text_hash = self._get_text_hash(text)
        if text_hash in self._lemma_cache and self._is_cache_valid(text_hash):
            return self._lemma_cache[text_hash].copy()

        return None

    def _cache_lemmatization(self, text: str, result: Dict[str, any]):
        """Сохранение результата лемматизации в кэш"""
        if not self.config.enable_caching:
            return

        text_hash = self._get_text_hash(text)

        # Управление размером кэша
        if len(self._lemma_cache) >= self.config.cache_max_size:
            # Удаляем самый старый элемент
            oldest_hash = min(self._cache_timestamps.keys(),
                            key=lambda k: self._cache_timestamps[k])
            del self._lemma_cache[oldest_hash]
            del self._cache_timestamps[oldest_hash]

        import time
        self._lemma_cache[text_hash] = result.copy()
        self._cache_timestamps[text_hash] = time.time()

    def clear_cache(self):
        """Очистка кэша лемматизации"""
        if self.config.enable_caching:
            self._lemma_cache.clear()
            self._cache_timestamps.clear()
            logger.info("Lemmatization cache cleared")

    def get_cache_stats(self) -> Dict[str, any]:
        """Получение статистики использования кэша"""
        if not self.config.enable_caching:
            return {'caching_enabled': False}

        import time
        current_time = time.time()
        valid_entries = sum(1 for timestamp in self._cache_timestamps.values()
                          if (current_time - timestamp) < self.config.cache_ttl_seconds)

        return {
            'caching_enabled': True,
            'cache_size': len(self._lemma_cache),
            'cache_max_size': self.config.cache_max_size,
            'valid_entries': valid_entries,
            'expired_entries': len(self._lemma_cache) - valid_entries,
            'cache_hit_potential': f"{valid_entries}/{len(self._lemma_cache)}" if self._lemma_cache else "0/0"
        }

    def get_lemmatization_stats(self, result: Dict[str, any]) -> Dict[str, int]:
        """Получение статистики лемматизации"""
        return {
            'original_tokens': len(result['tokens']),
            'filtered_tokens': len(result['filtered_lemmas']),
            'reduction_ratio': round((1 - len(result['filtered_lemmas']) / max(len(result['tokens']), 1)) * 100, 2),
            'unique_lemmas': len(set(result['filtered_lemmas']))
        }

    def _create_simple_fallback(self):
        """
        Создание простого fallback для лемматизации без SpaCy

        Returns:
            Объект с минимальным интерфейсом для работы
        """
        class SimpleFallback:
            def __init__(self):
                self.Defaults = type('Defaults', (), {'stop_words': set()})()

            def __call__(self, text):
                """Простая обработка текста без лемматизации"""
                # Создаем простой объект документа
                doc = SimpleDoc(text)
                return doc

        class SimpleDoc:
            def __init__(self, text):
                self.text = text
                # Простое разбиение на токены
                import re
                words = re.findall(r'\b\w+\b', text.lower())
                self.tokens = [SimpleToken(word) for word in words]

            def __iter__(self):
                return iter(self.tokens)

        class SimpleToken:
            def __init__(self, text):
                self.text = text
                self.lemma_ = text.lower()  # Простая "лемматизация" - приведение к нижнему регистру
                self.pos_ = "UNKNOWN"
                self.is_alpha = text.isalpha()
                self.is_digit = text.isdigit()
                self.is_punct = not text.isalnum()

        logger.warning("Using simple fallback lemmatizer (no SpaCy)")
        return SimpleFallback()

    def _setup_simple_stopwords(self):
        """Настройка простых стоп-слов для fallback режима"""
        # Базовые русские стоп-слова
        basic_stopwords = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
            'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
            'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
            'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
            'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
            'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
            'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
            'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }

        self._stopwords = basic_stopwords
        logger.info("Simple stopwords initialized for fallback mode")

    def __del__(self):
        """Корректное закрытие ThreadPoolExecutor при удалении объекта"""
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass  # Игнорируем ошибки при закрытии
