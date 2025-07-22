"""
Модуль лемматизации с использованием SpaCy
"""

import spacy
import logging
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from ..models import get_model_manager

logger = logging.getLogger(__name__)


@dataclass
class LemmatizerConfig:
    """Конфигурация лемматизатора"""
    model_name: str = "ru_core_news_lg"
    preserve_technical_terms: bool = True
    custom_stopwords: Set[str] = field(default_factory=set)
    min_token_length: int = 2
    preserve_numbers: bool = True


class Lemmatizer:
    """Класс для лемматизации текста с использованием SpaCy"""

    def __init__(self, config: LemmatizerConfig = None):
        self.config = config or LemmatizerConfig()
        self.model_manager = get_model_manager()
        self._nlp = None
        self._stopwords = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ленивая инициализация модели"""
        if self._initialized:
            return

        # Получаем модель через менеджер
        self._nlp = await self.model_manager.get_spacy_model(self.config.model_name)
        self._setup_stopwords()
        self._initialized = True
        logger.info(f"Lemmatizer initialized with model: {self.config.model_name}")

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

        if not text or not isinstance(text, str):
            return {
                'original': text or '',
                'lemmatized': '',
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }

        # Обработка текста через SpaCy
        doc = self._nlp(text)
        return self._process_doc(doc, text)

    def lemmatize_text(self, text: str) -> Dict[str, any]:
        """
        Синхронная лемматизация текста (для обратной совместимости)

        Args:
            text: Входной текст

        Returns:
            Dict с результатами лемматизации
        """
        # Синхронная реализация без asyncio.run()
        try:
            # Проверяем, есть ли уже запущенный event loop
            loop = asyncio.get_running_loop()
            # Если есть, используем синхронную версию
            return self._lemmatize_text_sync(text)
        except RuntimeError:
            # Если нет активного loop, можем использовать asyncio.run
            return asyncio.run(self.lemmatize_text_async(text))

    def _lemmatize_text_sync(self, text: str) -> Dict[str, any]:
        """
        Синхронная версия лемматизации без использования asyncio
        """
        if not text or not text.strip():
            return {
                'original': text,
                'lemmatized': text,
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'filtered_lemmas': []
            }

        try:
            # Инициализируем модель если нужно (синхронно)
            if self._nlp is None:
                import spacy
                try:
                    self._nlp = spacy.load(self.config.model_name)
                except OSError:
                    logger.warning(f"Model {self.config.model_name} not found, using default")
                    self._nlp = spacy.load("ru_core_news_sm")

            # Обрабатываем текст
            doc = self._nlp(text)
            return self._process_doc(doc, text)

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
        
        # Пропускаем стоп-слова
        if token.lemma_.lower() in self._stopwords:
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
        
        if token.lemma_.lower() in technical_units:
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

                # Позволяем другим задачам выполняться
                if i % (batch_size * 5) == 0:  # Каждые 500 текстов
                    await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Error in async batch lemmatization: {e}")
                # Добавляем результаты с ошибками
                for text in batch:
                    results.append({
                        'original': text,
                        'lemmatized': text,
                        'tokens': [],
                        'lemmas': [],
                        'pos_tags': [],
                        'filtered_lemmas': []
                    })

        return results

    def lemmatize_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Синхронная пакетная лемматизация (для обратной совместимости)"""
        try:
            # Проверяем, есть ли уже запущенный event loop
            loop = asyncio.get_running_loop()
            # Если есть, используем синхронную версию
            return self._lemmatize_batch_sync(texts)
        except RuntimeError:
            # Если нет активного loop, можем использовать asyncio.run
            return asyncio.run(self.lemmatize_batch_async(texts))

    def _lemmatize_batch_sync(self, texts: List[str]) -> List[Dict[str, any]]:
        """Синхронная пакетная лемматизация без asyncio"""
        results = []
        for text in texts:
            result = self._lemmatize_text_sync(text)
            results.append(result)
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
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)
            
            if self._should_include_token(token):
                filtered_lemmas.append(token.lemma_.lower())
        
        return {
            'original': original_text,
            'lemmatized': ' '.join(filtered_lemmas),
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'filtered_lemmas': filtered_lemmas
        }
    
    def get_lemmatization_stats(self, result: Dict[str, any]) -> Dict[str, int]:
        """Получение статистики лемматизации"""
        return {
            'original_tokens': len(result['tokens']),
            'filtered_tokens': len(result['filtered_lemmas']),
            'reduction_ratio': round((1 - len(result['filtered_lemmas']) / max(len(result['tokens']), 1)) * 100, 2),
            'unique_lemmas': len(set(result['filtered_lemmas']))
        }
