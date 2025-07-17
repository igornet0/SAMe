"""
Модуль лемматизации с использованием SpaCy
"""

import spacy
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LemmatizerConfig:
    """Конфигурация лемматизатора"""
    model_name: str = "ru_core_news_lg"
    preserve_technical_terms: bool = True
    custom_stopwords: Set[str] = None
    min_token_length: int = 2
    preserve_numbers: bool = True


class Lemmatizer:
    """Класс для лемматизации текста с использованием SpaCy"""
    
    def __init__(self, config: LemmatizerConfig = None):
        self.config = config or LemmatizerConfig()
        self.nlp = None
        self._load_model()
        self._setup_stopwords()
    
    def _load_model(self):
        """Загрузка модели SpaCy"""
        try:
            self.nlp = spacy.load(self.config.model_name)
            logger.info(f"Loaded SpaCy model: {self.config.model_name}")
        except OSError:
            logger.error(f"SpaCy model {self.config.model_name} not found. Please install it:")
            logger.error(f"python -m spacy download {self.config.model_name}")
            raise
    
    def _setup_stopwords(self):
        """Настройка стоп-слов"""
        # Базовые стоп-слова из SpaCy
        self.stopwords = set(self.nlp.Defaults.stop_words)
        
        # Добавляем кастомные стоп-слова
        if self.config.custom_stopwords:
            self.stopwords.update(self.config.custom_stopwords)
        
        # Технические стоп-слова для МТР
        technical_stopwords = {
            'изделие', 'деталь', 'элемент', 'часть', 'компонент',
            'материал', 'оборудование', 'устройство', 'прибор',
            'система', 'блок', 'модуль', 'узел', 'механизм'
        }
        self.stopwords.update(technical_stopwords)
    
    def lemmatize_text(self, text: str) -> Dict[str, any]:
        """
        Лемматизация текста
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с результатами лемматизации
        """
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
        doc = self.nlp(text)
        
        tokens = []
        lemmas = []
        pos_tags = []
        filtered_lemmas = []
        
        for token in doc:
            # Пропускаем пунктуацию и пробелы
            if token.is_punct or token.is_space:
                continue
            
            # Сохраняем информацию о токене
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)
            
            # Фильтрация для финального результата
            if self._should_include_token(token):
                filtered_lemmas.append(token.lemma_.lower())
        
        # Формируем лемматизированный текст
        lemmatized_text = ' '.join(filtered_lemmas)
        
        return {
            'original': text,
            'lemmatized': lemmatized_text,
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'filtered_lemmas': filtered_lemmas
        }
    
    def _should_include_token(self, token) -> bool:
        """Определяет, должен ли токен быть включен в результат"""
        # Пропускаем короткие токены
        if len(token.text) < self.config.min_token_length:
            return False
        
        # Пропускаем стоп-слова
        if token.lemma_.lower() in self.stopwords:
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
    
    def lemmatize_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Пакетная лемматизация"""
        results = []
        
        # Используем nlp.pipe для эффективной обработки
        docs = list(self.nlp.pipe(texts))
        
        for i, doc in enumerate(docs):
            try:
                result = self._process_doc(doc, texts[i])
                results.append(result)
            except Exception as e:
                logger.error(f"Error lemmatizing text {i}: {e}")
                results.append({
                    'original': texts[i],
                    'lemmatized': texts[i],
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
