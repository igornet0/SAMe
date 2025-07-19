"""
Модуль очистки текста от артефактов, HTML-тегов, спецсимволов
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Конфигурация для очистки текста"""
    remove_html: bool = True
    remove_special_chars: bool = True
    remove_extra_spaces: bool = True
    remove_numbers: bool = False
    preserve_technical_terms: bool = True
    custom_patterns: List[str] = field(default_factory=list)


class TextCleaner:
    """Класс для очистки текста от различных артефактов"""
    
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компиляция регулярных выражений для оптимизации"""
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'html_entities': re.compile(r'&[a-zA-Z0-9#]+;'),
            'special_chars': re.compile(r'[^\w\s\-\.\,\(\)\[\]]', re.UNICODE),
            'extra_spaces': re.compile(r'\s+'),
            'numbers_only': re.compile(r'^\d+$'),
            'technical_units': re.compile(r'\b\d+\s*(мм|см|м|кг|г|т|л|мл|В|А|Вт|кВт|МВт|Гц|кГц|МГц|ГГц|°C|°F|К|Па|кПа|МПа|ГПа|бар|атм|об/мин|м/с|км/ч)\b', re.IGNORECASE),
            'ocr_artifacts': re.compile(r'[|]{2,}|_{3,}|\.{4,}|—{2,}'),
        }
    
    def clean_text(self, text: str) -> Dict[str, str]:
        """
        Основной метод очистки текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Dict с этапами очистки: raw, html_cleaned, special_cleaned, normalized
        """
        if not text or not isinstance(text, str):
            return {
                'raw': text or '',
                'html_cleaned': '',
                'special_cleaned': '',
                'normalized': ''
            }
        
        result = {'raw': text}
        
        # Этап 1: Удаление HTML
        if self.config.remove_html:
            cleaned = self._remove_html(text)
            result['html_cleaned'] = cleaned
        else:
            result['html_cleaned'] = text
        
        # Этап 2: Удаление спецсимволов
        if self.config.remove_special_chars:
            cleaned = self._remove_special_chars(result['html_cleaned'])
            result['special_cleaned'] = cleaned
        else:
            result['special_cleaned'] = result['html_cleaned']
        
        # Этап 3: Нормализация пробелов
        if self.config.remove_extra_spaces:
            cleaned = self._normalize_spaces(result['special_cleaned'])
            result['normalized'] = cleaned
        else:
            result['normalized'] = result['special_cleaned']
        
        return result
    
    def _remove_html(self, text: str) -> str:
        """Удаление HTML тегов и сущностей"""
        # Удаляем HTML теги
        text = self.patterns['html_tags'].sub(' ', text)
        # Удаляем HTML сущности
        text = self.patterns['html_entities'].sub(' ', text)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Удаление специальных символов с сохранением технических терминов"""
        if self.config.preserve_technical_terms:
            # Сохраняем технические единицы измерения
            technical_terms = self.patterns['technical_units'].findall(text)
            
        # Удаляем OCR артефакты
        text = self.patterns['ocr_artifacts'].sub(' ', text)
        
        # Удаляем специальные символы
        text = self.patterns['special_chars'].sub(' ', text)
        
        return text
    
    def _normalize_spaces(self, text: str) -> str:
        """Нормализация пробелов"""
        # Заменяем множественные пробелы на одинарные
        text = self.patterns['extra_spaces'].sub(' ', text)
        # Убираем пробелы в начале и конце
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Пакетная очистка текстов"""
        results = []
        for text in texts:
            try:
                result = self.clean_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error cleaning text: {e}")
                results.append({
                    'raw': text,
                    'html_cleaned': text,
                    'special_cleaned': text,
                    'normalized': text
                })
        return results
    
    def get_cleaning_stats(self, original: str, cleaned: str) -> Dict[str, int]:
        """Получение статистики очистки"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'removed_chars': len(original) - len(cleaned),
            'compression_ratio': round((len(original) - len(cleaned)) / len(original) * 100, 2) if original else 0
        }
