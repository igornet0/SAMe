#!/usr/bin/env python3
"""
Модуль для нормализации технических терминов
Обрабатывает специфические технические термины, которые плохо лемматизируются стандартными методами
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TechnicalTermsNormalizerConfig:
    """Конфигурация нормализатора технических терминов"""
    preserve_compound_terms: bool = True  # Сохранять составные термины (штуцер-гайка)
    normalize_technical_abbreviations: bool = True  # Нормализовать технические сокращения
    preserve_technical_hyphens: bool = True  # Сохранять дефисы в технических терминах
    technical_token_prefix: str = "TECH_"  # Префикс для технических токенов
    case_sensitive: bool = False


class TechnicalTermsNormalizer:
    """Класс для нормализации технических терминов"""
    
    def __init__(self, config: TechnicalTermsNormalizerConfig = None):
        self.config = config or TechnicalTermsNormalizerConfig()
        
        # Словарь технических терминов с их базовыми формами
        self.technical_terms_dict = {
            # Штрихкод и связанные термины
            'штрихкодирования': 'штрихкод',
            'штрихкод': 'штрихкод',
            'штрих-кода': 'штрихкод',
            'штрих-код': 'штрихкод',
            'штрих': 'штрихкод',  # в контексте штрих-кода
            'штрикод': 'штрихкод',
            
            # Штуцер и связанные термины
            'штуцеру': 'штуцер',
            'штуцером': 'штуцер',
            'штуцеров': 'штуцер',
            'штуцерных': 'штуцер',
            'штуцерный': 'штуцер',
            'штуцерная': 'штуцер',
            'штуцерами': 'штуцер',
            'штуцера': 'штуцер',
            'штуцер': 'штуцер',
            'штуц': 'штуцер',
            
            # Чугун и связанные термины
            'чугуну': 'чугун',
            'чугунными': 'чугун',
            'чугунным': 'чугун',
            'чугунный': 'чугун',
            'чугунные': 'чугун',
            'чугунная': 'чугун',
            'чугуна': 'чугун',
            'чугун': 'чугун',
            'чуг': 'чугун',
        }
        
        # Составные технические термины (сохраняем как есть)
        self.compound_technical_terms = {
            'штуцер-штуцер': 'штуцер-штуцер',
            'штуцер-елочка': 'штуцер-елочка',
            'штуцер-гайка': 'штуцер-гайка',
            'штуцерно-торцовый': 'штуцер-торцовый',
            'штуцерно-торцевое': 'штуцер-торцовый',
            'штрих-код': 'штрихкод',
            'штрих-кода': 'штрихкод',
        }
        
        # Технические сокращения
        self.technical_abbreviations = {
            'шт': 'штука',
            'кг': 'килограмм', 
            'мм': 'миллиметр',
            'см': 'сантиметр',
            'м': 'метр',
            'л': 'литр',
            'мл': 'миллилитр',
            'в': 'вольт',
            'вт': 'ватт',
            'а': 'ампер',
            'гц': 'герц',
            'бар': 'бар',
            'атм': 'атмосфера',
        }
        
        # Паттерны для поиска технических терминов
        self._init_patterns()
        
        logger.info(f"TechnicalTermsNormalizer initialized with {len(self.technical_terms_dict)} terms")
    
    def _init_patterns(self):
        """Инициализация регулярных выражений"""
        
        # Создаем паттерны для составных терминов (приоритет выше)
        compound_terms = list(self.compound_technical_terms.keys())
        compound_pattern = '|'.join(re.escape(term) for term in sorted(compound_terms, key=len, reverse=True))
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        self.compound_pattern = re.compile(rf'\b({compound_pattern})\b', flags)
        
        # Создаем паттерны для простых технических терминов
        simple_terms = list(self.technical_terms_dict.keys())
        simple_pattern = '|'.join(re.escape(term) for term in sorted(simple_terms, key=len, reverse=True))
        self.simple_pattern = re.compile(rf'\b({simple_pattern})\b', flags)
        
        # Паттерн для технических сокращений
        if self.config.normalize_technical_abbreviations:
            abbrev_terms = list(self.technical_abbreviations.keys())
            abbrev_pattern = '|'.join(re.escape(term) for term in abbrev_terms)
            self.abbrev_pattern = re.compile(rf'\b({abbrev_pattern})\b', flags)
    
    def normalize_technical_terms(self, text: str) -> Dict[str, any]:
        """
        Нормализация технических терминов в тексте
        
        Args:
            text: Исходный текст
            
        Returns:
            Dict с результатами нормализации
        """
        if not text or not isinstance(text, str):
            return {
                'original': text,
                'normalized': text,
                'technical_terms_found': [],
                'technical_terms_count': 0,
                'processing_successful': True
            }
        
        result = {
            'original': text,
            'normalized': text,
            'technical_terms_found': [],
            'technical_terms_count': 0,
            'processing_successful': True
        }
        
        try:
            normalized_text = text
            terms_found = []
            
            # Сначала обрабатываем составные термины (приоритет выше)
            if self.config.preserve_compound_terms:
                def replace_compound(match):
                    original_term = match.group(1)
                    normalized_term = self.compound_technical_terms.get(original_term.lower(), original_term)
                    terms_found.append(f"{original_term} -> {normalized_term}")
                    return normalized_term
                
                normalized_text = self.compound_pattern.sub(replace_compound, normalized_text)
            
            # Затем обрабатываем простые технические термины
            def replace_simple(match):
                original_term = match.group(1)
                normalized_term = self.technical_terms_dict.get(original_term.lower(), original_term)
                if normalized_term != original_term:
                    terms_found.append(f"{original_term} -> {normalized_term}")
                return normalized_term
            
            normalized_text = self.simple_pattern.sub(replace_simple, normalized_text)
            
            # Обрабатываем технические сокращения
            if self.config.normalize_technical_abbreviations:
                def replace_abbrev(match):
                    original_abbrev = match.group(1)
                    full_term = self.technical_abbreviations.get(original_abbrev.lower(), original_abbrev)
                    if full_term != original_abbrev:
                        terms_found.append(f"{original_abbrev} -> {full_term}")
                    return full_term
                
                normalized_text = self.abbrev_pattern.sub(replace_abbrev, normalized_text)
            
            result.update({
                'normalized': normalized_text.strip(),
                'technical_terms_found': terms_found,
                'technical_terms_count': len(terms_found)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in technical terms normalization: {e}")
            result.update({
                'processing_successful': False,
                'error': str(e)
            })
            return result
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Извлечение всех технических терминов из текста без замены
        
        Args:
            text: Исходный текст
            
        Returns:
            Список найденных технических терминов
        """
        result = self.normalize_technical_terms(text)
        return result.get('technical_terms_found', [])
    
    def has_technical_terms(self, text: str) -> bool:
        """
        Проверка наличия технических терминов в тексте
        
        Args:
            text: Исходный текст
            
        Returns:
            True если найдены технические термины
        """
        result = self.normalize_technical_terms(text)
        return result.get('technical_terms_count', 0) > 0
    
    def add_custom_technical_terms(self, terms_dict: Dict[str, str]):
        """
        Добавление пользовательских технических терминов
        
        Args:
            terms_dict: Словарь {исходный_термин: нормализованный_термин}
        """
        self.technical_terms_dict.update(terms_dict)
        self._init_patterns()  # Пересоздаем паттерны
        logger.info(f"Added {len(terms_dict)} custom technical terms")
    
    def add_custom_compound_terms(self, terms_dict: Dict[str, str]):
        """
        Добавление пользовательских составных терминов
        
        Args:
            terms_dict: Словарь {исходный_термин: нормализованный_термин}
        """
        self.compound_technical_terms.update(terms_dict)
        self._init_patterns()  # Пересоздаем паттерны
        logger.info(f"Added {len(terms_dict)} custom compound terms")
    
    def get_technical_terms_statistics(self, texts: List[str]) -> Dict[str, int]:
        """
        Получение статистики по техническим терминам в коллекции текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Словарь {термин: количество_вхождений}
        """
        terms_stats = {}
        
        for text in texts:
            terms = self.extract_technical_terms(text)
            for term in terms:
                terms_stats[term] = terms_stats.get(term, 0) + 1
        
        return dict(sorted(terms_stats.items(), key=lambda x: x[1], reverse=True))


def create_technical_terms_normalizer(config: TechnicalTermsNormalizerConfig = None) -> TechnicalTermsNormalizer:
    """Фабричная функция для создания нормализатора технических терминов"""
    return TechnicalTermsNormalizer(config)


# Предустановленные конфигурации
DEFAULT_CONFIG = TechnicalTermsNormalizerConfig()

AGGRESSIVE_CONFIG = TechnicalTermsNormalizerConfig(
    preserve_compound_terms=True,
    normalize_technical_abbreviations=True,
    preserve_technical_hyphens=True
)

CONSERVATIVE_CONFIG = TechnicalTermsNormalizerConfig(
    preserve_compound_terms=False,
    normalize_technical_abbreviations=False,
    preserve_technical_hyphens=False
)
