"""
Модуль нормализации текста для унификации наименований МТР
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NormalizerConfig:
    """Конфигурация нормализатора"""
    standardize_units: bool = True
    normalize_abbreviations: bool = True
    unify_technical_terms: bool = True
    remove_brand_names: bool = False
    standardize_numbers: bool = True


class TextNormalizer:
    """Класс для нормализации технических наименований"""
    
    def __init__(self, config: NormalizerConfig = None):
        self.config = config or NormalizerConfig()
        self._setup_normalization_rules()
    
    def _setup_normalization_rules(self):
        """Настройка правил нормализации"""
        
        # Стандартизация единиц измерения
        self.unit_mappings = {
            # Длина
            'миллиметр': 'мм', 'миллиметра': 'мм', 'миллиметров': 'мм',
            'сантиметр': 'см', 'сантиметра': 'см', 'сантиметров': 'см',
            'метр': 'м', 'метра': 'м', 'метров': 'м',
            'километр': 'км', 'километра': 'км', 'километров': 'км',
            
            # Масса
            'грамм': 'г', 'грамма': 'г', 'граммов': 'г',
            'килограмм': 'кг', 'килограмма': 'кг', 'килограммов': 'кг',
            'тонна': 'т', 'тонны': 'т', 'тонн': 'т',
            
            # Объем
            'литр': 'л', 'литра': 'л', 'литров': 'л',
            'миллилитр': 'мл', 'миллилитра': 'мл', 'миллилитров': 'мл',
            
            # Электрические
            'вольт': 'В', 'вольта': 'В', 'вольтов': 'В',
            'ампер': 'А', 'ампера': 'А', 'амперов': 'А',
            'ватт': 'Вт', 'ватта': 'Вт', 'ваттов': 'Вт',
            'киловатт': 'кВт', 'киловатта': 'кВт', 'киловаттов': 'кВт',
            
            # Давление
            'паскаль': 'Па', 'паскаля': 'Па', 'паскалей': 'Па',
            'килопаскаль': 'кПа', 'килопаскаля': 'кПа', 'килопаскалей': 'кПа',
            'мегапаскаль': 'МПа', 'мегапаскаля': 'МПа', 'мегапаскалей': 'МПа',
            'атмосфера': 'атм', 'атмосферы': 'атм', 'атмосфер': 'атм',
        }
        
        # Нормализация аббревиатур
        self.abbreviation_mappings = {
            'эл': 'электрический', 'электр': 'электрический',
            'мех': 'механический', 'механ': 'механический',
            'гидр': 'гидравлический', 'гидравл': 'гидравлический',
            'пневм': 'пневматический', 'пневмат': 'пневматический',
            'автом': 'автоматический', 'авт': 'автоматический',
            'ручн': 'ручной', 'руч': 'ручной',
            'стандарт': 'стандартный', 'станд': 'стандартный',
            'спец': 'специальный', 'специал': 'специальный',
        }
        
        # Унификация технических терминов
        self.technical_term_mappings = {
            # Крепеж
            'болт': ['болт', 'винт крепежный', 'крепежный болт'],
            'гайка': ['гайка', 'гайка крепежная'],
            'шайба': ['шайба', 'шайба плоская', 'шайба круглая'],
            'винт': ['винт', 'винт крепежный', 'крепежный винт'],
            
            # Трубы и фитинги
            'труба': ['труба', 'трубка', 'трубопровод'],
            'фитинг': ['фитинг', 'соединение', 'переходник'],
            'муфта': ['муфта', 'муфта соединительная'],
            
            # Электрика
            'кабель': ['кабель', 'провод', 'проводник'],
            'разъем': ['разъем', 'соединитель', 'коннектор'],
            'выключатель': ['выключатель', 'переключатель'],
        }
        
        # Компиляция регулярных выражений
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компиляция регулярных выражений"""
        self.patterns = {
            # Числа с единицами измерения
            'number_unit': re.compile(r'(\d+(?:[.,]\d+)?)\s*([а-яё]+)', re.IGNORECASE),
            
            # Диапазоны значений
            'range_values': re.compile(r'(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)', re.IGNORECASE),
            
            # Технические характеристики в скобках
            'tech_specs': re.compile(r'\(([^)]+)\)', re.IGNORECASE),
            
            # Артикулы и коды
            'article_codes': re.compile(r'\b[A-Z0-9]{3,}-?[A-Z0-9]*\b'),
            
            # Множественные пробелы
            'multiple_spaces': re.compile(r'\s+'),
        }
    
    def normalize_text(self, text: str) -> Dict[str, str]:
        """
        Основной метод нормализации текста
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с этапами нормализации
        """
        if not text or not isinstance(text, str):
            return {
                'original': text or '',
                'units_normalized': '',
                'abbreviations_normalized': '',
                'terms_unified': '',
                'final_normalized': ''
            }
        
        result = {'original': text}
        current_text = text.lower()
        
        # Этап 1: Нормализация единиц измерения
        if self.config.standardize_units:
            current_text = self._normalize_units(current_text)
            result['units_normalized'] = current_text
        else:
            result['units_normalized'] = current_text
        
        # Этап 2: Нормализация аббревиатур
        if self.config.normalize_abbreviations:
            current_text = self._normalize_abbreviations(current_text)
            result['abbreviations_normalized'] = current_text
        else:
            result['abbreviations_normalized'] = current_text
        
        # Этап 3: Унификация технических терминов
        if self.config.unify_technical_terms:
            current_text = self._unify_technical_terms(current_text)
            result['terms_unified'] = current_text
        else:
            result['terms_unified'] = current_text
        
        # Этап 4: Финальная очистка
        current_text = self._final_cleanup(current_text)
        result['final_normalized'] = current_text
        
        return result
    
    def _normalize_units(self, text: str) -> str:
        """Нормализация единиц измерения"""
        for full_unit, short_unit in self.unit_mappings.items():
            # Заменяем полные названия на сокращения
            pattern = r'\b' + re.escape(full_unit) + r'\b'
            text = re.sub(pattern, short_unit, text, flags=re.IGNORECASE)
        
        # Стандартизация числовых значений с единицами
        def replace_number_unit(match):
            number = match.group(1).replace(',', '.')
            unit = match.group(2)
            
            # Проверяем, есть ли единица в наших маппингах
            normalized_unit = self.unit_mappings.get(unit.lower(), unit)
            return f"{number} {normalized_unit}"
        
        text = self.patterns['number_unit'].sub(replace_number_unit, text)
        
        return text
    
    def _normalize_abbreviations(self, text: str) -> str:
        """Нормализация аббревиатур"""
        for abbr, full_form in self.abbreviation_mappings.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def _unify_technical_terms(self, text: str) -> str:
        """Унификация технических терминов"""
        for canonical_term, variants in self.technical_term_mappings.items():
            for variant in variants:
                if variant != canonical_term:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    text = re.sub(pattern, canonical_term, text, flags=re.IGNORECASE)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Финальная очистка текста"""
        # Удаляем множественные пробелы
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # Убираем пробелы в начале и конце
        text = text.strip()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Пакетная нормализация"""
        results = []
        for text in texts:
            try:
                result = self.normalize_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error normalizing text: {e}")
                results.append({
                    'original': text,
                    'units_normalized': text,
                    'abbreviations_normalized': text,
                    'terms_unified': text,
                    'final_normalized': text
                })
        return results
    
    def extract_technical_specs(self, text: str) -> List[str]:
        """Извлечение технических характеристик из текста"""
        specs = []
        
        # Извлекаем содержимое скобок
        bracket_content = self.patterns['tech_specs'].findall(text)
        specs.extend(bracket_content)
        
        # Извлекаем диапазоны значений
        ranges = self.patterns['range_values'].findall(text)
        for range_match in ranges:
            specs.append(f"{range_match[0]}-{range_match[1]}")
        
        return specs
    
    def get_normalization_stats(self, original: str, normalized: str) -> Dict[str, any]:
        """Получение статистики нормализации"""
        return {
            'original_length': len(original),
            'normalized_length': len(normalized),
            'compression_ratio': round((len(original) - len(normalized)) / len(original) * 100, 2) if original else 0,
            'extracted_specs': len(self.extract_technical_specs(original))
        }
