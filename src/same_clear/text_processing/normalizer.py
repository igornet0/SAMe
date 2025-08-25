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

    # Настройки для обработки числовых токенов
    reduce_numeric_weight: bool = True
    numeric_token_replacement: str = "<NUM>"  # Заменять числа на токен
    preserve_units_with_numbers: bool = True  # Сохранять "10 мм", "5 кг" и т.д.
    normalize_ranges: bool = True  # "10-15" -> "<NUM>-<NUM>"


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
        # ИСПРАВЛЕНИЕ: НЕ приводим весь текст к нижнему регистру
        # Это разрушает марки материалов и бренды
        current_text = text  # Сохраняем оригинальный регистр

        # ИСПРАВЛЕНИЕ: Глобальная защита марок материалов от разрушения
        protected_materials = {}
        current_text = self._protect_material_grades_global(current_text, protected_materials)

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
        
        # Этап 4: Обработка числовых токенов
        if self.config.reduce_numeric_weight:
            current_text = self._process_numeric_tokens(current_text)
            result['numeric_processed'] = current_text
        else:
            result['numeric_processed'] = current_text

        # Этап 5: Финальная очистка
        current_text = self._final_cleanup(current_text)

        # ИСПРАВЛЕНИЕ: Восстанавливаем защищенные марки материалов
        current_text = self._restore_protected_materials_global(current_text, protected_materials)

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

    def _process_numeric_tokens(self, text: str) -> str:
        """Обработка числовых токенов для снижения их веса в поиске"""
        if not self.config.reduce_numeric_weight:
            return text

        # ИСПРАВЛЕНИЕ: Защищаем марки материалов от разрушения
        protected_materials = []
        material_patterns = [
            # Марки стали: 09Г2С-14, 12Х18Н10Т, 20Х23Н18
            r'\b\d{2}[А-Я]\d*[А-Я]*-?\d*\b',
            # Стандартные стали: Ст3пс, Ст20, У8А
            r'\b[СУ]т?\d+[а-я]*\b',
            # Легированные стали: 40Х, 45ХН, 30ХГСА
            r'\b\d{2}[ХГСНМТВКЮЛБЦЧФЭРАИОУЫЯЕЁЖЗШЩЪЬЭЮЯхгснмтвкюлбцчфэраиоуыяеёжзшщъьэюя]+\b',
            # Цветные сплавы: АМг6, Д16Т, ВТ1-0
            r'\b[АВДЛМ][А-Я]*\d+[А-Я]*-?\d*\b',
        ]

        def protect_material(match):
            material = match.group(0)
            # Проверяем, что это действительно марка материала
            if (len(material) >= 3 and
                not material.isdigit() and
                material.lower() not in ['гост', 'дин', 'iso']):
                protected_materials.append(material)
                return f"__MATERIAL_{len(protected_materials)-1}__"
            return material

        # Защищаем марки материалов
        for pattern in material_patterns:
            text = re.sub(pattern, protect_material, text, flags=re.IGNORECASE)

        # Сохраняем числа с единицами измерения если нужно
        if self.config.preserve_units_with_numbers:
            # Заменяем числа без единиц на токены, но сохраняем с единицами
            # Сначала защищаем числа с единицами
            protected_patterns = []

            # Находим все числа с единицами и заменяем их временными маркерами
            unit_pattern = r'(\d+(?:[.,]\d+)?)\s*(мм|см|м|км|кг|г|т|л|мл|В|А|Вт|кВт|МВт|Гц|кГц|МГц|ГГц|°C|°F|К|Па|кПа|МПа|ГПа|бар|атм|об/мин|м/с|км/ч)\b'

            def protect_units(match):
                protected_patterns.append(match.group(0))
                return f"__PROTECTED_{len(protected_patterns)-1}__"

            text = re.sub(unit_pattern, protect_units, text, flags=re.IGNORECASE)

        # Обрабатываем диапазоны
        if self.config.normalize_ranges:
            text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\b',
                         f'{self.config.numeric_token_replacement}-{self.config.numeric_token_replacement}', text)

        # Заменяем оставшиеся отдельные числа
        text = re.sub(r'\b\d+(?:[.,]\d+)?\b', self.config.numeric_token_replacement, text)

        # Восстанавливаем защищенные числа с единицами
        if self.config.preserve_units_with_numbers:
            for i, pattern in enumerate(protected_patterns):
                text = text.replace(f"__PROTECTED_{i}__", pattern)

        # ИСПРАВЛЕНИЕ: Восстанавливаем защищенные марки материалов
        for i, material in enumerate(protected_materials):
            text = text.replace(f"__MATERIAL_{i}__", material)

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

    def _protect_material_grades_global(self, text: str, protected_materials: dict) -> str:
        """
        ИСПРАВЛЕНИЕ: Глобальная защита марок материалов от разрушения на всех этапах нормализации
        Примеры: 09Г2С-14, 12Х18Н10Т, Ст3пс, 40Х, 20Х23Н18
        """
        # Паттерны для марок материалов
        material_patterns = [
            # Марки стали: 09Г2С-14, 12Х18Н10Т, 20Х23Н18
            r'\b\d{2}[А-Я]\d*[А-Я]*-?\d*\b',
            # Стандартные стали: Ст3пс, Ст20, У8А
            r'\b[СУ]т?\d+[а-я]*\b',
            # Легированные стали: 40Х, 45ХН, 30ХГСА
            r'\b\d{2}[ХГСНМТВКЮЛБЦЧФЭРАИОУЫЯЕЁЖЗШЩЪЬЭЮЯхгснмтвкюлбцчфэраиоуыяеёжзшщъьэюя]+\b',
            # Цветные сплавы: АМг6, Д16Т, ВТ1-0
            r'\b[АВДЛМ][А-Я]*\d+[А-Я]*-?\d*\b',
            # Дополнительные марки: Ц6Хр, 09Г2С
            r'\b[А-Я]\d+[А-Я][а-я]*\b',
        ]

        protected_text = text
        material_counter = 0

        for pattern in material_patterns:
            def protect_material(match):
                nonlocal material_counter
                material = match.group(0)

                # Проверяем, что это действительно марка материала
                if (len(material) >= 3 and
                    not material.isdigit() and
                    material.upper() not in ['ГОСТ', 'ДИН', 'ISO', 'ТУ', 'ACME', 'TMK', 'FMC']):

                    placeholder = f"__GLOBAL_MATERIAL_{material_counter}__"
                    protected_materials[placeholder] = material
                    material_counter += 1
                    return placeholder

                return material

            protected_text = re.sub(pattern, protect_material, protected_text, flags=re.IGNORECASE)

        return protected_text

    def _restore_protected_materials_global(self, text: str, protected_materials: dict) -> str:
        """
        ИСПРАВЛЕНИЕ: Восстанавливает глобально защищенные марки материалов
        """
        restored_text = text

        for placeholder, material in protected_materials.items():
            restored_text = restored_text.replace(placeholder, material)

        return restored_text
