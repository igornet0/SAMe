"""
Модуль обработки единиц измерения и технических параметров
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from fractions import Fraction

logger = logging.getLogger(__name__)


@dataclass
class UnitsConfig:
    """Конфигурация обработки единиц измерения"""
    normalize_fractions: bool = True  # 1/2" → 0.5"
    convert_to_metric: bool = False   # дюймы → мм
    extract_parameters: bool = True   # Извлекать параметры как структуру
    preserve_original: bool = True    # Сохранять оригинальное значение
    standardize_units: bool = True    # мм, MM, Мм → мм

    # Phase 2: Enhanced Number Normalization
    use_typed_prefixes: bool = True   # Использовать типизированные префиксы
    preserve_fractions: bool = True   # 1/2 → frac_1-2
    preserve_ranges: bool = True      # 0-1000°C → range_0-1000-c
    preserve_ratios: bool = True      # 40:1 → ratio_40-1
    preserve_dimensions: bool = True  # 89×8мм → dims=[89,8] unit="мм"
    preserve_compound_units: bool = True  # м³/ч, об/мин как единое целое


class UnitsProcessor:
    """Процессор единиц измерения и технических параметров"""
    
    def __init__(self, config: UnitsConfig = None):
        self.config = config or UnitsConfig()
        self._compile_patterns()
        self._init_conversion_tables()
    
    def _compile_patterns(self):
        """Компиляция регулярных выражений"""
        self.patterns = {
            # Дробные дюймы: 1/2", 3/4", 1-1/2"
            'fractional_inches': re.compile(
                r'(\d+(?:-\d+)?/\d+)\s*(?:"|дюйм|inch)', 
                re.IGNORECASE
            ),
            
            # Простые дюймы: 10", 5.5", 0,05" (ИСПРАВЛЕНИЕ: добавлены запятые)
            'decimal_inches': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(?:"|дюйм|inch)',
                re.IGNORECASE
            ),
            
            # Размеры: 245х10,03мм, 65х35мм (ИСПРАВЛЕНИЕ: включаем единицу в паттерн)
            'dimensions': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)(?:\s*[хx×]\s*(\d+(?:[.,]\d+)?))?\s*(мм|см|м|дюйм)?',
                re.IGNORECASE
            ),
            
            # Единицы измерения с числами
            'units_with_numbers': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(мм|см|м|км|дм|мкм|нм|г|кг|т|мг|л|мл|куб\.?м|м3|см3|мм3|В|А|Вт|кВт|МВт|Гц|кГц|МГц|ГГц|°C|°F|К|Па|кПа|МПа|ГПа|бар|атм|об/мин|м/с|км/ч|шт|упак|компл|набор)',
                re.IGNORECASE
            ),
            
            # Диапазоны: 10-15мм, 5,5-7,2кг
            'ranges': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м|км|г|кг|т|л|мл|В|А|Вт|кВт|°C|°F|Па|кПа|МПа|бар|атм)',
                re.IGNORECASE
            ),
            
            # Технические коды: ТУ 14-3Р-82-2022, ГОСТ 123-456
            'tech_codes': re.compile(
                r'(ТУ|ГОСТ|ОСТ|СТО|СТП|ТИ|РД)\s+([0-9А-Яа-я\-\.]+)',
                re.IGNORECASE
            ),
            
            # Артикулы: 4-730-059, SCM-6066-71
            'article_codes': re.compile(
                r'\b([A-Za-zА-Яа-я]*\d+[-\.]?\d*[-\.]?\d*[A-Za-zА-Яа-я]*)\b'
            ),

            # Phase 2: Enhanced patterns for typed prefixes

            # Простые дроби: 1/2, 3/4 (без дюймов)
            'simple_fractions': re.compile(
                r'\b(\d+)/(\d+)\b',
                re.IGNORECASE
            ),

            # Соотношения/передаточные числа: 40:1, 1:10
            'ratios': re.compile(
                r'\b(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)\b',
                re.IGNORECASE
            ),

            # Phase 3: Enhanced compound units with more patterns
            # ИСПРАВЛЕНИЕ: добавлены варианты м3/ч и м³/ч для полного покрытия
            'compound_units': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(м³/ч|м3/ч|об/мин|л/мин|км/ч|м/с|кг/м³|кг/м3|Вт/м²|Вт/м2|м³/с|м3/с|л/с|об/с|кг/ч|т/ч)',
                re.IGNORECASE
            ),

            # Атмосферы и другие единицы давления
            'pressure_units': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(атм|бар|торр|psi)',
                re.IGNORECASE
            ),

            # Объемы в литрах
            'volume_units': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(л|мл|куб\.?м|м³|см³|мм³)',
                re.IGNORECASE
            ),

            # Коэффициенты и безразмерные величины
            'dimensionless_coefficients': re.compile(
                r'коэффициент\s+(\d+(?:[.,]\d+)?)|к-т\s+(\d+(?:[.,]\d+)?)',
                re.IGNORECASE
            )
        }
    
    def _init_conversion_tables(self):
        """Инициализация таблиц конверсии"""
        # Конверсия дюймов в мм
        self.inch_to_mm = 25.4
        
        # Стандартизация единиц
        self.unit_standardization = {
            'мм': ['mm', 'MM', 'Мм', 'мМ'],
            'см': ['cm', 'CM', 'См', 'сМ'],
            'м': ['m', 'M', 'М'],
            'кг': ['kg', 'KG', 'Кг', 'кГ'],
            'г': ['g', 'G', 'Г'],
            'л': ['l', 'L', 'Л'],
            'В': ['v', 'V', 'в', 'вольт', 'volt'],
            'А': ['a', 'A', 'а', 'ампер', 'amp'],
            'Вт': ['w', 'W', 'вт', 'ватт', 'watt'],
            '°C': ['c', 'C', 'с', 'цельсий', 'celsius']
        }
        
        # Обратный словарь для быстрого поиска
        self.unit_reverse_map = {}
        for standard, variants in self.unit_standardization.items():
            for variant in variants:
                self.unit_reverse_map[variant.lower()] = standard
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Основной метод обработки единиц измерения
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с обработанным текстом и извлеченными параметрами
        """
        if not text or not isinstance(text, str):
            return {
                'original': text or '',
                'processed': '',
                'extracted_parameters': []
            }
        
        result = {
            'original': text,
            'processed': text,
            'extracted_parameters': []
        }
        
        processed_text = text

        # Phase 2: Enhanced processing with typed prefixes

        # Phase 3: Enhanced complex unit recognition (приоритет - обрабатываем первыми)

        # Обработка составных единиц
        if self.config.preserve_compound_units:
            processed_text, compound_params = self._process_compound_units_enhanced(processed_text)
            result['extracted_parameters'].extend(compound_params)

        # Обработка единиц давления
        processed_text, pressure_params = self._process_pressure_units(processed_text)
        result['extracted_parameters'].extend(pressure_params)

        # Обработка объемов
        processed_text, volume_params = self._process_volume_units(processed_text)
        result['extracted_parameters'].extend(volume_params)

        # Обработка безразмерных коэффициентов
        processed_text, coeff_params = self._process_dimensionless_coefficients(processed_text)
        result['extracted_parameters'].extend(coeff_params)

        # ИСПРАВЛЕНИЕ: Защищаем марки материалов от разрушения
        processed_text = self._protect_material_grades(processed_text)

        # Обработка простых дробей
        if self.config.preserve_fractions:
            processed_text, simple_fraction_params = self._process_simple_fractions(processed_text)
            result['extracted_parameters'].extend(simple_fraction_params)

        # Обработка соотношений/передаточных чисел (ПРИОРИТЕТ: обрабатываем первыми)
        if self.config.preserve_ratios:
            processed_text, ratio_params = self._process_ratios_protected(processed_text)
            result['extracted_parameters'].extend(ratio_params)

        # Обработка диапазонов (с типизированными префиксами)
        # ИСПРАВЛЕНИЕ: исключаем уже обработанные соотношения
        if self.config.preserve_ranges:
            processed_text, range_params = self._process_ranges_enhanced(processed_text)
            result['extracted_parameters'].extend(range_params)
        else:
            # Fallback to old method if enhanced is disabled
            processed_text, range_params = self._process_ranges(processed_text)
            result['extracted_parameters'].extend(range_params)

        # Обработка дробных дюймов
        if self.config.normalize_fractions:
            processed_text, fraction_params = self._process_fractional_inches(processed_text)
            result['extracted_parameters'].extend(fraction_params)

        # Обработка десятичных дюймов
        processed_text, decimal_params = self._process_decimal_inches(processed_text)
        result['extracted_parameters'].extend(decimal_params)

        # Обработка размеров (с улучшенной структурой)
        if self.config.preserve_dimensions:
            processed_text, dimension_params = self._process_dimensions_enhanced(processed_text)
            result['extracted_parameters'].extend(dimension_params)
        else:
            # Fallback to old method if enhanced is disabled
            processed_text, dimension_params = self._process_dimensions(processed_text)
            result['extracted_parameters'].extend(dimension_params)

        # Обработка единиц с числами
        processed_text, unit_params = self._process_units_with_numbers(processed_text)
        result['extracted_parameters'].extend(unit_params)

        # Обработка технических кодов
        processed_text, tech_params = self._process_tech_codes(processed_text)
        result['extracted_parameters'].extend(tech_params)

        # Восстанавливаем защищенные составные единицы
        processed_text = self._restore_protected_compounds(processed_text)

        result['processed'] = processed_text.strip()
        
        return result
    
    def _process_fractional_inches(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка дробных дюймов"""
        parameters = []
        
        def replace_fraction(match):
            fraction_str = match.group(1)
            try:
                # Обработка смешанных дробей: 1-1/2 → 1.5
                if '-' in fraction_str:
                    whole, frac = fraction_str.split('-', 1)
                    decimal_value = float(whole) + float(Fraction(frac))
                else:
                    decimal_value = float(Fraction(fraction_str))
                
                # Добавляем параметр
                param = {
                    'type': 'размер',
                    'value': decimal_value,
                    'unit': 'дюйм',
                    'original': match.group(0)
                }
                
                if self.config.convert_to_metric:
                    param['value_mm'] = round(decimal_value * self.inch_to_mm, 2)
                    param['unit_metric'] = 'мм'
                
                parameters.append(param)
                
                # Возвращаем нормализованное значение
                if self.config.convert_to_metric:
                    return f"{param['value_mm']} мм"
                else:
                    return f"{decimal_value} дюйм"
                    
            except (ValueError, ZeroDivisionError):
                logger.warning(f"Failed to parse fraction: {fraction_str}")
                return match.group(0)
        
        processed_text = self.patterns['fractional_inches'].sub(replace_fraction, text)
        return processed_text, parameters
    
    def _process_decimal_inches(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка десятичных дюймов"""
        parameters = []
        
        def replace_decimal(match):
            value_str = match.group(1)
            try:
                # ИСПРАВЛЕНИЕ: сохраняем точность, нормализуя запятые в точки
                normalized_value_str = value_str.replace(',', '.')
                decimal_value = float(normalized_value_str)

                param = {
                    'type': 'размер',
                    'value': decimal_value,
                    'unit': 'дюйм',
                    'original': match.group(0),
                    'value_str': normalized_value_str  # Сохраняем строковое представление
                }

                if self.config.convert_to_metric:
                    param['value_mm'] = round(decimal_value * self.inch_to_mm, 2)
                    param['unit_metric'] = 'мм'

                parameters.append(param)

                if self.config.convert_to_metric:
                    return f"{param['value_mm']} мм"
                else:
                    # ИСПРАВЛЕНИЕ: используем строковое представление для сохранения точности
                    return f"{normalized_value_str} дюйм"

            except ValueError:
                logger.warning(f"Failed to parse decimal inch: {value_str}")
                return match.group(0)
        
        processed_text = self.patterns['decimal_inches'].sub(replace_decimal, text)
        return processed_text, parameters
    
    def _process_dimensions(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка размеров типа 245х10,03"""
        parameters = []
        
        def replace_dimension(match):
            dim1 = match.group(1).replace(',', '.')
            dim2 = match.group(2).replace(',', '.')
            dim3 = match.group(3).replace(',', '.') if match.group(3) else None
            
            try:
                # Определяем тип размера
                if dim3:
                    # Трехмерный размер
                    param = {
                        'type': 'размеры_3d',
                        'length': float(dim1),
                        'width': float(dim2),
                        'height': float(dim3),
                        'unit': 'мм',  # По умолчанию мм
                        'original': match.group(0)
                    }
                else:
                    # Двумерный размер
                    param = {
                        'type': 'размеры_2d',
                        'length': float(dim1),
                        'width': float(dim2),
                        'unit': 'мм',  # По умолчанию мм
                        'original': match.group(0)
                    }
                
                parameters.append(param)
                
                # Возвращаем нормализованный вид
                if dim3:
                    return f"{dim1}×{dim2}×{dim3}"
                else:
                    return f"{dim1}×{dim2}"
                    
            except ValueError:
                logger.warning(f"Failed to parse dimensions: {match.group(0)}")
                return match.group(0)
        
        processed_text = self.patterns['dimensions'].sub(replace_dimension, text)
        return processed_text, parameters

    def _process_units_with_numbers(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка единиц измерения с числами"""
        parameters = []

        def replace_unit(match):
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2).lower()

            try:
                value = float(value_str)

                # Стандартизация единицы
                standard_unit = self.unit_reverse_map.get(unit, unit)

                param = {
                    'type': 'измерение',
                    'value': value,
                    'unit': standard_unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                # Возвращаем стандартизованный вид
                return f"{value} {standard_unit}"

            except ValueError:
                logger.warning(f"Failed to parse unit value: {value_str}")
                return match.group(0)

        processed_text = self.patterns['units_with_numbers'].sub(replace_unit, text)
        return processed_text, parameters

    def _process_ranges(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка диапазонов значений"""
        parameters = []

        def replace_range(match):
            min_val = match.group(1).replace(',', '.')
            max_val = match.group(2).replace(',', '.')
            unit = match.group(3).lower()

            try:
                min_value = float(min_val)
                max_value = float(max_val)

                # Стандартизация единицы
                standard_unit = self.unit_reverse_map.get(unit, unit)

                param = {
                    'type': 'диапазон',
                    'min_value': min_value,
                    'max_value': max_value,
                    'unit': standard_unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                # Возвращаем стандартизованный вид
                return f"{min_value}–{max_value} {standard_unit}"

            except ValueError:
                logger.warning(f"Failed to parse range: {match.group(0)}")
                return match.group(0)

        processed_text = self.patterns['ranges'].sub(replace_range, text)
        return processed_text, parameters

    def _process_tech_codes(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка технических кодов"""
        parameters = []

        def replace_tech_code(match):
            code_type = match.group(1).upper()
            code_value = match.group(2)

            param = {
                'type': 'технический_код',
                'code_type': code_type,
                'code_value': code_value,
                'original': match.group(0)
            }

            parameters.append(param)

            # Возвращаем стандартизованный вид
            return f"{code_type} {code_value}"

        processed_text = self.patterns['tech_codes'].sub(replace_tech_code, text)
        return processed_text, parameters

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная обработка текстов"""
        results = []
        for text in texts:
            try:
                result = self.process_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text '{text}': {e}")
                results.append({
                    'original': text,
                    'processed': text,
                    'extracted_parameters': []
                })
        return results

    # Phase 2: Enhanced processing methods with typed prefixes

    def _process_compound_units(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка составных единиц: м³/ч, об/мин, л/мин"""
        parameters = []

        def replace_compound(match):
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2)

            try:
                value = float(value_str)

                param = {
                    'type': 'составная_единица',
                    'value': value,
                    'unit': unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                # Возвращаем с сохранением составной единицы
                return f"{value}{unit}"

            except ValueError:
                logger.warning(f"Failed to parse compound unit: {value_str}")
                return match.group(0)

        processed_text = self.patterns['compound_units'].sub(replace_compound, text)
        return processed_text, parameters

    def _process_simple_fractions(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка простых дробей: 1/2 → frac_1-2"""
        parameters = []
        protected_fractions = []

        def replace_fraction(match):
            numerator = match.group(1)
            denominator = match.group(2)

            # ИСПРАВЛЕНИЕ: Контекстное различение дробей и размеров одежды
            context_start = max(0, match.start() - 30)
            context_end = min(len(text), match.end() + 30)
            context = text[context_start:context_end].lower()

            # Проверяем контекст на размеры одежды
            clothing_keywords = ['костюм', 'размер', 'куртка', 'брюки', 'полукомбинезон',
                               'одежда', 'рост', 'обхват', 'муж', 'жен']

            # Проверяем диапазоны размеров одежды (обычно 150-200)
            try:
                num_val = int(numerator)
                den_val = int(denominator)

                # Размеры одежды: числа в диапазоне 150-200, близкие друг к другу
                if (150 <= num_val <= 200 and 150 <= den_val <= 200 and
                    abs(num_val - den_val) <= 20 and
                    any(keyword in context for keyword in clothing_keywords)):

                    # Это размер одежды, не дробь!
                    param = {
                        'type': 'размер_одежды',
                        'min_size': min(num_val, den_val),
                        'max_size': max(num_val, den_val),
                        'original': match.group(0)
                    }
                    parameters.append(param)

                    if self.config.use_typed_prefixes:
                        size_token = f"size_{numerator}-{denominator}"
                        placeholder = f"__SIZE_{len(protected_fractions)}__"
                        protected_fractions.append(size_token)
                        return placeholder
                    else:
                        return match.group(0)  # Оставляем как есть

            except ValueError:
                pass

            # Обычная дробь
            param = {
                'type': 'дробь',
                'numerator': int(numerator),
                'denominator': int(denominator),
                'decimal_value': float(numerator) / float(denominator),
                'original': match.group(0)
            }

            parameters.append(param)

            if self.config.use_typed_prefixes:
                # ИСПРАВЛЕНИЕ: защищаем дроби от дальнейшей обработки
                fraction_token = f"frac_{numerator}-{denominator}"
                placeholder = f"__FRACTION_{len(protected_fractions)}__"
                protected_fractions.append(fraction_token)
                return placeholder
            else:
                return f"{param['decimal_value']}"

        processed_text = self.patterns['simple_fractions'].sub(replace_fraction, text)

        # Сохраняем информацию о защищенных дробях
        if not hasattr(self, '_protected_fractions'):
            self._protected_fractions = {}

        for i, fraction in enumerate(protected_fractions):
            placeholder = f"__FRACTION_{i}__"
            self._protected_fractions[placeholder] = fraction

        return processed_text, parameters

    def _process_ratios(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка соотношений: 40:1 → ratio_40-1"""
        parameters = []

        def replace_ratio(match):
            first = match.group(1).replace(',', '.')
            second = match.group(2).replace(',', '.')

            try:
                first_val = float(first)
                second_val = float(second)

                param = {
                    'type': 'соотношение',
                    'first': first_val,
                    'second': second_val,
                    'ratio': first_val / second_val if second_val != 0 else None,
                    'original': match.group(0)
                }

                parameters.append(param)

                if self.config.use_typed_prefixes:
                    return f"ratio_{first}-{second}"
                else:
                    return f"{first}:{second}"

            except ValueError:
                logger.warning(f"Failed to parse ratio: {first}:{second}")
                return match.group(0)

        processed_text = self.patterns['ratios'].sub(replace_ratio, text)
        return processed_text, parameters

    def _process_ratios_protected(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка соотношений с защитой от конфликтов с диапазонами"""
        parameters = []
        protected_ratios = []

        def replace_ratio(match):
            first = match.group(1).replace(',', '.')
            second = match.group(2).replace(',', '.')

            try:
                first_val = float(first)
                second_val = float(second)

                param = {
                    'type': 'соотношение',
                    'first': first_val,
                    'second': second_val,
                    'ratio': first_val / second_val if second_val != 0 else None,
                    'original': match.group(0)
                }

                parameters.append(param)

                if self.config.use_typed_prefixes:
                    ratio_token = f"ratio_{first}-{second}"
                    # Защищаем соотношение от дальнейшей обработки
                    placeholder = f"__RATIO_{len(protected_ratios)}__"
                    protected_ratios.append(ratio_token)
                    return placeholder
                else:
                    return f"{first}:{second}"

            except ValueError:
                logger.warning(f"Failed to parse ratio: {first}:{second}")
                return match.group(0)

        processed_text = self.patterns['ratios'].sub(replace_ratio, text)

        # Сохраняем информацию о защищенных соотношениях
        if not hasattr(self, '_protected_ratios'):
            self._protected_ratios = {}

        for i, ratio in enumerate(protected_ratios):
            placeholder = f"__RATIO_{i}__"
            self._protected_ratios[placeholder] = ratio

        return processed_text, parameters

    def _process_ranges_enhanced(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка диапазонов: 0-1000°C → range_0-1000-c"""
        parameters = []

        def replace_range(match):
            start_str = match.group(1).replace(',', '.')
            end_str = match.group(2).replace(',', '.')
            unit = match.group(3).lower()

            try:
                start_val = float(start_str)
                end_val = float(end_str)

                param = {
                    'type': 'диапазон',
                    'start': start_val,
                    'end': end_val,
                    'unit': unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                if self.config.use_typed_prefixes:
                    # Упрощаем единицу для префикса
                    unit_short = unit.replace('°c', 'c').replace('°f', 'f')
                    return f"range_{start_str}-{end_str}-{unit_short}"
                else:
                    return f"{start_val}-{end_val} {unit}"

            except ValueError:
                logger.warning(f"Failed to parse range: {start_str}-{end_str} {unit}")
                return match.group(0)

        processed_text = self.patterns['ranges'].sub(replace_range, text)
        return processed_text, parameters

    def _process_dimensions_enhanced(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка размеров: 89×8мм → dims=[89,8] unit="мм" """
        parameters = []

        def replace_dimension(match):
            dim1_str = match.group(1).replace(',', '.')
            dim2_str = match.group(2).replace(',', '.')
            dim3_str = match.group(3).replace(',', '.') if match.group(3) else None
            unit_from_pattern = match.group(4) if match.group(4) else None

            try:
                dim1 = float(dim1_str)
                dim2 = float(dim2_str)
                dim3 = float(dim3_str) if dim3_str else None

                # ИСПРАВЛЕНИЕ: используем единицу из паттерна или по умолчанию
                unit = unit_from_pattern.lower() if unit_from_pattern else 'мм'

                if dim3:
                    param = {
                        'type': 'размеры_3d',
                        'dimensions': [dim1, dim2, dim3],
                        'unit': unit,
                        'original': match.group(0),
                        # Phase 3: Store as JSON array format
                        'dims': [dim1, dim2, dim3],
                        'unit_structured': unit
                    }
                    parameters.append(param)
                    if self.config.use_typed_prefixes:
                        return f"dims=[{dim1},{dim2},{dim3}] unit=\"{unit}\""
                    else:
                        return f"{dim1}×{dim2}×{dim3}"
                else:
                    param = {
                        'type': 'размеры_2d',
                        'dimensions': [dim1, dim2],
                        'unit': unit,
                        'original': match.group(0),
                        # Phase 3: Store as JSON array format
                        'dims': [dim1, dim2],
                        'unit_structured': unit
                    }
                    parameters.append(param)
                    if self.config.use_typed_prefixes:
                        return f"dims=[{dim1},{dim2}] unit=\"{unit}\""
                    else:
                        return f"{dim1}×{dim2}"

            except ValueError:
                logger.warning(f"Failed to parse dimensions: {match.group(0)}")
                return match.group(0)

        processed_text = self.patterns['dimensions'].sub(replace_dimension, text)
        return processed_text, parameters

    # Phase 3: Enhanced complex unit recognition methods

    def _process_compound_units_enhanced(self, text: str) -> Tuple[str, List[Dict]]:
        """Enhanced обработка составных единиц: м³/ч, об/мин, л/мин"""
        parameters = []
        protected_compounds = []

        def replace_compound(match):
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2)

            try:
                value = float(value_str)

                param = {
                    'type': 'составная_единица',
                    'value': value,
                    'unit': unit,
                    'original': match.group(0),
                    'structured': True  # Phase 3: mark as structured
                }

                parameters.append(param)

                # Phase 3: Preserve compound units as single tokens with protection
                compound_token = f"{value}{unit}"
                placeholder = f"__COMPOUND_{len(protected_compounds)}__"
                protected_compounds.append(compound_token)
                return placeholder

            except ValueError:
                logger.warning(f"Failed to parse compound unit: {value_str}")
                return match.group(0)

        processed_text = self.patterns['compound_units'].sub(replace_compound, text)

        # Сохраняем информацию о защищенных составных единицах для восстановления
        if not hasattr(self, '_protected_compounds'):
            self._protected_compounds = {}

        for i, compound in enumerate(protected_compounds):
            placeholder = f"__COMPOUND_{i}__"
            self._protected_compounds[placeholder] = compound

        return processed_text, parameters

    def _restore_protected_compounds(self, text: str) -> str:
        """Восстанавливает защищенные составные единицы, соотношения и дроби"""
        restored_text = text

        # Восстанавливаем защищенные составные единицы
        if hasattr(self, '_protected_compounds'):
            for placeholder, compound in self._protected_compounds.items():
                restored_text = restored_text.replace(placeholder, compound)
            # Очищаем кэш после использования
            self._protected_compounds = {}

        # Восстанавливаем защищенные соотношения
        if hasattr(self, '_protected_ratios'):
            for placeholder, ratio in self._protected_ratios.items():
                restored_text = restored_text.replace(placeholder, ratio)
            # Очищаем кэш после использования
            self._protected_ratios = {}

        # Восстанавливаем защищенные дроби
        if hasattr(self, '_protected_fractions'):
            for placeholder, fraction in self._protected_fractions.items():
                restored_text = restored_text.replace(placeholder, fraction)
            # Очищаем кэш после использования
            self._protected_fractions = {}

        # ИСПРАВЛЕНИЕ: Восстанавливаем защищенные марки материалов
        if hasattr(self, '_protected_materials'):
            for placeholder, material in self._protected_materials.items():
                restored_text = restored_text.replace(placeholder, material)
            # Очищаем кэш после использования
            self._protected_materials = {}

        return restored_text

    def _protect_material_grades(self, text: str) -> str:
        """
        ИСПРАВЛЕНИЕ: Защищает марки материалов от разрушения
        Примеры: 09Г2С-14, 12Х18Н10Т, Ст3пс, 40Х
        """
        if not hasattr(self, '_protected_materials'):
            self._protected_materials = {}

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
        ]

        protected_text = text
        material_counter = 0

        for pattern in material_patterns:
            def protect_material(match):
                nonlocal material_counter
                material = match.group(0)

                # Проверяем, что это действительно марка материала, а не случайное совпадение
                # Исключаем очевидно не материалы
                if (len(material) < 3 or
                    material.isdigit() or
                    material.lower() in ['гост', 'дин', 'iso']):
                    return material

                placeholder = f"__MATERIAL_{material_counter}__"
                self._protected_materials[placeholder] = material
                material_counter += 1
                return placeholder

            protected_text = re.sub(pattern, protect_material, protected_text, flags=re.IGNORECASE)

        return protected_text

    def _process_pressure_units(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка единиц давления: атм, бар, торр, psi"""
        parameters = []

        def replace_pressure(match):
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2).lower()

            try:
                value = float(value_str)

                param = {
                    'type': 'давление',
                    'value': value,
                    'unit': unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                # Normalize pressure units
                if unit == 'атм':
                    return f"{value}атм"
                else:
                    return f"{value} {unit}"

            except ValueError:
                logger.warning(f"Failed to parse pressure: {value_str}")
                return match.group(0)

        processed_text = self.patterns['pressure_units'].sub(replace_pressure, text)
        return processed_text, parameters

    def _process_volume_units(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка объемов: л, мл, м³, см³"""
        parameters = []

        def replace_volume(match):
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2).lower()

            try:
                value = float(value_str)

                param = {
                    'type': 'объем',
                    'value': value,
                    'unit': unit,
                    'original': match.group(0)
                }

                parameters.append(param)

                # Standardize volume units
                if unit in ['куб.м', 'куб м']:
                    return f"{value}м³"
                else:
                    return f"{value}{unit}"

            except ValueError:
                logger.warning(f"Failed to parse volume: {value_str}")
                return match.group(0)

        processed_text = self.patterns['volume_units'].sub(replace_volume, text)
        return processed_text, parameters

    def _process_dimensionless_coefficients(self, text: str) -> Tuple[str, List[Dict]]:
        """Обработка безразмерных коэффициентов"""
        parameters = []

        def replace_coefficient(match):
            # Может быть в группе 1 или 2 в зависимости от паттерна
            value_str = (match.group(1) or match.group(2)).replace(',', '.')

            try:
                value = float(value_str)

                param = {
                    'type': 'коэффициент',
                    'value': value,
                    'unit': 'безразмерный',
                    'original': match.group(0)
                }

                parameters.append(param)

                return f"коэффициент {value}"

            except ValueError:
                logger.warning(f"Failed to parse coefficient: {value_str}")
                return match.group(0)

        processed_text = self.patterns['dimensionless_coefficients'].sub(replace_coefficient, text)
        return processed_text, parameters

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Получение статистики обработки"""
        total_texts = len(results)
        total_parameters = sum(len(r['extracted_parameters']) for r in results)

        # Подсчет по типам параметров
        param_types = {}
        for result in results:
            for param in result['extracted_parameters']:
                param_type = param['type']
                param_types[param_type] = param_types.get(param_type, 0) + 1

        return {
            'total_texts': total_texts,
            'total_parameters': total_parameters,
            'avg_parameters_per_text': total_parameters / total_texts if total_texts > 0 else 0,
            'parameter_types': param_types
        }
