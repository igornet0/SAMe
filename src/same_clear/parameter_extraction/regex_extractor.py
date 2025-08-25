"""
Модуль извлечения параметров с использованием регулярных выражений
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Типы параметров"""
    DIMENSION = "dimension"
    WEIGHT = "weight"
    VOLUME = "volume"
    ELECTRICAL = "electrical"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FREQUENCY = "frequency"
    SPEED = "speed"
    MATERIAL = "material"
    COLOR = "color"
    BRAND = "brand"
    MODEL = "model"
    ARTICLE = "article"
    CAPACITY = "capacity"  # Емкость аккумуляторов
    THREAD = "thread"      # Резьба
    PROTECTION = "protection"  # Степень защиты
    VOLTAGE = "voltage"    # Напряжение
    CURRENT = "current"    # Ток
    QUANTITY = "quantity"  # Количество
    CHARACTERISTIC = "characteristic"  # Характеристики


@dataclass
class ParameterPattern:
    """Паттерн для извлечения параметра"""
    name: str
    pattern: str
    parameter_type: ParameterType
    unit: Optional[str] = None
    description: str = ""
    priority: int = 1
    compiled_pattern: re.Pattern = field(init=False)
    
    def __post_init__(self):
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.UNICODE)


@dataclass
class ExtractedParameter:
    """Извлеченный параметр"""
    name: str
    value: Union[str, float, int]
    unit: Optional[str]
    parameter_type: ParameterType
    confidence: float
    source_text: str
    position: Tuple[int, int]  # Позиция в тексте (start, end)


class RegexParameterExtractor:
    """Класс для извлечения параметров с помощью регулярных выражений"""
    
    def __init__(self):
        self.patterns: List[ParameterPattern] = []
        self._setup_default_patterns()
        logger.info("RegexParameterExtractor initialized")
    
    def _setup_default_patterns(self):
        """Настройка стандартных паттернов для МТР"""

        # Размеры и габариты - более точные паттерны
        dimension_patterns = [
            ParameterPattern(
                name="diameter",
                pattern=r"(?:диаметр|диам\.?|ø)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Диаметр изделия",
                priority=3
            ),
            ParameterPattern(
                name="length",
                pattern=r"(?:длина|длин\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Длина изделия",
                priority=3
            ),
            ParameterPattern(
                name="width",
                pattern=r"(?:ширина|шир\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Ширина изделия",
                priority=3
            ),
            ParameterPattern(
                name="height",
                pattern=r"(?:высота|выс\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Высота изделия",
                priority=3
            ),
            ParameterPattern(
                name="thickness",
                pattern=r"(?:толщина|толщ\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Толщина изделия",
                priority=3
            ),
            # Размеры в формате "1200мм"
            ParameterPattern(
                name="dimension_direct",
                pattern=r"(\d{2,4})\s*(мм|см|м)(?!\w)",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Размер изделия",
                priority=2
            ),
        ]
        
        # Вес - улучшенные паттерны
        weight_patterns = [
            ParameterPattern(
                name="weight",
                pattern=r"(?:вес|масса)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(г|кг|т)",
                parameter_type=ParameterType.WEIGHT,
                unit="кг",
                description="Вес изделия",
                priority=3
            ),
            # Вес в граммах (с пробелом и слитно)
            ParameterPattern(
                name="weight_g",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(г|гр|g)(?!\w)",
                parameter_type=ParameterType.WEIGHT,
                unit="г",
                description="Вес в граммах",
                priority=3
            ),
            # Вес в граммах слитно
            ParameterPattern(
                name="weight_g_compact",
                pattern=r"(\d+(?:[.,]\d+)?)(г|гр)(?!\w)",
                parameter_type=ParameterType.WEIGHT,
                unit="г",
                description="Вес в граммах (слитно)",
                priority=4
            ),
            # Вес в килограммах (с пробелом и слитно)
            ParameterPattern(
                name="weight_kg",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(кг|kg)(?!\w)",
                parameter_type=ParameterType.WEIGHT,
                unit="кг",
                description="Вес в килограммах",
                priority=3
            ),
            # Вес в килограммах слитно
            ParameterPattern(
                name="weight_kg_compact",
                pattern=r"(\d+(?:[.,]\d+)?)(кг|kg)(?!\w)",
                parameter_type=ParameterType.WEIGHT,
                unit="кг",
                description="Вес в килограммах (слитно)",
                priority=4
            ),
        ]
        
        # Электрические характеристики - улучшенные паттерны
        electrical_patterns = [
            # Напряжение
            ParameterPattern(
                name="voltage_explicit",
                pattern=r"(?:напряжение|напр\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(в|v|вольт)",
                parameter_type=ParameterType.VOLTAGE,
                unit="В",
                description="Напряжение (явное)",
                priority=3
            ),
            ParameterPattern(
                name="voltage_direct",
                pattern=r"(?<![\w\-])(\d{1,5}(?:[.,]\d+)?)\s*(в|v)(?!\w)",  # Ограничиваем до 5 цифр и исключаем артикулы
                parameter_type=ParameterType.VOLTAGE,
                unit="В",
                description="Напряжение (прямое)",
                priority=3
            ),
            # Ток
            ParameterPattern(
                name="current_explicit",
                pattern=r"(?:ток)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(а|a|ампер)",
                parameter_type=ParameterType.CURRENT,
                unit="А",
                description="Сила тока (явная)",
                priority=3
            ),
            ParameterPattern(
                name="current_direct",
                pattern=r"(?<![\w\-])(\d{1,4}(?:[.,]\d+)?)\s*(а|a)(?!\w)",  # Ограничиваем до 4 цифр и исключаем артикулы
                parameter_type=ParameterType.CURRENT,
                unit="А",
                description="Сила тока (прямая)",
                priority=3
            ),
            ParameterPattern(
                name="power",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(вт|w)(?!\w)",
                parameter_type=ParameterType.ELECTRICAL,
                unit="Вт",
                description="Мощность",
                priority=3
            ),
            ParameterPattern(
                name="power_kw",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(квт|kw)(?!\w)",
                parameter_type=ParameterType.ELECTRICAL,
                unit="кВт",
                description="Мощность в киловаттах",
                priority=3
            ),

        ]
        
        # Давление - более точные паттерны
        pressure_patterns = [
            ParameterPattern(
                name="pressure",
                pattern=r"(?:давление|давл\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(па|pa|кпа|kpa|мпа|mpa|бар|bar|атм|atm)",
                parameter_type=ParameterType.PRESSURE,
                unit="МПа",
                description="Рабочее давление",
                priority=3
            ),
            # Степень защиты IP
            ParameterPattern(
                name="ip_rating",
                pattern=r"ip\s*(\d{2})",
                parameter_type=ParameterType.ARTICLE,
                description="Степень защиты IP",
                priority=3
            ),
        ]
        
        # Температура - более строгие паттерны
        temperature_patterns = [
            ParameterPattern(
                name="temperature",
                pattern=r"(?:температура|темп\.?)\s*[=:]?\s*([+-]?\d{1,3}(?:[.,]\d+)?)\s*(°?c|°?f)",
                parameter_type=ParameterType.TEMPERATURE,
                unit="°C",
                description="Рабочая температура",
                priority=3
            ),
            # Цветовая температура - отдельный паттерн
            ParameterPattern(
                name="color_temperature",
                pattern=r"(\d{4,5})\s*k(?!\w)",
                parameter_type=ParameterType.TEMPERATURE,
                unit="K",
                description="Цветовая температура",
                priority=3
            ),
        ]
        
        # Частота
        frequency_patterns = [
            ParameterPattern(
                name="frequency_explicit",
                pattern=r"(?:частота|част\.?)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(гц|hz|кгц|khz|мгц|mhz|ггц|ghz)",
                parameter_type=ParameterType.FREQUENCY,
                unit="Гц",
                description="Частота (явная)",
                priority=3
            ),
            ParameterPattern(
                name="frequency_direct",
                pattern=r"(\d{1,4}(?:[.,]\d+)?)\s*(гц|hz|кгц|khz|мгц|mhz|ггц|ghz)(?!\w)",
                parameter_type=ParameterType.FREQUENCY,
                unit="Гц",
                description="Частота (прямая)",
                priority=3
            ),
        ]
        
        # Материал и цвет - очищенные паттерны
        material_patterns = [
            ParameterPattern(
                name="material",
                pattern=r"(?:материал|мат\.?)\s*[=:]?\s*([а-яё\s]{3,20})(?:\s|$|,|;)",
                parameter_type=ParameterType.MATERIAL,
                description="Материал изготовления",
                priority=3
            ),
            ParameterPattern(
                name="steel_grade",
                pattern=r"(?:сталь|марка\s+стали)\s+([а-я0-9\-]{2,10})",
                parameter_type=ParameterType.MATERIAL,
                description="Марка стали",
                priority=3
            ),
            # Конкретные материалы
            ParameterPattern(
                name="material_specific",
                pattern=r"(?:^|,|\s)(медный|стальной|пластиковый|алюминиевый|железный|деревянный|резиновый)(?:,|\s|$)",
                parameter_type=ParameterType.MATERIAL,
                description="Конкретный материал",
                priority=3
            ),
            # Цвета - улучшенный паттерн
            ParameterPattern(
                name="color",
                pattern=r"(?:^|,|\s|цвет\s*)(белый|черный|красный|синий|зеленый|желтый|серый|коричневый|серебристый|золотой|прозрачный)(?:,|\s|$)",
                parameter_type=ParameterType.COLOR,
                description="Цвет изделия",
                priority=2
            ),
        ]
        
        # Артикулы и коды
        article_patterns = [
            ParameterPattern(
                name="article",
                pattern=r"(?:артикул|арт\.?|код|code)\s*[=:]?\s*([a-z0-9\-_]+)",
                parameter_type=ParameterType.ARTICLE,
                description="Артикул изделия",
                priority=3
            ),
            ParameterPattern(
                name="gost",
                pattern=r"(гост\s+\d+(?:\.\d+)*(?:\-\d+)?)",
                parameter_type=ParameterType.ARTICLE,
                description="ГОСТ стандарт",
                priority=3
            ),
            # Модели и бренды
            ParameterPattern(
                name="model_code",
                pattern=r"(?:модель|model)\s*[=:]?\s*([a-z0-9\-_]+)",
                parameter_type=ParameterType.MODEL,
                description="Модель изделия",
                priority=3
            ),
            # Объем в мл/л
            ParameterPattern(
                name="volume_ml",
                pattern=r"(\d+)\s*(мл|ml)(?!\w)",
                parameter_type=ParameterType.VOLUME,
                unit="мл",
                description="Объем в миллилитрах",
                priority=3
            ),
            ParameterPattern(
                name="volume_l",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(л|l)(?!\w)",
                parameter_type=ParameterType.VOLUME,
                unit="л",
                description="Объем в литрах",
                priority=3
            ),
        ]

        # Емкость аккумуляторов - расширенные паттерны
        capacity_patterns = [
            ParameterPattern(
                name="battery_capacity_mah",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(мач|mah|ма·ч|ma·h)(?!\w)",
                parameter_type=ParameterType.CAPACITY,
                unit="мАч",
                description="Емкость аккумулятора в мАч",
                priority=3
            ),
            # Слитное написание мАч (более широкий паттерн)
            ParameterPattern(
                name="battery_capacity_mah_compact",
                pattern=r"(\d+)(мач|mah)(?!\w)",
                parameter_type=ParameterType.CAPACITY,
                unit="мАч",
                description="Емкость аккумулятора в мАч (слитно)",
                priority=4
            ),
            # Емкость в скобках
            ParameterPattern(
                name="battery_capacity_mah_brackets",
                pattern=r"\((\d+(?:[.,]\d+)?)(мач|mah)\)",
                parameter_type=ParameterType.CAPACITY,
                unit="мАч",
                description="Емкость аккумулятора в мАч (в скобках)",
                priority=4
            ),
            ParameterPattern(
                name="battery_capacity_ah",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(ач|ah|а·ч|a·h)(?!\w)",
                parameter_type=ParameterType.CAPACITY,
                unit="Ач",
                description="Емкость аккумулятора в Ач",
                priority=3
            ),
            # Емкость слитно
            ParameterPattern(
                name="battery_capacity_ah_compact",
                pattern=r"(\d+(?:[.,]\d+)?)(ач|ah)(?!\w)",
                parameter_type=ParameterType.CAPACITY,
                unit="Ач",
                description="Емкость аккумулятора в Ач (слитно)",
                priority=4
            ),
            ParameterPattern(
                name="battery_capacity_wh",
                pattern=r"(\d+(?:[.,]\d+)?)\s*(втч|wh|вт·ч|w·h)(?!\w)",
                parameter_type=ParameterType.CAPACITY,
                unit="Втч",
                description="Емкость аккумулятора в Втч",
                priority=3
            ),
        ]

        # Резьба
        thread_patterns = [
            ParameterPattern(
                name="metric_thread",
                pattern=r"м(\d+)(?:х|x)(\d+(?:[.,]\d+)?)",
                parameter_type=ParameterType.THREAD,
                description="Метрическая резьба",
                priority=3
            ),
        ]

        # Количество
        quantity_patterns = [
            ParameterPattern(
                name="quantity_pieces",
                pattern=r"(\d+)\s*(шт|штук|штуки)(?!\w)",
                parameter_type=ParameterType.QUANTITY,
                unit="шт",
                description="Количество штук",
                priority=3
            ),
            ParameterPattern(
                name="quantity_package",
                pattern=r"(?:уп\.?|упаковка)\s*(\d{1,4})\s*(шт|штук)?",  # Ограничиваем до 4 цифр
                parameter_type=ParameterType.QUANTITY,
                unit="шт",
                description="Количество в упаковке",
                priority=3
            ),
            ParameterPattern(
                name="quantity_set",
                pattern=r"(\d+)\s*(компл|комплект|набор)(?!\w)",
                parameter_type=ParameterType.QUANTITY,
                unit="компл",
                description="Количество комплектов",
                priority=3
            ),
        ]

        # Размеры (комбинированные)
        dimension_combined_patterns = [
            ParameterPattern(
                name="dimensions_3d",
                pattern=r"(\d+)(?:х|x)(\d+)(?:х|x)(\d+)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                description="Размеры 3D (длина x ширина x высота)",
                priority=3
            ),
            ParameterPattern(
                name="dimensions_2d",
                pattern=r"(\d+)(?:х|x)(\d+)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                description="Размеры 2D (длина x ширина)",
                priority=2
            ),
        ]

        # Характеристики автоматов
        characteristic_patterns = [
            ParameterPattern(
                name="breaker_characteristic",
                pattern=r"(?:характеристика|тип)\s*([a-d]|[а-г])(?!\w)",
                parameter_type=ParameterType.CHARACTERISTIC,
                description="Характеристика автомата",
                priority=3
            ),
            ParameterPattern(
                name="characteristic_direct",
                pattern=r"(?:^|,|\s)([a-d]|[а-г])(?:,|\s|$)",
                parameter_type=ParameterType.CHARACTERISTIC,
                description="Характеристика (прямая)",
                priority=2
            ),
            ParameterPattern(
                name="breaker_type_direct",
                pattern=r"(\d+a?)\s*-\s*(\d+)\s*-\s*([a-d]|[а-г])(?!\w)",
                parameter_type=ParameterType.CHARACTERISTIC,
                description="Тип автомата (прямой)",
                priority=3
            ),
        ]

        # Слитные значения (комбинированные паттерны)
        combined_patterns = [
            # Вес + объем (например, "1,8кг10л")
            ParameterPattern(
                name="weight_volume_combined",
                pattern=r"(\d+(?:[.,]\d+)?)(кг|kg)(\d+(?:[.,]\d+)?)(л|l)(?!\w)",
                parameter_type=ParameterType.WEIGHT,  # Основной тип - вес
                description="Комбинированный вес и объем",
                priority=4
            ),
            # Мощность + обороты (например, "0.37квт1000обмин")
            ParameterPattern(
                name="power_rpm_combined",
                pattern=r"(\d+(?:[.,]\d+)?)(квт|kw)(\d+)(об/?мин|rpm)(?!\w)",
                parameter_type=ParameterType.ELECTRICAL,
                description="Комбинированная мощность и обороты",
                priority=4
            ),
            # Ток + напряжение (например, "32a220v")
            ParameterPattern(
                name="current_voltage_combined",
                pattern=r"(\d+(?:[.,]\d+)?)(а|a)(\d+(?:[.,]\d+)?)(в|v)(?!\w)",
                parameter_type=ParameterType.CURRENT,
                description="Комбинированный ток и напряжение",
                priority=4
            ),
            # Размеры слитно (например, "326х175х208")
            ParameterPattern(
                name="dimensions_compact",
                pattern=r"(\d+)х(\d+)х?(\d+)?\s*(мм|см|м)?(?!\w)",
                parameter_type=ParameterType.DIMENSION,
                description="Размеры слитно",
                priority=4
            ),
        ]
        
        # Объединяем все паттерны
        all_patterns = (
            dimension_patterns + weight_patterns + electrical_patterns +
            pressure_patterns + temperature_patterns + frequency_patterns +
            material_patterns + article_patterns + capacity_patterns + thread_patterns +
            quantity_patterns + dimension_combined_patterns + characteristic_patterns +
            combined_patterns
        )
        
        self.patterns.extend(all_patterns)
        logger.info(f"Loaded {len(self.patterns)} default patterns")
    
    def extract_parameters(self, text: str) -> List[ExtractedParameter]:
        """
        Извлечение параметров из текста
        
        Args:
            text: Входной текст
            
        Returns:
            Список извлеченных параметров
        """
        if not text or not isinstance(text, str):
            return []
        
        extracted_params = []
        
        for pattern in self.patterns:
            matches = pattern.compiled_pattern.finditer(text)
            
            for match in matches:
                try:
                    param = self._create_parameter(pattern, match, text)
                    if param:
                        extracted_params.append(param)

                        # Извлекаем дополнительные параметры из комбинированных паттернов
                        additional_params = self._extract_additional_from_combined(pattern, match, text)
                        extracted_params.extend(additional_params)
                except Exception as e:
                    logger.warning(f"Error extracting parameter {pattern.name}: {e}")
        
        # Удаляем дубликаты и сортируем по приоритету
        extracted_params = self._deduplicate_parameters(extracted_params)

        # Валидация параметров
        extracted_params = self._validate_parameters(extracted_params)

        extracted_params.sort(key=lambda x: (x.parameter_type.value, -x.confidence))

        return extracted_params
    
    def _create_parameter(self, pattern: ParameterPattern, match: re.Match, text: str) -> Optional[ExtractedParameter]:
        """Создание объекта параметра из совпадения"""
        groups = match.groups()

        if not groups:
            return None

        # Проверка на ложные срабатывания
        if self._is_false_positive(pattern, match, text):
            return None

        # Основное значение (первая группа)
        raw_value = groups[0].replace(',', '.')

        # Единица измерения (вторая группа, если есть)
        unit = groups[1] if len(groups) > 1 and groups[1] else pattern.unit

        # Преобразование значения
        if pattern.parameter_type in [ParameterType.DIMENSION, ParameterType.WEIGHT,
                                    ParameterType.ELECTRICAL, ParameterType.PRESSURE,
                                    ParameterType.TEMPERATURE, ParameterType.FREQUENCY,
                                    ParameterType.VOLUME, ParameterType.CAPACITY,
                                    ParameterType.VOLTAGE, ParameterType.CURRENT,
                                    ParameterType.QUANTITY]:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
        elif pattern.parameter_type == ParameterType.THREAD:
            # Для резьбы объединяем группы
            if len(groups) >= 2:
                value = f"М{groups[0]}x{groups[1]}"
            else:
                value = raw_value.strip()
        elif pattern.name in ['dimensions_3d', 'dimensions_2d']:
            # Для размеров объединяем группы
            if pattern.name == 'dimensions_3d' and len(groups) >= 3:
                value = f"{groups[0]}x{groups[1]}x{groups[2]}"
                unit = groups[3] if len(groups) > 3 and groups[3] else "мм"
            elif pattern.name == 'dimensions_2d' and len(groups) >= 2:
                value = f"{groups[0]}x{groups[1]}"
                unit = groups[2] if len(groups) > 2 and groups[2] else "мм"
            else:
                value = raw_value.strip()
        elif pattern.name == 'breaker_type_direct':
            # Для характеристик автоматов объединяем группы
            if len(groups) >= 3:
                value = f"{groups[0]}-{groups[1]}-{groups[2]}"
            else:
                value = raw_value.strip()
        elif pattern.name in ['weight_volume_combined', 'power_rpm_combined', 'current_voltage_combined']:
            # Для комбинированных паттернов извлекаем первое значение
            try:
                value = float(groups[0].replace(',', '.'))
                unit = groups[1]
            except (ValueError, IndexError):
                value = raw_value.strip()
        elif pattern.name == 'dimensions_compact':
            # Для компактных размеров
            if len(groups) >= 2:
                if groups[2]:  # Есть третий размер
                    value = f"{groups[0]}x{groups[1]}x{groups[2]}"
                else:
                    value = f"{groups[0]}x{groups[1]}"
                unit = groups[3] if len(groups) > 3 and groups[3] else "мм"
            else:
                value = raw_value.strip()
        else:
            value = raw_value.strip()

        # Расчет уверенности
        confidence = self._calculate_confidence(pattern, match, text)

        main_param = ExtractedParameter(
            name=pattern.name,
            value=value,
            unit=unit,
            parameter_type=pattern.parameter_type,
            confidence=confidence,
            source_text=match.group(0),
            position=(match.start(), match.end())
        )

        return main_param

    def _extract_additional_from_combined(self, pattern: ParameterPattern, match: re.Match, text: str) -> List[ExtractedParameter]:
        """Извлечение дополнительных параметров из комбинированных паттернов"""
        additional_params = []
        groups = match.groups()

        try:
            if pattern.name == 'weight_volume_combined' and len(groups) >= 4:
                # Извлекаем объем из комбинированного паттерна
                volume_value = float(groups[2].replace(',', '.'))
                volume_unit = groups[3]

                volume_param = ExtractedParameter(
                    name="volume_from_combined",
                    value=volume_value,
                    unit=volume_unit,
                    parameter_type=ParameterType.VOLUME,
                    confidence=0.8,
                    source_text=match.group(0),
                    position=(match.start(), match.end())
                )
                additional_params.append(volume_param)

            elif pattern.name == 'power_rpm_combined' and len(groups) >= 4:
                # Извлекаем обороты из комбинированного паттерна
                rpm_value = float(groups[2])

                rpm_param = ExtractedParameter(
                    name="rpm_from_combined",
                    value=rpm_value,
                    unit="об/мин",
                    parameter_type=ParameterType.SPEED,
                    confidence=0.8,
                    source_text=match.group(0),
                    position=(match.start(), match.end())
                )
                additional_params.append(rpm_param)

            elif pattern.name == 'current_voltage_combined' and len(groups) >= 4:
                # Извлекаем напряжение из комбинированного паттерна
                voltage_value = float(groups[2].replace(',', '.'))
                voltage_unit = groups[3]

                voltage_param = ExtractedParameter(
                    name="voltage_from_combined",
                    value=voltage_value,
                    unit=voltage_unit,
                    parameter_type=ParameterType.VOLTAGE,
                    confidence=0.8,
                    source_text=match.group(0),
                    position=(match.start(), match.end())
                )
                additional_params.append(voltage_param)

        except (ValueError, IndexError):
            pass

        return additional_params

    def _calculate_confidence(self, pattern: ParameterPattern, match: re.Match, text: str) -> float:
        """Расчет уверенности в извлеченном параметре"""
        base_confidence = 0.7
        
        # Бонус за точность паттерна
        if len(match.group(0)) > 5:
            base_confidence += 0.1
        
        # Бонус за наличие единиц измерения
        if len(match.groups()) > 1 and match.groups()[1]:
            base_confidence += 0.1
        
        # Бонус за приоритет паттерна
        base_confidence += pattern.priority * 0.05
        
        # Штраф за слишком короткое совпадение
        if len(match.group(0)) < 3:
            base_confidence -= 0.2
        
        return min(1.0, max(0.0, base_confidence))

    def _is_false_positive(self, pattern: ParameterPattern, match: re.Match, text: str) -> bool:
        """Проверка на ложные срабатывания"""
        matched_text = match.group(0).lower()
        value = match.groups()[0] if match.groups() else ""

        # Контекст вокруг совпадения
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end].lower()

        # Исключения для размеров
        if pattern.parameter_type == ParameterType.DIMENSION:
            # Исключаем цветовую температуру (например, "6500k")
            if re.search(r'\d+k\b', context):
                return True
            # Исключаем IP рейтинги (например, "ip40")
            if re.search(r'ip\s*\d+', context):
                return True
            # Исключаем модели (например, "w80dm")
            if re.search(r'[a-z]\d+[a-z]', context):
                return True

        # Исключения для мощности
        if pattern.parameter_type == ParameterType.ELECTRICAL and pattern.name == "power":
            # Исключаем цветовую температуру
            if re.search(r'\d+k\b', context):
                return True

        # Исключения для давления
        if pattern.parameter_type == ParameterType.PRESSURE:
            # Исключаем IP рейтинги
            if re.search(r'ip\s*\d+', context):
                return True

        # Исключения для температуры
        if pattern.parameter_type == ParameterType.TEMPERATURE and pattern.name == "temperature":
            # Исключаем артикулы и модели с большими числами
            if isinstance(value, str):
                try:
                    temp_value = float(value.replace(',', '.'))
                    # Исключаем нереальные температуры
                    if abs(temp_value) > 1000:
                        return True
                    # Исключаем если это похоже на артикул
                    if re.search(r'[a-z]\d+[a-z]|elt-\d+|\d+-\d+', context):
                        return True
                except ValueError:
                    pass

        # Исключения для веса
        if pattern.parameter_type == ParameterType.WEIGHT:
            # Исключаем только явные артикулы (но не названия продуктов)
            if re.search(r'артикул|код', context):
                return True
            # Исключаем если это часть артикула
            if isinstance(value, str):
                try:
                    weight_value = float(value.replace(',', '.'))
                    # Исключаем нереально большие веса (больше 1000 кг)
                    if weight_value > 1000 and pattern.unit == "кг":
                        return True
                    # Исключаем если это похоже на номер модели в контексте Loctite (только для больших чисел)
                    if re.search(r'loctite.*prism', context) and weight_value > 100:
                        return True
                except ValueError:
                    pass

        # Исключения для емкости
        if pattern.parameter_type == ParameterType.CAPACITY:
            # Исключаем если это не в контексте аккумулятора или устройства
            battery_context = re.search(r'аккумулятор|батарея|battery|акб|power\s*bank|ноутбук|laptop|телефон|phone|рация|радиостанция|шуруповерт|matebook|probook|greenworks|makita|li-ion', context)
            if not battery_context:
                return True

        # Исключения для частоты
        if pattern.parameter_type == ParameterType.FREQUENCY:
            # Исключаем если это похоже на артикул
            if re.search(r'[a-z]\d+[a-z]|f\d+-\d+|\d+f\d+', context):
                return True
            # Проверяем реалистичность частоты
            if isinstance(value, str):
                try:
                    freq_value = float(value.replace(',', '.'))
                    # Исключаем нереалистичные частоты (меньше 1 Гц или больше 100 ГГц)
                    if freq_value < 1 or freq_value > 100000000000:
                        return True
                except ValueError:
                    pass

        # Исключения для напряжения и тока
        if pattern.parameter_type in [ParameterType.VOLTAGE, ParameterType.CURRENT]:
            # Исключаем если это похоже на артикул или модель
            if re.search(r'elt-\d+|модель|model|артикул|арт\.?|код|code', context):
                return True

            # Исключаем большие числа, которые явно являются артикулами/кодами
            if isinstance(value, str):
                try:
                    num_value = float(value.replace(',', '.'))
                    # Исключаем нереалистично большие значения для тока (>1000А) и напряжения (>100000В)
                    if pattern.parameter_type == ParameterType.CURRENT and num_value > 1000:
                        return True
                    elif pattern.parameter_type == ParameterType.VOLTAGE and num_value > 100000:
                        return True

                    # Исключаем числа с 6+ цифрами (скорее всего артикулы)
                    if len(value.replace('.', '').replace(',', '')) >= 6:
                        return True
                except ValueError:
                    pass

        # Исключения для количественных параметров
        if pattern.parameter_type == ParameterType.QUANTITY:
            # Исключаем если это похоже на артикул или код
            if re.search(r'артикул|арт\.?|код|code|модель|model', context):
                return True

            # Исключаем нереалистично большие количества (>10000)
            if isinstance(value, str):
                try:
                    qty_value = float(value.replace(',', '.'))
                    if qty_value > 10000:  # Нереалистично большое количество в упаковке
                        return True
                except ValueError:
                    pass

        # Исключения для материалов
        if pattern.parameter_type == ParameterType.MATERIAL:
            # Исключаем системы и модели
            if re.search(r'система|system|модель|model', context):
                return True
            # Исключаем слишком короткие совпадения
            if isinstance(value, str) and len(value.strip()) < 3:
                return True
            # Исключаем фрагменты слов (расширенный список)
            if isinstance(value, str):
                clean_value = value.strip().lower()
                forbidden_fragments = [
                    'ический', 'ор', 'ь', 'ем', 'ема', 'ный', 'ая', 'ое', 'ые',
                    'ический выключатель', 'ор латр', 'ор регулируемый',
                    'дифференциальный', 'иэк'
                ]
                if clean_value in forbidden_fragments:
                    return True
                # Исключаем если это часть составного слова
                if re.search(r'автомат|трансформатор|выключатель', context) and len(clean_value) < 8:
                    return True

        # Общие исключения для коротких значений
        if isinstance(value, str) and len(value.strip()) < 2:
            return True

        return False

    def _deduplicate_parameters(self, parameters: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Удаление дублирующихся параметров"""
        seen = {}
        result = []

        for param in parameters:
            # Более сложная логика дедупликации
            if param.parameter_type == ParameterType.DIMENSION:
                # Для размеров учитываем и значение
                key = (param.parameter_type, param.value, param.unit)
            elif param.parameter_type in [ParameterType.WEIGHT, ParameterType.VOLUME,
                                        ParameterType.CAPACITY, ParameterType.VOLTAGE,
                                        ParameterType.CURRENT]:
                # Для числовых параметров группируем по типу и значению
                key = (param.parameter_type, param.value, param.unit)
            else:
                key = (param.name, param.parameter_type)

            if key not in seen or param.confidence > seen[key].confidence:
                seen[key] = param

        # Дополнительная проверка на похожие параметры
        final_result = []
        for param in seen.values():
            # Проверяем, нет ли уже похожего параметра
            is_duplicate = False
            for existing in final_result:
                if (param.parameter_type == existing.parameter_type and
                    param.value == existing.value and
                    param.unit == existing.unit):
                    # Оставляем параметр с более высокой уверенностью
                    if param.confidence > existing.confidence:
                        final_result.remove(existing)
                        final_result.append(param)
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_result.append(param)

        return final_result
    
    def add_pattern(self, pattern: ParameterPattern):
        """Добавление нового паттерна"""
        self.patterns.append(pattern)
        logger.info(f"Added pattern: {pattern.name}")
    
    def remove_pattern(self, pattern_name: str):
        """Удаление паттерна по имени"""
        self.patterns = [p for p in self.patterns if p.name != pattern_name]
        logger.info(f"Removed pattern: {pattern_name}")
    
    def _validate_parameters(self, parameters: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Валидация извлеченных параметров на реалистичность"""
        validated_params = []

        for param in parameters:
            if self._is_parameter_realistic(param):
                validated_params.append(param)
            else:
                logger.debug(f"Parameter {param.name}={param.value} failed validation")

        return validated_params

    def _is_parameter_realistic(self, param: ExtractedParameter) -> bool:
        """Проверка реалистичности параметра"""
        if not isinstance(param.value, (int, float)):
            return True  # Для нечисловых значений не проверяем

        value = float(param.value)

        # Проверки по типам параметров
        if param.parameter_type == ParameterType.VOLTAGE:
            # Напряжение: от 0.1В до 100000В
            return 0.1 <= value <= 100000

        elif param.parameter_type == ParameterType.CURRENT:
            # Ток: от 0.001А до 1000А (более реалистичный диапазон)
            return 0.001 <= value <= 1000

        elif param.parameter_type == ParameterType.FREQUENCY:
            # Частота: от 0.1Гц до 100ГГц
            return 0.1 <= value <= 100000000000

        elif param.parameter_type == ParameterType.ELECTRICAL:
            # Мощность: от 0.1Вт до 1000000Вт
            return 0.1 <= value <= 1000000

        elif param.parameter_type == ParameterType.WEIGHT:
            # Вес: от 0.001г до 100000кг
            if param.unit == "г":
                return 0.001 <= value <= 100000000  # до 100 тонн в граммах
            elif param.unit == "кг":
                return 0.001 <= value <= 100000  # до 100 тонн
            return True

        elif param.parameter_type == ParameterType.CAPACITY:
            # Емкость аккумуляторов
            if param.unit in ["мАч", "mah"]:
                return 1 <= value <= 1000000  # от 1мАч до 1000Ач
            elif param.unit in ["Ач", "ah"]:
                return 0.001 <= value <= 1000  # от 1мАч до 1000Ач
            return True

        elif param.parameter_type == ParameterType.QUANTITY:
            # Количество: от 1 до 10000 (более реалистичный диапазон для упаковок)
            return 1 <= value <= 10000

        # Для остальных типов не проверяем
        return True

    def get_patterns_by_type(self, parameter_type: ParameterType) -> List[ParameterPattern]:
        """Получение паттернов по типу параметра"""
        return [p for p in self.patterns if p.parameter_type == parameter_type]
    
    def extract_parameters_batch(self, texts: List[str]) -> List[List[ExtractedParameter]]:
        """Пакетное извлечение параметров"""
        results = []
        for text in texts:
            try:
                params = self.extract_parameters(text)
                results.append(params)
            except Exception as e:
                logger.error(f"Error extracting parameters from text: {e}")
                results.append([])
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики экстрактора"""
        type_counts = {}
        for pattern in self.patterns:
            type_name = pattern.parameter_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            'total_patterns': len(self.patterns),
            'patterns_by_type': type_counts,
            'supported_types': [t.value for t in ParameterType]
        }
    
    def save_patterns(self, filepath: str):
        """Сохранение паттернов в файл"""
        patterns_data = []
        for pattern in self.patterns:
            patterns_data.append({
                'name': pattern.name,
                'pattern': pattern.pattern,
                'parameter_type': pattern.parameter_type.value,
                'unit': pattern.unit,
                'description': pattern.description,
                'priority': pattern.priority
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Patterns saved to {filepath}")
    
    def load_patterns(self, filepath: str):
        """Загрузка паттернов из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        
        loaded_patterns = []
        for data in patterns_data:
            pattern = ParameterPattern(
                name=data['name'],
                pattern=data['pattern'],
                parameter_type=ParameterType(data['parameter_type']),
                unit=data.get('unit'),
                description=data.get('description', ''),
                priority=data.get('priority', 1)
            )
            loaded_patterns.append(pattern)
        
        self.patterns.extend(loaded_patterns)
        logger.info(f"Loaded {len(loaded_patterns)} patterns from {filepath}")
