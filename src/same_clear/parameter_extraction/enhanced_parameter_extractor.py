"""
Расширенный модуль извлечения параметров с улучшенными возможностями
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Расширенные типы параметров"""
    # Основные физические параметры
    DIMENSION = "dimension"
    WEIGHT = "weight"
    VOLUME = "volume"
    AREA = "area"
    LENGTH = "length"
    WIDTH = "width"
    HEIGHT = "height"
    DIAMETER = "diameter"
    THICKNESS = "thickness"
    
    # Электрические параметры
    ELECTRICAL = "electrical"
    VOLTAGE = "voltage"
    CURRENT = "current"
    POWER = "power"
    FREQUENCY = "frequency"
    RESISTANCE = "resistance"
    CAPACITANCE = "capacitance"
    INDUCTANCE = "inductance"
    
    # Механические параметры
    PRESSURE = "pressure"
    FORCE = "force"
    TORQUE = "torque"
    SPEED = "speed"
    ROTATION = "rotation"
    TORQUE_RATING = "torque_rating"
    
    # Температурные параметры
    TEMPERATURE = "temperature"
    TEMPERATURE_RANGE = "temperature_range"
    
    # Химические параметры
    PH = "ph"
    CONCENTRATION = "concentration"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    
    # Технические характеристики
    THREAD = "thread"
    PROTECTION = "protection"
    CAPACITY = "capacity"
    EFFICIENCY = "efficiency"
    LIFETIME = "lifetime"
    WARRANTY = "warranty"
    
    # Материалы и покрытия
    MATERIAL = "material"
    COATING = "coating"
    COLOR = "color"
    FINISH = "finish"
    
    # Идентификация
    BRAND = "brand"
    MODEL = "model"
    ARTICLE = "article"
    SERIAL = "serial"
    PART_NUMBER = "part_number"
    
    # Количественные параметры
    QUANTITY = "quantity"
    PACK_SIZE = "pack_size"
    BATCH_SIZE = "batch_size"
    
    # Специфические параметры
    GOST = "gost"
    STANDARD = "standard"
    CERTIFICATION = "certification"
    COMPLIANCE = "compliance"
    
    # Функциональные параметры
    FUNCTION = "function"
    APPLICATION = "application"
    ENVIRONMENT = "environment"
    OPERATING_CONDITIONS = "operating_conditions"
    
    # Общие параметры
    CHARACTERISTIC = "characteristic"

@dataclass
class EnhancedParameterPattern:
    """Расширенный паттерн для извлечения параметра"""
    name: str
    patterns: List[str]  # Множественные паттерны
    parameter_type: ParameterType
    unit: Optional[str] = None
    description: str = ""
    priority: int = 1
    context_keywords: List[str] = field(default_factory=list)
    exclusion_keywords: List[str] = field(default_factory=list)
    value_transformations: Dict[str, Any] = field(default_factory=dict)
    compiled_patterns: List[re.Pattern] = field(init=False)
    
    def __post_init__(self):
        self.compiled_patterns = []
        for pattern in self.patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.UNICODE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

@dataclass
class ExtractedParameter:
    """Извлеченный параметр с расширенной информацией"""
    name: str
    value: Union[str, float, int]
    unit: Optional[str] = None
    parameter_type: ParameterType = ParameterType.CHARACTERISTIC
    confidence: float = 1.0
    context: str = ""
    position: Tuple[int, int] = (0, 0)
    raw_match: str = ""
    transformations_applied: List[str] = field(default_factory=list)

class EnhancedParameterExtractor:
    """Расширенный извлекатель параметров"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.unit_conversions = self._initialize_unit_conversions()
        self.material_keywords = self._initialize_material_keywords()
        self.brand_keywords = self._initialize_brand_keywords()
        self.function_keywords = self._initialize_function_keywords()
        
        logger.info("EnhancedParameterExtractor initialized")
    
    def _initialize_patterns(self) -> List[EnhancedParameterPattern]:
        """Инициализация расширенных паттернов"""
        patterns = []
        
        # Размеры и габариты
        patterns.extend([
            EnhancedParameterPattern(
                name="dimensions",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)\s*[хx×]?\s*(\d+(?:[.,]\d+)?)?\s*(?:мм|mm|см|cm|м|m)',
                    r'(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|см|cm|м|m)',
                    r'размер[ы]?\s*(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)',
                    r'габарит[ы]?\s*(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)'
                ],
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                priority=3,
                context_keywords=["размер", "габарит", "длина", "ширина", "высота"]
            ),
            EnhancedParameterPattern(
                name="diameter",
                patterns=[
                    r'[DdØ]\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm)',
                    r'диаметр\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm)',
                    r'[Ø]\s*(\d+(?:[.,]\d+)?)',
                    r'ф\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm)'
                ],
                parameter_type=ParameterType.DIAMETER,
                unit="мм",
                priority=3
            ),
            EnhancedParameterPattern(
                name="length",
                patterns=[
                    r'длина\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|см|cm|м|m)',
                    r'L\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm)',
                    r'длин[аы]\s*(\d+(?:[.,]\d+)?)'
                ],
                parameter_type=ParameterType.LENGTH,
                unit="мм",
                priority=2
            ),
            EnhancedParameterPattern(
                name="width",
                patterns=[
                    r'ширина\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|см|cm)',
                    r'шир[инаы]\s*(\d+(?:[.,]\d+)?)'
                ],
                parameter_type=ParameterType.WIDTH,
                unit="мм",
                priority=2
            ),
            EnhancedParameterPattern(
                name="height",
                patterns=[
                    r'высота\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|см|cm)',
                    r'выс[отаы]\s*(\d+(?:[.,]\d+)?)'
                ],
                parameter_type=ParameterType.HEIGHT,
                unit="мм",
                priority=2
            )
        ])
        
        # Вес и масса
        patterns.extend([
            EnhancedParameterPattern(
                name="weight",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:кг|kg|г|g|т|t)',
                    r'вес\s*(\d+(?:[.,]\d+)?)\s*(?:кг|kg|г|g)',
                    r'масса\s*(\d+(?:[.,]\d+)?)\s*(?:кг|kg|г|g)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:кг|kg|г|g)\s*вес'
                ],
                parameter_type=ParameterType.WEIGHT,
                unit="кг",
                priority=3,
                context_keywords=["вес", "масса", "тяжелый", "легкий"]
            )
        ])
        
        # Объем и емкость
        patterns.extend([
            EnhancedParameterPattern(
                name="volume",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:л|l|мл|ml|дм3|dm3|см3|cm3)',
                    r'объем\s*(\d+(?:[.,]\d+)?)\s*(?:л|l|мл|ml)',
                    r'емкость\s*(\d+(?:[.,]\d+)?)\s*(?:л|l|мл|ml)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:л|l|мл|ml)\s*объем'
                ],
                parameter_type=ParameterType.VOLUME,
                unit="л",
                priority=3,
                context_keywords=["объем", "емкость", "вместимость"]
            ),
            EnhancedParameterPattern(
                name="capacity",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:Ач|Ah|мАч|mAh|Втч|Wh)',
                    r'емкость\s*(\d+(?:[.,]\d+)?)\s*(?:Ач|Ah|мАч|mAh)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:Ач|Ah|мАч|mAh)\s*емкость'
                ],
                parameter_type=ParameterType.CAPACITY,
                unit="Ач",
                priority=3,
                context_keywords=["емкость", "аккумулятор", "батарея"]
            )
        ])
        
        # Электрические параметры
        patterns.extend([
            EnhancedParameterPattern(
                name="voltage",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:В|V|кВ|kV)',
                    r'напряжение\s*(\d+(?:[.,]\d+)?)\s*(?:В|V)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:В|V)\s*напряжение'
                ],
                parameter_type=ParameterType.VOLTAGE,
                unit="В",
                priority=3,
                context_keywords=["напряжение", "вольт", "электрический"]
            ),
            EnhancedParameterPattern(
                name="current",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:А|A|мА|mA|кА|kA)',
                    r'ток\s*(\d+(?:[.,]\d+)?)\s*(?:А|A)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:А|A)\s*ток'
                ],
                parameter_type=ParameterType.CURRENT,
                unit="А",
                priority=3,
                context_keywords=["ток", "ампер", "электрический"]
            ),
            EnhancedParameterPattern(
                name="power",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:Вт|W|кВт|kW|МВт|MW)',
                    r'мощность\s*(\d+(?:[.,]\d+)?)\s*(?:Вт|W)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:Вт|W|кВт|kW)\s*мощность'
                ],
                parameter_type=ParameterType.POWER,
                unit="Вт",
                priority=3,
                context_keywords=["мощность", "ватт", "энергия"]
            ),
            EnhancedParameterPattern(
                name="frequency",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:Гц|Hz|кГц|kHz|МГц|MHz)',
                    r'частота\s*(\d+(?:[.,]\d+)?)\s*(?:Гц|Hz)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:Гц|Hz)\s*частота'
                ],
                parameter_type=ParameterType.FREQUENCY,
                unit="Гц",
                priority=2,
                context_keywords=["частота", "герц", "радио"]
            )
        ])
        
        # Давление и сила
        patterns.extend([
            EnhancedParameterPattern(
                name="pressure",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:бар|bar|Па|Pa|кПа|kPa|МПа|MPa|атм|atm)',
                    r'давление\s*(\d+(?:[.,]\d+)?)\s*(?:бар|bar)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:бар|bar)\s*давление'
                ],
                parameter_type=ParameterType.PRESSURE,
                unit="бар",
                priority=3,
                context_keywords=["давление", "бар", "паскаль"]
            ),
            EnhancedParameterPattern(
                name="force",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:Н|N|кН|kN|МН|MN)',
                    r'сила\s*(\d+(?:[.,]\d+)?)\s*(?:Н|N)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:Н|N)\s*сила'
                ],
                parameter_type=ParameterType.FORCE,
                unit="Н",
                priority=2,
                context_keywords=["сила", "ньютон", "тяга"]
            )
        ])
        
        # Температура
        patterns.extend([
            EnhancedParameterPattern(
                name="temperature",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:°C|°F|К|K|градус)',
                    r'температура\s*(\d+(?:[.,]\d+)?)\s*(?:°C|°F)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:°C|°F)\s*температура',
                    r'до\s*[-\-]?(\d+(?:[.,]\d+)?)\s*(?:°C|°F)',
                    r'от\s*(\d+(?:[.,]\d+)?)\s*(?:°C|°F)'
                ],
                parameter_type=ParameterType.TEMPERATURE,
                unit="°C",
                priority=2,
                context_keywords=["температура", "градус", "мороз", "жара"]
            )
        ])
        
        # Резьба и крепеж
        patterns.extend([
            EnhancedParameterPattern(
                name="thread",
                patterns=[
                    r'[Мм]\s*(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)',
                    r'резьба\s*[Мм]\s*(\d+(?:[.,]\d+)?)',
                    r'[Мм]\s*(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)',
                    r'болт\s*[Мм]\s*(\d+(?:[.,]\d+)?)',
                    r'гайка\s*[Мм]\s*(\d+(?:[.,]\d+)?)'
                ],
                parameter_type=ParameterType.THREAD,
                unit="мм",
                priority=3,
                context_keywords=["резьба", "болт", "гайка", "винт", "крепеж"]
            )
        ])
        
        # Степень защиты
        patterns.extend([
            EnhancedParameterPattern(
                name="protection",
                patterns=[
                    r'IP\s*(\d{2})',
                    r'степень\s*защиты\s*IP\s*(\d{2})',
                    r'защита\s*IP\s*(\d{2})'
                ],
                parameter_type=ParameterType.PROTECTION,
                unit="IP",
                priority=3,
                context_keywords=["защита", "пыль", "влага", "водонепроницаемый"]
            )
        ])
        
        # ГОСТ и стандарты
        patterns.extend([
            EnhancedParameterPattern(
                name="gost",
                patterns=[
                    r'ГОСТ\s*(\d+(?:[-\d]*)?)',
                    r'ГОСТ\s*(\d+(?:[-\d]*)?)\s*(\d{4})?',
                    r'стандарт\s*ГОСТ\s*(\d+(?:[-\d]*)?)'
                ],
                parameter_type=ParameterType.GOST,
                unit="ГОСТ",
                priority=4,
                context_keywords=["ГОСТ", "стандарт", "норма", "требование"]
            ),
            EnhancedParameterPattern(
                name="standard",
                patterns=[
                    r'ISO\s*(\d+(?:[-\d]*)?)',
                    r'EN\s*(\d+(?:[-\d]*)?)',
                    r'DIN\s*(\d+(?:[-\d]*)?)',
                    r'ANSI\s*(\d+(?:[-\d]*)?)'
                ],
                parameter_type=ParameterType.STANDARD,
                unit="стандарт",
                priority=3,
                context_keywords=["стандарт", "норма", "требование"]
            )
        ])
        
        # Количество и упаковка
        patterns.extend([
            EnhancedParameterPattern(
                name="quantity",
                patterns=[
                    r'(\d+(?:[.,]\d+)?)\s*(?:шт|pcs|штук|piece)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:уп|pack|упаковка)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:компл|set|набор)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:пара|pair)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:м|meter|метр)'
                ],
                parameter_type=ParameterType.QUANTITY,
                unit="шт",
                priority=2,
                context_keywords=["количество", "штук", "упаковка", "набор"]
            )
        ])
        
        # Цвет и отделка
        patterns.extend([
            EnhancedParameterPattern(
                name="color",
                patterns=[
                    r'(?:цвет|color)\s*([а-яё\w\s]+?)(?:\s|,|$)',
                    r'([а-яё\w\s]+?)\s*(?:цвет|color)',
                    r'(?:окраска|окрашен)\s*([а-яё\w\s]+?)(?:\s|,|$)'
                ],
                parameter_type=ParameterType.COLOR,
                unit="цвет",
                priority=1,
                context_keywords=["цвет", "окраска", "краска", "отделка"]
            )
        ])
        
        return patterns
    
    def _initialize_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Инициализация конвертации единиц измерения"""
        return {
            "weight": {
                "г": 0.001, "g": 0.001,
                "кг": 1.0, "kg": 1.0,
                "т": 1000.0, "t": 1000.0
            },
            "volume": {
                "мл": 0.001, "ml": 0.001,
                "л": 1.0, "l": 1.0,
                "дм3": 1.0, "dm3": 1.0,
                "см3": 0.001, "cm3": 0.001
            },
            "length": {
                "мм": 1.0, "mm": 1.0,
                "см": 10.0, "cm": 10.0,
                "м": 1000.0, "m": 1000.0
            },
            "power": {
                "Вт": 1.0, "W": 1.0,
                "кВт": 1000.0, "kW": 1000.0,
                "МВт": 1000000.0, "MW": 1000000.0
            },
            "voltage": {
                "В": 1.0, "V": 1.0,
                "кВ": 1000.0, "kV": 1000.0
            },
            "current": {
                "А": 1.0, "A": 1.0,
                "мА": 0.001, "mA": 0.001,
                "кА": 1000.0, "kA": 1000.0
            }
        }
    
    def _initialize_material_keywords(self) -> Dict[str, List[str]]:
        """Инициализация ключевых слов материалов"""
        return {
            "металл": ["сталь", "железо", "алюминий", "медь", "латунь", "бронза", "титан", "никель"],
            "пластик": ["пластик", "полиэтилен", "полипропилен", "ПВХ", "полистирол", "акрил"],
            "дерево": ["дерево", "древесина", "фанера", "ДСП", "МДФ", "массив"],
            "стекло": ["стекло", "оргстекло", "плексиглас"],
            "резина": ["резина", "каучук", "силикон", "эластомер"],
            "керамика": ["керамика", "фарфор", "фаянс"],
            "композит": ["композит", "карбон", "кевлар", "стекловолокно"]
        }
    
    def _initialize_brand_keywords(self) -> List[str]:
        """Инициализация ключевых слов брендов"""
        return [
            'neox', 'osairous', 'yealink', 'sanfor', 'санфор', 'биолан', 'нэфис',
            'персил', 'dallas', 'премиум', 'маяк', 'chint', 'andeli', 'grass',
            'kraft', 'reoflex', 'керхер', 'huawei', 'honor', 'ВЫСОТА', 'ugreen',
            'alisafox', 'маякавто', 'техноавиа', 'восток-сервис', 'attache', 'камаз',
            'зубр', 'hp', 'ekf', 'dexp', 'matrix', 'siemens', 'комус', 'gigant',
            'hyundai', 'iveco', 'stayer', 'brauberg', 'makita', 'bentec', 'сибртех',
            'bosch', 'rexant', 'sampa', 'kyocera', 'avrora', 'derrick', 'cummins',
            'economy', 'samsung', 'ofite', 'professional', 'caterpillar', 'intel',
            'proxima', 'core', 'shantui', 'king', 'office', 'петролеум', 'трейл',
            'skf', 'форвелд', 'скаймастер', 'tony', 'kentek', 'ресанта', 'dexter',
            'electric', 'оттм'
        ]
    
    def _initialize_function_keywords(self) -> Dict[str, List[str]]:
        """Инициализация ключевых слов функций"""
        return {
            "сварка": ["сварка", "сварочный", "электрод", "припой"],
            "резка": ["резка", "резать", "нож", "лезвие", "диск"],
            "сверление": ["сверление", "сверло", "дрель", "бур"],
            "шлифовка": ["шлифовка", "шлифовать", "наждак", "абразив"],
            "покраска": ["покраска", "краска", "покрытие", "лак"],
            "очистка": ["очистка", "чистка", "моющий", "детергент"],
            "смазка": ["смазка", "смазывать", "масло", "смазочный"],
            "изоляция": ["изоляция", "изолятор", "диэлектрик", "непроводящий"],
            "защита": ["защита", "защитный", "броня", "щит"],
            "крепление": ["крепление", "крепеж", "болт", "гайка", "винт"]
        }
    
    def extract_parameters(self, text: str) -> List[ExtractedParameter]:
        """Извлечение параметров из текста с расширенными возможностями"""
        if not text:
            return []
        
        text_lower = text.lower()
        extracted_params = []
        used_positions = set()
        
        # Извлечение по паттернам
        for pattern in self.patterns:
            for compiled_pattern in pattern.compiled_patterns:
                matches = compiled_pattern.finditer(text)
                
                for match in matches:
                    # Проверка на пересечение с уже извлеченными параметрами
                    match_pos = (match.start(), match.end())
                    if any(self._positions_overlap(match_pos, used_pos) for used_pos in used_positions):
                        continue
                    
                    # Проверка контекста
                    if not self._check_context(text_lower, match, pattern):
                        continue
                    
                    # Извлечение значений
                    values = self._extract_values_from_match(match, pattern)
                    if not values:
                        continue
                    
                    # Создание параметра
                    param = self._create_parameter(
                        pattern, values, match, text, match_pos
                    )
                    
                    if param:
                        extracted_params.append(param)
                        used_positions.add(match_pos)
        
        # Извлечение материалов
        material_params = self._extract_materials(text)
        extracted_params.extend(material_params)
        
        # Извлечение брендов
        brand_params = self._extract_brands(text)
        extracted_params.extend(brand_params)
        
        # Извлечение функций
        function_params = self._extract_functions(text)
        extracted_params.extend(function_params)
        
        # Сортировка по приоритету и позиции
        extracted_params.sort(key=lambda p: (p.parameter_type.value, -p.confidence, p.position[0]))
        
        return extracted_params
    
    def _positions_overlap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Проверка пересечения позиций"""
        return not (pos1[1] <= pos2[0] or pos2[1] <= pos1[0])
    
    def _check_context(self, text: str, match: re.Match, pattern: EnhancedParameterPattern) -> bool:
        """Проверка контекста для паттерна"""
        # Проверка исключающих ключевых слов
        if pattern.exclusion_keywords:
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]
            
            for exclusion in pattern.exclusion_keywords:
                if exclusion.lower() in context:
                    return False
        
        # Проверка включающих ключевых слов
        if pattern.context_keywords:
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end]
            
            has_context = any(keyword.lower() in context for keyword in pattern.context_keywords)
            if not has_context:
                return False
        
        return True
    
    def _extract_values_from_match(self, match: re.Match, pattern: EnhancedParameterPattern) -> List[Any]:
        """Извлечение значений из совпадения"""
        values = []
        
        # Обработка групп в регулярном выражении
        if match.groups():
            for group in match.groups():
                if group:
                    try:
                        # Попытка преобразования в число
                        if '.' in group or ',' in group:
                            value = float(group.replace(',', '.'))
                        else:
                            value = int(group)
                        values.append(value)
                    except ValueError:
                        values.append(group.strip())
        
        return values
    
    def _create_parameter(self, pattern: EnhancedParameterPattern, values: List[Any], 
                         match: re.Match, text: str, position: Tuple[int, int]) -> Optional[ExtractedParameter]:
        """Создание параметра из извлеченных данных"""
        if not values:
            return None
        
        # Определение основного значения
        main_value = values[0] if len(values) == 1 else values
        
        # Применение трансформаций
        transformed_value = self._apply_transformations(main_value, pattern)
        transformations = []
        
        if transformed_value != main_value:
            transformations.append("value_transformation")
        
        # Определение контекста
        context_start = max(0, match.start() - 30)
        context_end = min(len(text), match.end() + 30)
        context = text[context_start:context_end].strip()
        
        # Расчет уверенности
        confidence = self._calculate_confidence(pattern, match, text, values)
        
        return ExtractedParameter(
            name=pattern.name,
            value=transformed_value,
            unit=pattern.unit,
            parameter_type=pattern.parameter_type,
            confidence=confidence,
            context=context,
            position=position,
            raw_match=match.group(0),
            transformations_applied=transformations
        )
    
    def _apply_transformations(self, value: Any, pattern: EnhancedParameterPattern) -> Any:
        """Применение трансформаций к значению"""
        if not pattern.value_transformations:
            return value
        
        # Конвертация единиц измерения
        if "unit_conversion" in pattern.value_transformations:
            # Логика конвертации единиц
            pass
        
        return value
    
    def _calculate_confidence(self, pattern: EnhancedParameterPattern, match: re.Match, 
                            text: str, values: List[Any]) -> float:
        """Расчет уверенности в извлеченном параметре"""
        confidence = 0.5  # Базовая уверенность
        
        # Увеличение уверенности за приоритет паттерна
        confidence += pattern.priority * 0.1
        
        # Увеличение уверенности за контекст
        if pattern.context_keywords:
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end].lower()
            
            context_matches = sum(1 for keyword in pattern.context_keywords 
                                if keyword.lower() in context)
            confidence += context_matches * 0.1
        
        # Увеличение уверенности за качество совпадения
        if match.groups():
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_materials(self, text: str) -> List[ExtractedParameter]:
        """Извлечение материалов"""
        materials = []
        text_lower = text.lower()
        
        for material_type, keywords in self.material_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Поиск позиции ключевого слова
                    start = text_lower.find(keyword)
                    if start != -1:
                        end = start + len(keyword)
                        materials.append(ExtractedParameter(
                            name="material",
                            value=material_type,
                            parameter_type=ParameterType.MATERIAL,
                            confidence=0.8,
                            context=text[max(0, start-20):min(len(text), end+20)],
                            position=(start, end),
                            raw_match=keyword
                        ))
                        break  # Один материал на тип
        
        return materials
    
    def _extract_brands(self, text: str) -> List[ExtractedParameter]:
        """Извлечение брендов"""
        brands = []
        text_lower = text.lower()
        
        for brand in self.brand_keywords:
            if brand.lower() in text_lower:
                start = text_lower.find(brand.lower())
                if start != -1:
                    end = start + len(brand)
                    brands.append(ExtractedParameter(
                        name="brand",
                        value=brand,
                        parameter_type=ParameterType.BRAND,
                        confidence=0.9,
                        context=text[max(0, start-20):min(len(text), end+20)],
                        position=(start, end),
                        raw_match=brand
                    ))
        
        return brands
    
    def _extract_functions(self, text: str) -> List[ExtractedParameter]:
        """Извлечение функций"""
        functions = []
        text_lower = text.lower()
        
        for function_type, keywords in self.function_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start = text_lower.find(keyword)
                    if start != -1:
                        end = start + len(keyword)
                        functions.append(ExtractedParameter(
                            name="function",
                            value=function_type,
                            parameter_type=ParameterType.FUNCTION,
                            confidence=0.7,
                            context=text[max(0, start-20):min(len(text), end+20)],
                            position=(start, end),
                            raw_match=keyword
                        ))
                        break  # Одна функция на тип
        
        return functions
    
    def get_parameter_summary(self, parameters: List[ExtractedParameter]) -> Dict[str, Any]:
        """Получение сводки по извлеченным параметрам"""
        summary = {
            "total_parameters": len(parameters),
            "parameter_types": defaultdict(int),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "units_found": set(),
            "materials_found": set(),
            "brands_found": set(),
            "functions_found": set()
        }
        
        for param in parameters:
            summary["parameter_types"][param.parameter_type.value] += 1
            
            if param.confidence >= 0.8:
                summary["high_confidence"] += 1
            elif param.confidence >= 0.6:
                summary["medium_confidence"] += 1
            else:
                summary["low_confidence"] += 1
            
            if param.unit:
                summary["units_found"].add(param.unit)
            
            if param.parameter_type == ParameterType.MATERIAL:
                summary["materials_found"].add(param.value)
            elif param.parameter_type == ParameterType.BRAND:
                summary["brands_found"].add(param.value)
            elif param.parameter_type == ParameterType.FUNCTION:
                summary["functions_found"].add(param.value)
        
        # Преобразование set в list для JSON сериализации
        summary["units_found"] = list(summary["units_found"])
        summary["materials_found"] = list(summary["materials_found"])
        summary["brands_found"] = list(summary["brands_found"])
        summary["functions_found"] = list(summary["functions_found"])
        
        return summary
