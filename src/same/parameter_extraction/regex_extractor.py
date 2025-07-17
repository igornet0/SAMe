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
        
        # Размеры и габариты
        dimension_patterns = [
            ParameterPattern(
                name="diameter",
                pattern=r"(?:диаметр|диам\.?|ø|d)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Диаметр изделия"
            ),
            ParameterPattern(
                name="length",
                pattern=r"(?:длина|длин\.?|l)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Длина изделия"
            ),
            ParameterPattern(
                name="width",
                pattern=r"(?:ширина|шир\.?|w)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Ширина изделия"
            ),
            ParameterPattern(
                name="height",
                pattern=r"(?:высота|выс\.?|h)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Высота изделия"
            ),
            ParameterPattern(
                name="thickness",
                pattern=r"(?:толщина|толщ\.?|t)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(мм|см|м)?",
                parameter_type=ParameterType.DIMENSION,
                unit="мм",
                description="Толщина изделия"
            ),
        ]
        
        # Вес
        weight_patterns = [
            ParameterPattern(
                name="weight",
                pattern=r"(?:вес|масса|m)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(г|кг|т)?",
                parameter_type=ParameterType.WEIGHT,
                unit="кг",
                description="Вес изделия"
            ),
        ]
        
        # Электрические характеристики
        electrical_patterns = [
            ParameterPattern(
                name="voltage",
                pattern=r"(?:напряжение|напр\.?|u|v)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(в|v|вольт)?",
                parameter_type=ParameterType.ELECTRICAL,
                unit="В",
                description="Напряжение"
            ),
            ParameterPattern(
                name="current",
                pattern=r"(?:ток|i|a)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(а|a|ампер)?",
                parameter_type=ParameterType.ELECTRICAL,
                unit="А",
                description="Сила тока"
            ),
            ParameterPattern(
                name="power",
                pattern=r"(?:мощность|мощн\.?|p|w)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(вт|w|квт|kw|мвт|mw)?",
                parameter_type=ParameterType.ELECTRICAL,
                unit="Вт",
                description="Мощность"
            ),
        ]
        
        # Давление
        pressure_patterns = [
            ParameterPattern(
                name="pressure",
                pattern=r"(?:давление|давл\.?|p)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(па|pa|кпа|kpa|мпа|mpa|бар|bar|атм|atm)?",
                parameter_type=ParameterType.PRESSURE,
                unit="МПа",
                description="Рабочее давление"
            ),
        ]
        
        # Температура
        temperature_patterns = [
            ParameterPattern(
                name="temperature",
                pattern=r"(?:температура|темп\.?|t)\s*[=:]?\s*([+-]?\d+(?:[.,]\d+)?)\s*(°?c|°?f|к|k)?",
                parameter_type=ParameterType.TEMPERATURE,
                unit="°C",
                description="Рабочая температура"
            ),
        ]
        
        # Частота
        frequency_patterns = [
            ParameterPattern(
                name="frequency",
                pattern=r"(?:частота|част\.?|f)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(гц|hz|кгц|khz|мгц|mhz|ггц|ghz)?",
                parameter_type=ParameterType.FREQUENCY,
                unit="Гц",
                description="Частота"
            ),
        ]
        
        # Материал
        material_patterns = [
            ParameterPattern(
                name="material",
                pattern=r"(?:материал|мат\.?)\s*[=:]?\s*([а-яё\s]+?)(?:\s|$|,|;)",
                parameter_type=ParameterType.MATERIAL,
                description="Материал изготовления"
            ),
            ParameterPattern(
                name="steel_grade",
                pattern=r"(?:сталь|ст\.?)\s*([а-я0-9\-]+)",
                parameter_type=ParameterType.MATERIAL,
                description="Марка стали"
            ),
        ]
        
        # Артикулы и коды
        article_patterns = [
            ParameterPattern(
                name="article",
                pattern=r"(?:артикул|арт\.?|код|code)\s*[=:]?\s*([a-z0-9\-_]+)",
                parameter_type=ParameterType.ARTICLE,
                description="Артикул изделия"
            ),
            ParameterPattern(
                name="gost",
                pattern=r"(гост\s+\d+(?:\.\d+)*(?:\-\d+)?)",
                parameter_type=ParameterType.ARTICLE,
                description="ГОСТ стандарт"
            ),
        ]
        
        # Объединяем все паттерны
        all_patterns = (
            dimension_patterns + weight_patterns + electrical_patterns +
            pressure_patterns + temperature_patterns + frequency_patterns +
            material_patterns + article_patterns
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
                except Exception as e:
                    logger.warning(f"Error extracting parameter {pattern.name}: {e}")
        
        # Удаляем дубликаты и сортируем по приоритету
        extracted_params = self._deduplicate_parameters(extracted_params)
        extracted_params.sort(key=lambda x: (x.parameter_type.value, -x.confidence))
        
        return extracted_params
    
    def _create_parameter(self, pattern: ParameterPattern, match: re.Match, text: str) -> Optional[ExtractedParameter]:
        """Создание объекта параметра из совпадения"""
        groups = match.groups()
        
        if not groups:
            return None
        
        # Основное значение (первая группа)
        raw_value = groups[0].replace(',', '.')
        
        # Единица измерения (вторая группа, если есть)
        unit = groups[1] if len(groups) > 1 and groups[1] else pattern.unit
        
        # Преобразование значения
        if pattern.parameter_type in [ParameterType.DIMENSION, ParameterType.WEIGHT, 
                                    ParameterType.ELECTRICAL, ParameterType.PRESSURE,
                                    ParameterType.TEMPERATURE, ParameterType.FREQUENCY]:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
        else:
            value = raw_value.strip()
        
        # Расчет уверенности
        confidence = self._calculate_confidence(pattern, match, text)
        
        return ExtractedParameter(
            name=pattern.name,
            value=value,
            unit=unit,
            parameter_type=pattern.parameter_type,
            confidence=confidence,
            source_text=match.group(0),
            position=(match.start(), match.end())
        )
    
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
    
    def _deduplicate_parameters(self, parameters: List[ExtractedParameter]) -> List[ExtractedParameter]:
        """Удаление дублирующихся параметров"""
        seen = {}
        result = []
        
        for param in parameters:
            key = (param.name, param.parameter_type)
            
            if key not in seen or param.confidence > seen[key].confidence:
                seen[key] = param
        
        return list(seen.values())
    
    def add_pattern(self, pattern: ParameterPattern):
        """Добавление нового паттерна"""
        self.patterns.append(pattern)
        logger.info(f"Added pattern: {pattern.name}")
    
    def remove_pattern(self, pattern_name: str):
        """Удаление паттерна по имени"""
        self.patterns = [p for p in self.patterns if p.name != pattern_name]
        logger.info(f"Removed pattern: {pattern_name}")
    
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
