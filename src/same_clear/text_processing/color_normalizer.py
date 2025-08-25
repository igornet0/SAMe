#!/usr/bin/env python3
"""
Модуль для нормализации цветовых терминов в текстах товаров
Заменяет все цветовые характеристики на унифицированный токен <COLOR>
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ColorNormalizerConfig:
    """Конфигурация нормализатора цветов"""
    color_token: str = "<COLOR>"
    preserve_color_info: bool = False  # Если True, сохраняет информацию о цвете в отдельном поле
    case_sensitive: bool = False
    normalize_compound_colors: bool = True  # ярко-синий -> <COLOR>
    normalize_color_variations: bool = True  # синеватый, синенький -> <COLOR>
    min_color_word_length: int = 3


class ColorNormalizer:
    """Класс для нормализации цветовых терминов"""
    
    def __init__(self, config: ColorNormalizerConfig = None):
        self.config = config or ColorNormalizerConfig()
        
        # Основные цвета с различными формами
        self.base_colors = {
            # Основная палитра - добавляем все формы
            'красный', 'красная', 'красное', 'красные', 'красного', 'красной', 'красным', 'красными',
            'синий', 'синяя', 'синее', 'синие', 'синего', 'синей', 'синим', 'синими', 'синь',
            'зеленый', 'зеленая', 'зеленое', 'зеленые', 'зеленого', 'зеленой', 'зеленым', 'зелеными',
            'желтый', 'желтая', 'желтое', 'желтые', 'желтого', 'желтой', 'желтым', 'желтыми',
            'черный', 'черная', 'черное', 'черные', 'черного', 'черной', 'черным', 'черными', 'чёрный', 'чёрная', 'чёрное', 'чёрные',
            'белый', 'белая', 'белое', 'белые', 'белого', 'белой', 'белым', 'белыми',
            'серый', 'серая', 'серое', 'серые', 'серого', 'серой', 'серым', 'серыми',

            # Дополнительные цвета
            'оранжевый', 'оранжевая', 'оранжевое', 'оранжевые',
            'фиолетовый', 'фиолетовая', 'фиолетовое', 'фиолетовые',
            'розовый', 'розовая', 'розовое', 'розовые',
            'коричневый', 'коричневая', 'коричневое', 'коричневые',
            'голубой', 'голубая', 'голубое', 'голубые',

            # Специальные цвета
            'бежевый', 'бежевая', 'бежевое', 'бежевые',
            'кремовый', 'кремовая', 'кремовое', 'кремовые',
            'золотой', 'золотая', 'золотое', 'золотые', 'золотистый', 'золотистая',
            'серебряный', 'серебряная', 'серебряное', 'серебряные', 'серебристый', 'серебристая',
            'бронзовый', 'бронзовая', 'бронзовое', 'бронзовые',

            # Яркие цвета
            'малиновый', 'малиновая', 'малиновое', 'малиновые',
            'вишневый', 'вишневая', 'вишневое', 'вишневые',
            'лимонный', 'лимонная', 'лимонное', 'лимонные',
            'салатовый', 'салатовая', 'салатовое', 'салатовые',
            'мятный', 'мятная', 'мятное', 'мятные',
            'бирюзовый', 'бирюзовая', 'бирюзовое', 'бирюзовые',
            'изумрудный', 'изумрудная', 'изумрудное', 'изумрудные',
            'сиреневый', 'сиреневая', 'сиреневое', 'сиреневые',
            'лиловый', 'лиловая', 'лиловое', 'лиловые',
            'бордовый', 'бордовая', 'бордовое', 'бордовые',

            # Специальные названия
            'марсала', 'хаки', 'терракотовый', 'терракотовая', 'песочный', 'песочная',
            'слоновая', 'слоновый', 'кость', 'костяной', 'костяная',

            # Технические цвета
            'прозрачный', 'прозрачная', 'прозрачное', 'прозрачные',
            'матовый', 'матовая', 'матовое', 'матовые',
            'глянцевый', 'глянцевая', 'глянцевое', 'глянцевые',
            'металлик', 'металлический', 'металлическая', 'металлическое', 'металлические',
            'перламутровый', 'перламутровая', 'перламутровое', 'перламутровые',
            'хром', 'хромированный', 'хромированная', 'хромированное', 'хромированные',
            'никель', 'никелированный', 'никелированная',
            'латунь', 'латунный', 'латунная', 'латунное', 'латунные',
            'медь', 'медный', 'медная', 'медное', 'медные',
            'алюминий', 'алюминиевый', 'алюминиевая', 'алюминиевое', 'алюминиевые',

            # Цветовые характеристики
            'цветной', 'цветная', 'цветное', 'цветные', 'цвет',
            'разноцветный', 'разноцветная', 'разноцветное', 'разноцветные',
            'многоцветный', 'многоцветная', 'многоцветное', 'многоцветные',
            'однотонный', 'однотонная', 'однотонное', 'однотонные',
            'двухцветный', 'двухцветная', 'двухцветное', 'двухцветные',
            'трехцветный', 'трехцветная', 'трехцветное', 'трехцветные'
        }
        
        # Модификаторы цвета
        self.color_modifiers = {
            # Наречия
            'ярко', 'темно', 'светло', 'бледно', 'насыщенно', 'глубоко',
            'нежно', 'мягко', 'интенсивно', 'приглушенно', 'тускло',
            'слабо', 'сильно', 'густо', 'ярче', 'темнее', 'светлее',

            # Прилагательные (мужской род)
            'яркий', 'темный', 'светлый', 'бледный', 'насыщенный', 'глубокий',
            'нежный', 'мягкий', 'интенсивный', 'приглушенный', 'тусклый',
            'слабый', 'сильный', 'густой',

            # Прилагательные (женский род)
            'яркая', 'темная', 'светлая', 'бледная', 'насыщенная', 'глубокая',
            'нежная', 'мягкая', 'интенсивная', 'приглушенная', 'тусклая',
            'слабая', 'сильная', 'густая',

            # Прилагательные (средний род)
            'яркое', 'темное', 'светлое', 'бледное', 'насыщенное', 'глубокое',
            'нежное', 'мягкое', 'интенсивное', 'приглушенное', 'тусклое',
            'слабое', 'сильное', 'густое',

            # Прилагательные (множественное число)
            'яркие', 'темные', 'светлые', 'бледные', 'насыщенные', 'глубокие',
            'нежные', 'мягкие', 'интенсивные', 'приглушенные', 'тусклые',
            'слабые', 'сильные', 'густые'
        }
        
        # Суффиксы для вариаций цветов
        self.color_suffixes = {
            'оватый', 'еватый', 'енький', 'онький', 'астый', 'истый',
            'оват', 'еват', 'енек', 'онек', 'аст', 'ист'
        }
        
        # Инициализируем паттерны
        self._color_patterns = []
        self._compound_patterns = []
        self._init_patterns()
        
        logger.info(f"ColorNormalizer initialized with {len(self.base_colors)} base colors")
    
    def _init_patterns(self):
        """Инициализация регулярных выражений для поиска цветов"""
        
        # Создаем паттерны для основных цветов с учетом склонений
        color_patterns = []
        
        for color in self.base_colors:
            if len(color) < self.config.min_color_word_length:
                continue
                
            # Основная форма
            color_patterns.append(color)
            
            # Добавляем вариации с суффиксами
            if self.config.normalize_color_variations:
                for suffix in self.color_suffixes:
                    color_patterns.append(f"{color[:-2]}{suffix}")  # убираем окончание и добавляем суффикс
        
        # Создаем паттерн для простых цветов
        colors_regex = '|'.join(sorted(color_patterns, key=len, reverse=True))
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        self._color_patterns.append(re.compile(rf'\b({colors_regex})\b', flags))
        
        # Создаем паттерны для составных цветов (модификатор + цвет)
        if self.config.normalize_compound_colors:
            modifiers_regex = '|'.join(self.color_modifiers)
            compound_pattern = rf'\b({modifiers_regex})[-\s]({colors_regex})\b'
            self._compound_patterns.append(re.compile(compound_pattern, flags))
            
            # Паттерн для цвет + модификатор (например, "синий яркий")
            reverse_compound_pattern = rf'\b({colors_regex})[-\s]({modifiers_regex})\b'
            self._compound_patterns.append(re.compile(reverse_compound_pattern, flags))
    
    def normalize_colors(self, text: str) -> Dict[str, any]:
        """
        Нормализация цветовых терминов в тексте
        
        Args:
            text: Исходный текст
            
        Returns:
            Dict с результатами нормализации
        """
        if not text or not isinstance(text, str):
            return {
                'original': text,
                'normalized': text,
                'colors_found': [],
                'colors_count': 0,
                'processing_successful': True
            }
        
        result = {
            'original': text,
            'normalized': text,
            'colors_found': [],
            'colors_count': 0,
            'processing_successful': True
        }
        
        try:
            normalized_text = text
            colors_found = []
            
            # Сначала обрабатываем составные цвета
            for pattern in self._compound_patterns:
                matches = pattern.findall(normalized_text)
                for match in matches:
                    if isinstance(match, tuple):
                        color_phrase = ' '.join(match)
                    else:
                        color_phrase = match
                    
                    colors_found.append(color_phrase.lower())
                    # Заменяем найденное на токен
                    normalized_text = pattern.sub(self.config.color_token, normalized_text)
            
            # Затем обрабатываем простые цвета
            for pattern in self._color_patterns:
                matches = pattern.findall(normalized_text)
                for match in matches:
                    colors_found.append(match.lower())
                    # Заменяем найденное на токен
                    normalized_text = pattern.sub(self.config.color_token, normalized_text)
            
            # Убираем дублирующиеся токены цвета
            normalized_text = re.sub(rf'{re.escape(self.config.color_token)}\s*{re.escape(self.config.color_token)}+', 
                                   self.config.color_token, normalized_text)
            
            result.update({
                'normalized': normalized_text.strip(),
                'colors_found': list(set(colors_found)),  # убираем дубликаты
                'colors_count': len(colors_found)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in color normalization: {e}")
            result.update({
                'processing_successful': False,
                'error': str(e)
            })
            return result
    
    def extract_colors(self, text: str) -> List[str]:
        """
        Извлечение всех цветовых терминов из текста без замены
        
        Args:
            text: Исходный текст
            
        Returns:
            Список найденных цветов
        """
        result = self.normalize_colors(text)
        return result.get('colors_found', [])
    
    def has_colors(self, text: str) -> bool:
        """
        Проверка наличия цветовых терминов в тексте
        
        Args:
            text: Исходный текст
            
        Returns:
            True если найдены цвета
        """
        result = self.normalize_colors(text)
        return result.get('colors_count', 0) > 0
    
    def get_color_statistics(self, texts: List[str]) -> Dict[str, int]:
        """
        Получение статистики по цветам в коллекции текстов
        
        Args:
            texts: Список текстов
            
        Returns:
            Словарь {цвет: количество_вхождений}
        """
        color_stats = {}
        
        for text in texts:
            colors = self.extract_colors(text)
            for color in colors:
                color_stats[color] = color_stats.get(color, 0) + 1
        
        return dict(sorted(color_stats.items(), key=lambda x: x[1], reverse=True))
    
    def add_custom_colors(self, colors: List[str]):
        """
        Добавление пользовательских цветов в словарь
        
        Args:
            colors: Список дополнительных цветов
        """
        self.base_colors.update(colors)
        self._init_patterns()  # Пересоздаем паттерны
        logger.info(f"Added {len(colors)} custom colors")
    
    def add_custom_modifiers(self, modifiers: List[str]):
        """
        Добавление пользовательских модификаторов цвета
        
        Args:
            modifiers: Список дополнительных модификаторов
        """
        self.color_modifiers.update(modifiers)
        self._init_patterns()  # Пересоздаем паттерны
        logger.info(f"Added {len(modifiers)} custom modifiers")


def create_color_normalizer(config: ColorNormalizerConfig = None) -> ColorNormalizer:
    """Фабричная функция для создания нормализатора цветов"""
    return ColorNormalizer(config)


# Предустановленные конфигурации
DEFAULT_CONFIG = ColorNormalizerConfig()

AGGRESSIVE_CONFIG = ColorNormalizerConfig(
    normalize_compound_colors=True,
    normalize_color_variations=True,
    min_color_word_length=2
)

CONSERVATIVE_CONFIG = ColorNormalizerConfig(
    normalize_compound_colors=False,
    normalize_color_variations=False,
    min_color_word_length=4
)
