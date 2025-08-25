"""
Модуль парсинга технических кодов, артикулов и стандартов
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TechCodesConfig:
    """Конфигурация обработки технических кодов"""
    parse_gost: bool = True           # Парсинг ГОСТ
    parse_tu: bool = True             # Парсинг ТУ
    parse_articles: bool = True       # Парсинг артикулов
    parse_drawings: bool = True       # Парсинг чертежей
    preserve_structure: bool = True   # Сохранять структуру кода
    normalize_separators: bool = True # Нормализовать разделители

    # Phase 4: Two-layer code generation
    enable_two_layer_generation: bool = True  # Двухслойная генерация кодов
    fix_gost_ost_confusion: bool = True       # Исправить путаницу ГОСТ/ОСТ
    prevent_code_truncation: bool = True      # Предотвратить обрезание кодов
    separate_technical_descriptive: bool = True  # Разделить технические и описательные

    # Standards layer (preserve exactly)
    preserve_standards_exactly: bool = True   # ГОСТ|ТУ|DIN|ISO точно как есть

    # Article codes layer (dimensional/power tokens)
    extract_dimensional_codes: bool = True    # Размерные коды (DN50, M10)
    extract_power_codes: bool = True          # Мощностные коды (5кВт, 1500об/мин)
    exclude_descriptive_words: bool = True    # Исключить описательные слова


class TechCodesProcessor:
    """Процессор технических кодов и артикулов"""
    
    def __init__(self, config: TechCodesConfig = None):
        self.config = config or TechCodesConfig()
        self._compile_patterns()
        self._init_code_types()
    
    def _compile_patterns(self):
        """Компиляция регулярных выражений"""
        self.patterns = {
            # ГОСТ: ГОСТ 123-456-78, ГОСТ Р 52857-2007
            'gost': re.compile(
                r'(ГОСТ)\s*(Р)?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*[-–—]?\s*(\d{2,4})?',
                re.IGNORECASE
            ),
            
            # ТУ: ТУ 14-3Р-82-2022, ТУ 2296-001-12345678-2019
            'tu': re.compile(
                r'(ТУ)\s+([0-9А-Яа-яA-Za-z\-\.]+)',
                re.IGNORECASE
            ),
            
            # ОСТ, СТО, СТП
            'other_standards': re.compile(
                r'(ОСТ|СТО|СТП|РД|ТИ|МИ)\s+([0-9А-Яа-яA-Za-z\-\.]+)',
                re.IGNORECASE
            ),
            
            # Артикулы: 4-730-059, SCM-6066-71, АБВ.123.456
            'articles': re.compile(
                r'\b([A-Za-zА-Яа-я]{0,5}\d+[-\.\s]*\d*[-\.\s]*\d*[A-Za-zА-Яа-я]*)\b'
            ),
            
            # Чертежи: Ч-123.456.789, DWG-001-002
            'drawings': re.compile(
                r'\b([ЧчDWGdwg]+[-\s]*\d+[-\.\s]*\d*[-\.\s]*\d*)\b'
            ),
            
            # Сложные артикулы с буквами: КП-65х35ф/65х35
            'complex_articles': re.compile(
                r'\b([А-Яа-яA-Za-z]{1,5}[-\s]*\d+[хx×]\d+[А-Яа-яA-Za-z]*(?:/\d+[хx×]\d+)?)\b'
            ),
            
            # Серийные номера: S/N 123456, SN:789012
            'serial_numbers': re.compile(
                r'(S/?N|SN|серийный\s*номер|сер\.?\s*№?)\s*[:=]?\s*([A-Za-z0-9\-]+)',
                re.IGNORECASE
            ),

            # Phase 4: Enhanced patterns for two-layer generation

            # Standards layer - точное распознавание стандартов
            'standards_exact': re.compile(
                r'\b(ГОСТ|ТУ|ОСТ|СТО|СТП|DIN|ISO|EN|ANSI|ASTM|IEC)\s+([0-9А-Яа-яA-Za-z\-\.\/\s]+?)(?=\s|$|[,;.])',
                re.IGNORECASE
            ),

            # Dimensional codes - размерные коды
            'dimensional_codes': re.compile(
                r'\b(DN|PN|M|G|R|Ø)\s*(\d+(?:[.,]\d+)?)',
                re.IGNORECASE
            ),

            # Power codes - мощностные коды
            'power_codes': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*(кВт|МВт|об/мин|л\.с\.|HP|rpm)',
                re.IGNORECASE
            ),

            # Technical parameters without descriptive words
            'technical_parameters': re.compile(
                r'\b(\d+(?:[.,]\d+)?)\s*(мм|см|м|кг|т|л|В|А|Вт|°C|°F|бар|атм|Па|кПа|МПа)\b',
                re.IGNORECASE
            )
        }
    
    def _init_code_types(self):
        """Инициализация типов кодов"""
        self.code_types = {
            'ГОСТ': {
                'description': 'Государственный стандарт',
                'format': 'ГОСТ [Р] XXXXX-YYYY-ZZZZ',
                'components': ['type', 'category', 'number', 'year']
            },
            'ТУ': {
                'description': 'Технические условия',
                'format': 'ТУ XXXX-XXX-XXXXXXXX-YYYY',
                'components': ['type', 'code']
            },
            'ОСТ': {
                'description': 'Отраслевой стандарт',
                'format': 'ОСТ XXXXX-XX',
                'components': ['type', 'code']
            },
            'СТО': {
                'description': 'Стандарт организации',
                'format': 'СТО XXXXX-YYYY',
                'components': ['type', 'code']
            }
        }
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Основной метод парсинга технических кодов
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с результатами парсинга
        """
        if not text or not isinstance(text, str):
            return {
                'original': text or '',
                'processed': '',
                'extracted_codes': []
            }
        
        result = {
            'original': text,
            'processed': text,
            'extracted_codes': []
        }
        
        processed_text = text

        # Phase 4: Two-layer code generation approach

        if self.config.enable_two_layer_generation:
            # Layer 1: Standards (preserve exactly, no lemmatization)
            if self.config.preserve_standards_exactly:
                processed_text, standards_codes = self._parse_standards_exact(processed_text)
                result['extracted_codes'].extend(standards_codes)

            # Layer 2: Article codes (dimensional/power tokens without descriptive words)
            if self.config.extract_dimensional_codes:
                processed_text, dimensional_codes = self._parse_dimensional_codes(processed_text)
                result['extracted_codes'].extend(dimensional_codes)

            if self.config.extract_power_codes:
                processed_text, power_codes = self._parse_power_codes(processed_text)
                result['extracted_codes'].extend(power_codes)

            # Technical parameters (clean, without descriptive words)
            processed_text, tech_params = self._parse_technical_parameters(processed_text)
            result['extracted_codes'].extend(tech_params)

        # Fallback to old methods if two-layer is disabled
        else:
            # Парсинг ГОСТ
            if self.config.parse_gost:
                processed_text, gost_codes = self._parse_gost(processed_text)
                result['extracted_codes'].extend(gost_codes)

            # Парсинг ТУ
            if self.config.parse_tu:
                processed_text, tu_codes = self._parse_tu(processed_text)
                result['extracted_codes'].extend(tu_codes)

            # Парсинг других стандартов
            processed_text, other_codes = self._parse_other_standards(processed_text)
            result['extracted_codes'].extend(other_codes)
        
        # Парсинг артикулов
        if self.config.parse_articles:
            processed_text, articles = self._parse_articles(processed_text)
            result['extracted_codes'].extend(articles)
        
        # Парсинг чертежей
        if self.config.parse_drawings:
            processed_text, drawings = self._parse_drawings(processed_text)
            result['extracted_codes'].extend(drawings)
        
        # Парсинг серийных номеров
        processed_text, serial_nums = self._parse_serial_numbers(processed_text)
        result['extracted_codes'].extend(serial_nums)
        
        result['processed'] = processed_text.strip()
        
        return result
    
    def _parse_gost(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг ГОСТ стандартов"""
        codes = []
        
        def replace_gost(match):
            gost_type = match.group(1)  # ГОСТ
            category = match.group(2)   # Р (если есть)
            number = match.group(3)     # Основной номер
            sub_number = match.group(4) # Подномер
            year = match.group(5)       # Год
            
            code_info = {
                'type': 'ГОСТ',
                'category': 'Р' if category else 'обычный',
                'number': number,
                'sub_number': sub_number,
                'year': year,
                'original': match.group(0),
                'structured': True
            }
            
            # Формируем нормализованный вид
            normalized = f"ГОСТ"
            if category:
                normalized += f" {category}"
            normalized += f" {number}-{sub_number}"
            if year:
                normalized += f"-{year}"
            
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['gost'].sub(replace_gost, text)
        return processed_text, codes
    
    def _parse_tu(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг ТУ (технических условий)"""
        codes = []
        
        def replace_tu(match):
            tu_type = match.group(1)  # ТУ
            code = match.group(2)     # Код
            
            code_info = {
                'type': 'ТУ',
                'code': code,
                'original': match.group(0),
                'structured': True
            }
            
            # Пытаемся разобрать структуру ТУ
            tu_parts = code.split('-')
            if len(tu_parts) >= 3:
                code_info.update({
                    'group': tu_parts[0],
                    'subgroup': tu_parts[1] if len(tu_parts) > 1 else None,
                    'number': tu_parts[2] if len(tu_parts) > 2 else None,
                    'year': tu_parts[3] if len(tu_parts) > 3 else None
                })
            
            normalized = f"ТУ {code}"
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['tu'].sub(replace_tu, text)
        return processed_text, codes
    
    def _parse_other_standards(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг других стандартов (ОСТ, СТО, СТП и т.д.)"""
        codes = []
        
        def replace_standard(match):
            std_type = match.group(1).upper()
            code = match.group(2)
            
            code_info = {
                'type': std_type,
                'code': code,
                'original': match.group(0),
                'structured': True
            }
            
            normalized = f"{std_type} {code}"
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['other_standards'].sub(replace_standard, text)
        return processed_text, codes
    
    def _parse_articles(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг артикулов"""
        codes = []
        processed_text = text
        
        # Сначала обрабатываем сложные артикулы
        def replace_complex_article(match):
            article = match.group(1)
            
            code_info = {
                'type': 'артикул_сложный',
                'code': article,
                'original': match.group(0),
                'structured': False
            }
            
            # Нормализуем разделители
            if self.config.normalize_separators:
                normalized = re.sub(r'[-\s]+', '-', article)
                normalized = re.sub(r'[хx]', '×', normalized)
            else:
                normalized = article
            
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['complex_articles'].sub(replace_complex_article, processed_text)
        
        # Затем простые артикулы
        def replace_article(match):
            article = match.group(1)

            # Пропускаем если это просто число
            if article.isdigit() and len(article) < 3:
                return match.group(0)

            # ИСПРАВЛЕНИЕ: Фильтрация ложных артикулов
            if self._is_false_article(article):
                return match.group(0)  # Не извлекаем как артикул

            code_info = {
                'type': 'артикул',
                'code': article,
                'original': match.group(0),
                'structured': False
            }
            
            # Нормализуем разделители
            if self.config.normalize_separators:
                # ИСПРАВЛЕНИЕ: НЕ заменяем точки в десятичных числах
                # Заменяем только множественные разделители, сохраняя десятичные точки
                normalized = re.sub(r'[-\s]+', '-', article)  # Убрали \. из паттерна
                normalized = re.sub(r'[хx]', '×', normalized)
            else:
                normalized = article
            
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['articles'].sub(replace_article, processed_text)
        return processed_text, codes
    
    def _parse_drawings(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг номеров чертежей"""
        codes = []
        
        def replace_drawing(match):
            drawing = match.group(1)
            
            code_info = {
                'type': 'чертеж',
                'code': drawing,
                'original': match.group(0),
                'structured': False
            }
            
            normalized = re.sub(r'[-\s]+', '-', drawing.upper())
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['drawings'].sub(replace_drawing, text)
        return processed_text, codes
    
    def _parse_serial_numbers(self, text: str) -> Tuple[str, List[Dict]]:
        """Парсинг серийных номеров"""
        codes = []
        
        def replace_serial(match):
            prefix = match.group(1)
            number = match.group(2)
            
            code_info = {
                'type': 'серийный_номер',
                'prefix': prefix,
                'number': number,
                'original': match.group(0),
                'structured': True
            }
            
            normalized = f"S/N {number}"
            code_info['normalized'] = normalized
            codes.append(code_info)
            
            return normalized
        
        processed_text = self.patterns['serial_numbers'].sub(replace_serial, text)
        return processed_text, codes

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
                    'extracted_codes': []
                })
        return results

    def validate_code(self, code_type: str, code: str) -> Dict[str, Any]:
        """
        Валидация технического кода

        Args:
            code_type: Тип кода (ГОСТ, ТУ и т.д.)
            code: Код для валидации

        Returns:
            Dict с результатами валидации
        """
        validation_result = {
            'valid': False,
            'code_type': code_type,
            'code': code,
            'errors': [],
            'suggestions': []
        }

        if code_type == 'ГОСТ':
            # Проверка формата ГОСТ
            gost_match = self.patterns['gost'].match(f"ГОСТ {code}")
            if gost_match:
                validation_result['valid'] = True
            else:
                validation_result['errors'].append("Неверный формат ГОСТ")
                validation_result['suggestions'].append("Формат: ГОСТ [Р] XXXXX-YYYY[-ZZZZ]")

        elif code_type == 'ТУ':
            # Проверка формата ТУ
            if re.match(r'^[0-9А-Яа-яA-Za-z\-\.]+$', code):
                validation_result['valid'] = True
            else:
                validation_result['errors'].append("Неверный формат ТУ")
                validation_result['suggestions'].append("Формат: ТУ XXXX-XXX-XXXXXXXX-YYYY")

        return validation_result

    def _is_false_article(self, candidate: str) -> bool:
        """
        ИСПРАВЛЕНИЕ: Фильтрация ложных артикулов
        Определяет, является ли кандидат ложным артикулом
        """
        # Паттерны ложных артикулов
        false_patterns = [
            r'^\d+-\d+-шт$',           # количества: 2-0-шт
            r'^\d+-\d+кг$',            # веса: 1505-0кг
            r'^\d+-\d+мВО$',           # характеристики: 220-0-мВО
            r'^\d+-применяемый$',      # части описаний: 220-применяемый
            r'^\d+-\d+г$',             # граммы: 2-0-г
            r'^\d+-\d+В$',             # вольты: 14-0В
            r'^\d+-\d+м$',             # метры: 6500-0м
            r'^\d+-\d+мм$',            # миллиметры: 10-03мм (но это уже исправлено)
            r'^\d+-\d+квт$',           # киловатты: 4-0квт
            r'^\d+-\d+мпа$',           # мегапаскали: 35-0мпа
            r'^\d+-пробки$',           # части описаний: 2-пробки
            r'^\d+-комплект$',         # комплекты: 1-0-комплект
            r'^[А-Я]\d+-$',            # неполные коды: К52-
            r'^\d+-$',                 # неполные числа: 245-
            r'^dims=',                 # технические токены: dims=[...]
            r'^range_',                # диапазоны: range_20295-85
            r'^frac_',                 # дроби: frac_1-2

            # ИСПРАВЛЕНИЕ: Дополнительные паттерны ложных артикулов
            r'^\d+\.\d+-мВО$',         # характеристики с точкой: 220.0-мВО
            r'^\d+-ЭНЕРГО$',           # части названий брендов: 4-ЭНЕРГО
            r'^\d+-с$',                # части описаний: 200-с
            r'^\d+-применяемый$',      # применяемый: 220-применяемый
            r'^\d+\.\d+-комплект$',    # комплекты с точкой: 1.0-комплект
            r'^[А-Я]+\d*\.\d+-$',      # неполные коды с точкой: Ш1.0-
            r'^\d+\.\d+$',             # просто числа с точкой: 20.0, 45.0
            r'^[А-Я]\d+[А-Я]*р$',      # части слов: Ц6Хр
            r'^\d+п$',                 # части единиц: 61п
            r'^\d+-предметов$',        # количество предметов: 13-предметов
            r'^К\d+-dims$',            # технические токены: К52-dims
            r'^\d+-ТУ$',               # части стандартов: 85-ТУ, 2012-ТУ
            r'^Fig\d+-$',              # части кодов: Fig1502-
            r'^НКТ\d+-\d+$',           # части обозначений: НКТ60-4
            r'^СТ\d+-$',               # части обозначений: СТ110-
        ]

        # Проверяем каждый паттерн
        for pattern in false_patterns:
            if re.match(pattern, candidate, re.IGNORECASE):
                return True

        # Дополнительные проверки

        # Слишком короткие (менее 3 символов)
        if len(candidate) < 3:
            return True

        # Только цифры и дефисы без букв (вероятно, размеры или количества)
        if re.match(r'^[\d\-]+$', candidate) and '-' in candidate:
            parts = candidate.split('-')
            # Если все части - числа, это вероятно не артикул
            if all(part.isdigit() for part in parts if part):
                return True

        # Единицы измерения в конце
        units_suffixes = ['шт', 'кг', 'г', 'мм', 'см', 'м', 'л', 'мл', 'В', 'А', 'квт', 'мпа', 'атм']
        for suffix in units_suffixes:
            if candidate.lower().endswith(suffix):
                return True

        # ИСПРАВЛЕНИЕ: Дополнительные проверки ложных артикулов

        # Части названий брендов и моделей
        brand_parts = ['энерго', 'рекорд', 'мастер', 'зубр', 'acme', 'tmk', 'fmc', 'centum']
        if candidate.lower() in brand_parts:
            return True

        # Характеристики с единицами (число-единица)
        if re.match(r'^\d+\.\d+-[а-я]+$', candidate, re.IGNORECASE):
            return True

        # Части технических обозначений
        tech_parts = ['dims', 'range', 'frac', 'size', 'применяемый', 'комплект', 'предметов']
        for part in tech_parts:
            if part in candidate.lower():
                return True

        # Слишком общие коды (только цифры и один дефис)
        if re.match(r'^\d+-\d+$', candidate) and len(candidate) <= 6:
            return True

        # Коды, заканчивающиеся дефисом (неполные)
        if candidate.endswith('-') and len(candidate) <= 8:
            return True

        return False

    def get_code_info(self, code_type: str) -> Dict[str, Any]:
        """Получение информации о типе кода"""
        return self.code_types.get(code_type.upper(), {
            'description': 'Неизвестный тип кода',
            'format': 'Не определен',
            'components': []
        })

    # Phase 4: Enhanced two-layer processing methods

    def _parse_standards_exact(self, text: str) -> Tuple[str, List[Dict]]:
        """Layer 1: Точное распознавание стандартов (ГОСТ|ТУ|DIN|ISO)"""
        codes = []

        def replace_standard(match):
            std_type = match.group(1).upper()
            code_value = match.group(2).strip()

            # Phase 4: Fix ГОСТ/ОСТ confusion
            if self.config.fix_gost_ost_confusion:
                if std_type == 'ОСТ' and 'ГОСТ' in text.upper():
                    std_type = 'ГОСТ'

            code_info = {
                'type': 'стандарт',
                'standard_type': std_type,
                'code': code_value,
                'original': match.group(0),
                'layer': 'standards',  # Phase 4: mark layer
                'preserve_exactly': True  # Phase 4: no lemmatization
            }

            # Phase 4: Prevent code truncation
            if self.config.prevent_code_truncation:
                normalized = f"{std_type} {code_value}"
                # Remove trailing dashes
                normalized = normalized.rstrip('-')
            else:
                normalized = match.group(0)

            code_info['normalized'] = normalized
            codes.append(code_info)

            return normalized

        processed_text = self.patterns['standards_exact'].sub(replace_standard, text)
        return processed_text, codes

    def _parse_dimensional_codes(self, text: str) -> Tuple[str, List[Dict]]:
        """Layer 2: Размерные коды (DN50, PN16, M10)"""
        codes = []

        def replace_dimensional(match):
            prefix = match.group(1).upper()
            value = match.group(2)

            code_info = {
                'type': 'размерный_код',
                'prefix': prefix,
                'value': float(value.replace(',', '.')),
                'original': match.group(0),
                'layer': 'article_codes',  # Phase 4: mark layer
                'category': 'dimensional'
            }

            normalized = f"{prefix}{value}"
            code_info['normalized'] = normalized
            codes.append(code_info)

            return normalized

        processed_text = self.patterns['dimensional_codes'].sub(replace_dimensional, text)
        return processed_text, codes

    def _parse_power_codes(self, text: str) -> Tuple[str, List[Dict]]:
        """Layer 2: Мощностные коды (5кВт, 1500об/мин)"""
        codes = []

        def replace_power(match):
            value = match.group(1).replace(',', '.')
            unit = match.group(2)

            code_info = {
                'type': 'мощностной_код',
                'value': float(value),
                'unit': unit,
                'original': match.group(0),
                'layer': 'article_codes',  # Phase 4: mark layer
                'category': 'power'
            }

            normalized = f"{value}{unit}"
            code_info['normalized'] = normalized
            codes.append(code_info)

            return normalized

        processed_text = self.patterns['power_codes'].sub(replace_power, text)
        return processed_text, codes

    def _parse_technical_parameters(self, text: str) -> Tuple[str, List[Dict]]:
        """Layer 2: Технические параметры без описательных слов"""
        codes = []

        def replace_technical(match):
            value = match.group(1).replace(',', '.')
            unit = match.group(2)

            # Phase 4: Exclude descriptive words - only keep technical values
            if self.config.exclude_descriptive_words:
                # Skip if surrounded by descriptive words
                descriptive_words = ['резиновый', 'металлический', 'стальной', 'пластиковый',
                                   'круглый', 'квадратный', 'эластичный', 'жесткий']

                # Simple check - this could be enhanced
                surrounding_text = text[max(0, match.start()-20):match.end()+20].lower()
                if any(word in surrounding_text for word in descriptive_words):
                    return match.group(0)  # Keep original, don't extract as code

            code_info = {
                'type': 'технический_параметр',
                'value': float(value),
                'unit': unit,
                'original': match.group(0),
                'layer': 'article_codes',  # Phase 4: mark layer
                'category': 'technical'
            }

            normalized = f"{value}{unit}"
            code_info['normalized'] = normalized
            codes.append(code_info)

            return normalized

        processed_text = self.patterns['technical_parameters'].sub(replace_technical, text)
        return processed_text, codes

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Получение статистики парсинга"""
        total_texts = len(results)
        total_codes = sum(len(r['extracted_codes']) for r in results)

        # Подсчет по типам кодов
        code_types = {}
        for result in results:
            for code in result['extracted_codes']:
                code_type = code['type']
                code_types[code_type] = code_types.get(code_type, 0) + 1

        return {
            'total_texts': total_texts,
            'total_codes': total_codes,
            'avg_codes_per_text': total_codes / total_texts if total_texts > 0 else 0,
            'code_types': code_types,
            'most_common_types': sorted(code_types.items(), key=lambda x: x[1], reverse=True)
        }
