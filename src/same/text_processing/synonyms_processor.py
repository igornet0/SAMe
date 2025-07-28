"""
Модуль нормализации синонимов и технических терминов
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SynonymsConfig:
    """Конфигурация обработки синонимов"""
    normalize_materials: bool = True      # каучуковый → резиновый
    normalize_shapes: bool = True         # круглый → цилиндрический
    normalize_functions: bool = True      # запорный → перекрывающий
    normalize_brands: bool = False        # Оставляем бренды как есть
    case_sensitive: bool = False          # Регистронезависимый поиск
    preserve_technical_terms: bool = True # Сохранять специальные термины


class SynonymsProcessor:
    """Процессор нормализации синонимов"""
    
    def __init__(self, config: SynonymsConfig = None):
        self.config = config or SynonymsConfig()
        self._init_synonym_dictionaries()
        self._compile_patterns()
    
    def _init_synonym_dictionaries(self):
        """Инициализация словарей синонимов"""
        
        # Материалы
        self.material_synonyms = {
            'резиновый': ['каучуковый', 'эластичный', 'гуммированный', 'латексный'],
            'стальной': ['металлический', 'железный', 'из стали'],
            'пластиковый': ['пластмассовый', 'полимерный', 'из пластика'],
            'алюминиевый': ['алюминевый', 'из алюминия', 'дюралевый'],
            'медный': ['из меди', 'бронзовый', 'латунный'],
            'керамический': ['керамичный', 'из керамики', 'фарфоровый'],
            'стеклянный': ['из стекла', 'стекло'],
            'деревянный': ['из дерева', 'древесный'],
            'бумажный': ['из бумаги', 'картонный'],
            'тканевый': ['текстильный', 'из ткани', 'матерчатый']
        }
        
        # Формы и геометрия
        self.shape_synonyms = {
            'цилиндрический': ['круглый', 'трубчатый', 'цилиндровый'],
            'прямоугольный': ['четырехугольный', 'квадратный'],
            'сферический': ['шарообразный', 'круглый', 'шаровой'],
            'конический': ['конусный', 'конусообразный'],
            'плоский': ['плоскостной', 'ровный'],
            'изогнутый': ['кривой', 'гнутый', 'согнутый'],
            'спиральный': ['винтовой', 'закрученный', 'витой']
        }
        
        # Функции и назначение
        self.function_synonyms = {
            'запорный': ['перекрывающий', 'отсекающий', 'блокирующий'],
            'регулирующий': ['настроечный', 'управляющий', 'контрольный'],
            'фильтрующий': ['очищающий', 'фильтровальный'],
            'соединительный': ['соединяющий', 'крепежный', 'стыковочный'],
            'уплотнительный': ['герметизирующий', 'изолирующий'],
            'защитный': ['предохранительный', 'охранный', 'безопасности'],
            'измерительный': ['контрольный', 'мерительный', 'датчиковый'],
            'приводной': ['движущий', 'передаточный', 'силовой']
        }
        
        # Размеры и характеристики
        self.characteristic_synonyms = {
            'большой': ['крупный', 'увеличенный', 'габаритный'],
            'малый': ['маленький', 'мини', 'компактный', 'миниатюрный'],
            'тонкий': ['узкий', 'тонкостенный'],
            'толстый': ['широкий', 'толстостенный'],
            'длинный': ['удлиненный', 'протяженный'],
            'короткий': ['укороченный', 'компактный'],
            'легкий': ['облегченный', 'малый по весу'],
            'тяжелый': ['усиленный', 'массивный']
        }
        
        # Технические термины (специальные случаи)
        self.technical_synonyms = {
            'подшипник': ['подшипниковый узел', 'опора'],
            'втулка': ['гильза', 'вкладыш', 'муфта'],
            'прокладка': ['уплотнение', 'сальник', 'манжета'],
            'фильтр': ['фильтрующий элемент', 'очиститель'],
            'клапан': ['вентиль', 'затвор', 'заслонка'],
            'насос': ['помпа', 'нагнетатель'],
            'двигатель': ['мотор', 'привод', 'движок'],
            'редуктор': ['понижающий редуктор', 'замедлитель'],
            'муфта': ['соединение', 'сцепление', 'переходник']
        }
        
        # Объединяем все словари
        self.all_synonyms = {}
        if self.config.normalize_materials:
            self.all_synonyms.update(self.material_synonyms)
        if self.config.normalize_shapes:
            self.all_synonyms.update(self.shape_synonyms)
        if self.config.normalize_functions:
            self.all_synonyms.update(self.function_synonyms)
        if self.config.preserve_technical_terms:
            self.all_synonyms.update(self.technical_synonyms)
        
        self.all_synonyms.update(self.characteristic_synonyms)
        
        # Создаем обратный индекс для быстрого поиска
        self.reverse_synonyms = {}
        for canonical, synonyms in self.all_synonyms.items():
            for synonym in synonyms:
                self.reverse_synonyms[synonym.lower()] = canonical
    
    def _compile_patterns(self):
        """Компиляция паттернов для поиска"""
        # Создаем паттерн для поиска всех синонимов
        all_terms = list(self.reverse_synonyms.keys())
        # Сортируем по длине (длинные сначала) для правильного матчинга
        all_terms.sort(key=len, reverse=True)
        
        # Экранируем специальные символы
        escaped_terms = [re.escape(term) for term in all_terms]
        
        # Создаем паттерн с границами слов
        pattern = r'\b(' + '|'.join(escaped_terms) + r')\b'
        
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        self.synonym_pattern = re.compile(pattern, flags)
    
    def process_text(self, text: str) -> Dict[str, any]:
        """
        Основной метод нормализации синонимов
        
        Args:
            text: Входной текст
            
        Returns:
            Dict с результатами обработки
        """
        if not text or not isinstance(text, str):
            return {
                'original': text or '',
                'normalized': '',
                'replacements': []
            }
        
        result = {
            'original': text,
            'normalized': text,
            'replacements': []
        }
        
        # Поиск и замена синонимов
        def replace_synonym(match):
            found_term = match.group(1).lower()
            canonical_term = self.reverse_synonyms.get(found_term)
            
            if canonical_term:
                # Сохраняем информацию о замене
                replacement_info = {
                    'original': match.group(1),
                    'canonical': canonical_term,
                    'position': match.start()
                }
                result['replacements'].append(replacement_info)
                
                # Возвращаем канонический термин с сохранением регистра первой буквы
                if match.group(1)[0].isupper():
                    return canonical_term.capitalize()
                else:
                    return canonical_term
            
            return match.group(1)
        
        # Применяем замены
        normalized_text = self.synonym_pattern.sub(replace_synonym, text)
        result['normalized'] = normalized_text
        
        return result
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, any]]:
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
                    'normalized': text,
                    'replacements': []
                })
        return results
    
    def add_custom_synonyms(self, canonical: str, synonyms: List[str]):
        """
        Добавление пользовательских синонимов
        
        Args:
            canonical: Каноническая форма
            synonyms: Список синонимов
        """
        self.all_synonyms[canonical] = synonyms
        
        # Обновляем обратный индекс
        for synonym in synonyms:
            self.reverse_synonyms[synonym.lower()] = canonical
        
        # Перекомпилируем паттерны
        self._compile_patterns()
        
        logger.info(f"Added custom synonyms for '{canonical}': {synonyms}")
    
    def get_canonical_form(self, term: str) -> str:
        """
        Получение канонической формы термина
        
        Args:
            term: Исходный термин
            
        Returns:
            Каноническая форма или исходный термин
        """
        return self.reverse_synonyms.get(term.lower(), term)
    
    def get_statistics(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Получение статистики нормализации"""
        total_texts = len(results)
        total_replacements = sum(len(r['replacements']) for r in results)
        
        # Подсчет по типам замен
        replacement_stats = {}
        for result in results:
            for replacement in result['replacements']:
                canonical = replacement['canonical']
                replacement_stats[canonical] = replacement_stats.get(canonical, 0) + 1
        
        return {
            'total_texts': total_texts,
            'total_replacements': total_replacements,
            'avg_replacements_per_text': total_replacements / total_texts if total_texts > 0 else 0,
            'most_common_replacements': sorted(replacement_stats.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]
        }
