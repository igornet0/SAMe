"""
Оптимизированная версия модуля для классификации токенов.
Включает улучшения производительности на основе анализа.
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional
from functools import lru_cache
from pathlib import Path


class OptimizedTokenClassifier:
    """
    Оптимизированный классификатор токенов с улучшенной производительностью.
    
    Улучшения:
    - O(1) поиск для всех типов токенов включая parameters
    - LRU кэш для часто используемых токенов
    - Предкомпилированные регулярные выражения
    - Frozen sets для неизменяемых данных
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация оптимизированного классификатора.
        
        Args:
            config_path: путь к JSON файлу с конфигурацией токенов
        """
        # Основные множества токенов
        self.pass_names: frozenset[str] = frozenset()
        self.stop_words: frozenset[str] = frozenset()
        self.names_model: frozenset[str] = frozenset()
        self.articles: frozenset[str] = frozenset()
        
        # Обратный индекс для параметров: token -> category
        self.parameter_index: Dict[str, str] = {}
        self.parameters: Dict[str, frozenset[str]] = {}
        
        # Предкомпилированные регулярные выражения
        self._compile_regex_patterns()
        
        # Загружаем конфигурацию
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()
        
        # Создаем обратный индекс после загрузки данных
        self._build_parameter_index()
    
    def _compile_regex_patterns(self):
        """Предкомпиляция регулярных выражений для лучшей производительности."""
        self.token_split_pattern = re.compile(r'[,.()[\]/+\-]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.trim_pattern = re.compile(r'^\s+|\s+$')
    
    def _build_parameter_index(self):
        """Создание обратного индекса для O(1) поиска параметров."""
        self.parameter_index = {}
        for category, tokens in self.parameters.items():
            for token in tokens:
                self.parameter_index[token.lower()] = category
    
    def _load_default_config(self):
        """Загрузка стандартной конфигурации с frozen sets."""
        # Основные названия товаров
        self.pass_names = frozenset({
            'светильник', 'панель', 'призма', 'система', 'контроллер', 'микросота',
            'спрей', 'ванной', 'комнаты', 'средство', 'чистящее', 'сочное', 'яблоко',
            'стиральный', 'порошок', 'строп', 'цепной', 'сухой', 'паек', 'офицерский',
            'армейские', 'будни', 'одноветвевой', 'крепления', 'груза', 'автоаптечка',
            'ремонтта', 'покрышек', 'камер', 'нагрузка', 'оснащен', 'стандартный',
            'крепежными', 'винтами', 'автобензин', 'автозапчасть', 'автолампа', 
            'автолампочка', 'автошампунь', 'автоэмаль', 'автомобиля', 'фарная', 
            'автомат', 'характеристика', 'автоматический', 'выключатель',
            'автотрансформатор', 'регулируемый', 'active', 'мойки', 'компонент', 
            'агрегат', 'адаптер', 'переходник', 'переходники', 'пистолета', 'резьбы',
            'ноутбука', 'штуцер', 'компьютера', 'пк', 'приемник', 'адаптеры', 
            'набор', 'костюм', 'фильтр', 'переводник', 'кольцо', 'кольцо', 
            'уплотнительное', 'комплект', 'сборе', 'подшипник', 'труба', 'рукав',
            'клапан', 'уплотнение', 'ключ', 'болт', 'сапоги', 'давления', 'услуга',
            'втулка', 'прокладка', 'резьбой', 'кабель', 'насос', 'насоса', 'гайка',
            'муфта', 'шланг', 'хомут', 'шайба', 'манжета', 'кран', 'блок', 
            'соединение', 'топливный', 'куртка', 'лента', 'перчатки', 'ботинки',
            'ремень', 'патрубок', 'фитинг', 'винт', 'строп', 'датчик', 'челюсть', 'вал'
        })
        
        # Служебные слова
        self.stop_words = frozenset({
            'на', 'от', 'для', 'по', 'до', 'и', 'с', 'в', 'из', 'под', 'к',
            'без', 'со', 'н', 'не'
        })
        
        # Названия брендов/моделей
        self.names_model = frozenset({
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
        })
        
        # Артикулы
        self.articles = frozenset({
            'артикул', 'дпо-108', 'w80dm', 'Dect-T393-4XAA', '6F22-BL1', 'dect',
            '8/0-c', 'арк-г', 'а24-5-1', '103', 'ва47-29', 'nb1-63dc', 'dc500b',
            'tdgc2-0.5k', 'аосн-2-220', 'hw-200325cp0', 'hw-200325cpo', 'hq-200325epo',
            'hq-200325ep0', 'hn-200325epo', 'hn-200325ep0', 'hp-s-1047a', 'elt-122000',
            'гост', 'din', '28919-91', '9833-73', '7360-2015', '3668-015-26602587-2008',
            '000ту', '3668-025-77020022-2015', '18829-73', '633-80', '7798-70', 'код',
            '8752-79', '25573-82'
        })
        
        # Технические параметры по категориям (сокращенная версия для примера)
        self.parameters = {
            'TYPE': frozenset({
                'тип', 'типа', 'led', 'светодиодная', '1сц', 'дифференциальный',
                'авдт', 'латр', 'активная', 'пена', 'безконтактной', 'бесконтактной',
                'magic', 'full hd', 'hd', 'блютуз', 'aux', 'rca', 'тв', 'ударные',
                'рост', 'размер', 'высота', 'защиты', 'рабочий', 'летний', 'высокого',
                'гидравлический', 'мужской', 'по металлу', 'масляный', 'универсальный',
                'силовой', 'воздушный', 'защитный', 'шаровой', 'электрический'
            }),
            'CATEGORY': frozenset({
                'потолочный', 'потолочная', 'иэк', '1-п', 'постоянного тока', 'поясах',
                'c2a', 'акр.', 'акрил', 'truck', 'грузовиков', 'воздушно-отопительный',
                'cfp', 'зимний', 'присоединительной', 'мво', 'оцинкованная', 'оцинкованный',
                'оцинкованное', 'противоэнцефалитный', 'l'
            }),
            'CURRENT': frozenset({'32a', '63a', '20a', '16а', '32а', '2a', '10а'}),
            'VOLTAGE': frozenset({'24v', '220в', '0в-300в', '24в', '230в', '380в', 'напряжения', 'вт'}),
            'COLOR': frozenset({
                'цвет', 'белый', 'ультрабелый', 'super red', 'red', 'красная', 'желтая',
                'черная', 'белая', 'черный', 'синий', 'серый', 'оранжевый', 'красный',
                'желтый', 'зеленый', 'темно-синий', 'синяя'
            }),
            'MATERIAL': frozenset({
                'корпус', 'железо', 'серебро', 'gold', 'сталь', 'стальной', 'тканей',
                'металлический', 'пластик', 'нержавеющая', 'пластиковая', 'пластиковый',
                'материал', 'стальная', 'медный', 'резиновый', 'металл', 'металлическая',
                'ткань', 'стали', 'текстильный', 'алюминиевый', 'латунный', 'дуб', 'металла'
            })
        }
    
    @lru_cache(maxsize=1024)
    def classify_token(self, token: str) -> Tuple[str, str]:
        """
        Оптимизированная классификация токена с кэшированием.
        
        Args:
            token: токен для классификации
            
        Returns:
            Tuple[категория, подкатегория]
        """
        token_lower = token.lower()
        
        # O(1) поиск во всех основных категориях
        if token_lower in self.pass_names:
            return 'PRODUCT_NAME', 'pass_names'
        elif token_lower in self.stop_words:
            return 'STOP_WORD', 'stop_words'
        elif token_lower in self.names_model:
            return 'BRAND_MODEL', 'names_model'
        elif token_lower in self.articles:
            return 'ARTICLE', 'articles'
        elif token_lower in self.parameter_index:
            # O(1) поиск параметров через обратный индекс
            return 'PARAMETER', self.parameter_index[token_lower]
        else:
            return 'UNKNOWN', 'unknown'
    
    def split_complex_token(self, token: str) -> List[str]:
        """
        Оптимизированное разбиение сложного токена с предкомпилированными regex.
        
        Args:
            token: сложный токен для разбиения
            
        Returns:
            Список простых токенов
        """
        if len(token) < 3:
            return [token]
        
        # Используем предкомпилированные паттерны
        token = self.token_split_pattern.sub(' ', token)
        tokens = [t.strip() for t in token.split() if t.strip()]
        
        return tokens if tokens else [token]
    
    def load_config(self, config_path: str):
        """
        Загрузка конфигурации из JSON файла с созданием frozen sets.
        
        Args:
            config_path: путь к JSON файлу
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.pass_names = frozenset(data.get('pass_names', []))
            self.stop_words = frozenset(data.get('stop_world', []))
            self.names_model = frozenset(data.get('names_model', []))
            self.articles = frozenset(data.get('arcticles', []))
            
            # Создаем frozen sets для параметров
            self.parameters = {}
            for key, values in data.get('parameters', {}).items():
                self.parameters[key] = frozenset(values)
                
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            self._load_default_config()
    
    def get_token_statistics(self) -> Dict[str, int]:
        """
        Получение статистики по токенам.
        
        Returns:
            Словарь с количеством токенов по категориям
        """
        stats = {
            'product_names': len(self.pass_names),
            'stop_words': len(self.stop_words),
            'brands_models': len(self.names_model),
            'articles': len(self.articles),
            'parameters_total': sum(len(v) for v in self.parameters.values()),
            'parameter_categories': len(self.parameters),
            'parameter_index_size': len(self.parameter_index)
        }
        
        # Добавляем статистику по категориям параметров
        for param_type, values in self.parameters.items():
            stats[f'param_{param_type.lower()}'] = len(values)
        
        return stats
    
    def clear_cache(self):
        """Очистка кэша классификации."""
        self.classify_token.cache_clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Получение информации о кэше."""
        cache_info = self.classify_token.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'maxsize': cache_info.maxsize,
            'currsize': cache_info.currsize,
            'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if cache_info.hits + cache_info.misses > 0 else 0
        }


def create_optimized_token_classifier(config_path: Optional[str] = None) -> OptimizedTokenClassifier:
    """
    Фабричная функция для создания оптимизированного классификатора токенов.
    
    Args:
        config_path: путь к конфигурационному файлу
        
    Returns:
        Экземпляр OptimizedTokenClassifier
    """
    return OptimizedTokenClassifier(config_path)

