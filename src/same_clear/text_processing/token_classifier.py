"""
Модуль для классификации и предобработки токенов товарных наименований.
Используется для нормализации текста перед поиском аналогов.
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


class TokenClassifier:
    """
    Классификатор токенов для предобработки товарных наименований.
    
    Разделяет токены на категории:
    - pass_names: названия товаров (первое слово)
    - stop_words: служебные слова
    - names_model: названия брендов/моделей
    - articles: артикулы
    - parameters: технические параметры по категориям
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация классификатора.
        
        Args:
            config_path: путь к JSON файлу с конфигурацией токенов
        """
        self.pass_names: Set[str] = set()
        self.stop_words: Set[str] = set()
        self.names_model: Set[str] = set()
        self.articles: Set[str] = set()
        self.parameters: Dict[str, Set[str]] = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_default_config(self):
        """Загрузка стандартной конфигурации токенов."""
        # Основные названия товаров (первое слово)
        self.pass_names = {
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
        }
        
        # Служебные слова
        self.stop_words = {
            'на', 'от', 'для', 'по', 'до', 'и', 'с', 'в', 'из', 'под', 'к',
            'без', 'со', 'н', 'не'
        }
        
        # Названия брендов/моделей
        self.names_model = {
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
        }
        
        # Артикулы
        self.articles = {
            'артикул', 'дпо-108', 'w80dm', 'Dect-T393-4XAA', '6F22-BL1', 'dect',
            '8/0-c', 'арк-г', 'а24-5-1', '103', 'ва47-29', 'nb1-63dc', 'dc500b',
            'tdgc2-0.5k', 'аосн-2-220', 'hw-200325cp0', 'hw-200325cpo', 'hq-200325epo',
            'hq-200325ep0', 'hn-200325epo', 'hn-200325ep0', 'hp-s-1047a', 'elt-122000',
            'гост', 'din', '28919-91', '9833-73', '7360-2015', '3668-015-26602587-2008',
            '000ту', '3668-025-77020022-2015', '18829-73', '633-80', '7798-70', 'код',
            '8752-79', '25573-82'
        }
        
        # Технические параметры по категориям
        self.parameters = {
            'TYPE': {
                'тип', 'типа', 'led', 'светодиодная', '1сц', '(одноветвевой стандартный)',
                'рпс-у', 'lock', 'сзи', 'нсд', 'скн', 'евро-95', 'экто-95', 'н7',
                'галогеновая', 'дифференциальный', 'авдт', 'латр', 'toy 3p0', 'ао',
                'foam', 'суперпена', 'активная', 'пена', 'безконтактной', 'бесконтактной',
                'magic', 'full hd', 'hd', 'easy!lock', 'внеш.', 'блютуз', 'aux', 'rca',
                'музыки', 'игр', 'smart яндекс тв', 'тв', 'ударные', 'рост', 'размер',
                'высота', 'защиты', 'ст40хн2ма', 'опз', 'рабочий', 'летний', 'высокого',
                'пропиткой', 'гидравлический', 'мужской', 'по металлу', 'кл.', 'масляный',
                'р', 'кожаные', 'длина', 'а4', 'мв', 'итр', 'универсальный', 'силовой',
                'воздушный', 'защитный', 'шаровой', 'электрический', 'огнестойкий', 'man',
                'мужские', 'pro', 'зимние', 'эксп', 'особ', 'обратный', 'утепленные',
                'пониженных', 'шестигранная', 'канатный', 'светодиодный', 'защитные',
                'толщина', 'механических', 'ударный', 'прямой', 'ок', 'комбинированный',
                'цанговый', 'замковое', 'брс', 'утепленный', 'отрезной', 'рвд', 'нкт',
                'левая', 'навесной', 'fmc', 'круглый', 'мужская', 'летние', 'композитный',
                'правая', 'тмк', 'jic', 'левый', 'tdm', 'nbr', 'правый', 'прозрачный',
                'жестким', 'магнитный', 'тпу', 'усиленный', 'самоклеящаяся', 'червячный',
                'междусменного', 'санях', 'ширина', 'жесткий', 'npt', 'натуральном', 'plus',
                'огнестойком', 'бурильная', 'соединительная', 'промывочный', 'антистатических',
                'женский', 'sata', 'нижняя', 'нмво', 'верхний', 'фланцевый', 'статического',
                'двусторонний', 'защитой', 'пу', 'армированный', 'резьбовой', 'факторов',
                'забойный', 'стандарт', 'верхняя', 'плюс', 'утепленная', 'свп', 'защитное',
                'прямая', 'стп', 'комбинированные', 'гальваническое', 'fmt', 'en',
                'быстроразъемное', 'механический', 'водяной'
            },
            'CATEGORY': {
                'потолочный', 'потолочная', 'иэк', '1-п', 'постоянного тока', 'поясах',
                'c2a', 'акр.', 'акрил', 'truck', 'грузовиков', 'воздушно-отопительный',
                'cfp', 'зимний', 'присоединительной', 'мво', 'оцинкованная', 'оцинкованный',
                'оцинкованное', 'противоэнцефалитный', 'l'
            },
            'CHARACTERISTIC': {
                'c', 'd', 'dk', 'ту', 'up', 'пвх', 'tmk', 'мв', 'класс', 'л', 'iek',
                'pf', 'а', 'хл', 's', 'lt'
            },
            'DIAMETER': {'d-36мм', 'диаметр', 'ду', 'диаметром'},
            'BREAKING_CAPACITY': {'6ка', '6ka', '4,5ка', '10ка'},
            'CURRENT': {'32a', '63a', '20a', '16а', '32а', '2a', '10а'},
            'VOLTAGE': {'24v', '220в', '0в-300в', '24в', '230в', '380в', 'напряжения', 'вт'},
            'POLES': {'1p', '2p', '3p', '3n+p', '4p'},
            'POWER': {'30вт', '24-70w', '50w', 'квт', '200дж', 'ва'},
            'COUNT': {'шт', 'уп.10', '2-4(пар)', 'час', 'предметов', '100шт', 'листов', '10шт', '2шт'},
            'TEMPERATURE': {'6500k', 'градусов', '6500к'},
            'WEIGHT': {'500мл', '400г', '24кг', '6кг', 'кг', 'тонн', 'мл', 'см', '50мм', '3м', '50м', '1л'},
            'LENGTH': {'1200мм', '4м', '30мм', '6мм', 'мм', 'м'},
            'COLOR': {
                'цвет', 'белый', 'ультрабелый', 'super red', 'red', 'красная', 'желтая',
                'черная', 'белая', 'черный', 'синий', 'серый', 'оранжевый', 'красный',
                'желтый', 'зеленый', 'темно-синий', 'синяя'
            },
            'MATERIAL': {
                'корпус', 'железо', 'серебро', 'gold', 'сталь', 'стальной', 'тканей',
                'металлический', 'пластик', 'нержавеющая', 'пластиковая', 'пластиковый',
                'материал', 'стальная', 'медный', 'резиновый', 'металл', 'металлическая',
                'ткань', 'стали', 'текстильный', 'алюминиевый', 'латунный', 'дуб', 'металла'
            },
            'IP': {'ip40', 'ip-dect', 'ip54', 'ip65', 'ip44'},
            'THREAD': {'м22', 'м16*1,5'},
            'COUNTRY': {'россия'},
            'SPEED': {'мбит/с'},
            'FREQUENCY': set(),
            'CASH': {
                '2.1', ')', '10)', '3', '8', 'уп', '10', '103-6ка-3n+p', '32a-30-c',
                '63a-30-c', '4', '5ка', '/', 'постоянного', 'тока', 'tdgc2', '0', '5k',
                'аосн-2-220', '20', '23', '1035', '601', 'л.', '0,8', '1', '-', 'х',
                '1,5', 'акр', 'toy', '3p0', 'super', '2к', '22внут.', 'wi-f', 'м16*1',
                '5', 'wi', 'fi', '5,3', '182-188', '170-176', '2', 'металлу', 'п',
                '194-200', 'кл', '158-164', '100', '2"', '6', '50', 'р', 'x', '52-54',
                '56-58', '60-62', '64-66', '12', '206-212', '218-224', '4"', '68-70',
                '72-74', '1"', '48-50', '25', '44-46', '200', '500', '40', '104-108',
                'з-102', '96-100', '9', '30', '7', '16', '1.2.3', '18', '15', '40-42',
                '8"', '1/2', 'цвет:', '00', '70', '№', ',', '+', 'з-86', '250', '60',
                'з-133', '11', '32', '112-116', '150', '№10', '1-3', '125', 'рабочее',
                '24', '80', '(', '400', '300', '45', '90', '42', '43', 'м,'
            },
            'WAIT': {'ожидание', 'отстой', 'сутки'},
            'PRESSURE': {'ру16'}
        }
    
    def load_config(self, config_path: str):
        """
        Загрузка конфигурации из JSON файла.
        
        Args:
            config_path: путь к JSON файлу
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.pass_names = set(data.get('pass_names', []))
            self.stop_words = set(data.get('stop_world', []))
            self.names_model = set(data.get('names_model', []))
            self.articles = set(data.get('arcticles', []))
            
            # Преобразуем параметры в множества
            self.parameters = {}
            for key, values in data.get('parameters', {}).items():
                self.parameters[key] = set(values)
                
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            self._load_default_config()
    
    def save_config(self, config_path: str):
        """
        Сохранение конфигурации в JSON файл.
        
        Args:
            config_path: путь к файлу для сохранения
        """
        data = {
            'pass_names': list(self.pass_names),
            'stop_world': list(self.stop_words),
            'names_model': list(self.names_model),
            'arcticles': list(self.articles),
            'parameters': {k: list(v) for k, v in self.parameters.items()}
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def classify_token(self, token: str) -> Tuple[str, str]:
        """
        Классификация токена по категориям.
        
        Args:
            token: токен для классификации
            
        Returns:
            Tuple[категория, подкатегория]
        """
        token_lower = token.lower()
        
        if token_lower in self.pass_names:
            return 'PRODUCT_NAME', 'pass_names'
        elif token_lower in self.stop_words:
            return 'STOP_WORD', 'stop_words'
        elif token_lower in self.names_model:
            return 'BRAND_MODEL', 'names_model'
        elif token_lower in self.articles:
            return 'ARTICLE', 'articles'
        else:
            # Проверяем параметры
            for param_type, values in self.parameters.items():
                if token_lower in values:
                    return 'PARAMETER', param_type
            
            return 'UNKNOWN', 'unknown'
    
    def split_complex_token(self, token: str) -> List[str]:
        """
        Разбиение сложного токена на простые.
        
        Args:
            token: сложный токен для разбиения
            
        Returns:
            Список простых токенов
        """
        if len(token) < 3:
            return [token]
        
        # Заменяем разделители на пробелы
        token = re.sub(r'[,.()[\]/+\-]', ' ', token)
        
        # Разбиваем по пробелам и фильтруем пустые
        tokens = [t.strip() for t in token.split() if t.strip()]
        
        return tokens if tokens else [token]
    
    def normalize_product_name(self, text: str) -> Dict[str, List[str]]:
        """
        Нормализация названия товара с разделением на компоненты.
        
        Args:
            text: исходный текст названия товара
            
        Returns:
            Словарь с классифицированными компонентами
        """
        # Приводим к нижнему регистру и разбиваем на токены
        tokens = text.lower().split()
        
        if not tokens:
            return {}
        
        # Первый токен - название товара
        product_name = tokens[0] if tokens else ""
        remaining_tokens = tokens[1:] if len(tokens) > 1 else []
        
        result = {
            'product_name': [product_name],
            'parameters': [],
            'brands': [],
            'articles': [],
            'stop_words': [],
            'unknown': []
        }
        
        # Обрабатываем оставшиеся токены
        for token in remaining_tokens:
            # Разбиваем сложные токены
            simple_tokens = self.split_complex_token(token)
            
            for simple_token in simple_tokens:
                if not simple_token:
                    continue
                    
                category, subcategory = self.classify_token(simple_token)
                
                if category == 'PRODUCT_NAME':
                    result['parameters'].append(simple_token)
                elif category == 'BRAND_MODEL':
                    result['brands'].append(simple_token)
                elif category == 'ARTICLE':
                    result['articles'].append(simple_token)
                elif category == 'STOP_WORD':
                    result['stop_words'].append(simple_token)
                elif category == 'PARAMETER':
                    result['parameters'].append(simple_token)
                else:
                    result['unknown'].append(simple_token)
        
        return result
    
    def extract_searchable_parameters(self, text: str) -> List[str]:
        """
        Извлечение параметров для поиска аналогов.
        
        Args:
            text: название товара
            
        Returns:
            Список параметров для поиска
        """
        normalized = self.normalize_product_name(text)
        
        # Объединяем параметры и бренды (исключаем название товара и служебные слова)
        searchable = []
        searchable.extend(normalized['parameters'])
        searchable.extend(normalized['brands'])
        searchable.extend(normalized['articles'])
        
        return [param for param in searchable if param]
    
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
            'parameter_categories': len(self.parameters)
        }
        
        # Добавляем статистику по категориям параметров
        for param_type, values in self.parameters.items():
            stats[f'param_{param_type.lower()}'] = len(values)
        
        return stats


def create_token_classifier(config_path: Optional[str] = None) -> TokenClassifier:
    """
    Фабричная функция для создания классификатора токенов.
    
    Args:
        config_path: путь к конфигурационному файлу
        
    Returns:
        Экземпляр TokenClassifier
    """
    return TokenClassifier(config_path)

