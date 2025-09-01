"""
Специализированный модуль для поиска дубликатов и аналогов
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class DuplicateSearchConfig:
    """Конфигурация поиска дубликатов"""
    exact_match_threshold: float = 0.98
    fuzzy_match_threshold: float = 0.45  # Радикально снижен для максимального покрытия
    parameter_similarity_threshold: float = 0.45  # Радикально снижен для максимального покрытия
    semantic_similarity_threshold: float = 0.40  # Радикально снижен для максимального покрытия
    min_duplicate_group_size: int = 2
    max_duplicate_group_size: int = 50
    enable_parameter_check: bool = True
    enable_brand_check: bool = True
    enable_semantic_check: bool = True  # Новый флаг
    # Весовые коэффициенты для разных типов сравнения
    exact_weight: float = 1.0
    fuzzy_weight: float = 0.8
    semantic_weight: float = 0.7
    parameter_weight: float = 0.6
    brand_weight: float = 0.5

@dataclass
class AnalogSearchConfig:
    """Конфигурация поиска аналогов"""
    exact_analog_threshold: float = 0.70  # Снижен для большего покрытия
    close_analog_threshold: float = 0.55  # Снижен для большего покрытия
    possible_analog_threshold: float = 0.25  # Радикально снижен для максимального покрытия
    # Улучшенные весовые коэффициенты
    semantic_weight: float = 0.45  # Увеличен для лучшего семантического поиска
    fuzzy_weight: float = 0.40     # Увеличен для лучшего нечеткого поиска
    parameter_weight: float = 0.30  # Увеличен для технических товаров
    brand_weight: float = 0.10     # Снижен с 0.2
    category_weight: float = 0.15  # Увеличен для лучшей категорийной группировки
    max_analogs_per_item: int = 100  # Радикально увеличен для максимального покрытия
    # Новые параметры
    enable_hierarchical_search: bool = True
    enable_parameter_priority: bool = True

@dataclass
class DuplicateResult:
    """Результат поиска дубликатов"""
    main_index: int
    main_name: str
    duplicate_indices: List[int]
    duplicate_names: List[str]
    similarity_scores: List[float]
    duplicate_type: str  # exact, fuzzy, parameter_based

@dataclass
class AnalogResult:
    """Результат поиска аналогов"""
    reference_index: int
    reference_name: str
    analogs: List[Dict[str, Any]]  # index, name, similarity, type, confidence

class AdvancedDuplicateDetector:
    """Продвинутый детектор дубликатов"""
    
    def __init__(self, config: DuplicateSearchConfig):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=None,
            lowercase=True
        )
        
    def detect_duplicates(self, df: pd.DataFrame, name_column: str = 'processed_name') -> List[DuplicateResult]:
        """Поиск дубликатов с использованием множественных алгоритмов"""
        logger.info("Starting advanced duplicate detection...")
        
        results = []
        processed_indices = set()
        
        # Получение названий
        names = df[name_column].fillna('').astype(str).tolist()
        
        # TF-IDF векторизация для семантического сравнения
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(names)
            tfidf_similarity = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            tfidf_similarity = None
        
        # Поиск дубликатов с прогресс-баром
        for idx in tqdm(range(len(names)), desc="Detecting duplicates"):
            if idx in processed_indices:
                continue
                
            current_name = names[idx]
            if not current_name or current_name.strip() == '':
                continue
            
            duplicates = []
            duplicate_indices = []
            similarity_scores = []
            duplicate_types = []
            
            # Поиск дубликатов среди оставшихся элементов
            for compare_idx in range(idx + 1, len(names)):
                if compare_idx in processed_indices:
                    continue
                    
                compare_name = names[compare_idx]
                if not compare_name or compare_name.strip() == '':
                    continue
                
                # Множественные проверки дубликатов
                duplicate_type, similarity = self._check_duplicate(
                    current_name, compare_name, idx, compare_idx, tfidf_similarity
                )
                
                if duplicate_type and similarity >= self.config.fuzzy_match_threshold:
                    duplicates.append(compare_name)
                    duplicate_indices.append(compare_idx)
                    similarity_scores.append(similarity)
                    duplicate_types.append(duplicate_type)
            
            if duplicates:
                # Определение основного типа дубликатов в группе
                main_duplicate_type = self._determine_main_duplicate_type(duplicate_types)
                
                result = DuplicateResult(
                    main_index=idx,
                    main_name=current_name,
                    duplicate_indices=duplicate_indices,
                    duplicate_names=duplicates,
                    similarity_scores=similarity_scores,
                    duplicate_type=main_duplicate_type
                )
                results.append(result)
                
                # Отметка обработанных индексов
                processed_indices.add(idx)
                processed_indices.update(duplicate_indices)
        
        logger.info(f"Found {len(results)} duplicate groups")
        return results
    
    def _check_duplicate(self, name1: str, name2: str, idx1: int, idx2: int, 
                        tfidf_similarity: Optional[np.ndarray] = None) -> Tuple[Optional[str], float]:
        """УЛУЧШЕННАЯ проверка дубликатов с многоуровневым сравнением"""
        
        # Уровень 1: Точное совпадение
        if name1.lower().strip() == name2.lower().strip():
            return "exact", 1.0
        
        # Уровень 2: Проверка ключевых параметров (ГОСТ, размеры, бренды)
        key_param_similarity = self._check_key_parameters_similarity(name1, name2)
        if key_param_similarity >= 0.9:
            return "parameter_exact", key_param_similarity
        
        # Уровень 3: Комбинированная оценка схожести
        similarity_scores = {}
        
        # 3.1 Нечеткое сравнение с RapidFuzz (улучшенное)
        fuzzy_ratio = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_partial = fuzz.partial_ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_token_sort = fuzz.token_sort_ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_token_set = fuzz.token_set_ratio(name1.lower(), name2.lower()) / 100.0
        
        # Взвешенная комбинация нечетких метрик
        fuzzy_score = (
            fuzzy_ratio * 0.3 + 
            fuzzy_partial * 0.2 + 
            fuzzy_token_sort * 0.3 + 
            fuzzy_token_set * 0.2
        )
        similarity_scores['fuzzy'] = fuzzy_score
        
        # 3.2 TF-IDF семантическое сравнение
        semantic_score = 0.0
        if tfidf_similarity is not None and self.config.enable_semantic_check:
            try:
                semantic_score = tfidf_similarity[idx1, idx2]
            except IndexError:
                pass
        similarity_scores['semantic'] = semantic_score
        
        # 3.3 Параметрическое сравнение
        param_similarity = self._check_parameter_similarity(name1, name2)
        similarity_scores['parameter'] = param_similarity
        
        # 3.4 Брендовое сравнение
        brand_similarity = self._check_brand_similarity(name1, name2)
        similarity_scores['brand'] = brand_similarity
        
        # Уровень 4: Взвешенная комбинация всех метрик
        weighted_score = (
            self.config.fuzzy_weight * similarity_scores['fuzzy'] +
            self.config.semantic_weight * similarity_scores['semantic'] +
            self.config.parameter_weight * similarity_scores['parameter'] +
            self.config.brand_weight * similarity_scores['brand']
        )
        
        # Нормализация весов
        total_weight = (self.config.fuzzy_weight + self.config.semantic_weight + 
                       self.config.parameter_weight + self.config.brand_weight)
        weighted_score /= total_weight
        
        # Уровень 5: Определение типа дубликата на основе комбинированной оценки
        if weighted_score >= self.config.fuzzy_match_threshold:
            # Определяем доминирующий тип схожести
            dominant_type = max(similarity_scores.items(), key=lambda x: x[1])[0]
            return f"hybrid_{dominant_type}", weighted_score
        elif weighted_score >= self.config.parameter_similarity_threshold:
            return "parameter_based", weighted_score
        elif weighted_score >= self.config.semantic_similarity_threshold:
            return "semantic", weighted_score
        
        return None, 0.0
    
    def _check_key_parameters_similarity(self, name1: str, name2: str) -> float:
        """Проверка схожести ключевых параметров (ГОСТ, размеры, бренды)"""
        # Извлечение ГОСТов
        gost1 = re.findall(r'ГОСТ\s*\d+[-\d]*', name1.upper())
        gost2 = re.findall(r'ГОСТ\s*\d+[-\d]*', name2.upper())
        
        # Извлечение размеров (М10х50, 25т, 100W и т.д.)
        sizes1 = re.findall(r'[Мм]\d+[хx]\d+|\d+[тt]|\d+[Ww]|\d+[Аа]', name1)
        sizes2 = re.findall(r'[Мм]\d+[хx]\d+|\d+[тt]|\d+[Ww]|\d+[Аа]', name2)
        
        # Извлечение брендов
        brands1 = re.findall(r'\b[A-Za-zА-Яа-я]{3,}\b', name1)
        brands2 = re.findall(r'\b[A-Za-zА-Яа-я]{3,}\b', name2)
        
        # Проверка совпадений
        gost_match = len(set(gost1).intersection(set(gost2))) > 0 if gost1 and gost2 else False
        size_match = len(set(sizes1).intersection(set(sizes2))) > 0 if sizes1 and sizes2 else False
        brand_match = len(set(brands1).intersection(set(brands2))) > 0 if brands1 and brands2 else False
        
        # Взвешенная оценка
        score = 0.0
        if gost_match:
            score += 0.5  # ГОСТ - самый важный параметр
        if size_match:
            score += 0.3  # Размеры - важный параметр
        if brand_match:
            score += 0.2  # Бренд - дополнительный параметр
        
        return score
    
    def _check_brand_similarity(self, name1: str, name2: str) -> float:
        """Проверка схожести брендов"""
        # Список известных брендов
        known_brands = ['neox', 'osairous', 'yealink', 'sanfor', 'санфор', 'биолан', 'нэфис', 
                       'персил', 'dallas', 'премиум', 'маяк', 'chint', 'andeli', 'grass', 
                       'kraft', 'reoflex', 'керхер', 'huawei', 'honor', 'ВЫСОТА', 'ugreen', 
                       'alisafox', 'маякавто', 'техноавиа', 'восток-сервис', 'attache', 'камаз', 
                       'зубр', 'hp', 'ekf', 'dexp', 'matrix', 'siemens', 'комус', 'gigant', 
                       'hyundai', 'iveco', 'stayer', 'brauberg', 'makita', 'bentec', 'сибртех', 
                       'bosch', 'rexant', 'sampa', 'kyocera', 'avrora', 'derrick', 'cummins', 
                       'economy', 'samsung', 'ofite', 'professional', 'caterpillar', 'intel', 
                       'proxima', 'core', 'shantui', 'king', 'office', 'петролеум', 'трейл', 
                       'skf', 'форвелд', 'скаймастер', 'tony', 'kentek', 'ресанта', 'dexter', 
                       'electric', 'оттм']
        
        # Поиск брендов в названиях
        brands1 = [brand for brand in known_brands if brand.lower() in name1.lower()]
        brands2 = [brand for brand in known_brands if brand.lower() in name2.lower()]
        
        if not brands1 or not brands2:
            return 0.0
        
        # Проверка совпадений брендов
        common_brands = set(brands1).intersection(set(brands2))
        if common_brands:
            return 1.0  # Точное совпадение бренда
        
        return 0.0
    
    def _check_parameter_similarity(self, name1: str, name2: str) -> float:
        """УЛУЧШЕННАЯ проверка схожести на основе параметров"""
        # Извлечение числовых параметров с контекстом
        param_patterns = [
            r'(\d+)\s*[хx]\s*(\d+)',  # Размеры М10х50
            r'(\d+)\s*[тt]',          # Тоннаж 25т
            r'(\d+)\s*[Ww]',          # Мощность 100W
            r'(\d+)\s*[Аа]',          # Ток 16А
            r'(\d+)\s*[Vv]',          # Напряжение 24V
            r'(\d+)\s*[кг]',          # Вес 10кг
            r'(\d+)\s*[л]',           # Объем 1л
            r'(\d+)\s*[мм]',          # Размеры 150мм
        ]
        
        params1 = []
        params2 = []
        
        for pattern in param_patterns:
            params1.extend(re.findall(pattern, name1))
            params2.extend(re.findall(pattern, name2))
        
        # Преобразование в множества для сравнения
        params1_set = set(str(param) for param in params1)
        params2_set = set(str(param) for param in params2)
        
        if not params1_set or not params2_set:
            return 0.0
        
        # Расчет схожести параметров
        common_params = params1_set.intersection(params2_set)
        total_params = params1_set.union(params2_set)
        
        if not total_params:
            return 0.0
        
        return len(common_params) / len(total_params)
    
    def _determine_main_duplicate_type(self, types: List[str]) -> str:
        """Определение основного типа дубликатов в группе"""
        if not types:
            return "unknown"
        
        # Приоритет типов
        type_priority = {"exact": 3, "fuzzy": 2, "parameter_based": 1, "semantic": 1}
        
        # Выбор типа с наивысшим приоритетом
        return max(types, key=lambda t: type_priority.get(t, 0))

class AdvancedAnalogFinder:
    """Продвинутый поисковик аналогов"""
    
    def __init__(self, config: AnalogSearchConfig):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=None,
            lowercase=True
        )
    
    def find_analogs(self, df: pd.DataFrame, name_column: str = 'processed_name',
                    category_column: str = 'category', brand_column: str = 'model_brand') -> List[AnalogResult]:
        """Поиск аналогов с использованием гибридного подхода"""
        logger.info("Starting advanced analog detection...")
        
        results = []
        names = df[name_column].fillna('').astype(str).tolist()
        categories = df[category_column].fillna('').astype(str).tolist()
        brands = df[brand_column].fillna('').astype(str).tolist()
        
        # TF-IDF векторизация
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(names)
            tfidf_similarity = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            tfidf_similarity = None
        
        # Поиск аналогов с прогресс-баром
        for idx in tqdm(range(len(names)), desc="Finding analogs"):
            reference_name = names[idx]
            if not reference_name or reference_name.strip() == '':
                continue
            
            analogs = []
            
            # Поиск аналогов среди других товаров
            for compare_idx in range(len(names)):
                if compare_idx == idx:
                    continue
                    
                compare_name = names[compare_idx]
                if not compare_name or compare_name.strip() == '':
                    continue
                
                # Гибридная оценка схожести
                similarity, confidence = self._calculate_hybrid_similarity(
                    reference_name, compare_name,
                    categories[idx], categories[compare_idx],
                    brands[idx], brands[compare_idx],
                    idx, compare_idx, tfidf_similarity
                )
                
                if similarity >= self.config.possible_analog_threshold:
                    analog_type = self._determine_analog_type(similarity)
                    
                    analogs.append({
                        'index': compare_idx,
                        'name': compare_name,
                        'similarity': similarity,
                        'type': analog_type,
                        'confidence': confidence
                    })
            
            if analogs:
                # Сортировка по схожести и ограничение количества
                analogs.sort(key=lambda x: x['similarity'], reverse=True)
                analogs = analogs[:self.config.max_analogs_per_item]
                
                result = AnalogResult(
                    reference_index=idx,
                    reference_name=reference_name,
                    analogs=analogs
                )
                results.append(result)
        
        logger.info(f"Found {len(results)} analog groups")
        return results
    
    def _calculate_hybrid_similarity(self, name1: str, name2: str,
                                   category1: str, category2: str,
                                   brand1: str, brand2: str,
                                   idx1: int, idx2: int,
                                   tfidf_similarity: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """УЛУЧШЕННЫЙ расчет гибридной схожести с приоритизацией параметров"""
        
        # 1. Семантическая схожесть (TF-IDF)
        semantic_similarity = 0.0
        if tfidf_similarity is not None:
            try:
                semantic_similarity = tfidf_similarity[idx1, idx2]
            except IndexError:
                pass
        
        # 2. Улучшенная нечеткая схожесть (RapidFuzz)
        fuzzy_ratio = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_partial = fuzz.partial_ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_token_sort = fuzz.token_sort_ratio(name1.lower(), name2.lower()) / 100.0
        fuzzy_token_set = fuzz.token_set_ratio(name1.lower(), name2.lower()) / 100.0
        
        # Взвешенная комбинация нечетких метрик
        fuzzy_similarity = (
            fuzzy_ratio * 0.25 + 
            fuzzy_partial * 0.25 + 
            fuzzy_token_sort * 0.25 + 
            fuzzy_token_set * 0.25
        )
        
        # 3. Улучшенная схожесть категорий
        category_similarity = self._calculate_category_similarity(category1, category2)
        
        # 4. Улучшенная схожесть брендов
        brand_similarity = self._calculate_brand_similarity_advanced(brand1, brand2, name1, name2)
        
        # 5. Улучшенная параметрическая схожесть
        param_similarity = self._calculate_parameter_similarity_advanced(name1, name2)
        
        # 6. Функциональная схожесть (новый параметр)
        functional_similarity = self._calculate_functional_similarity(name1, name2)
        
        # Взвешенная комбинация с учетом приоритетов
        if self.config.enable_parameter_priority:
            # Приоритет параметрам для технических товаров
            if self._is_technical_item(name1) and self._is_technical_item(name2):
                weights = {
                    'semantic': self.config.semantic_weight * 0.8,
                    'fuzzy': self.config.fuzzy_weight * 0.8,
                    'parameter': self.config.parameter_weight * 1.5,  # Увеличенный вес
                    'brand': self.config.brand_weight * 1.2,
                    'category': self.config.category_weight * 1.1,
                    'functional': 0.15  # Новый вес
                }
            else:
                weights = {
                    'semantic': self.config.semantic_weight,
                    'fuzzy': self.config.fuzzy_weight,
                    'parameter': self.config.parameter_weight,
                    'brand': self.config.brand_weight,
                    'category': self.config.category_weight,
                    'functional': 0.1
                }
        else:
            weights = {
                'semantic': self.config.semantic_weight,
                'fuzzy': self.config.fuzzy_weight,
                'parameter': self.config.parameter_weight,
                'brand': self.config.brand_weight,
                'category': self.config.category_weight,
                'functional': 0.1
            }
        
        # Взвешенная комбинация
        total_similarity = (
            weights['semantic'] * semantic_similarity +
            weights['fuzzy'] * fuzzy_similarity +
            weights['parameter'] * param_similarity +
            weights['brand'] * brand_similarity +
            weights['category'] * category_similarity +
            weights['functional'] * functional_similarity
        )
        
        # Нормализация
        total_weight = sum(weights.values())
        total_similarity /= total_weight
        
        # Расчет уверенности с учетом качества каждого компонента
        confidence = self._calculate_confidence_advanced(
            semantic_similarity, fuzzy_similarity, param_similarity,
            brand_similarity, category_similarity, functional_similarity
        )
        
        return total_similarity, confidence
    
    def _calculate_category_similarity(self, category1: str, category2: str) -> float:
        """Улучшенная схожесть категорий"""
        if not category1 or not category2:
            return 0.0
        
        if category1 == category2:
            return 1.0
        
        # Схожие категории
        similar_categories = {
            'электрика': ['освещение', 'электротехника', 'электрооборудование'],
            'металлопрокат': ['металл', 'сталь', 'железо'],
            'сантехника': ['водоснабжение', 'отопление', 'канализация'],
            'инструменты': ['инструмент', 'оборудование'],
            'химия': ['химические_вещества', 'растворители', 'краски'],
            'крепеж': ['метизы', 'болты', 'гайки', 'шайбы'],
            'общие_товары': ['прочее', 'разное', 'прочие_товары']
        }
        
        # Проверка на схожие категории
        for main_cat, similar_cats in similar_categories.items():
            if (category1 == main_cat and category2 in similar_cats) or \
               (category2 == main_cat and category1 in similar_cats):
                return 0.7
        
        return 0.0
    
    def _calculate_brand_similarity_advanced(self, brand1: str, brand2: str, name1: str, name2: str) -> float:
        """Улучшенная схожесть брендов"""
        # Прямое сравнение брендов
        if brand1 and brand2 and brand1 == brand2:
            return 1.0
        
        # Поиск брендов в названиях
        known_brands = ['neox', 'osairous', 'yealink', 'sanfor', 'санфор', 'биолан', 'нэфис', 
                       'персил', 'dallas', 'премиум', 'маяк', 'chint', 'andeli', 'grass', 
                       'kraft', 'reoflex', 'керхер', 'huawei', 'honor', 'ВЫСОТА', 'ugreen', 
                       'alisafox', 'маякавто', 'техноавиа', 'восток-сервис', 'attache', 'камаз', 
                       'зубр', 'hp', 'ekf', 'dexp', 'matrix', 'siemens', 'комус', 'gigant', 
                       'hyundai', 'iveco', 'stayer', 'brauberg', 'makita', 'bentec', 'сибртех', 
                       'bosch', 'rexant', 'sampa', 'kyocera', 'avrora', 'derrick', 'cummins', 
                       'economy', 'samsung', 'ofite', 'professional', 'caterpillar', 'intel', 
                       'proxima', 'core', 'shantui', 'king', 'office', 'петролеум', 'трейл', 
                       'skf', 'форвелд', 'скаймастер', 'tony', 'kentek', 'ресанта', 'dexter', 
                       'electric', 'оттм']
        
        brands1 = [brand for brand in known_brands if brand.lower() in name1.lower()]
        brands2 = [brand for brand in known_brands if brand.lower() in name2.lower()]
        
        if not brands1 or not brands2:
            return 0.0
        
        # Проверка совпадений брендов
        common_brands = set(brands1).intersection(set(brands2))
        if common_brands:
            return 1.0
        
        # Проверка на схожие бренды (например, разные модели одного производителя)
        brand_groups = {
            'hp': ['hewlett', 'packard'],
            'samsung': ['samsung', 'galaxy'],
            'huawei': ['huawei', 'honor'],
            'bosch': ['bosch', 'blue'],
            'makita': ['makita', 'lxt']
        }
        
        for group_name, group_brands in brand_groups.items():
            if any(brand in brands1 for brand in group_brands) and \
               any(brand in brands2 for brand in group_brands):
                return 0.8
        
        return 0.0
    
    def _calculate_parameter_similarity_advanced(self, name1: str, name2: str) -> float:
        """Улучшенная схожесть параметров с приоритизацией"""
        # Извлечение параметров с контекстом
        param_patterns = [
            (r'(\d+)\s*[хx]\s*(\d+)', 0.3),  # Размеры М10х50
            (r'(\d+)\s*[тt]', 0.25),         # Тоннаж 25т
            (r'(\d+)\s*[Ww]', 0.2),          # Мощность 100W
            (r'(\d+)\s*[Аа]', 0.15),         # Ток 16А
            (r'(\d+)\s*[Vv]', 0.1),          # Напряжение 24V
        ]
        
        params1 = {}
        params2 = {}
        
        for pattern, weight in param_patterns:
            matches1 = re.findall(pattern, name1)
            matches2 = re.findall(pattern, name2)
            
            if matches1:
                params1[pattern] = (matches1, weight)
            if matches2:
                params2[pattern] = (matches2, weight)
        
        if not params1 or not params2:
            return 0.0
        
        # Взвешенное сравнение параметров
        total_score = 0.0
        total_weight = 0.0
        
        for pattern in params1:
            if pattern in params2:
                matches1, weight1 = params1[pattern]
                matches2, weight2 = params2[pattern]
                
                # Сравнение совпадающих параметров
                common_matches = set(str(m) for m in matches1).intersection(set(str(m) for m in matches2))
                total_matches = set(str(m) for m in matches1).union(set(str(m) for m in matches2))
                
                if total_matches:
                    similarity = len(common_matches) / len(total_matches)
                    weight = (weight1 + weight2) / 2
                    total_score += similarity * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_functional_similarity(self, name1: str, name2: str) -> float:
        """Расчет функциональной схожести"""
        # Функциональные группы
        functional_groups = {
            'грузоподъемные': ['кран', 'подъемник', 'тельфер', 'лебедка'],
            'транспортные': ['тягач', 'трал', 'автомобиль', 'грузовик'],
            'электротехнические': ['автомат', 'выключатель', 'розетка', 'кабель'],
            'инструментальные': ['отвертка', 'ключ', 'дрель', 'шуруповерт'],
            'химические': ['краска', 'растворитель', 'клей', 'герметик'],
            'металлические': ['болт', 'гайка', 'шайба', 'швеллер', 'уголок'],
            'осветительные': ['лампа', 'светильник', 'фонарь', 'прожектор']
        }
        
        # Определение функциональных групп для каждого названия
        groups1 = set()
        groups2 = set()
        
        for group_name, keywords in functional_groups.items():
            if any(keyword in name1.lower() for keyword in keywords):
                groups1.add(group_name)
            if any(keyword in name2.lower() for keyword in keywords):
                groups2.add(group_name)
        
        if not groups1 or not groups2:
            return 0.0
        
        # Схожесть функциональных групп
        common_groups = groups1.intersection(groups2)
        total_groups = groups1.union(groups2)
        
        return len(common_groups) / len(total_groups)
    
    def _is_technical_item(self, name: str) -> bool:
        """Определение технического товара"""
        technical_keywords = ['автомат', 'выключатель', 'кран', 'насос', 'двигатель', 
                            'трансформатор', 'датчик', 'контроллер', 'адаптер', 'аккумулятор']
        return any(keyword in name.lower() for keyword in technical_keywords)
    
    def _calculate_confidence_advanced(self, semantic: float, fuzzy: float, param: float,
                                     brand: float, category: float, functional: float) -> float:
        """Улучшенный расчет уверенности"""
        # Взвешенная уверенность с учетом качества каждого компонента
        weights = [0.2, 0.2, 0.25, 0.15, 0.1, 0.1]  # Параметры имеют больший вес
        scores = [semantic, fuzzy, param, brand, category, functional]
        
        # Расчет взвешенной средней
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, semantic: float, fuzzy: float, 
                            category: float, brand: float, param: float) -> float:
        """Расчет уверенности в результате"""
        # Простая средняя уверенность
        return (semantic + fuzzy + category + brand + param) / 5.0
    
    def _determine_analog_type(self, similarity: float) -> str:
        """УЛУЧШЕННОЕ определение типа аналога"""
        if similarity >= self.config.exact_analog_threshold:
            return "точный аналог"
        elif similarity >= self.config.close_analog_threshold:
            return "близкий аналог"
        elif similarity >= self.config.possible_analog_threshold:
            return "возможный аналог"
        else:
            return "нет аналогов"

class EnhancedTreeBuilder:
    """УЛУЧШЕННЫЙ построитель деревьев товаров с иерархической структурой"""
    
    def __init__(self):
        self.max_tree_depth = 8  # Увеличена глубина для более глубоких деревьев
        self.min_group_size = 2
        self.enable_hierarchical_grouping = True
        self.enable_functional_grouping = True
    
    def build_trees(self, duplicate_results: List[DuplicateResult], 
                   analog_results: List[AnalogResult]) -> List[Dict[str, Any]]:
        """УЛУЧШЕННОЕ построение деревьев товаров с иерархической структурой"""
        logger.info("Building enhanced product trees...")
        
        trees = []
        
        # Группировка по функциональным категориям
        if self.enable_functional_grouping:
            functional_groups = self._group_by_functional_category(analog_results)
        else:
            functional_groups = {'all': analog_results}
        
        # Построение деревьев для каждой функциональной группы
        for group_name, group_results in functional_groups.items():
            group_trees = self._build_group_trees(group_results, duplicate_results)
            trees.extend(group_trees)
        
        logger.info(f"Built {len(trees)} enhanced product trees")
        return trees
    
    def _group_by_functional_category(self, analog_results: List[AnalogResult]) -> Dict[str, List[AnalogResult]]:
        """Группировка по функциональным категориям"""
        functional_groups = {
            'грузоподъемные': [],
            'транспортные': [],
            'электротехнические': [],
            'инструментальные': [],
            'химические': [],
            'металлические': [],
            'осветительные': [],
            'прочие': []
        }
        
        for result in analog_results:
            name = result.reference_name.lower()
            assigned = False
            
            # Определение функциональной группы
            if any(keyword in name for keyword in ['кран', 'подъемник', 'тельфер', 'лебедка']):
                functional_groups['грузоподъемные'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['тягач', 'трал', 'автомобиль', 'грузовик']):
                functional_groups['транспортные'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['автомат', 'выключатель', 'розетка', 'кабель']):
                functional_groups['электротехнические'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['отвертка', 'ключ', 'дрель', 'шуруповерт']):
                functional_groups['инструментальные'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['краска', 'растворитель', 'клей', 'герметик']):
                functional_groups['химические'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['болт', 'гайка', 'шайба', 'швеллер', 'уголок']):
                functional_groups['металлические'].append(result)
                assigned = True
            elif any(keyword in name for keyword in ['лампа', 'светильник', 'фонарь', 'прожектор']):
                functional_groups['осветительные'].append(result)
                assigned = True
            
            if not assigned:
                functional_groups['прочие'].append(result)
        
        # Удаление пустых групп
        return {k: v for k, v in functional_groups.items() if v}
    
    def _build_group_trees(self, group_results: List[AnalogResult], 
                          duplicate_results: List[DuplicateResult]) -> List[Dict[str, Any]]:
        """Построение деревьев для функциональной группы"""
        trees = []
        
        for analog_result in tqdm(group_results, desc="Building group trees"):
            tree = {
                'root_index': analog_result.reference_index,
                'root_name': analog_result.reference_name,
                'functional_group': self._determine_functional_group(analog_result.reference_name),
                'duplicates': [],
                'exact_analogs': [],
                'close_analogs': [],
                'possible_analogs': [],
                'children': [],
                'hierarchy_level': 1
            }
            
            # Распределение аналогов по типам с сортировкой по схожести
            analogs_by_type = {
                "точный аналог": [],
                "близкий аналог": [],
                "возможный аналог": []
            }
            
            for analog in analog_result.analogs:
                analog_type = analog['type']
                if analog_type in analogs_by_type:
                    analogs_by_type[analog_type].append(analog)
            
            # Сортировка по схожести внутри каждого типа
            for analog_type in analogs_by_type:
                analogs_by_type[analog_type].sort(key=lambda x: x['similarity'], reverse=True)
            
            tree['exact_analogs'] = analogs_by_type["точный аналог"]
            tree['close_analogs'] = analogs_by_type["близкий аналог"]
            tree['possible_analogs'] = analogs_by_type["возможный аналог"]
            
            # Добавление дубликатов (если есть)
            for dup_result in duplicate_results:
                if dup_result.main_index == analog_result.reference_index:
                    for i, dup_idx in enumerate(dup_result.duplicate_indices):
                        tree['duplicates'].append({
                            'index': dup_idx,
                            'name': dup_result.duplicate_names[i],
                            'similarity': dup_result.similarity_scores[i],
                            'type': 'дубль'
                        })
                    break
            
            # Построение иерархии (если включено)
            if self.enable_hierarchical_grouping:
                tree['children'] = self._build_hierarchy_levels(
                    tree['exact_analogs'] + tree['close_analogs'] + tree['possible_analogs']
                )
            
            trees.append(tree)
        
        return trees
    
    def _determine_functional_group(self, name: str) -> str:
        """Определение функциональной группы товара"""
        name_lower = name.lower()
        
        if any(keyword in name_lower for keyword in ['кран', 'подъемник', 'тельфер', 'лебедка']):
            return 'грузоподъемные'
        elif any(keyword in name_lower for keyword in ['тягач', 'трал', 'автомобиль', 'грузовик']):
            return 'транспортные'
        elif any(keyword in name_lower for keyword in ['автомат', 'выключатель', 'розетка', 'кабель']):
            return 'электротехнические'
        elif any(keyword in name_lower for keyword in ['отвертка', 'ключ', 'дрель', 'шуруповерт']):
            return 'инструментальные'
        elif any(keyword in name_lower for keyword in ['краска', 'растворитель', 'клей', 'герметик']):
            return 'химические'
        elif any(keyword in name_lower for keyword in ['болт', 'гайка', 'шайба', 'швеллер', 'уголок']):
            return 'металлические'
        elif any(keyword in name_lower for keyword in ['лампа', 'светильник', 'фонарь', 'прожектор']):
            return 'осветительные'
        else:
            return 'прочие'
    
    def _build_hierarchy_levels(self, analogs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Построение иерархических уровней"""
        if not analogs:
            return []
        
        # Группировка по схожести для создания уровней
        hierarchy = []
        current_level = []
        current_similarity = analogs[0]['similarity'] if analogs else 0
        
        for analog in analogs:
            # Если схожесть значительно отличается, создаем новый уровень
            if abs(analog['similarity'] - current_similarity) > 0.1:
                if current_level:
                    hierarchy.append({
                        'level': len(hierarchy) + 2,
                        'similarity_range': f"{current_similarity:.2f}-{current_level[0]['similarity']:.2f}",
                        'items': current_level
                    })
                current_level = [analog]
                current_similarity = analog['similarity']
            else:
                current_level.append(analog)
        
        # Добавление последнего уровня
        if current_level:
            hierarchy.append({
                'level': len(hierarchy) + 2,
                'similarity_range': f"{current_similarity:.2f}-{current_level[0]['similarity']:.2f}",
                'items': current_level
            })
        
        return hierarchy
    
    def export_trees_to_text(self, trees: List[Dict[str, Any]], 
                           df: pd.DataFrame, output_file: str):
        """Экспорт деревьев в текстовый файл"""
        logger.info(f"Exporting trees to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for tree in trees:
                # Корень дерева
                root_name = df.iloc[tree['root_index']].iloc[0]  # Первая колонка с названием
                f.write(f"- {tree['root_index']} | {root_name} (None)\n")
                
                # Дубликаты
                for dup in tree['duplicates']:
                    dup_name = df.iloc[dup['index']].iloc[0]
                    f.write(f"    - {dup['index']} | {dup_name} [дубль] ({dup['similarity']:.4f})\n")
                
                # Точные аналоги
                for analog in tree['exact_analogs']:
                    analog_name = df.iloc[analog['index']].iloc[0]
                    f.write(f"    - {analog['index']} | {analog_name} [аналог] ({analog['similarity']:.4f})\n")
                
                # Близкие аналоги
                for analog in tree['close_analogs']:
                    analog_name = df.iloc[analog['index']].iloc[0]
                    f.write(f"    - {analog['index']} | {analog_name} [близкий аналог] ({analog['similarity']:.4f})\n")
                
                # Возможные аналоги
                for analog in tree['possible_analogs']:
                    analog_name = df.iloc[analog['index']].iloc[0]
                    f.write(f"    - {analog['index']} | {analog_name} [возможный аналог] ({analog['similarity']:.4f})\n")
                
                f.write("\n")
        
        logger.info(f"Trees exported to {output_file}")

class DuplicateAnalogSearchEngine:
    """Главный движок поиска дубликатов и аналогов"""
    
    def __init__(self, duplicate_config: DuplicateSearchConfig = None, 
                 analog_config: AnalogSearchConfig = None):
        self.duplicate_config = duplicate_config or DuplicateSearchConfig()
        self.analog_config = analog_config or AnalogSearchConfig()
        
        self.duplicate_detector = AdvancedDuplicateDetector(self.duplicate_config)
        self.analog_finder = AdvancedAnalogFinder(self.analog_config)
        self.tree_builder = EnhancedTreeBuilder()
        
        logger.info("DuplicateAnalogSearchEngine initialized")
    
    async def process_catalog(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Полная обработка каталога"""
        logger.info("Starting catalog processing...")
        
        results = {
            'duplicates': [],
            'analogs': [],
            'trees': [],
            'statistics': {}
        }
        
        # Поиск дубликатов
        duplicate_results = self.duplicate_detector.detect_duplicates(df)
        results['duplicates'] = duplicate_results
        
        # Поиск аналогов
        analog_results = self.analog_finder.find_analogs(df)
        results['analogs'] = analog_results
        
        # Построение деревьев
        trees = self.tree_builder.build_trees(duplicate_results, analog_results)
        results['trees'] = trees
        
        # Статистика
        results['statistics'] = self._calculate_statistics(duplicate_results, analog_results, trees)
        
        logger.info("Catalog processing completed")
        return results
    
    def _calculate_statistics(self, duplicates: List[DuplicateResult], 
                            analogs: List[AnalogResult], 
                            trees: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет статистики"""
        stats = {
            'total_duplicate_groups': len(duplicates),
            'total_analog_groups': len(analogs),
            'total_trees': len(trees),
            'total_duplicates': sum(len(d.duplicate_indices) for d in duplicates),
            'total_analogs': sum(len(a.analogs) for a in analogs),
            'analog_types': {}
        }
        
        # Статистика по типам аналогов
        for analog_result in analogs:
            for analog in analog_result.analogs:
                analog_type = analog['type']
                stats['analog_types'][analog_type] = stats['analog_types'].get(analog_type, 0) + 1
        
        return stats

