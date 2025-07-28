"""
Модуль классификации товаров по категориям для улучшения поиска аналогов
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CategoryClassifierConfig:
    """Конфигурация классификатора категорий"""
    use_keyword_matching: bool = True
    use_pattern_matching: bool = True
    min_confidence: float = 0.6
    default_category: str = "общие_товары"
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8


class CategoryClassifier:
    """Классификатор товаров по категориям на основе ключевых слов и паттернов"""
    
    def __init__(self, config: CategoryClassifierConfig = None):
        self.config = config or CategoryClassifierConfig()
        self._setup_category_rules()
        
        logger.info("CategoryClassifier initialized")
    
    def _setup_category_rules(self):
        """Настройка правил категоризации с данными из каталога"""

        # Загружаем улучшенные ключевые слова из каталога
        try:
            from enhanced_classifier_config import ENHANCED_CATEGORY_KEYWORDS, CATEGORY_EXCEPTIONS, CATEGORY_PRIORITY
            enhanced_keywords = ENHANCED_CATEGORY_KEYWORDS
            self.category_exceptions = CATEGORY_EXCEPTIONS
            self.category_priority = CATEGORY_PRIORITY
        except ImportError:
            logger.warning("Enhanced classifier config not found, using default keywords")
            enhanced_keywords = {}
            self.category_exceptions = {}
            self.category_priority = {}

        # Базовые категории и их ключевые слова (объединяем с данными каталога)
        base_keywords = {
            "средства_защиты": {
                "ледоход", "ледоходы", "шипы", "антискольжение", "противоскользящие",
                "каска", "шлем", "очки", "перчатки", "респиратор", "маска",
                "жилет", "костюм", "комбинезон", "сапоги", "ботинки", "защитные",
                "профессиональный", "рабочий", "защита", "безопасность"
            },

            "крепеж": {
                "болт", "гайка", "шуруп", "саморез", "винт", "заклепка",
                "дюбель", "анкер", "шпилька", "шайба", "гровер", "скоба",
                "крепежный", "метизы", "резьбовой", "оцинкованный"
            },

            "электрика": {
                "кабель", "провод", "розетка", "выключатель", "лампа", "светильник",
                "трансформатор", "автомат", "предохранитель", "реле", "контактор",
                "щит", "короб", "гофра", "изолятор", "электрический", "электротехнический"
            },

            "сантехника": {
                "труба", "фитинг", "кран", "вентиль", "задвижка", "клапан",
                "насос", "фильтр", "бойлер", "радиатор", "батарея", "унитаз",
                "раковина", "ванна", "душ", "смеситель", "сантехнический", "водопроводный"
            },

            "инструменты": {
                "отвертка", "ключ", "молоток", "пила", "дрель", "шуруповерт",
                "болгарка", "перфоратор", "лобзик", "рубанок", "стамеска",
                "плоскогубцы", "кусачки", "тиски", "струбцина", "инструмент", "набор"
            },

            "металлопрокат": {
                "швеллер", "балка", "уголок", "профиль", "труба", "лист",
                "полоса", "круг", "квадрат", "арматура", "сетка", "проволока",
                "металлический", "стальной", "железный", "алюминиевый"
            },

            "химия": {
                "растворитель", "сольвент", "ацетон", "спирт", "кислота",
                "щелочь", "реагент", "присадка", "антифриз", "масло",
                "смазка", "очиститель", "обезжириватель", "химический", "краска"
            },

            "текстиль": {
                "ткань", "полотно", "брезент", "тент", "полог", "пленка",
                "мешок", "рукав", "лента", "веревка", "канат", "трос",
                "текстильный", "хлопчатобумажный", "синтетический"
            }
        }

        # Объединяем базовые ключевые слова с данными из каталога
        self.category_keywords = {}
        for category in base_keywords:
            combined_keywords = set(base_keywords[category])
            if category in enhanced_keywords:
                combined_keywords.update(enhanced_keywords[category])
            self.category_keywords[category] = combined_keywords
        
        # Паттерны для более точной классификации
        self.category_patterns = {
            "средства_защиты": [
                r'\bледоход\w*\s+проф\w*',  # ледоход проф
                r'\bледоход\w*\s+\d+',      # ледоход 10
                r'\bшип\w*\s+\d+',          # шипы 10
            ],
            
            "металлопрокат": [
                r'\bшвеллер\s+\d+',         # швеллер 10
                r'\bуголок\s+\d+x\d+',      # уголок 50x50
                r'\bлист\s+\d+мм',          # лист 2мм
            ],
            
            "химия": [
                r'\bсольвент\s+\d+',        # сольвент 10
                r'\bрастворитель\s+\d+',    # растворитель 646
            ],
            
            "текстиль": [
                r'\bполог\s+\d+\s*x?\s*\d*', # полог 10x10
                r'\bтент\s+\d+\s*x?\s*\d*',  # тент 5x3
            ]
        }
        
        # Компилируем паттерны
        self.compiled_patterns = {}
        for category, patterns in self.category_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Классификация текста по категории с использованием данных каталога

        Args:
            text: Текст для классификации

        Returns:
            Tuple[category, confidence]
        """
        if not text or not isinstance(text, str):
            return self.config.default_category, 0.0

        text_lower = text.lower().strip()

        # Проверяем исключения для известных проблемных товаров
        if hasattr(self, 'category_exceptions'):
            for exception_text, exception_category in self.category_exceptions.items():
                if exception_text.lower() in text_lower:
                    logger.debug(f"Applied exception rule: '{text}' → {exception_category}")
                    return exception_category, 0.95  # Высокая уверенность для исключений

        category_scores = defaultdict(float)

        # Поиск по ключевым словам
        if self.config.use_keyword_matching:
            for category, keywords in self.category_keywords.items():
                score = self._calculate_keyword_score(text_lower, keywords)
                if score > 0:
                    category_scores[category] += score * 0.7  # Вес ключевых слов

        # Поиск по паттернам
        if self.config.use_pattern_matching:
            for category, patterns in self.compiled_patterns.items():
                score = self._calculate_pattern_score(text, patterns)
                if score > 0:
                    category_scores[category] += score * 0.3  # Вес паттернов

        # Выбираем категорию с максимальным скором
        if category_scores:
            # Применяем приоритеты категорий при равных скорах
            best_candidates = []
            max_score = max(category_scores.values())

            for category, score in category_scores.items():
                if abs(score - max_score) < 0.1:  # Считаем равными если разница < 0.1
                    priority = self.category_priority.get(category, 999)
                    best_candidates.append((category, score, priority))

            # Сортируем по приоритету (меньше = выше приоритет)
            best_candidates.sort(key=lambda x: (x[2], -x[1]))

            if best_candidates:
                category, confidence, _ = best_candidates[0]

                if confidence >= self.config.min_confidence:
                    logger.debug(f"Classified '{text}' as {category} with confidence {confidence:.3f}")
                    return category, confidence

        logger.debug(f"No confident classification for '{text}', using default category")
        return self.config.default_category, 0.0
    
    def _calculate_keyword_score(self, text: str, keywords: Set[str]) -> float:
        """Вычисление скора на основе ключевых слов"""
        found_keywords = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in text:
                found_keywords += 1
        
        return found_keywords / total_keywords if total_keywords > 0 else 0.0
    
    def _calculate_pattern_score(self, text: str, patterns: List[re.Pattern]) -> float:
        """Вычисление скора на основе паттернов"""
        matched_patterns = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if pattern.search(text):
                matched_patterns += 1
        
        return matched_patterns / total_patterns if total_patterns > 0 else 0.0
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Пакетная классификация"""
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying text: {e}")
                results.append((self.config.default_category, 0.0))
        
        return results
    
    def get_category_stats(self, classifications: List[Tuple[str, float]]) -> Dict[str, int]:
        """Получение статистики по категориям"""
        stats = defaultdict(int)
        for category, confidence in classifications:
            if confidence >= self.config.min_confidence:
                stats[category] += 1
            else:
                stats[self.config.default_category] += 1
        
        return dict(stats)
    
    def validate_with_catalog_stats(self, text: str, predicted_category: str) -> float:
        """
        Валидация классификации на основе статистики каталога

        Args:
            text: Исходный текст
            predicted_category: Предсказанная категория

        Returns:
            Коэффициент уверенности (0.0-1.0)
        """
        # Базовая валидация - проверяем наличие ключевых слов категории
        if predicted_category in self.category_keywords:
            keywords = self.category_keywords[predicted_category]
            text_lower = text.lower()

            # Подсчитываем совпадения ключевых слов
            matches = sum(1 for keyword in keywords if keyword in text_lower)

            if matches > 0:
                # Высокая уверенность если найдены специфичные ключевые слова
                if predicted_category == "средства_защиты" and any(word in text_lower for word in ["ледоход", "шипы", "противоскользящ"]):
                    return 0.95
                elif predicted_category == "химия" and any(word in text_lower for word in ["сольвент", "растворитель"]):
                    return 0.90
                elif predicted_category == "металлопрокат" and any(word in text_lower for word in ["швеллер", "уголок", "балка"]):
                    return 0.90
                else:
                    return min(0.85, 0.5 + (matches / len(keywords)) * 0.5)

        return 0.3  # Низкая уверенность если нет совпадений

    def get_enhanced_classification(self, text: str) -> Dict[str, any]:
        """
        Расширенная классификация с дополнительной информацией

        Args:
            text: Текст для классификации

        Returns:
            Словарь с результатами классификации и метаданными
        """
        category, confidence = self.classify(text)
        validation_confidence = self.validate_with_catalog_stats(text, category)

        # Получаем альтернативные категории
        text_lower = text.lower()
        all_scores = defaultdict(float)

        for cat, keywords in self.category_keywords.items():
            score = self._calculate_keyword_score(text_lower, keywords)
            if score > 0:
                all_scores[cat] = score

        # Сортируем по убыванию скора
        sorted_alternatives = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'primary_category': category,
            'confidence': confidence,
            'validation_confidence': validation_confidence,
            'final_confidence': min(confidence, validation_confidence),
            'alternatives': sorted_alternatives[:3],  # Топ-3 альтернативы
            'used_exception': any(exc.lower() in text_lower for exc in getattr(self, 'category_exceptions', {})),
            'matched_keywords': self._get_matched_keywords(text_lower, category)
        }

    def _get_matched_keywords(self, text_lower: str, category: str) -> List[str]:
        """Получение списка совпавших ключевых слов"""
        if category not in self.category_keywords:
            return []

        matched = []
        for keyword in self.category_keywords[category]:
            if keyword in text_lower:
                matched.append(keyword)

        return matched

    def add_category_keywords(self, category: str, keywords: Set[str]):
        """Добавление ключевых слов для категории"""
        if category not in self.category_keywords:
            self.category_keywords[category] = set()
        
        self.category_keywords[category].update(keywords)
        logger.info(f"Added {len(keywords)} keywords to category '{category}'")
    
    def get_categories(self) -> List[str]:
        """Получение списка всех категорий"""
        return list(self.category_keywords.keys())
