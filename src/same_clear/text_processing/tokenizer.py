"""
Модуль токенизации текста для системы поиска аналогов

Этот модуль предоставляет различные методы токенизации текста:
- Простая токенизация по пробелам
- Токенизация с использованием регулярных выражений
- Токенизация с использованием SpaCy
- Токенизация с сохранением технических терминов
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Импорт векторизатора
try:
    from .token_vectorizer import TokenVectorizer, VectorizerConfig
except ImportError:
    TokenVectorizer = None
    VectorizerConfig = None

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Конфигурация токенизатора"""
    # Тип токенизатора
    tokenizer_type: str = "regex"  # simple, regex, spacy, technical
    
    # Настройки для regex токенизатора
    word_pattern: str = r'\b\w+\b'
    preserve_numbers: bool = True
    preserve_technical_codes: bool = True
    min_token_length: int = 1
    max_token_length: int = 50
    
    # Настройки для SpaCy токенизатора
    spacy_model: str = "ru_core_news_lg"
    preserve_entities: bool = True
    
    # Фильтрация токенов
    remove_stopwords: bool = False
    remove_punctuation: bool = True
    lowercase: bool = True
    
    # Технические термины для сохранения
    preserve_technical_terms: bool = True
    technical_patterns: List[str] = None
    
    # Производительность
    enable_caching: bool = True
    cache_size: int = 10000
    max_workers: int = 2

    # Векторизация токенов
    enable_vectorization: bool = False
    vectorization_type: str = "simple"  # simple, embedding, hybrid
    vocabulary_size: int = 50000
    embedding_dim: int = 128
    use_pretrained_embeddings: bool = False
    
    def __post_init__(self):
        if self.technical_patterns is None:
            self.technical_patterns = [
                # Составные технические коды (высокий приоритет)
                r'\b[А-ЯA-Z]\d+[-]\d+[-]\d+\b',    # А24-5-1, Н7-24-70
                r'\b[А-ЯA-Z]{2,}\s*\d+[-]\d+\b',   # ГОСТ 7798-70, ISO 9001-2015
                r'\b[А-ЯA-Z]+\d+[xх×]\d+[.,]?\d*\b', # М10х50, 3x2.5
                r'\b\d+[xх×]\d+[.,]\d+\b',         # 3x2.5, 10х20.5

                # Электрические характеристики
                r'\b\d+[Vв]\b',                    # 24V, 220В
                r'\b\d+[Wвт]\b',                   # 50W, 21Вт
                r'\b\d+[kк][Wвт]?\b',              # 6500k, 1кВт
                r'\b\d+[Гг][цц]\b',                # 50Гц, 60ГЦ

                # Стандарты и рейтинги
                r'\bIP\d+\b',                      # IP40, IP65
                r'\b[А-ЯA-Z]{2,}\d+[А-ЯA-Z]*\b',  # ГОСТ123, ISO9001, DIN456
                r'\bDN\d+\b',                      # DN50, DN100
                r'\bPN\d+\b',                      # PN16, PN25

                # Размеры и единицы
                r'\b[А-ЯA-Z]+\d+[А-ЯA-Z]*\b',     # М10, Н7
                r'\b\d+[А-ЯA-Z]+\d*\b',           # 10М, 220В (если не покрыто выше)
                r'\b\d+[.,]\d+\b',                # 10.5, 10,5
                r'\b\d+[°]\w*\b',                 # 90°, 45°C
                r'\b\d+мм\b',                     # 1200мм, 50мм
            ]


class Tokenizer:
    """Класс для токенизации текста с различными стратегиями"""
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self._nlp = None
        self._stopwords = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="tokenizer")

        # Кэш токенизации
        self._token_cache = {} if self.config.enable_caching else None

        # Векторизатор токенов
        self._vectorizer = None
        if self.config.enable_vectorization and TokenVectorizer is not None:
            vectorizer_config = VectorizerConfig(
                vectorization_type=self.config.vectorization_type,
                vocabulary_size=self.config.vocabulary_size,
                embedding_dim=self.config.embedding_dim,
                use_pretrained_embeddings=self.config.use_pretrained_embeddings
            )
            self._vectorizer = TokenVectorizer(vectorizer_config)
            logger.info(f"Token vectorizer initialized with type: {self.config.vectorization_type}")

        # Компиляция регулярных выражений
        self._compile_patterns()

        logger.info(f"Tokenizer initialized with type: {self.config.tokenizer_type}")
    
    def _compile_patterns(self):
        """Компиляция регулярных выражений для оптимизации"""
        self._word_pattern = re.compile(self.config.word_pattern, re.UNICODE)
        
        # Технические паттерны
        self._technical_patterns = []
        if self.config.preserve_technical_terms and self.config.technical_patterns:
            for pattern in self.config.technical_patterns:
                try:
                    self._technical_patterns.append(re.compile(pattern, re.UNICODE))
                except re.error as e:
                    logger.warning(f"Invalid technical pattern '{pattern}': {e}")
        
        # Паттерн для удаления пунктуации
        if self.config.remove_punctuation:
            self._punct_pattern = re.compile(r'[^\w\s]', re.UNICODE)
    
    async def _ensure_initialized(self):
        """Асинхронная инициализация SpaCy модели"""
        if self._initialized or self.config.tokenizer_type != "spacy":
            return
        
        try:
            # Импорт и загрузка SpaCy модели
            import spacy
            from same_search.models import get_model_manager
            
            model_manager = get_model_manager()
            self._nlp = await model_manager.get_spacy_model(self.config.spacy_model)
            
            # Инициализация стоп-слов
            if self.config.remove_stopwords:
                self._stopwords = set(self._nlp.Defaults.stop_words)
            
            self._initialized = True
            logger.info(f"SpaCy model '{self.config.spacy_model}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SpaCy model: {e}")
            # Fallback к regex токенизатору
            self.config.tokenizer_type = "regex"
            logger.warning("Falling back to regex tokenizer")
    
    def tokenize_text(self, text: str, include_vectors: bool = None) -> Dict[str, Any]:
        """
        Токенизация текста

        Args:
            text: Входной текст
            include_vectors: Включить векторизацию (по умолчанию из конфига)

        Returns:
            Dict с результатами токенизации
        """
        if not text or not isinstance(text, str):
            base_result = {
                'original': text if text is not None else '',
                'tokens': [],
                'token_count': 0,
                'unique_tokens': [],
                'unique_count': 0
            }

            # Добавляем пустую векторизацию если нужно
            if self._should_include_vectors(include_vectors):
                base_result['vectorization'] = self._get_empty_vectorization()

            return base_result
        
        # Проверяем кэш
        if self._token_cache is not None:
            cache_key = f"{text}_{self.config.tokenizer_type}_{include_vectors}"
            if cache_key in self._token_cache:
                return self._token_cache[cache_key]

        # Выбираем метод токенизации
        if self.config.tokenizer_type == "simple":
            result = self._tokenize_simple(text)
        elif self.config.tokenizer_type == "regex":
            result = self._tokenize_regex(text)
        elif self.config.tokenizer_type == "spacy":
            result = self._tokenize_spacy_sync(text)
        elif self.config.tokenizer_type == "technical":
            result = self._tokenize_technical(text)
        else:
            logger.warning(f"Unknown tokenizer type: {self.config.tokenizer_type}, using regex")
            result = self._tokenize_regex(text)

        # Добавляем векторизацию если нужно
        if self._should_include_vectors(include_vectors):
            result['vectorization'] = self._vectorize_tokens(result['tokens'])

        # Кэшируем результат
        if self._token_cache is not None:
            if len(self._token_cache) >= self.config.cache_size:
                # Простая стратегия очистки кэша - удаляем первый элемент
                self._token_cache.pop(next(iter(self._token_cache)))
            self._token_cache[cache_key] = result

        return result
    
    def _tokenize_simple(self, text: str) -> Dict[str, Any]:
        """Простая токенизация по пробелам"""
        tokens = text.split()
        
        # Фильтрация токенов
        filtered_tokens = self._filter_tokens(tokens)
        
        return self._create_result(text, filtered_tokens)
    
    def _tokenize_regex(self, text: str) -> Dict[str, Any]:
        """Токенизация с использованием регулярных выражений"""
        # Сначала извлекаем технические термины и запоминаем их позиции
        technical_tokens = set()  # Используем set для избежания дублирования
        technical_positions = []

        if self.config.preserve_technical_terms:
            for pattern in self._technical_patterns:
                for match in pattern.finditer(text):
                    technical_token = match.group()
                    if technical_token not in technical_tokens:
                        technical_tokens.add(technical_token)
                        technical_positions.append((match.start(), match.end(), technical_token))

        # Сортируем по позиции
        technical_positions.sort()

        # Основная токенизация - извлекаем все слова
        word_tokens = []
        for match in self._word_pattern.finditer(text):
            word_token = match.group()
            word_start, word_end = match.start(), match.end()

            # Проверяем, не является ли это слово частью технического термина
            is_part_of_technical = False
            for tech_start, tech_end, tech_token in technical_positions:
                if word_start >= tech_start and word_end <= tech_end:
                    is_part_of_technical = True
                    break

            if not is_part_of_technical:
                word_tokens.append(word_token)

        # Объединяем обычные токены и технические термины
        all_tokens = word_tokens + list(technical_tokens)

        # Фильтрация токенов
        filtered_tokens = self._filter_tokens(all_tokens)

        return self._create_result(text, filtered_tokens)
    
    def _tokenize_spacy_sync(self, text: str) -> Dict[str, Any]:
        """Синхронная токенизация с использованием SpaCy"""
        if not self._initialized:
            # Fallback к regex если SpaCy не инициализирован
            return self._tokenize_regex(text)
        
        try:
            doc = self._nlp(text)
            tokens = []
            
            for token in doc:
                if token.is_space:
                    continue
                
                # Сохраняем именованные сущности
                if self.config.preserve_entities and token.ent_type_:
                    tokens.append(token.text)
                elif not token.is_punct or not self.config.remove_punctuation:
                    tokens.append(token.text)
            
            # Фильтрация токенов
            filtered_tokens = self._filter_tokens(tokens)
            
            return self._create_result(text, filtered_tokens)
            
        except Exception as e:
            logger.error(f"Error in SpaCy tokenization: {e}")
            return self._tokenize_regex(text)
    
    def _tokenize_technical(self, text: str) -> Dict[str, Any]:
        """Специализированная токенизация для технических текстов с сохранением логического порядка"""
        # Используем улучшенную regex токенизацию
        result = self._tokenize_regex(text)

        # Дополнительная обработка для технических терминов
        tokens = result['tokens']

        # Убираем дублирование токенов, сохраняя порядок
        unique_tokens = []
        seen = set()
        for token in tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)

        # Группируем связанные технические термины
        grouped_tokens = self._group_technical_terms(unique_tokens)

        # ИСПРАВЛЕНИЕ: Сохраняем логический порядок токенов
        ordered_tokens = self._preserve_logical_order(text, grouped_tokens)

        return self._create_result(text, ordered_tokens)

    def _preserve_logical_order(self, text: str, tokens: List[str]) -> List[str]:
        """
        Сохранение логического порядка токенов вместо случайной перестановки

        Логический порядок:
        1. Основные термины (существительные, названия)
        2. Технические характеристики (размеры, мощность, напряжение)
        3. Стандарты и коды (ГОСТ, IP, модели)
        4. Дополнительные характеристики (цвет, материал)
        """
        import re

        # Категории токенов
        main_terms = []          # Основные термины
        technical_specs = []     # Технические характеристики
        standards_codes = []     # Стандарты и коды
        additional_props = []    # Дополнительные свойства

        # Паттерны для категоризации
        main_term_patterns = [
            r'^[А-ЯЁ][а-яё]+$',  # Русские слова с заглавной буквы
            r'^[A-Z][a-z]+$',    # Английские слова с заглавной буквы
        ]

        technical_spec_patterns = [
            r'^\d+[WВвт]+$',      # Мощность: 50W, 100Вт
            r'^\d+[VВв]$',        # Напряжение: 24V, 220В
            r'^\d+[Aa]$',         # Ток: 32A, 63А
            r'^\d+[Kк]$',         # Температура: 6500K
            r'^\d+х\d+',          # Размеры: 10х50, 620х340х780
            r'^М\d+х\d+$',        # Резьба: М10х50
            r'^\d+мм$',           # Размеры в мм: 1200мм
            r'^\d+[.,]\d+',       # Дробные числа: 2.1, 3,5
        ]

        standards_codes_patterns = [
            r'^ГОСТ\s*\d+',       # ГОСТ стандарты
            r'^IP\d+$',           # IP рейтинги
            r'^[А-Я]+\d+',        # Технические коды: АВДТ103
            r'^\d+-\d+',          # Номера стандартов: 7798-70
            r'^[A-Z]+\d+[A-Z]*$', # Модели: W80DM, А24-5-1
        ]

        additional_props_patterns = [
            r'^(черный|белый|серый|красный|синий|зеленый)$',  # Цвета
            r'^(оцинкованный|пластиковый|металлический)$',    # Материалы
        ]

        def matches_patterns(token: str, patterns: List[str]) -> bool:
            """Проверка соответствия токена паттернам"""
            return any(re.match(pattern, token, re.IGNORECASE) for pattern in patterns)

        # Категоризация токенов
        for token in tokens:
            if matches_patterns(token, standards_codes_patterns):
                standards_codes.append(token)
            elif matches_patterns(token, technical_spec_patterns):
                technical_specs.append(token)
            elif matches_patterns(token, additional_props_patterns):
                additional_props.append(token)
            elif matches_patterns(token, main_term_patterns):
                main_terms.append(token)
            else:
                # Неопределенные токены добавляем к основным терминам
                main_terms.append(token)

        # Собираем в логическом порядке
        ordered_tokens = main_terms + technical_specs + standards_codes + additional_props

        # Убираем дубликаты, сохраняя порядок
        seen = set()
        final_tokens = []
        for token in ordered_tokens:
            if token not in seen:
                final_tokens.append(token)
                seen.add(token)

        return final_tokens

    def _group_technical_terms(self, tokens: List[str]) -> List[str]:
        """Улучшенная группировка связанных технических терминов"""
        import re

        grouped = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Проверяем возможность группировки с следующими токенами
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]

                # УЛУЧШЕНИЕ: Группировка метрических резьб М10х50
                if (token == 'М' and re.match(r'^\d+х\d+$', next_token)):
                    grouped.append(f"М{next_token}")
                    i += 2
                    continue

                # УЛУЧШЕНИЕ: Группировка ГОСТ стандартов
                if (token == 'ГОСТ' and re.match(r'^\d+$', next_token)):
                    # Проверяем полный формат ГОСТ XXXX-YY
                    if i < len(tokens) - 2:
                        third_token = tokens[i + 2]
                        if re.match(r'^\d+$', third_token):
                            grouped.append(f"ГОСТ {next_token}-{third_token}")
                            i += 3
                            continue
                    grouped.append(f"ГОСТ {next_token}")
                    i += 2
                    continue

                # УЛУЧШЕНИЕ: Группировка технических кодов АВДТ
                if (re.match(r'^[А-Я]+$', token) and re.match(r'^\d+$', next_token)):
                    # Проверяем сложные коды типа АВДТ 103-6кА-3N+P
                    if i < len(tokens) - 2:
                        third_token = tokens[i + 2]
                        # Ищем паттерн типа 6кА-3N+P
                        if re.match(r'^\d+кА', third_token):
                            # Собираем полный технический код
                            full_code = f"{token} {next_token}-{third_token}"
                            # Проверяем дополнительные части
                            j = i + 3
                            while j < len(tokens) and re.match(r'^[A-Z0-9+\-]+$', tokens[j]):
                                full_code += f" {tokens[j]}"
                                j += 1
                            grouped.append(full_code)
                            i = j
                            continue
                    grouped.append(f"{token} {next_token}")
                    i += 2
                    continue

                # УЛУЧШЕНИЕ: Группировка размерностей 620х340х780
                if (re.match(r'^\d+$', token) and re.match(r'^х\d+$', next_token)):
                    dimension = token + next_token
                    j = i + 2
                    # Собираем все части размерности
                    while j < len(tokens) and re.match(r'^х\d+$', tokens[j]):
                        dimension += tokens[j]
                        j += 1
                    grouped.append(dimension)
                    i = j
                    continue

                # УЛУЧШЕНИЕ: Группировка IP рейтингов
                if (token == 'IP' and re.match(r'^\d+$', next_token)):
                    grouped.append(f"IP{next_token}")
                    i += 2
                    continue

                # Стандартная группировка: буква + число
                if (re.match(r'^[A-ZА-Я]+$', token) and re.match(r'^\d+$', next_token) and
                    len(token) <= 3 and len(next_token) <= 4):
                    grouped.append(f"{token}{next_token}")
                    i += 2
                    continue

            # Если группировка не применима, добавляем токен как есть
            grouped.append(token)
            i += 1

        return grouped
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Фильтрация токенов согласно конфигурации"""
        filtered = []
        
        for token in tokens:
            # Проверка длины
            if len(token) < self.config.min_token_length or len(token) > self.config.max_token_length:
                continue
            
            # Приведение к нижнему регистру
            if self.config.lowercase:
                token = token.lower()
            
            # Удаление стоп-слов
            if self.config.remove_stopwords and self._stopwords and token.lower() in self._stopwords:
                continue
            
            # Удаление пунктуации
            if self.config.remove_punctuation and hasattr(self, '_punct_pattern'):
                token = self._punct_pattern.sub('', token)
                if not token:  # Пропускаем пустые токены после удаления пунктуации
                    continue
            
            filtered.append(token)
        
        return filtered
    
    def _create_result(self, original_text: str, tokens: List[str]) -> Dict[str, Any]:
        """Создание результата токенизации"""
        unique_tokens = list(set(tokens))
        
        return {
            'original': original_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'unique_tokens': unique_tokens,
            'unique_count': len(unique_tokens),
            'tokenizer_type': self.config.tokenizer_type
        }
    
    async def tokenize_text_async(self, text: str) -> Dict[str, Any]:
        """Асинхронная токенизация текста"""
        await self._ensure_initialized()
        
        # Выполняем синхронную токенизацию в ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.tokenize_text, text)
    
    def tokenize_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетная токенизация текстов"""
        return [self.tokenize_text(text) for text in texts]
    
    async def tokenize_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Асинхронная пакетная токенизация"""
        await self._ensure_initialized()
        
        # Выполняем пакетную обработку в ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.tokenize_batch, texts)
    
    def get_token_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Получение статистики токенизации для набора текстов"""
        all_tokens = []
        all_unique_tokens = set()
        token_counts = []
        
        for text in texts:
            result = self.tokenize_text(text)
            all_tokens.extend(result['tokens'])
            all_unique_tokens.update(result['unique_tokens'])
            token_counts.append(result['token_count'])
        
        return {
            'total_texts': len(texts),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(all_unique_tokens),
            'avg_tokens_per_text': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens_per_text': min(token_counts) if token_counts else 0,
            'max_tokens_per_text': max(token_counts) if token_counts else 0
        }
    
    def _should_include_vectors(self, include_vectors: bool = None) -> bool:
        """Определяет, нужно ли включать векторизацию"""
        if include_vectors is not None:
            return include_vectors and self._vectorizer is not None
        return self.config.enable_vectorization and self._vectorizer is not None

    def _vectorize_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Векторизация списка токенов"""
        if self._vectorizer is None or not self._vectorizer.is_fitted:
            return self._get_empty_vectorization()

        try:
            return self._vectorizer.vectorize_tokens(tokens)
        except Exception as e:
            logger.warning(f"Error in token vectorization: {e}")
            return self._get_empty_vectorization()

    def _get_empty_vectorization(self) -> Dict[str, Any]:
        """Возвращает пустую структуру векторизации"""
        return {
            'original_tokens': [],
            'token_count': 0,
            'token_ids': [],
            'ids_vector': None,
            'token_embeddings': None,
            'aggregated_embedding': None
        }

    def fit_vectorizer(self, token_lists: List[List[str]]) -> bool:
        """
        Обучение векторизатора на списках токенов

        Args:
            token_lists: Список списков токенов для обучения

        Returns:
            True если обучение прошло успешно
        """
        if self._vectorizer is None:
            logger.warning("Vectorizer not initialized")
            return False

        try:
            self._vectorizer.fit(token_lists)
            logger.info("Vectorizer fitted successfully")
            return True
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            return False

    def get_token_vector(self, token: str) -> Dict[str, Any]:
        """
        Получение векторного представления токена

        Args:
            token: Токен для векторизации

        Returns:
            Словарь с информацией о токене и его векторе
        """
        if self._vectorizer is None or not self._vectorizer.is_fitted:
            return {'error': 'Vectorizer not available or not fitted'}

        return self._vectorizer.get_token_info(token)

    def get_vectorizer_stats(self) -> Dict[str, Any]:
        """Получение статистики векторизатора"""
        if self._vectorizer is None:
            return {'vectorizer_available': False}

        stats = self._vectorizer.get_vocabulary_stats()
        stats['vectorizer_available'] = True
        return stats

    def save_vectorizer(self, path: str = None) -> bool:
        """Сохранение векторизатора"""
        if self._vectorizer is None or not self._vectorizer.is_fitted:
            logger.warning("Vectorizer not available or not fitted")
            return False

        try:
            if path:
                self._vectorizer.config.vocabulary_path = path
            self._vectorizer._save_vocabulary()
            return True
        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
            return False

    def load_vectorizer(self, vocabulary_path: str, embeddings_path: str = None) -> bool:
        """Загрузка векторизатора"""
        if self._vectorizer is None:
            logger.warning("Vectorizer not initialized")
            return False

        try:
            self._vectorizer.load_vocabulary(vocabulary_path, embeddings_path)
            return True
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            return False

    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
