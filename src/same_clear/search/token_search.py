"""
Система поиска по токенам для проекта SAMe

Этот модуль предоставляет различные методы поиска:
- Поиск по ID токенов
- Семантический поиск по эмбеддингам
- Гибридный поиск (комбинация методов)
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, deque
import re
import hashlib
try:
    from datasketch import MinHashLSH, MinHash
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    MinHashLSH = None
    MinHash = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


class TrieNode:
    """Узел префиксного дерева"""
    def __init__(self):
        self.children = {}
        self.is_end_word = False
        self.record_indices = []  # Индексы записей, содержащих этот префикс


class Trie:
    """Префиксное дерево для быстрого поиска по началу строки"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, record_index: int):
        """Вставка слова в дерево с привязкой к индексу записи"""
        node = self.root
        word = word.lower().strip()

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.record_indices.append(record_index)

        node.is_end_word = True

    def search_prefix(self, prefix: str) -> List[int]:
        """Поиск всех записей, содержащих слова с данным префиксом"""
        node = self.root
        prefix = prefix.lower().strip()

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        return list(set(node.record_indices))  # Убираем дубликаты

    def get_all_words_with_prefix(self, prefix: str) -> List[str]:
        """Получение всех слов с данным префиксом"""
        node = self.root
        prefix = prefix.lower().strip()

        # Находим узел префикса
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Собираем все слова из поддерева
        words = []
        self._collect_words(node, prefix, words)
        return words

    def _collect_words(self, node: TrieNode, current_word: str, words: List[str]):
        """Рекурсивный сбор всех слов из поддерева"""
        if node.is_end_word:
            words.append(current_word)

        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)


@dataclass
class SearchResult:
    """Результат поиска"""
    code: str
    raw_name: str
    tokens: str
    token_vectors: str
    parameters: str
    score: float
    match_type: str  # 'token_id', 'semantic', 'hybrid'
    matched_tokens: List[str] = None
    similarity_score: float = 0.0


@dataclass
class SearchConfig:
    """Конфигурация поиска"""
    # Поиск по ID токенов
    token_id_weight: float = 0.6
    min_token_matches: int = 1

    # Семантический поиск
    semantic_weight: float = 0.4
    similarity_threshold: float = 0.3

    # Общие настройки
    max_results: int = 100
    enable_fuzzy_matching: bool = True
    boost_technical_terms: bool = True

    # Новые методы поиска
    # Trie (префиксный поиск)
    enable_trie_search: bool = True
    trie_min_prefix_length: int = 2
    trie_weight: float = 0.3

    # Inverted Index
    enable_inverted_index: bool = True
    inverted_index_weight: float = 0.4

    # TF-IDF
    enable_tfidf_search: bool = True
    tfidf_weight: float = 0.35
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)

    # MinHash LSH
    enable_lsh_search: bool = True
    lsh_weight: float = 0.25
    lsh_threshold: float = 0.6
    lsh_num_perm: int = 128

    # Spatial Indexing (FAISS)
    enable_spatial_search: bool = True
    spatial_weight: float = 0.3
    faiss_index_type: str = "flat"  # flat, ivf, hnsw

    # Advanced Embeddings
    enable_advanced_embeddings: bool = False  # Требует дополнительных моделей
    advanced_embedding_weight: float = 0.4

    # Graph-based Search
    enable_graph_search: bool = False  # Вычислительно затратный
    graph_weight: float = 0.2
    graph_similarity_threshold: float = 0.7


class TokenSearchEngine:
    """Движок поиска по токенам"""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()

        # Данные для поиска
        self.data_df = None
        self.token_vectors_cache = {}
        self.embeddings_cache = {}

        # Векторизатор для обработки запросов
        self.vectorizer = None
        self.tokenizer = None

        # Существующие индексы
        self.token_id_index = {}  # token_id -> list of record indices
        self.embeddings_matrix = None  # матрица эмбеддингов для всех записей

        # Новые структуры данных для поиска
        self.trie = Trie() if self.config.enable_trie_search else None
        self.inverted_index = defaultdict(set) if self.config.enable_inverted_index else None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # LSH только если datasketch доступен
        if DATASKETCH_AVAILABLE and self.config.enable_lsh_search:
            self.lsh_index = None
            self.minhashes = {}
        else:
            self.lsh_index = None
            self.minhashes = {}
            if self.config.enable_lsh_search:
                logger.warning("LSH search disabled: datasketch not available")
                self.config.enable_lsh_search = False

        # FAISS только если доступен
        if FAISS_AVAILABLE and self.config.enable_spatial_search:
            self.faiss_index = None
        else:
            self.faiss_index = None
            if self.config.enable_spatial_search:
                logger.warning("Spatial search disabled: faiss not available")
                self.config.enable_spatial_search = False

        # NetworkX только если доступен
        if NETWORKX_AVAILABLE and self.config.enable_graph_search:
            self.similarity_graph = None
        else:
            self.similarity_graph = None
            if self.config.enable_graph_search:
                logger.warning("Graph search disabled: networkx not available")
                self.config.enable_graph_search = False

        logger.info("TokenSearchEngine initialized with advanced search methods")
    
    def load_data(self, csv_file: str, vectorizer=None, tokenizer=None) -> bool:
        """
        Загрузка данных для поиска
        
        Args:
            csv_file: Путь к CSV файлу с обработанными данными
            vectorizer: Обученный векторизатор токенов
            tokenizer: Токенизатор для обработки запросов
            
        Returns:
            True если загрузка прошла успешно
        """
        try:
            logger.info(f"Loading search data from: {csv_file}")
            
            # Загружаем данные
            self.data_df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(self.data_df)} records")
            
            # Сохраняем векторизатор и токенизатор
            self.vectorizer = vectorizer
            self.tokenizer = tokenizer
            
            # Строим существующие индексы
            self._build_token_id_index()
            self._build_embeddings_matrix()

            # Строим новые индексы
            if self.config.enable_trie_search:
                self._build_trie_index()

            if self.config.enable_inverted_index:
                self._build_inverted_index()

            if self.config.enable_tfidf_search:
                self._build_tfidf_index()

            if self.config.enable_lsh_search and DATASKETCH_AVAILABLE:
                self._build_lsh_index()

            if self.config.enable_spatial_search and FAISS_AVAILABLE and self.embeddings_matrix is not None:
                self._build_faiss_index()

            if self.config.enable_graph_search and NETWORKX_AVAILABLE:
                self._build_similarity_graph()
            
            logger.info("Search engine ready")
            return True
            
        except Exception as e:
            logger.error(f"Error loading search data: {e}")
            return False
    
    def _build_token_id_index(self):
        """Построение индекса по ID токенов"""
        logger.info("Building token ID index...")

        self.token_id_index = {}

        for idx, row in self.data_df.iterrows():
            token_vectors = row.get('token_vectors', '')

            # Сначала пробуем извлечь из token_vectors в формате "IDs: [1, 2, 3, ...]"
            if pd.notna(token_vectors) and 'IDs:' in str(token_vectors):
                try:
                    ids_str = str(token_vectors).replace('IDs: ', '').strip('[]')
                    token_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]

                    # Добавляем в индекс
                    for token_id in token_ids:
                        if token_id not in self.token_id_index:
                            self.token_id_index[token_id] = []
                        self.token_id_index[token_id].append(idx)

                except Exception as e:
                    logger.warning(f"Error parsing token vectors for row {idx}: {e}")

            # Fallback: создаем ID токенов из BPE_Tokens если есть токенизатор
            elif self.tokenizer and 'BPE_Tokens' in row:
                try:
                    bpe_tokens = row.get('BPE_Tokens', '')
                    if pd.notna(bpe_tokens) and bpe_tokens:
                        # Парсим BPE токены из JSON-строки
                        if isinstance(bpe_tokens, str):
                            # Убираем квадратные скобки и кавычки, разделяем по запятым
                            clean_tokens = bpe_tokens.strip("[]'\"")
                            if clean_tokens:
                                tokens = [token.strip("'\" ") for token in clean_tokens.split(',')]

                                # Получаем ID токенов через токенизатор
                                token_ids = []
                                for token in tokens:
                                    if hasattr(self.tokenizer, 'token_to_id'):
                                        token_id = self.tokenizer.token_to_id(token.lower())
                                        if token_id is not None:
                                            token_ids.append(token_id)
                                    elif hasattr(self.tokenizer, 'vocabulary') and hasattr(self.tokenizer.vocabulary, 'get'):
                                        token_id = self.tokenizer.vocabulary.get(token.lower())
                                        if token_id is not None:
                                            token_ids.append(token_id)

                                # Добавляем в индекс
                                for token_id in token_ids:
                                    if token_id not in self.token_id_index:
                                        self.token_id_index[token_id] = []
                                    self.token_id_index[token_id].append(idx)

                except Exception as e:
                    logger.warning(f"Error creating token IDs from BPE tokens for row {idx}: {e}")

        logger.info(f"Token ID index built: {len(self.token_id_index)} unique token IDs")
    
    def _build_embeddings_matrix(self):
        """Построение матрицы эмбеддингов для семантического поиска"""
        logger.info("Building embeddings matrix...")

        # Сначала пробуем использовать векторизатор с token_embeddings
        if self.vectorizer and hasattr(self.vectorizer, 'token_embeddings'):
            embeddings_list = []
            successful_embeddings = 0

            for idx, row in self.data_df.iterrows():
                embedding_created = False

                # Метод 1: Из token_vectors в формате "IDs: [1, 2, 3, ...]"
                token_vectors = row.get('token_vectors', '')
                if pd.notna(token_vectors) and 'IDs:' in str(token_vectors):
                    try:
                        ids_str = str(token_vectors).replace('IDs: ', '').strip('[]')
                        token_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]

                        # Получаем эмбеддинги для токенов
                        token_embeddings = []
                        for token_id in token_ids:
                            if token_id < len(self.vectorizer.token_embeddings):
                                embedding = self.vectorizer.token_embeddings[token_id]
                                token_embeddings.append(embedding)

                        # Агрегируем эмбеддинги (среднее)
                        if token_embeddings:
                            aggregated = np.mean(token_embeddings, axis=0)
                            embeddings_list.append(aggregated)
                            successful_embeddings += 1
                            embedding_created = True

                    except Exception as e:
                        logger.warning(f"Error building embedding from token_vectors for row {idx}: {e}")

                # Метод 2: Из BPE_Tokens через токенизатор
                if not embedding_created and self.tokenizer and 'BPE_Tokens' in row:
                    try:
                        bpe_tokens = row.get('BPE_Tokens', '')
                        if pd.notna(bpe_tokens) and bpe_tokens:
                            # Парсим BPE токены
                            if isinstance(bpe_tokens, str):
                                clean_tokens = bpe_tokens.strip("[]'\"")
                                if clean_tokens:
                                    tokens = [token.strip("'\" ") for token in clean_tokens.split(',')]

                                    # Получаем эмбеддинги через токенизатор
                                    token_embeddings = []
                                    for token in tokens:
                                        token_id = None
                                        if hasattr(self.tokenizer, 'token_to_id'):
                                            token_id = self.tokenizer.token_to_id(token.lower())
                                        elif hasattr(self.tokenizer, 'vocabulary') and hasattr(self.tokenizer.vocabulary, 'get'):
                                            token_id = self.tokenizer.vocabulary.get(token.lower())

                                        if token_id is not None and token_id < len(self.vectorizer.token_embeddings):
                                            embedding = self.vectorizer.token_embeddings[token_id]
                                            token_embeddings.append(embedding)

                                    # Агрегируем эмбеддинги
                                    if token_embeddings:
                                        aggregated = np.mean(token_embeddings, axis=0)
                                        embeddings_list.append(aggregated)
                                        successful_embeddings += 1
                                        embedding_created = True

                    except Exception as e:
                        logger.warning(f"Error building embedding from BPE_Tokens for row {idx}: {e}")

                # Метод 3: Fallback - создаем случайный ненулевой вектор
                if not embedding_created:
                    # Создаем случайный вектор с небольшой нормой вместо нулевого
                    embedding_dim = getattr(self.vectorizer.config, 'embedding_dim', 128)
                    random_embedding = np.random.normal(0, 0.1, embedding_dim).astype('float32')
                    embeddings_list.append(random_embedding)

            if embeddings_list:
                self.embeddings_matrix = np.array(embeddings_list)
                logger.info(f"Embeddings matrix built: {self.embeddings_matrix.shape}")
                logger.info(f"Successful embeddings from tokens: {successful_embeddings}/{len(embeddings_list)}")

                # Проверяем нормы векторов
                norms = np.linalg.norm(self.embeddings_matrix, axis=1)
                zero_norm_count = np.sum(norms == 0)
                logger.info(f"Zero-norm vectors: {zero_norm_count}/{len(norms)}")
                logger.info(f"Average vector norm: {np.mean(norms):.4f}")
                return

        # Fallback: создаем эмбеддинги из BPE токенов с TF-IDF
        logger.warning("Vectorizer or embeddings not available, using TF-IDF fallback")
        return self._build_simple_embeddings_from_bpe()

    def _build_simple_embeddings_from_bpe(self):
        """Создание простых эмбеддингов из BPE токенов как fallback"""
        logger.info("Building simple embeddings from BPE tokens as fallback...")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            # Собираем тексты из BPE токенов или Raw_Name
            texts = []
            for idx, row in self.data_df.iterrows():
                text_parts = []

                # Пробуем использовать BPE токены
                bpe_tokens = row.get('BPE_Tokens', '')
                if pd.notna(bpe_tokens) and bpe_tokens:
                    # Преобразуем BPE токены в текст
                    if isinstance(bpe_tokens, str):
                        # Убираем квадратные скобки и кавычки
                        clean_tokens = bpe_tokens.strip("[]'\"")
                        if clean_tokens:
                            tokens = [token.strip("'\" ") for token in clean_tokens.split(',')]
                            text_parts.extend(tokens)

                # Добавляем также нормализованное имя для лучшего качества
                normalized_name = row.get('Normalized_Name', '')
                if pd.notna(normalized_name) and normalized_name:
                    text_parts.append(normalized_name)

                # Fallback к Raw_Name
                if not text_parts:
                    raw_name = row.get('Raw_Name', '')
                    if pd.notna(raw_name) and raw_name:
                        text_parts.append(raw_name)

                # Объединяем все части текста
                text = ' '.join(text_parts).lower() if text_parts else ''
                texts.append(text)

            # Создаем TF-IDF векторы с оптимизированными параметрами
            tfidf = TfidfVectorizer(
                max_features=min(5000, len(texts) * 50),  # Адаптивное количество признаков
                ngram_range=(1, 4),  # Расширяем до 4-грамм для технических терминов
                stop_words=None,  # Не используем стоп-слова для технических терминов
                min_df=1,
                max_df=0.90,  # Немного снижаем для сохранения важных терминов
                sublinear_tf=True,  # Логарифмическое масштабирование
                norm='l2',  # L2 нормализация
                analyzer='word',  # Анализ по словам
                token_pattern=r'(?u)\b\w+\b',  # Улучшенный паттерн токенов
                lowercase=True,
                use_idf=True,
                smooth_idf=True  # Сглаживание IDF
            )

            tfidf_matrix = tfidf.fit_transform(texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            logger.info(f"TF-IDF matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

            # Оптимальная размерность для SVD
            max_components = min(256, tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1)
            optimal_components = min(128, max_components)

            # Используем randomized SVD для лучшей производительности
            svd = TruncatedSVD(
                n_components=optimal_components,
                random_state=42,
                algorithm='randomized',  # Быстрее для больших матриц
                n_iter=7  # Больше итераций для лучшего качества
            )
            embeddings_matrix = svd.fit_transform(tfidf_matrix)

            # Масштабируем эмбеддинги для лучшего качества поиска
            # Используем explained variance для весов компонент
            explained_variance = svd.explained_variance_ratio_
            weights = np.sqrt(explained_variance)  # Квадратный корень для сглаживания
            embeddings_matrix = embeddings_matrix * weights

            # Нормализуем эмбеддинги
            from sklearn.preprocessing import normalize
            embeddings_matrix = normalize(embeddings_matrix, norm='l2')

            # Дополнительное масштабирование для увеличения схожести
            # Умножаем на константу для получения более высоких оценок
            embeddings_matrix = embeddings_matrix * 2.0

            self.embeddings_matrix = embeddings_matrix.astype('float32')
            logger.info(f"Enhanced embeddings matrix built: {self.embeddings_matrix.shape}")

            # Проверяем качество эмбеддингов
            norms = np.linalg.norm(self.embeddings_matrix, axis=1)
            zero_norm_count = np.sum(norms == 0)
            logger.info(f"Zero-norm vectors in fallback embeddings: {zero_norm_count}/{len(norms)}")
            logger.info(f"Average vector norm in fallback: {np.mean(norms):.4f}")
            logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
            logger.info(f"Top 5 explained variance ratios: {explained_variance[:5]}")

            # Сохраняем TF-IDF и SVD для создания эмбеддингов запросов
            self._fallback_tfidf = tfidf
            self._fallback_svd = svd

        except Exception as e:
            logger.error(f"Error building simple embeddings: {e}")
            import traceback
            traceback.print_exc()
            self.embeddings_matrix = None

    def _create_fallback_query_embedding(self, query: str) -> np.ndarray:
        """Создание улучшенного эмбеддинга запроса с использованием fallback метода"""
        try:
            if hasattr(self, '_fallback_tfidf') and hasattr(self, '_fallback_svd'):
                # Предобрабатываем запрос аналогично обучающим данным
                processed_query = query.lower().strip()

                # Используем сохраненные TF-IDF и SVD модели
                query_tfidf = self._fallback_tfidf.transform([processed_query])
                query_embedding = self._fallback_svd.transform(query_tfidf)

                # Применяем те же веса что и при обучении
                if hasattr(self._fallback_svd, 'explained_variance_ratio_'):
                    explained_variance = self._fallback_svd.explained_variance_ratio_
                    weights = np.sqrt(explained_variance)
                    query_embedding = query_embedding * weights

                # Нормализуем эмбеддинг запроса
                from sklearn.preprocessing import normalize
                query_embedding = normalize(query_embedding, norm='l2')

                # Применяем то же масштабирование что и к данным
                query_embedding = query_embedding * 2.0

                return query_embedding[0].astype('float32')
            else:
                logger.warning("Fallback TF-IDF/SVD models not available")
                return None
        except Exception as e:
            logger.error(f"Error creating fallback query embedding: {e}")
            return None

    def _create_query_embedding_from_tokens(self, query: str) -> np.ndarray:
        """Создание эмбеддинга запроса из токенов через векторизатор"""
        try:
            if not self.tokenizer or not self.vectorizer:
                return None

            # Токенизируем запрос
            if hasattr(self.tokenizer, 'tokenize'):
                tokens = self.tokenizer.tokenize(query.lower())
            else:
                # Простая токенизация как fallback
                tokens = query.lower().split()

            # Получаем эмбеддинги токенов
            token_embeddings = []
            for token in tokens:
                token_id = None
                if hasattr(self.tokenizer, 'token_to_id'):
                    token_id = self.tokenizer.token_to_id(token)
                elif hasattr(self.tokenizer, 'vocabulary') and hasattr(self.tokenizer.vocabulary, 'get'):
                    token_id = self.tokenizer.vocabulary.get(token)

                if (token_id is not None and
                    hasattr(self.vectorizer, 'token_embeddings') and
                    token_id < len(self.vectorizer.token_embeddings)):
                    embedding = self.vectorizer.token_embeddings[token_id]
                    token_embeddings.append(embedding)

            # Агрегируем эмбеддинги (среднее)
            if token_embeddings:
                aggregated = np.mean(token_embeddings, axis=0)
                return aggregated.astype('float32')
            else:
                return None

        except Exception as e:
            logger.error(f"Error creating query embedding from tokens: {e}")
            return None

    def _create_contextual_embedding(self, query: str) -> np.ndarray:
        """Создание контекстуального эмбеддинга с учетом семантики"""
        try:
            # Пробуем использовать sentence-transformers если доступен
            try:
                from sentence_transformers import SentenceTransformer

                # Используем легкую многоязычную модель
                if not hasattr(self, '_sentence_model'):
                    model_names = [
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "sentence-transformers/all-MiniLM-L6-v2"
                    ]

                    for model_name in model_names:
                        try:
                            self._sentence_model = SentenceTransformer(model_name)
                            logger.info(f"Loaded sentence transformer: {model_name}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")
                            continue

                    if not hasattr(self, '_sentence_model'):
                        return None

                # Создаем эмбеддинг запроса
                embedding = self._sentence_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

                # Приводим к нужной размерности если необходимо
                if hasattr(self, 'embeddings_matrix') and self.embeddings_matrix is not None:
                    target_dim = self.embeddings_matrix.shape[1]
                    current_dim = embedding.shape[1]

                    if current_dim != target_dim:
                        # Простое масштабирование или обрезание
                        if current_dim > target_dim:
                            embedding = embedding[:, :target_dim]
                        else:
                            # Дополняем нулями
                            padding = np.zeros((1, target_dim - current_dim))
                            embedding = np.hstack([embedding, padding])

                return embedding[0].astype('float32')

            except ImportError:
                logger.debug("sentence-transformers not available for contextual embeddings")
                return None

        except Exception as e:
            logger.error(f"Error creating contextual embedding: {e}")
            return None

    def _build_trie_index(self):
        """Построение префиксного дерева для быстрого поиска по началу слов"""
        try:
            logger.info("Building Trie index...")
            self.trie = Trie()

            for idx, row in self.data_df.iterrows():
                # Добавляем слова из названия товара
                raw_name = str(row.get('Raw_Name', ''))
                words = re.findall(r'\b\w+\b', raw_name.lower())

                for word in words:
                    if len(word) >= self.config.trie_min_prefix_length:
                        self.trie.insert(word, idx)

                # Добавляем токены если они есть
                tokens_str = str(row.get('tokenizer', ''))
                if tokens_str and tokens_str != 'nan':
                    try:
                        # Пробуем парсить как JSON список
                        if tokens_str.startswith('['):
                            tokens = eval(tokens_str)
                            for token in tokens:
                                if isinstance(token, str) and len(token) >= self.config.trie_min_prefix_length:
                                    self.trie.insert(token, idx)
                        else:
                            # Разбиваем по пробелам
                            tokens = tokens_str.split()
                            for token in tokens:
                                if len(token) >= self.config.trie_min_prefix_length:
                                    self.trie.insert(token, idx)
                    except:
                        pass

            logger.info("Trie index built successfully")
        except Exception as e:
            logger.error(f"Error building Trie index: {e}")
            self.trie = None

    def _build_inverted_index(self):
        """Построение обратного индекса для полнотекстового поиска"""
        try:
            logger.info("Building inverted index...")
            self.inverted_index = defaultdict(set)

            for idx, row in self.data_df.iterrows():
                # Индексируем слова из названия
                raw_name = str(row.get('Raw_Name', ''))
                words = re.findall(r'\b\w+\b', raw_name.lower())

                for word in words:
                    self.inverted_index[word].add(idx)

                # Индексируем токены
                tokens_str = str(row.get('tokenizer', ''))
                if tokens_str and tokens_str != 'nan':
                    try:
                        if tokens_str.startswith('['):
                            tokens = eval(tokens_str)
                            for token in tokens:
                                if isinstance(token, str):
                                    self.inverted_index[token.lower()].add(idx)
                        else:
                            tokens = tokens_str.split()
                            for token in tokens:
                                self.inverted_index[token.lower()].add(idx)
                    except:
                        pass

            logger.info(f"Inverted index built with {len(self.inverted_index)} terms")
        except Exception as e:
            logger.error(f"Error building inverted index: {e}")
            self.inverted_index = None

    def _build_tfidf_index(self):
        """Построение TF-IDF индекса"""
        try:
            logger.info("Building TF-IDF index...")

            # Собираем тексты для векторизации
            texts = []
            for idx, row in self.data_df.iterrows():
                text_parts = []

                # Добавляем название товара
                raw_name = str(row.get('Raw_Name', ''))
                if raw_name and raw_name != 'nan':
                    text_parts.append(raw_name)

                # Добавляем токены
                tokens_str = str(row.get('tokenizer', ''))
                if tokens_str and tokens_str != 'nan':
                    try:
                        if tokens_str.startswith('['):
                            tokens = eval(tokens_str)
                            text_parts.extend([str(t) for t in tokens if isinstance(t, str)])
                        else:
                            text_parts.extend(tokens_str.split())
                    except:
                        pass

                texts.append(' '.join(text_parts))

            # Создаем TF-IDF векторизатор
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                lowercase=True,
                stop_words=None  # Не используем стоп-слова для технических терминов
            )

            # Строим TF-IDF матрицу
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            logger.info(f"TF-IDF index built: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def _build_lsh_index(self):
        """Построение LSH индекса для быстрого поиска похожих элементов"""
        if not DATASKETCH_AVAILABLE:
            logger.warning("Cannot build LSH index: datasketch not available")
            return

        try:
            logger.info("Building LSH index...")

            # Создаем LSH индекс
            self.lsh_index = MinHashLSH(threshold=self.config.lsh_threshold, num_perm=self.config.lsh_num_perm)
            self.minhashes = {}

            for idx, row in self.data_df.iterrows():
                # Создаем набор токенов для MinHash
                tokens_set = set()

                # Добавляем слова из названия
                raw_name = str(row.get('Raw_Name', ''))
                words = re.findall(r'\b\w+\b', raw_name.lower())
                tokens_set.update(words)

                # Добавляем токены
                tokens_str = str(row.get('tokenizer', ''))
                if tokens_str and tokens_str != 'nan':
                    try:
                        if tokens_str.startswith('['):
                            tokens = eval(tokens_str)
                            tokens_set.update([str(t).lower() for t in tokens if isinstance(t, str)])
                        else:
                            tokens_set.update([t.lower() for t in tokens_str.split()])
                    except:
                        pass

                if tokens_set:
                    # Создаем MinHash для этого набора токенов
                    minhash = MinHash(num_perm=self.config.lsh_num_perm)
                    for token in tokens_set:
                        minhash.update(token.encode('utf8'))

                    self.minhashes[idx] = minhash
                    self.lsh_index.insert(idx, minhash)

            logger.info(f"LSH index built with {len(self.minhashes)} items")
        except Exception as e:
            logger.error(f"Error building LSH index: {e}")
            self.lsh_index = None
            self.minhashes = {}

    def _build_faiss_index(self):
        """Построение FAISS индекса для быстрого поиска ближайших соседей"""
        if not FAISS_AVAILABLE:
            logger.warning("Cannot build FAISS index: faiss not available")
            return

        try:
            logger.info("Building FAISS index...")

            if self.embeddings_matrix is None:
                logger.warning("No embeddings matrix available for FAISS index")
                return

            # Нормализуем эмбеддинги для косинусного сходства
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)

            # Проверяем количество нулевых векторов
            zero_norm_count = np.sum(norms == 0)
            if zero_norm_count > 0:
                logger.warning(f"Found {zero_norm_count} zero-norm vectors in embeddings matrix. These will be handled safely.")

            # Избегаем деления на ноль, заменяя нулевые нормы на 1
            norms = np.where(norms == 0, 1, norms)
            embeddings_normalized = self.embeddings_matrix / norms

            # Создаем FAISS индекс
            dimension = embeddings_normalized.shape[1]

            if self.config.faiss_index_type == "flat":
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусного сходства
            elif self.config.faiss_index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings_normalized) // 10))
                self.faiss_index.train(embeddings_normalized.astype('float32'))
            else:
                # По умолчанию используем flat
                self.faiss_index = faiss.IndexFlatIP(dimension)

            # Добавляем векторы в индекс
            self.faiss_index.add(embeddings_normalized.astype('float32'))

            logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            self.faiss_index = None

    def _build_similarity_graph(self):
        """Построение графа схожести для graph-based поиска"""
        if not NETWORKX_AVAILABLE:
            logger.warning("Cannot build similarity graph: networkx not available")
            return

        try:

            logger.info("Building similarity graph...")

            if self.embeddings_matrix is None:
                logger.warning("No embeddings matrix available for graph construction")
                return

            # Создаем граф
            self.similarity_graph = nx.Graph()

            # Добавляем узлы
            for idx in range(len(self.data_df)):
                self.similarity_graph.add_node(idx)

            # Вычисляем попарные сходства и добавляем рёбра
            similarities = cosine_similarity(self.embeddings_matrix)

            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    similarity = similarities[i][j]
                    if similarity >= self.config.graph_similarity_threshold:
                        self.similarity_graph.add_edge(i, j, weight=similarity)

            logger.info(f"Similarity graph built with {self.similarity_graph.number_of_nodes()} nodes and {self.similarity_graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error building similarity graph: {e}")
            self.similarity_graph = None

    def search_by_token_ids(self, token_ids: List[int], top_k: int = 10) -> List[SearchResult]:
        """
        Поиск по ID токенов
        
        Args:
            token_ids: Список ID токенов для поиска
            top_k: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        if not self.token_id_index:
            logger.warning("Token ID index not built")
            return []
        
        # Подсчитываем совпадения для каждой записи
        record_scores = {}
        
        for token_id in token_ids:
            if token_id in self.token_id_index:
                for record_idx in self.token_id_index[token_id]:
                    if record_idx not in record_scores:
                        record_scores[record_idx] = {'matches': 0, 'matched_tokens': []}
                    record_scores[record_idx]['matches'] += 1
                    record_scores[record_idx]['matched_tokens'].append(token_id)
        
        # Фильтруем по минимальному количеству совпадений
        filtered_records = {
            idx: data for idx, data in record_scores.items() 
            if data['matches'] >= self.config.min_token_matches
        }
        
        # Сортируем по количеству совпадений
        sorted_records = sorted(
            filtered_records.items(), 
            key=lambda x: x[1]['matches'], 
            reverse=True
        )[:top_k]
        
        # Создаем результаты
        results = []
        for record_idx, match_data in sorted_records:
            row = self.data_df.iloc[record_idx]
            
            # Вычисляем оценку (процент совпадающих токенов)
            total_query_tokens = len(token_ids)
            matched_tokens = match_data['matches']
            score = matched_tokens / total_query_tokens
            
            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=score,
                match_type='token_id',
                matched_tokens=match_data['matched_tokens']
            )
            results.append(result)
        
        logger.info(f"Token ID search found {len(results)} results")
        return results
    
    def search_by_similarity(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Семантический поиск по эмбеддингам
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        if self.embeddings_matrix is None:
            logger.warning("Embeddings matrix not available")
            return []
        
        # Вычисляем косинусное сходство
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Фильтруем по порогу сходства
        valid_indices = np.where(similarities >= self.config.similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            logger.info("No results above similarity threshold")
            return []
        
        # Сортируем по убыванию сходства
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
        
        # Создаем результаты
        results = []
        for idx in sorted_indices:
            row = self.data_df.iloc[idx]
            similarity_score = similarities[idx]
            
            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=similarity_score,
                match_type='semantic',
                similarity_score=similarity_score
            )
            results.append(result)
        
        logger.info(f"Semantic search found {len(results)} results")
        return results

    def search_hybrid(self, token_ids: List[int], query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Гибридный поиск (комбинация поиска по ID и семантического поиска)

        Args:
            token_ids: Список ID токенов для поиска
            query_embedding: Эмбеддинг запроса
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        # Получаем результаты от обоих методов
        token_results = self.search_by_token_ids(token_ids, top_k * 2)  # Берем больше для комбинирования
        semantic_results = self.search_by_similarity(query_embedding, top_k * 2)

        # Создаем словарь для объединения результатов
        combined_results = {}

        # Добавляем результаты поиска по токенам
        for result in token_results:
            code = result.code
            combined_results[code] = {
                'result': result,
                'token_score': result.score,
                'semantic_score': 0.0
            }

        # Добавляем результаты семантического поиска
        for result in semantic_results:
            code = result.code
            if code in combined_results:
                # Обновляем существующий результат
                combined_results[code]['semantic_score'] = result.score
            else:
                # Добавляем новый результат
                combined_results[code] = {
                    'result': result,
                    'token_score': 0.0,
                    'semantic_score': result.score
                }

        # Вычисляем комбинированные оценки
        final_results = []
        for code, data in combined_results.items():
            result = data['result']

            # Взвешенная комбинация оценок
            combined_score = (
                data['token_score'] * self.config.token_id_weight +
                data['semantic_score'] * self.config.semantic_weight
            )

            # Создаем новый результат с комбинированной оценкой
            hybrid_result = SearchResult(
                code=result.code,
                raw_name=result.raw_name,
                tokens=result.tokens,
                token_vectors=result.token_vectors,
                parameters=result.parameters,
                score=combined_score,
                match_type='hybrid',
                matched_tokens=getattr(result, 'matched_tokens', None),
                similarity_score=data['semantic_score']
            )
            final_results.append(hybrid_result)

        # Сортируем по комбинированной оценке
        final_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Hybrid search found {len(final_results)} results")
        return final_results[:top_k]

    def search_extended_hybrid(self, token_ids: List[int], query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Расширенный гибридный поиск с дополнительными методами поиска по ID токенов

        Включает:
        1. Точное совпадение токенов (высокий вес)
        2. Частичное совпадение токенов (средний вес)
        3. Семантический поиск (базовый вес)
        4. Поиск по подмножествам токенов
        5. Поиск по техническим паттернам

        Args:
            token_ids: Список ID токенов для поиска
            query_embedding: Эмбеддинг запроса
            top_k: Максимальное количество результатов

        Returns:
            Список результатов расширенного поиска
        """
        logger.info(f"Starting extended hybrid search with {len(token_ids)} tokens")

        # Веса для разных методов поиска
        weights = {
            'exact_match': 0.4,      # Точное совпадение всех токенов
            'partial_match': 0.25,   # Частичное совпадение токенов
            'semantic': 0.2,         # Семантическое сходство
            'subset_match': 0.1,     # Совпадение подмножеств
            'technical_boost': 0.05  # Бонус за технические термины
        }

        # Собираем результаты от всех методов
        all_results = {}

        # 1. Точное совпадение токенов (все токены присутствуют)
        exact_results = self._search_exact_token_match(token_ids)
        for result in exact_results:
            code = result.code
            if code not in all_results:
                all_results[code] = {'result': result, 'scores': {}}
            all_results[code]['scores']['exact_match'] = result.score

        # 2. Частичное совпадение токенов (большинство токенов присутствуют)
        partial_results = self._search_partial_token_match(token_ids, min_match_ratio=0.5)
        for result in partial_results:
            code = result.code
            if code not in all_results:
                all_results[code] = {'result': result, 'scores': {}}
            all_results[code]['scores']['partial_match'] = result.score

        # 3. Семантический поиск
        semantic_results = self.search_by_similarity(query_embedding, top_k * 2)
        for result in semantic_results:
            code = result.code
            if code not in all_results:
                all_results[code] = {'result': result, 'scores': {}}
            all_results[code]['scores']['semantic'] = result.score

        # 4. Поиск по подмножествам токенов
        subset_results = self._search_token_subsets(token_ids)
        for result in subset_results:
            code = result.code
            if code not in all_results:
                all_results[code] = {'result': result, 'scores': {}}
            all_results[code]['scores']['subset_match'] = result.score

        # 5. Технический бонус для специальных терминов
        technical_boost = self._calculate_technical_boost(token_ids, all_results)

        # Вычисляем комбинированные оценки
        final_results = []
        for code, data in all_results.items():
            result = data['result']
            scores = data['scores']

            # Взвешенная комбинация всех оценок
            combined_score = 0.0
            for method, weight in weights.items():
                if method == 'technical_boost':
                    combined_score += technical_boost.get(code, 0.0) * weight
                else:
                    combined_score += scores.get(method, 0.0) * weight

            # Создаем результат с расширенной информацией
            extended_result = SearchResult(
                code=result.code,
                raw_name=result.raw_name,
                tokens=result.tokens,
                token_vectors=result.token_vectors,
                parameters=result.parameters,
                score=combined_score,
                match_type='extended_hybrid',
                matched_tokens=getattr(result, 'matched_tokens', None),
                similarity_score=scores.get('semantic', 0.0)
            )

            # Добавляем детальную информацию о методах
            extended_result.method_scores = scores
            extended_result.technical_boost = technical_boost.get(code, 0.0)

            final_results.append(extended_result)

        # Сортируем по комбинированной оценке
        final_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Extended hybrid search found {len(final_results)} results")
        return final_results[:top_k]

    def _search_exact_token_match(self, token_ids: List[int]) -> List[SearchResult]:
        """
        Поиск записей с точным совпадением всех токенов

        Args:
            token_ids: Список ID токенов

        Returns:
            Список результатов с точным совпадением
        """
        if not token_ids:
            return []

        # Находим записи, содержащие ВСЕ токены
        record_matches = {}

        for token_id in token_ids:
            if token_id in self.token_id_index:
                for record_idx in self.token_id_index[token_id]:
                    if record_idx not in record_matches:
                        record_matches[record_idx] = set()
                    record_matches[record_idx].add(token_id)

        # Фильтруем только записи с полным совпадением
        exact_matches = {
            idx: tokens for idx, tokens in record_matches.items()
            if len(tokens) == len(token_ids)
        }

        results = []
        for record_idx, matched_tokens in exact_matches.items():
            row = self.data_df.iloc[record_idx]

            # Оценка = 1.0 для точного совпадения
            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=1.0,
                match_type='exact_token_match',
                matched_tokens=list(matched_tokens)
            )
            results.append(result)

        logger.info(f"Exact token match found {len(results)} results")
        return results

    def _search_partial_token_match(self, token_ids: List[int], min_match_ratio: float = 0.5) -> List[SearchResult]:
        """
        Поиск записей с частичным совпадением токенов

        Args:
            token_ids: Список ID токенов
            min_match_ratio: Минимальная доля совпадающих токенов

        Returns:
            Список результатов с частичным совпадением
        """
        if not token_ids:
            return []

        min_matches = max(1, int(len(token_ids) * min_match_ratio))

        # Подсчитываем совпадения для каждой записи
        record_matches = {}

        for token_id in token_ids:
            if token_id in self.token_id_index:
                for record_idx in self.token_id_index[token_id]:
                    if record_idx not in record_matches:
                        record_matches[record_idx] = set()
                    record_matches[record_idx].add(token_id)

        # Фильтруем по минимальному количеству совпадений
        partial_matches = {
            idx: tokens for idx, tokens in record_matches.items()
            if len(tokens) >= min_matches and len(tokens) < len(token_ids)  # Исключаем точные совпадения
        }

        results = []
        for record_idx, matched_tokens in partial_matches.items():
            row = self.data_df.iloc[record_idx]

            # Оценка = доля совпадающих токенов
            score = len(matched_tokens) / len(token_ids)

            result = SearchResult(
                code=row.get('Код', ''),
                raw_name=row.get('Raw_Name', ''),
                tokens=row.get('tokenizer', ''),
                token_vectors=row.get('token_vectors', ''),
                parameters=row.get('parameters', ''),
                score=score,
                match_type='partial_token_match',
                matched_tokens=list(matched_tokens)
            )
            results.append(result)

        # Сортируем по убыванию оценки
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Partial token match found {len(results)} results")
        return results

    def _search_token_subsets(self, token_ids: List[int]) -> List[SearchResult]:
        """
        Поиск по подмножествам токенов (комбинации из 2-3 токенов)

        Args:
            token_ids: Список ID токенов

        Returns:
            Список результатов поиска по подмножествам
        """
        if len(token_ids) < 2:
            return []

        from itertools import combinations

        results = []
        subset_scores = {}

        # Генерируем подмножества размером от 2 до min(3, len(token_ids))
        max_subset_size = min(3, len(token_ids))

        for subset_size in range(2, max_subset_size + 1):
            for token_subset in combinations(token_ids, subset_size):
                subset_results = self.search_by_token_ids(list(token_subset), top_k=20)

                for result in subset_results:
                    code = result.code
                    # Оценка = (размер подмножества / общее количество токенов) * оценка совпадения
                    subset_score = (subset_size / len(token_ids)) * result.score * 0.7  # Понижающий коэффициент

                    if code not in subset_scores or subset_score > subset_scores[code]['score']:
                        subset_scores[code] = {
                            'result': result,
                            'score': subset_score,
                            'subset_size': subset_size
                        }

        # Создаем финальные результаты
        for code, data in subset_scores.items():
            result = data['result']

            subset_result = SearchResult(
                code=result.code,
                raw_name=result.raw_name,
                tokens=result.tokens,
                token_vectors=result.token_vectors,
                parameters=result.parameters,
                score=data['score'],
                match_type='token_subset',
                matched_tokens=getattr(result, 'matched_tokens', None)
            )
            results.append(subset_result)

        # Сортируем по убыванию оценки
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Token subset search found {len(results)} results")
        return results[:20]  # Ограничиваем количество результатов

    def _calculate_technical_boost(self, token_ids: List[int], all_results: Dict) -> Dict[str, float]:
        """
        Вычисление технического бонуса для записей с техническими терминами

        Args:
            token_ids: Список ID токенов запроса
            all_results: Словарь всех результатов поиска

        Returns:
            Словарь с техническими бонусами для каждого кода товара
        """
        technical_boost = {}

        # Паттерны технических терминов (можно расширить)
        technical_patterns = {
            'gost': ['ГОСТ', 'гост'],
            'ip_rating': ['IP', 'ip'],
            'voltage': ['V', 'В', 'вольт'],
            'current': ['A', 'А', 'ампер'],
            'power': ['W', 'Вт', 'ватт'],
            'frequency': ['Hz', 'Гц', 'герц'],
            'temperature': ['K', 'К', '°C', '°К'],
            'thread': ['М', 'M', 'х', 'x'],
            'dimensions': ['мм', 'см', 'м', 'mm', 'cm'],
            'volume': ['л', 'мл', 'l', 'ml']
        }

        # Получаем токены запроса для анализа
        query_tokens = []
        if hasattr(self, 'tokenizer') and self.tokenizer:
            try:
                # Пытаемся получить текстовые токены из векторизатора
                for token_id in token_ids:
                    if (hasattr(self.tokenizer, '_vectorizer') and
                        self.tokenizer._vectorizer and
                        hasattr(self.tokenizer._vectorizer, 'id_to_token')):
                        token_text = self.tokenizer._vectorizer.id_to_token.get(token_id, '')
                        if token_text:
                            query_tokens.append(token_text.lower())
            except Exception as e:
                logger.warning(f"Could not extract token texts: {e}")

        # Анализируем каждый результат
        for code, data in all_results.items():
            result = data['result']
            boost = 0.0

            # Анализируем токены записи
            record_tokens = []
            if result.tokens:
                record_tokens = result.tokens.lower().split()

            # Анализируем параметры записи
            parameters_text = ""
            if result.parameters and isinstance(result.parameters, str):
                parameters_text = result.parameters.lower()

            # Подсчитываем технические совпадения
            technical_matches = 0
            total_technical_patterns = 0

            for pattern_type, patterns in technical_patterns.items():
                total_technical_patterns += len(patterns)

                for pattern in patterns:
                    pattern_lower = pattern.lower()

                    # Проверяем совпадения в токенах запроса
                    query_has_pattern = any(pattern_lower in token for token in query_tokens)

                    # Проверяем совпадения в токенах записи
                    record_has_pattern = any(pattern_lower in token for token in record_tokens)

                    # Проверяем совпадения в параметрах
                    params_has_pattern = pattern_lower in parameters_text

                    # Если паттерн есть и в запросе, и в записи - даем бонус
                    if query_has_pattern and (record_has_pattern or params_has_pattern):
                        technical_matches += 1

                        # Дополнительный бонус для важных технических терминов
                        if pattern_type in ['gost', 'ip_rating', 'thread']:
                            boost += 0.1
                        else:
                            boost += 0.05

            # Нормализуем бонус
            if technical_matches > 0:
                boost = min(boost, 0.3)  # Максимальный технический бонус 0.3
                technical_boost[code] = boost
            else:
                technical_boost[code] = 0.0

        logger.info(f"Technical boost calculated for {len(technical_boost)} results")
        return technical_boost

    def search_by_tokens(self, query: str, method: str = "hybrid", top_k: int = 10) -> List[SearchResult]:
        """
        Основная функция поиска по текстовому запросу

        Args:
            query: Текстовый запрос
            method: Метод поиска ('token_id', 'semantic', 'hybrid', 'extended_hybrid',
                   'advanced_hybrid', 'prefix', 'inverted_index', 'tfidf', 'lsh', 'spatial')
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.tokenizer or not self.vectorizer:
            logger.error("Tokenizer or vectorizer not available")
            # Fallback к простому текстовому поиску
            return self._fallback_text_search(query, top_k)

        logger.info(f"Searching for: '{query}' using method: {method}")

        try:
            # Токенизируем и векторизуем запрос
            tokenization_result = self.tokenizer.tokenize_text(query, include_vectors=True)

            if 'vectorization' not in tokenization_result:
                logger.error("Query vectorization failed")
                return []

            vectorization = tokenization_result['vectorization']

            # Извлекаем ID токенов и эмбеддинги
            token_ids = vectorization.get('token_ids', [])
            query_embedding = vectorization.get('aggregated_embedding')

            logger.info(f"Query tokens: {tokenization_result['tokens']}")
            logger.info(f"Query token IDs: {token_ids}")

            # Выполняем поиск в зависимости от метода
            if method == "token_id":
                return self.search_by_token_ids(token_ids, top_k)
            elif method == "semantic":
                if query_embedding is not None:
                    return self.search_by_similarity(query_embedding, top_k)
                else:
                    logger.warning("No embedding available, falling back to token ID search")
                    return self.search_by_token_ids(token_ids, top_k)
            elif method == "hybrid":
                if query_embedding is not None:
                    return self.search_hybrid(token_ids, query_embedding, top_k)
                else:
                    logger.warning("No embedding available, using token ID search only")
                    return self.search_by_token_ids(token_ids, top_k)
            elif method == "extended_hybrid":
                if query_embedding is not None:
                    return self.search_extended_hybrid(token_ids, query_embedding, top_k)
                else:
                    logger.warning("No embedding available, falling back to hybrid search")
                    return self.search_hybrid(token_ids, query_embedding, top_k) if query_embedding else self.search_by_token_ids(token_ids, top_k)
            elif method == "advanced_hybrid":
                return self.advanced_hybrid_search(query, top_k)
            elif method == "prefix":
                # Используем первое слово как префикс
                words = re.findall(r'\b\w+\b', query.lower())
                if words:
                    return self.prefix_search(words[0], top_k)
                else:
                    return []
            elif method == "inverted_index":
                return self.inverted_index_search(query, top_k)
            elif method == "tfidf":
                return self.tfidf_search(query, top_k)
            elif method == "lsh":
                return self.lsh_search(query, top_k)
            elif method == "spatial":
                # Пробуем разные методы создания эмбеддинга запроса в порядке качества
                embedding_to_use = None
                method_used = "none"

                # Метод 1: Контекстуальный эмбеддинг (лучшее качество)
                embedding_to_use = self._create_contextual_embedding(query)
                if embedding_to_use is not None:
                    method_used = "contextual"
                    logger.debug("Using contextual embedding for spatial search")

                # Метод 2: Используем эмбеддинг из токенизации (если доступен)
                if embedding_to_use is None and query_embedding is not None:
                    embedding_to_use = query_embedding
                    method_used = "vectorizer"
                    logger.debug("Using vectorizer embedding for spatial search")

                # Метод 3: Создаем эмбеддинг из токенов
                if embedding_to_use is None:
                    embedding_to_use = self._create_query_embedding_from_tokens(query)
                    if embedding_to_use is not None:
                        method_used = "token-based"
                        logger.debug("Using token-based embedding for spatial search")

                # Метод 4: Fallback к улучшенному TF-IDF эмбеддингу
                if embedding_to_use is None:
                    embedding_to_use = self._create_fallback_query_embedding(query)
                    if embedding_to_use is not None:
                        method_used = "tfidf-fallback"
                        logger.info("Using enhanced TF-IDF fallback embedding for spatial search")

                if embedding_to_use is not None:
                    logger.info(f"Spatial search using {method_used} embedding method")
                    return self.spatial_search(embedding_to_use, top_k)
                else:
                    logger.warning("No embedding available for spatial search - all methods failed")
                    return []
            else:
                logger.error(f"Unknown search method: {method}")
                return []

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def get_search_stats(self) -> Dict[str, Any]:
        """Получение статистики поискового движка"""
        if self.data_df is None:
            return {'status': 'not_loaded'}

        stats = {
            'status': 'ready',
            'total_records': len(self.data_df),
            'unique_token_ids': len(self.token_id_index),
            'embeddings_available': self.embeddings_matrix is not None,
            'config': {
                'token_id_weight': self.config.token_id_weight,
                'semantic_weight': self.config.semantic_weight,
                'similarity_threshold': self.config.similarity_threshold,
                'max_results': self.config.max_results
            },
            # Информация о новых индексах
            'advanced_search_methods': {
                'trie_available': self.trie is not None,
                'inverted_index_available': self.inverted_index is not None,
                'tfidf_available': self.tfidf_vectorizer is not None and self.tfidf_matrix is not None,
                'lsh_available': self.lsh_index is not None and DATASKETCH_AVAILABLE,
                'faiss_available': self.faiss_index is not None and FAISS_AVAILABLE,
                'graph_available': self.similarity_graph is not None and NETWORKX_AVAILABLE
            },
            # Информация о доступности библиотек
            'library_availability': {
                'datasketch': DATASKETCH_AVAILABLE,
                'faiss': FAISS_AVAILABLE,
                'networkx': NETWORKX_AVAILABLE
            }
        }

        if self.embeddings_matrix is not None:
            stats['embeddings_shape'] = self.embeddings_matrix.shape

        if self.inverted_index:
            stats['inverted_index_terms'] = len(self.inverted_index)

        if self.tfidf_matrix is not None:
            stats['tfidf_matrix_shape'] = self.tfidf_matrix.shape

        if self.lsh_index:
            stats['lsh_items'] = len(self.minhashes)

        if self.faiss_index:
            stats['faiss_total_vectors'] = self.faiss_index.ntotal

        if self.similarity_graph and nx is not None:
            stats['graph_nodes'] = self.similarity_graph.number_of_nodes()
            stats['graph_edges'] = self.similarity_graph.number_of_edges()

        return stats

    # Новые методы поиска

    def prefix_search(self, prefix: str, top_k: int = 10) -> List[SearchResult]:
        """
        Префиксный поиск с использованием Trie

        Args:
            prefix: Префикс для поиска
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.trie or len(prefix) < self.config.trie_min_prefix_length:
            return []

        try:
            # Получаем индексы записей с данным префиксом
            record_indices = self.trie.search_prefix(prefix)

            if not record_indices:
                return []

            # Создаем результаты
            results = []
            for idx in record_indices[:top_k]:
                if idx < len(self.data_df):
                    row = self.data_df.iloc[idx]

                    # Вычисляем оценку на основе длины префикса
                    raw_name = str(row.get('Raw_Name', ''))
                    score = len(prefix) / max(len(raw_name), 1)

                    result = SearchResult(
                        code=row.get('Код', ''),
                        raw_name=raw_name,
                        tokens=row.get('tokenizer', ''),
                        token_vectors=row.get('token_vectors', ''),
                        parameters=row.get('parameters', ''),
                        score=score,
                        match_type='prefix',
                        matched_tokens=[prefix]
                    )
                    results.append(result)

            # Сортируем по оценке
            results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Prefix search found {len(results)} results for '{prefix}'")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in prefix search: {e}")
            return []

    def inverted_index_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск с использованием обратного индекса

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.inverted_index:
            return []

        try:
            # Разбиваем запрос на токены
            query_tokens = re.findall(r'\b\w+\b', query.lower())

            if not query_tokens:
                return []

            # Находим пересечения для каждого токена
            candidate_sets = []
            for token in query_tokens:
                if token in self.inverted_index:
                    candidate_sets.append(self.inverted_index[token])

            if not candidate_sets:
                return []

            # Вычисляем оценки для кандидатов
            record_scores = {}

            for candidate_set in candidate_sets:
                for record_idx in candidate_set:
                    if record_idx not in record_scores:
                        record_scores[record_idx] = 0
                    record_scores[record_idx] += 1

            # Нормализуем оценки
            max_score = len(query_tokens)
            for idx in record_scores:
                record_scores[idx] = record_scores[idx] / max_score

            # Сортируем по оценке
            sorted_candidates = sorted(record_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Создаем результаты
            results = []
            for record_idx, score in sorted_candidates:
                if record_idx < len(self.data_df):
                    row = self.data_df.iloc[record_idx]

                    result = SearchResult(
                        code=row.get('Код', ''),
                        raw_name=row.get('Raw_Name', ''),
                        tokens=row.get('tokenizer', ''),
                        token_vectors=row.get('token_vectors', ''),
                        parameters=row.get('parameters', ''),
                        score=score,
                        match_type='inverted_index',
                        matched_tokens=query_tokens
                    )
                    results.append(result)

            logger.info(f"Inverted index search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in inverted index search: {e}")
            return []

    def tfidf_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        TF-IDF поиск с косинусным сходством

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []

        try:
            # Векторизуем запрос
            query_vector = self.tfidf_vectorizer.transform([query])

            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

            # Находим топ-K результатов
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Создаем результаты
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0:  # Только ненулевые сходства
                    row = self.data_df.iloc[idx]

                    result = SearchResult(
                        code=row.get('Код', ''),
                        raw_name=row.get('Raw_Name', ''),
                        tokens=row.get('tokenizer', ''),
                        token_vectors=row.get('token_vectors', ''),
                        parameters=row.get('parameters', ''),
                        score=similarity,
                        match_type='tfidf',
                        similarity_score=similarity
                    )
                    results.append(result)

            logger.info(f"TF-IDF search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []

    def lsh_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        LSH поиск для быстрого нахождения похожих элементов

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not DATASKETCH_AVAILABLE:
            logger.warning("LSH search not available: datasketch not installed")
            return []

        if not self.lsh_index or not self.minhashes:
            return []

        try:
            # Создаем MinHash для запроса
            query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

            if not query_tokens:
                return []

            query_minhash = MinHash(num_perm=self.config.lsh_num_perm)
            for token in query_tokens:
                query_minhash.update(token.encode('utf8'))

            # Ищем похожие элементы
            similar_indices = self.lsh_index.query(query_minhash)

            # Вычисляем точные Jaccard сходства
            results = []
            for idx in similar_indices:
                if idx in self.minhashes:
                    jaccard_sim = query_minhash.jaccard(self.minhashes[idx])

                    if jaccard_sim >= self.config.lsh_threshold:
                        row = self.data_df.iloc[idx]

                        result = SearchResult(
                            code=row.get('Код', ''),
                            raw_name=row.get('Raw_Name', ''),
                            tokens=row.get('tokenizer', ''),
                            token_vectors=row.get('token_vectors', ''),
                            parameters=row.get('parameters', ''),
                            score=jaccard_sim,
                            match_type='lsh',
                            similarity_score=jaccard_sim
                        )
                        results.append(result)

            # Сортируем по Jaccard сходству
            results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"LSH search found {len(results)} results")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in LSH search: {e}")
            return []

    def spatial_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Пространственный поиск с использованием FAISS

        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not FAISS_AVAILABLE:
            logger.warning("Spatial search not available: faiss not installed")
            return []

        if not self.faiss_index:
            return []

        try:
            # Нормализуем запрос для косинусного сходства
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning("Query embedding has zero norm, cannot perform spatial search")
                return []

            query_normalized = query_embedding / query_norm
            query_normalized = query_normalized.reshape(1, -1).astype('float32')

            # Ищем ближайших соседей
            similarities, indices = self.faiss_index.search(query_normalized, top_k)

            # Создаем результаты
            results = []
            scores_for_logging = []

            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and similarity > 0:  # Валидный индекс и положительное сходство
                    row = self.data_df.iloc[idx]
                    scores_for_logging.append(float(similarity))

                    result = SearchResult(
                        code=row.get('Код', ''),
                        raw_name=row.get('Raw_Name', ''),
                        tokens=row.get('tokenizer', ''),
                        token_vectors=row.get('token_vectors', ''),
                        parameters=row.get('parameters', ''),
                        score=float(similarity),
                        match_type='spatial',
                        similarity_score=float(similarity)
                    )
                    results.append(result)

            # Логируем статистику оценок
            if scores_for_logging:
                logger.info(f"Spatial search found {len(results)} results")
                logger.info(f"Similarity scores: min={min(scores_for_logging):.4f}, max={max(scores_for_logging):.4f}, avg={sum(scores_for_logging)/len(scores_for_logging):.4f}")
                logger.info(f"Top 3 scores: {sorted(scores_for_logging, reverse=True)[:3]}")
            else:
                logger.info("Spatial search found 0 results")

            return results

        except Exception as e:
            logger.error(f"Error in spatial search: {e}")
            return []

    def graph_search(self, reference_idx: int, top_k: int = 10) -> List[SearchResult]:
        """
        Graph-based поиск для нахождения связанных элементов

        Args:
            reference_idx: Индекс референсного элемента
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("Graph search not available: networkx not installed")
            return []

        if not self.similarity_graph:
            return []

        try:
            if reference_idx not in self.similarity_graph:
                return []

            # Получаем соседей с весами рёбер
            neighbors = []
            for neighbor in self.similarity_graph.neighbors(reference_idx):
                edge_data = self.similarity_graph.get_edge_data(reference_idx, neighbor)
                weight = edge_data.get('weight', 0)
                neighbors.append((neighbor, weight))

            # Сортируем по весу (сходству)
            neighbors.sort(key=lambda x: x[1], reverse=True)

            # Создаем результаты
            results = []
            for neighbor_idx, similarity in neighbors[:top_k]:
                row = self.data_df.iloc[neighbor_idx]

                result = SearchResult(
                    code=row.get('Код', ''),
                    raw_name=row.get('Raw_Name', ''),
                    tokens=row.get('tokenizer', ''),
                    token_vectors=row.get('token_vectors', ''),
                    parameters=row.get('parameters', ''),
                    score=similarity,
                    match_type='graph',
                    similarity_score=similarity
                )
                results.append(result)

            logger.info(f"Graph search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []

    def advanced_hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Продвинутый гибридный поиск, использующий все доступные методы

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        try:
            logger.info(f"Starting advanced hybrid search for: '{query}'")

            # Собираем результаты от всех методов
            all_results = {}

            # 1. Традиционные методы
            if self.tokenizer and self.vectorizer:
                # Токенизируем запрос
                token_ids = self.tokenizer.tokenize_to_ids(query)
                query_embedding = None

                if token_ids:
                    try:
                        query_embedding = self.vectorizer.vectorize_tokens(token_ids)
                    except:
                        pass

                # Token ID поиск
                token_results = self.search_by_token_ids(token_ids, top_k * 2)
                for result in token_results:
                    code = result.code
                    if code not in all_results:
                        all_results[code] = {'result': result, 'scores': {}}
                    all_results[code]['scores']['token_id'] = result.score

                # Семантический поиск
                if query_embedding is not None:
                    semantic_results = self.search_by_similarity(query_embedding, top_k * 2)
                    for result in semantic_results:
                        code = result.code
                        if code not in all_results:
                            all_results[code] = {'result': result, 'scores': {}}
                        all_results[code]['scores']['semantic'] = result.score

                    # Spatial поиск (FAISS)
                    if self.config.enable_spatial_search:
                        spatial_results = self.spatial_search(query_embedding, top_k)
                        for result in spatial_results:
                            code = result.code
                            if code not in all_results:
                                all_results[code] = {'result': result, 'scores': {}}
                            all_results[code]['scores']['spatial'] = result.score

            # 2. Новые методы поиска

            # Префиксный поиск
            if self.config.enable_trie_search:
                words = re.findall(r'\b\w+\b', query.lower())
                for word in words:
                    if len(word) >= self.config.trie_min_prefix_length:
                        prefix_results = self.prefix_search(word, top_k)
                        for result in prefix_results:
                            code = result.code
                            if code not in all_results:
                                all_results[code] = {'result': result, 'scores': {}}
                            all_results[code]['scores']['prefix'] = result.score

            # Inverted Index поиск
            if self.config.enable_inverted_index:
                inverted_results = self.inverted_index_search(query, top_k)
                for result in inverted_results:
                    code = result.code
                    if code not in all_results:
                        all_results[code] = {'result': result, 'scores': {}}
                    all_results[code]['scores']['inverted_index'] = result.score

            # TF-IDF поиск
            if self.config.enable_tfidf_search:
                tfidf_results = self.tfidf_search(query, top_k)
                for result in tfidf_results:
                    code = result.code
                    if code not in all_results:
                        all_results[code] = {'result': result, 'scores': {}}
                    all_results[code]['scores']['tfidf'] = result.score

            # LSH поиск
            if self.config.enable_lsh_search:
                lsh_results = self.lsh_search(query, top_k)
                for result in lsh_results:
                    code = result.code
                    if code not in all_results:
                        all_results[code] = {'result': result, 'scores': {}}
                    all_results[code]['scores']['lsh'] = result.score

            # 3. Комбинируем оценки
            final_results = []
            for code, data in all_results.items():
                result = data['result']
                scores = data['scores']

                # Взвешенная комбинация всех оценок
                combined_score = 0.0
                total_weight = 0.0

                # Традиционные методы
                if 'token_id' in scores:
                    combined_score += scores['token_id'] * self.config.token_id_weight
                    total_weight += self.config.token_id_weight

                if 'semantic' in scores:
                    combined_score += scores['semantic'] * self.config.semantic_weight
                    total_weight += self.config.semantic_weight

                # Новые методы
                if 'prefix' in scores and self.config.enable_trie_search:
                    combined_score += scores['prefix'] * self.config.trie_weight
                    total_weight += self.config.trie_weight

                if 'inverted_index' in scores and self.config.enable_inverted_index:
                    combined_score += scores['inverted_index'] * self.config.inverted_index_weight
                    total_weight += self.config.inverted_index_weight

                if 'tfidf' in scores and self.config.enable_tfidf_search:
                    combined_score += scores['tfidf'] * self.config.tfidf_weight
                    total_weight += self.config.tfidf_weight

                if 'lsh' in scores and self.config.enable_lsh_search:
                    combined_score += scores['lsh'] * self.config.lsh_weight
                    total_weight += self.config.lsh_weight

                if 'spatial' in scores and self.config.enable_spatial_search:
                    combined_score += scores['spatial'] * self.config.spatial_weight
                    total_weight += self.config.spatial_weight

                # Нормализуем оценку
                if total_weight > 0:
                    combined_score = combined_score / total_weight

                # Создаем результат с комбинированной оценкой
                hybrid_result = SearchResult(
                    code=result.code,
                    raw_name=result.raw_name,
                    tokens=result.tokens,
                    token_vectors=result.token_vectors,
                    parameters=result.parameters,
                    score=combined_score,
                    match_type='advanced_hybrid',
                    matched_tokens=getattr(result, 'matched_tokens', None),
                    similarity_score=scores.get('semantic', 0.0)
                )

                # Добавляем детальные оценки как атрибут
                hybrid_result.method_scores = scores

                final_results.append(hybrid_result)

            # Сортируем по комбинированной оценке
            final_results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Advanced hybrid search found {len(final_results)} results")
            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Error in advanced hybrid search: {e}")
            return []

    def _fallback_text_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Простой текстовый поиск как fallback когда токенизатор недоступен

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if self.data_df is None:
            logger.warning("No data available for fallback search")
            return []

        logger.info(f"Using fallback text search for query: '{query}'")

        try:
            # Простой поиск по подстроке в названиях
            query_lower = query.lower()
            results = []

            for idx, row in self.data_df.iterrows():
                # Ищем в разных полях названий
                search_fields = ['Raw_Name', 'Cleaned_Name', 'Lemmatized_Name', 'Normalized_Name']
                found = False
                score = 0.0

                for field in search_fields:
                    if field in row and pd.notna(row[field]):
                        text = str(row[field]).lower()
                        if query_lower in text:
                            found = True
                            # Простая оценка релевантности
                            score = max(score, len(query_lower) / len(text))

                if found:
                    result = SearchResult(
                        code=row.get('Код', ''),
                        raw_name=row.get('Raw_Name', ''),
                        tokens=row.get('tokenizer', ''),
                        token_vectors=row.get('token_vectors', ''),
                        parameters=row.get('parameters', ''),
                        score=score,
                        match_type='fallback_text',
                        similarity_score=score
                    )
                    results.append(result)

            # Сортируем по релевантности
            results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Fallback text search found {len(results)} results")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            return []
