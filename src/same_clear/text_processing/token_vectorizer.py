"""
Модуль векторизации токенов для системы поиска аналогов

Этот модуль предоставляет различные методы преобразования токенов в числовые векторы:
- Простая векторизация (токен → ID)
- Векторизация с эмбеддингами (токен → вектор)
- Гибридная векторизация (токен → ID + вектор)
"""

import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class VectorizerConfig:
    """Конфигурация векторизатора токенов"""
    vectorization_type: str = "simple"  # simple, embedding, hybrid
    vocabulary_size: int = 50000
    embedding_dim: int = 128
    min_token_frequency: int = 1
    max_token_frequency: float = 0.95  # Максимальная частота токена (для фильтрации стоп-слов)
    
    # Специальные токены
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    
    # Настройки эмбеддингов
    use_pretrained_embeddings: bool = True
    pretrained_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fallback_model_name: str = "all-MiniLM-L6-v2"  # Более легкая модель как fallback
    enable_embedding_cache: bool = True
    cache_dir: str = "embeddings_cache"
    
    # Сохранение и загрузка
    save_vocabulary: bool = True
    vocabulary_path: str = "vocabulary.json"
    embeddings_path: str = "token_embeddings.npy"


class TokenVectorizer:
    """Класс для векторизации токенов"""
    
    def __init__(self, config: VectorizerConfig = None):
        self.config = config or VectorizerConfig()
        
        # Словари для преобразования
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_frequencies = Counter()
        
        # Эмбеддинги токенов
        self.token_embeddings = None
        self.embedding_model = None
        
        # Статистика
        self.vocabulary_size = 0
        self.is_fitted = False
        
        logger.info(f"TokenVectorizer initialized with type: {self.config.vectorization_type}")
    
    def fit(self, token_lists: List[List[str]]) -> 'TokenVectorizer':
        """
        Обучение векторизатора на списках токенов
        
        Args:
            token_lists: Список списков токенов для обучения
            
        Returns:
            self для цепочки вызовов
        """
        logger.info(f"Fitting vectorizer on {len(token_lists)} token lists")
        
        # Подсчет частот токенов
        all_tokens = []
        for tokens in token_lists:
            all_tokens.extend(tokens)
            self.token_frequencies.update(tokens)
        
        logger.info(f"Found {len(self.token_frequencies)} unique tokens")
        
        # Фильтрация токенов по частоте
        filtered_tokens = self._filter_tokens_by_frequency()
        
        # Создание словаря токен → ID
        self._build_vocabulary(filtered_tokens)
        
        # Создание эмбеддингов если необходимо
        if self.config.vectorization_type in ["embedding", "hybrid"]:
            self._build_embeddings(filtered_tokens)
        
        self.is_fitted = True
        
        # Сохранение словаря
        if self.config.save_vocabulary:
            self._save_vocabulary()
        
        logger.info(f"Vectorizer fitted successfully. Vocabulary size: {self.vocabulary_size}")
        return self
    
    def _filter_tokens_by_frequency(self) -> List[str]:
        """Фильтрация токенов по частоте"""
        total_tokens = sum(self.token_frequencies.values())
        filtered_tokens = []
        
        for token, freq in self.token_frequencies.most_common():
            # Фильтруем слишком редкие токены
            if freq < self.config.min_token_frequency:
                break
            
            # Фильтруем слишком частые токены (стоп-слова)
            if freq / total_tokens > self.config.max_token_frequency:
                continue
            
            filtered_tokens.append(token)
            
            # Ограничиваем размер словаря
            if len(filtered_tokens) >= self.config.vocabulary_size - 4:  # -4 для специальных токенов
                break
        
        logger.info(f"Filtered vocabulary: {len(filtered_tokens)} tokens")
        return filtered_tokens
    
    def _build_vocabulary(self, tokens: List[str]):
        """Построение словаря токен → ID"""
        # Добавляем специальные токены
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.start_token,
            self.config.end_token
        ]
        
        # Создаем словарь
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Сначала специальные токены
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Затем обычные токены
        for i, token in enumerate(tokens, start=len(special_tokens)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.vocabulary_size = len(self.token_to_id)
        logger.info(f"Built vocabulary with {self.vocabulary_size} tokens")
    
    def _build_embeddings(self, tokens: List[str]):
        """Построение эмбеддингов токенов"""
        logger.info("Building token embeddings...")
        
        if self.config.use_pretrained_embeddings:
            self._build_pretrained_embeddings(tokens)
        else:
            self._build_random_embeddings()
    
    def _build_pretrained_embeddings(self, tokens: List[str]):
        """Построение эмбеддингов с использованием предобученной модели"""
        try:
            from sentence_transformers import SentenceTransformer
            import os

            # Создаем директорию для кэша если нужно
            if self.config.enable_embedding_cache:
                os.makedirs(self.config.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.config.cache_dir, f"embeddings_{hash(str(sorted(self.token_to_id.keys())))}.npy")

                # Проверяем наличие кэшированных эмбеддингов
                if os.path.exists(cache_path):
                    logger.info(f"Loading cached embeddings from {cache_path}")
                    self.token_embeddings = np.load(cache_path)
                    logger.info(f"Loaded cached embeddings shape: {self.token_embeddings.shape}")
                    return

            # Пробуем загрузить основную модель
            model_loaded = False
            for model_name in [self.config.pretrained_model_name, self.config.fallback_model_name]:
                try:
                    logger.info(f"Loading pretrained model: {model_name}")
                    self.embedding_model = SentenceTransformer(model_name)
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    continue

            if not model_loaded:
                raise Exception("No pretrained models could be loaded")

            # Получаем размерность эмбеддингов из модели
            sample_embedding = self.embedding_model.encode(["test"])
            actual_dim = sample_embedding.shape[1]

            if actual_dim != self.config.embedding_dim:
                logger.warning(f"Model embedding dim ({actual_dim}) != config dim ({self.config.embedding_dim})")
                self.config.embedding_dim = actual_dim

            # Создаем эмбеддинги для всех токенов
            all_tokens = list(self.token_to_id.keys())
            logger.info(f"Generating embeddings for {len(all_tokens)} tokens")

            # Фильтруем специальные токены для лучшего качества
            filtered_tokens = []
            special_tokens = {self.config.pad_token, self.config.unk_token,
                            self.config.start_token, self.config.end_token}

            for token in all_tokens:
                if token in special_tokens:
                    # Для специальных токенов используем осмысленные замены
                    if token == self.config.unk_token:
                        filtered_tokens.append("неизвестный")
                    elif token == self.config.pad_token:
                        filtered_tokens.append("")
                    else:
                        filtered_tokens.append(token)
                else:
                    filtered_tokens.append(token)

            # Генерируем эмбеддинги пакетами с оптимизацией
            batch_size = 64  # Оптимальный размер батча
            embeddings_list = []

            for i in range(0, len(filtered_tokens), batch_size):
                batch_tokens = filtered_tokens[i:i + batch_size]
                # Убираем пустые токены для кодирования
                non_empty_tokens = [t for t in batch_tokens if t.strip()]
                if non_empty_tokens:
                    batch_embeddings = self.embedding_model.encode(
                        non_empty_tokens,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Нормализуем на уровне модели
                    )

                    # Восстанавливаем размерность для пустых токенов
                    full_batch_embeddings = []
                    non_empty_idx = 0
                    for token in batch_tokens:
                        if token.strip():
                            full_batch_embeddings.append(batch_embeddings[non_empty_idx])
                            non_empty_idx += 1
                        else:
                            # Для пустых токенов создаем нулевой вектор
                            full_batch_embeddings.append(np.zeros(batch_embeddings.shape[1]))

                    embeddings_list.append(np.array(full_batch_embeddings))
                else:
                    # Если все токены пустые, создаем нулевые векторы
                    embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    embeddings_list.append(np.zeros((len(batch_tokens), embedding_dim)))

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_tokens)}/{len(filtered_tokens)} tokens")

            # Объединяем все эмбеддинги
            self.token_embeddings = np.vstack(embeddings_list).astype(np.float32)

            # Дополнительная нормализация для консистентности
            norms = np.linalg.norm(self.token_embeddings, axis=1, keepdims=True)
            self.token_embeddings = self.token_embeddings / (norms + 1e-8)

            logger.info(f"Generated embeddings shape: {self.token_embeddings.shape}")
            logger.info(f"Average embedding norm: {np.mean(np.linalg.norm(self.token_embeddings, axis=1)):.4f}")

            # Сохраняем в кэш если включен
            if self.config.enable_embedding_cache:
                np.save(cache_path, self.token_embeddings)
                logger.info(f"Cached embeddings saved to {cache_path}")

        except ImportError:
            logger.warning("sentence-transformers not available, using random embeddings")
            self._build_random_embeddings()
        except Exception as e:
            logger.error(f"Error building pretrained embeddings: {e}")
            self._build_random_embeddings()
    
    def _build_random_embeddings(self):
        """Построение улучшенных случайных эмбеддингов"""
        logger.info(f"Building improved random embeddings ({self.vocabulary_size}, {self.config.embedding_dim})")

        # Используем Xavier/Glorot инициализацию для лучшего качества
        np.random.seed(42)  # Для воспроизводимости

        # Xavier инициализация: std = sqrt(2 / (fan_in + fan_out))
        fan_in = self.config.embedding_dim
        fan_out = self.vocabulary_size
        std = np.sqrt(2.0 / (fan_in + fan_out))

        self.token_embeddings = np.random.normal(
            0, std, (self.vocabulary_size, self.config.embedding_dim)
        ).astype(np.float32)

        # Улучшенная нормализация с учетом распределения
        norms = np.linalg.norm(self.token_embeddings, axis=1, keepdims=True)
        # Добавляем небольшой шум для избежания идентичных векторов
        noise = np.random.normal(0, 0.01, self.token_embeddings.shape).astype(np.float32)
        self.token_embeddings = (self.token_embeddings + noise) / (norms + 1e-8)

        # Дополнительная проверка качества
        final_norms = np.linalg.norm(self.token_embeddings, axis=1)
        logger.info(f"Random embeddings stats: mean_norm={np.mean(final_norms):.4f}, std_norm={np.std(final_norms):.4f}")

        # Создаем более разнообразные эмбеддинги для специальных токенов
        special_tokens = {self.config.pad_token, self.config.unk_token,
                         self.config.start_token, self.config.end_token}

        for token, token_id in self.token_to_id.items():
            if token in special_tokens:
                # Специальные векторы для специальных токенов
                if token == self.config.pad_token:
                    self.token_embeddings[token_id] = np.zeros(self.config.embedding_dim)
                elif token == self.config.unk_token:
                    # Средний вектор для неизвестных токенов
                    self.token_embeddings[token_id] = np.mean(self.token_embeddings, axis=0)
                # START и END токены остаются случайными но нормализованными
    
    def vectorize_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Векторизация списка токенов
        
        Args:
            tokens: Список токенов для векторизации
            
        Returns:
            Словарь с результатами векторизации
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before vectorization")
        
        result = {
            'original_tokens': tokens,
            'token_count': len(tokens)
        }
        
        if self.config.vectorization_type in ["simple", "hybrid"]:
            # Простая векторизация: токен → ID
            token_ids = []
            for token in tokens:
                token_id = self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
                token_ids.append(token_id)
            
            result['token_ids'] = token_ids
            result['ids_vector'] = np.array(token_ids, dtype=np.int32)
        
        if self.config.vectorization_type in ["embedding", "hybrid"]:
            # Векторизация эмбеддингами
            if self.token_embeddings is not None:
                embeddings = []
                token_weights = []  # Веса для взвешенного усреднения

                for token in tokens:
                    token_id = self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
                    embedding = self.token_embeddings[token_id]
                    embeddings.append(embedding)

                    # Простая схема весов: более редкие токены получают больший вес
                    # (можно улучшить с помощью TF-IDF весов)
                    if token in self.token_to_id:
                        # Обратная частота токена (приблизительно)
                        weight = 1.0 / (1.0 + self.token_to_id[token] / len(self.token_to_id))
                    else:
                        weight = 0.5  # Средний вес для неизвестных токенов
                    token_weights.append(weight)

                result['token_embeddings'] = np.array(embeddings)
                result['token_weights'] = np.array(token_weights)

                # Улучшенное агрегированное представление
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    weights_array = np.array(token_weights)

                    # Взвешенное среднее
                    weighted_sum = np.sum(embeddings_array * weights_array.reshape(-1, 1), axis=0)
                    weight_sum = np.sum(weights_array)

                    if weight_sum > 0:
                        result['aggregated_embedding'] = weighted_sum / weight_sum
                    else:
                        result['aggregated_embedding'] = np.mean(embeddings_array, axis=0)

                    # Дополнительные агрегации для разнообразия
                    result['max_pooled_embedding'] = np.max(embeddings_array, axis=0)
                    result['min_pooled_embedding'] = np.min(embeddings_array, axis=0)

                    # Нормализуем финальный вектор
                    norm = np.linalg.norm(result['aggregated_embedding'])
                    if norm > 0:
                        result['aggregated_embedding'] = result['aggregated_embedding'] / norm
                else:
                    result['aggregated_embedding'] = np.zeros(self.config.embedding_dim)
                    result['max_pooled_embedding'] = np.zeros(self.config.embedding_dim)
                    result['min_pooled_embedding'] = np.zeros(self.config.embedding_dim)
        
        return result
    
    def vectorize_text(self, text: str, tokenizer_func) -> Dict[str, Any]:
        """
        Векторизация текста с токенизацией
        
        Args:
            text: Исходный текст
            tokenizer_func: Функция токенизации
            
        Returns:
            Словарь с результатами векторизации
        """
        # Токенизируем текст
        tokenization_result = tokenizer_func(text)
        tokens = tokenization_result.get('tokens', [])
        
        # Векторизуем токены
        vectorization_result = self.vectorize_tokens(tokens)
        
        # Объединяем результаты
        result = {
            'original_text': text,
            'tokenization': tokenization_result,
            'vectorization': vectorization_result
        }
        
        return result
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """Получение информации о токене"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting token info")
        
        token_id = self.token_to_id.get(token)
        if token_id is None:
            return {
                'token': token,
                'found': False,
                'id': self.token_to_id[self.config.unk_token],
                'frequency': 0
            }
        
        info = {
            'token': token,
            'found': True,
            'id': token_id,
            'frequency': self.token_frequencies.get(token, 0)
        }
        
        if self.token_embeddings is not None:
            info['embedding'] = self.token_embeddings[token_id]
            info['embedding_norm'] = np.linalg.norm(info['embedding'])
        
        return info
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Получение статистики словаря"""
        if not self.is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'vocabulary_size': self.vocabulary_size,
            'total_token_frequency': sum(self.token_frequencies.values()),
            'unique_tokens': len(self.token_frequencies),
            'config': {
                'vectorization_type': self.config.vectorization_type,
                'embedding_dim': self.config.embedding_dim,
                'min_token_frequency': self.config.min_token_frequency
            }
        }
    
    def _save_vocabulary(self):
        """Сохранение словаря в файл"""
        try:
            vocab_data = {
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'token_frequencies': dict(self.token_frequencies),
                'config': {
                    'vectorization_type': self.config.vectorization_type,
                    'vocabulary_size': self.vocabulary_size,
                    'embedding_dim': self.config.embedding_dim
                }
            }
            
            with open(self.config.vocabulary_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
            # Сохраняем эмбеддинги отдельно
            if self.token_embeddings is not None:
                np.save(self.config.embeddings_path, self.token_embeddings)
            
            logger.info(f"Vocabulary saved to {self.config.vocabulary_path}")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    def load_vocabulary(self, vocabulary_path: str = None, embeddings_path: str = None):
        """Загрузка словаря из файла"""
        vocab_path = vocabulary_path or self.config.vocabulary_path
        embed_path = embeddings_path or self.config.embeddings_path
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.token_to_id = vocab_data['token_to_id']
            self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            self.token_frequencies = Counter(vocab_data['token_frequencies'])
            self.vocabulary_size = vocab_data['config']['vocabulary_size']
            
            # Загружаем эмбеддинги если есть
            if Path(embed_path).exists():
                self.token_embeddings = np.load(embed_path)
                logger.info(f"Loaded embeddings shape: {self.token_embeddings.shape}")
            
            self.is_fitted = True
            logger.info(f"Vocabulary loaded from {vocab_path}")
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise
