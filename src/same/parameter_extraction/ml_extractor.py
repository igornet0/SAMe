"""
Модуль ML-извлечения параметров с использованием машинного обучения
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import spacy
import pickle
from pathlib import Path
import re

from .regex_extractor import ParameterType, ExtractedParameter

logger = logging.getLogger(__name__)


@dataclass
class MLExtractorConfig:
    """Конфигурация ML-экстрактора"""
    # Модель SpaCy
    spacy_model: str = "ru_core_news_lg"
    
    # Настройки классификатора
    classifier_type: str = "random_forest"  # random_forest, logistic_regression
    
    # Настройки векторизации
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    
    # Настройки обучения
    test_size: float = 0.2
    random_state: int = 42
    
    # Пороги уверенности
    min_confidence: float = 0.6
    
    # Настройки извлечения
    max_parameters_per_text: int = 20
    enable_context_features: bool = True


class MLParameterExtractor:
    """ML-экстрактор параметров"""
    
    def __init__(self, config: MLExtractorConfig = None):
        self.config = config or MLExtractorConfig()
        
        # Модели
        self.nlp = None
        self.vectorizer = None
        self.classifier = None
        self.value_extractors = {}  # Экстракторы значений для каждого типа параметра
        
        # Данные для обучения
        self.training_data = []
        self.is_trained = False
        
        # Загрузка SpaCy модели
        self._load_spacy_model()
        
        logger.info("MLParameterExtractor initialized")
    
    def _load_spacy_model(self):
        """Загрузка модели SpaCy"""
        try:
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info(f"Loaded SpaCy model: {self.config.spacy_model}")
        except OSError:
            logger.warning(f"SpaCy model {self.config.spacy_model} not found, using sm model")
            try:
                self.nlp = spacy.load("ru_core_news_sm")
            except OSError:
                logger.error("No Russian SpaCy model found. Please install: python -m spacy download ru_core_news_lg")
                raise
    
    def prepare_training_data(self, 
                            texts: List[str], 
                            annotations: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Подготовка данных для обучения
        
        Args:
            texts: Список текстов
            annotations: Список аннотаций для каждого текста
            
        Returns:
            Подготовленные данные для обучения
        """
        training_samples = []
        
        for text, text_annotations in zip(texts, annotations):
            # Обрабатываем текст через SpaCy
            doc = self.nlp(text)
            
            # Создаем примеры для каждого токена
            for token in doc:
                if token.is_alpha and len(token.text) > 2:
                    # Извлекаем признаки токена
                    features = self._extract_token_features(token, doc)
                    
                    # Определяем метку (есть ли параметр в этой позиции)
                    label = self._get_token_label(token, text_annotations)
                    
                    training_samples.append({
                        'text': text,
                        'token': token.text,
                        'features': features,
                        'label': label,
                        'parameter_type': label.get('parameter_type') if isinstance(label, dict) else None
                    })
        
        self.training_data = training_samples
        logger.info(f"Prepared {len(training_samples)} training samples")
        
        return training_samples
    
    def _extract_token_features(self, token, doc) -> Dict[str, Any]:
        """Извлечение признаков токена"""
        features = {
            # Базовые признаки токена
            'text': token.text.lower(),
            'lemma': token.lemma_.lower(),
            'pos': token.pos_,
            'tag': token.tag_,
            'is_digit': token.is_digit,
            'is_alpha': token.is_alpha,
            'is_upper': token.is_upper,
            'is_title': token.is_title,
            'length': len(token.text),
            
            # Морфологические признаки
            'shape': token.shape_,
            'is_stop': token.is_stop,
            'has_vector': token.has_vector,
            
            # Контекстные признаки
            'prev_token': token.nbor(-1).text.lower() if token.i > 0 else '',
            'next_token': token.nbor(1).text.lower() if token.i < len(doc) - 1 else '',
            'prev_pos': token.nbor(-1).pos_ if token.i > 0 else '',
            'next_pos': token.nbor(1).pos_ if token.i < len(doc) - 1 else '',
        }
        
        # Дополнительные контекстные признаки
        if self.config.enable_context_features:
            features.update(self._extract_context_features(token, doc))
        
        # Паттерн-основанные признаки
        features.update(self._extract_pattern_features(token))
        
        return features
    
    def _extract_context_features(self, token, doc) -> Dict[str, Any]:
        """Извлечение контекстных признаков"""
        context_features = {}
        
        # Окно контекста
        window_size = 3
        start_idx = max(0, token.i - window_size)
        end_idx = min(len(doc), token.i + window_size + 1)
        
        context_tokens = [t.text.lower() for t in doc[start_idx:end_idx] if t.i != token.i]
        context_features['context_tokens'] = ' '.join(context_tokens)
        
        # Наличие числовых значений в контексте
        context_features['has_numbers_in_context'] = any(t.is_digit for t in doc[start_idx:end_idx])
        
        # Наличие единиц измерения в контексте
        units = ['мм', 'см', 'м', 'кг', 'г', 'в', 'а', 'вт', 'па', 'бар', 'атм']
        context_features['has_units_in_context'] = any(
            any(unit in t.text.lower() for unit in units) 
            for t in doc[start_idx:end_idx]
        )
        
        return context_features
    
    def _extract_pattern_features(self, token) -> Dict[str, Any]:
        """Извлечение признаков на основе паттернов"""
        text = token.text.lower()
        
        pattern_features = {
            # Технические паттерны
            'is_technical_term': self._is_technical_term(text),
            'is_measurement_unit': self._is_measurement_unit(text),
            'is_material_name': self._is_material_name(text),
            'is_standard_reference': self._is_standard_reference(text),
            
            # Числовые паттерны
            'contains_digits': bool(re.search(r'\d', text)),
            'is_dimension_pattern': bool(re.search(r'\d+[x×]\d+', text)),
            'is_range_pattern': bool(re.search(r'\d+[-–]\d+', text)),
            
            # Специальные символы
            'has_special_chars': bool(re.search(r'[×x\-–°]', text)),
        }
        
        return pattern_features
    
    def _is_technical_term(self, text: str) -> bool:
        """Проверка, является ли токен техническим термином"""
        technical_terms = {
            'диаметр', 'длина', 'ширина', 'высота', 'толщина', 'вес', 'масса',
            'напряжение', 'ток', 'мощность', 'давление', 'температура', 'частота',
            'материал', 'сталь', 'алюминий', 'пластик', 'резина'
        }
        return text in technical_terms
    
    def _is_measurement_unit(self, text: str) -> bool:
        """Проверка, является ли токен единицей измерения"""
        units = {
            'мм', 'см', 'м', 'км', 'г', 'кг', 'т', 'л', 'мл',
            'в', 'а', 'вт', 'квт', 'па', 'кпа', 'мпа', 'бар', 'атм',
            'гц', 'кгц', 'мгц', '°c', '°f'
        }
        return text in units
    
    def _is_material_name(self, text: str) -> bool:
        """Проверка, является ли токен названием материала"""
        materials = {
            'сталь', 'железо', 'алюминий', 'медь', 'латунь', 'бронза',
            'пластик', 'резина', 'дерево', 'стекло', 'керамика'
        }
        return text in materials
    
    def _is_standard_reference(self, text: str) -> bool:
        """Проверка, является ли токен ссылкой на стандарт"""
        return bool(re.search(r'гост|din|iso|ту', text, re.IGNORECASE))
    
    def _get_token_label(self, token, annotations: List[Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Получение метки для токена на основе аннотаций"""
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        for annotation in annotations:
            ann_start = annotation.get('start', 0)
            ann_end = annotation.get('end', 0)
            
            # Проверяем пересечение
            if (token_start >= ann_start and token_end <= ann_end) or \
               (token_start <= ann_start and token_end >= ann_end) or \
               (token_start <= ann_start <= token_end) or \
               (token_start <= ann_end <= token_end):
                
                return {
                    'is_parameter': True,
                    'parameter_type': annotation.get('parameter_type', 'unknown'),
                    'parameter_name': annotation.get('parameter_name', ''),
                    'parameter_value': annotation.get('parameter_value', '')
                }
        
        return {'is_parameter': False}
    
    def train(self, training_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обучение ML-модели
        
        Args:
            training_data: Данные для обучения (опционально)
            
        Returns:
            Метрики обучения
        """
        if training_data:
            self.training_data = training_data
        
        if not self.training_data:
            raise ValueError("No training data available")
        
        logger.info("Starting ML model training")
        
        # Подготовка данных
        X, y = self._prepare_training_features()
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Векторизация текстовых признаков
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            lowercase=True
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Обучение классификатора
        if self.config.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.config.classifier_type}")
        
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Оценка модели
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'training_samples': len(self.training_data),
            'test_samples': len(y_test)
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        
        return metrics
    
    def _prepare_training_features(self) -> Tuple[List[str], List[str]]:
        """Подготовка признаков для обучения"""
        X = []  # Текстовые признаки
        y = []  # Метки
        
        for sample in self.training_data:
            # Объединяем текстовые признаки
            features = sample['features']
            text_features = [
                features.get('text', ''),
                features.get('lemma', ''),
                features.get('pos', ''),
                features.get('prev_token', ''),
                features.get('next_token', ''),
                features.get('context_tokens', '')
            ]
            
            X.append(' '.join(filter(None, text_features)))
            
            # Метка: есть ли параметр
            label = sample['label']
            if isinstance(label, dict):
                y.append('parameter' if label.get('is_parameter', False) else 'no_parameter')
            else:
                y.append('no_parameter')
        
        return X, y
    
    def extract_parameters(self, text: str) -> List[ExtractedParameter]:
        """
        Извлечение параметров с помощью ML
        
        Args:
            text: Входной текст
            
        Returns:
            Список извлеченных параметров
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        if not text or not isinstance(text, str):
            return []
        
        # Обрабатываем текст через SpaCy
        doc = self.nlp(text)
        
        # Извлекаем признаки для каждого токена
        token_features = []
        tokens = []
        
        for token in doc:
            if token.is_alpha and len(token.text) > 2:
                features = self._extract_token_features(token, doc)
                
                # Подготавливаем текстовые признаки
                text_features = [
                    features.get('text', ''),
                    features.get('lemma', ''),
                    features.get('pos', ''),
                    features.get('prev_token', ''),
                    features.get('next_token', ''),
                    features.get('context_tokens', '')
                ]
                
                token_features.append(' '.join(filter(None, text_features)))
                tokens.append(token)
        
        if not token_features:
            return []
        
        # Векторизация
        X = self.vectorizer.transform(token_features)
        
        # Предсказание
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        # Извлекаем параметры
        extracted_params = []
        
        for i, (token, prediction, proba) in enumerate(zip(tokens, predictions, probabilities)):
            if prediction == 'parameter':
                # Получаем уверенность
                confidence = max(proba)
                
                if confidence >= self.config.min_confidence:
                    # Определяем тип параметра и значение
                    param_type, param_value = self._determine_parameter_details(token, doc)
                    
                    param = ExtractedParameter(
                        name=self._generate_parameter_name(token, param_type),
                        value=param_value,
                        unit=self._extract_unit(token, doc),
                        parameter_type=param_type,
                        confidence=confidence,
                        source_text=token.text,
                        position=(token.idx, token.idx + len(token.text))
                    )
                    
                    extracted_params.append(param)
        
        # Ограничиваем количество параметров
        extracted_params = extracted_params[:self.config.max_parameters_per_text]
        
        return extracted_params
    
    def _determine_parameter_details(self, token, doc) -> Tuple[ParameterType, str]:
        """Определение типа параметра и его значения"""
        text = token.text.lower()
        
        # Простая эвристика для определения типа
        if any(dim in text for dim in ['диаметр', 'длина', 'ширина', 'высота', 'толщина']):
            return ParameterType.DIMENSION, self._extract_numeric_value(token, doc)
        elif any(elec in text for elec in ['напряжение', 'ток', 'мощность']):
            return ParameterType.ELECTRICAL, self._extract_numeric_value(token, doc)
        elif any(mat in text for mat in ['сталь', 'алюминий', 'пластик']):
            return ParameterType.MATERIAL, text
        else:
            return ParameterType.DIMENSION, self._extract_numeric_value(token, doc)
    
    def _extract_numeric_value(self, token, doc) -> str:
        """Извлечение числового значения рядом с токеном"""
        # Ищем числа в окрестности токена
        window_size = 3
        start_idx = max(0, token.i - window_size)
        end_idx = min(len(doc), token.i + window_size + 1)
        
        for t in doc[start_idx:end_idx]:
            if t.is_digit or re.search(r'\d+([.,]\d+)?', t.text):
                return t.text
        
        return token.text
    
    def _extract_unit(self, token, doc) -> Optional[str]:
        """Извлечение единицы измерения"""
        units = ['мм', 'см', 'м', 'кг', 'г', 'в', 'а', 'вт', 'па', 'бар']
        
        # Ищем единицы в окрестности токена
        window_size = 2
        start_idx = max(0, token.i - window_size)
        end_idx = min(len(doc), token.i + window_size + 1)
        
        for t in doc[start_idx:end_idx]:
            if t.text.lower() in units:
                return t.text.lower()
        
        return None
    
    def _generate_parameter_name(self, token, param_type: ParameterType) -> str:
        """Генерация имени параметра"""
        text = token.text.lower()
        
        if param_type == ParameterType.DIMENSION:
            if 'диаметр' in text:
                return 'diameter'
            elif 'длина' in text:
                return 'length'
            elif 'ширина' in text:
                return 'width'
            elif 'высота' in text:
                return 'height'
            else:
                return 'dimension'
        elif param_type == ParameterType.ELECTRICAL:
            if 'напряжение' in text:
                return 'voltage'
            elif 'ток' in text:
                return 'current'
            elif 'мощность' in text:
                return 'power'
            else:
                return 'electrical'
        elif param_type == ParameterType.MATERIAL:
            return 'material'
        else:
            return 'parameter'
    
    def save_model(self, filepath: str):
        """Сохранение обученной модели"""
        model_data = {
            'config': self.config,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained,
            'training_data': self.training_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"ML model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка обученной модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        self.training_data = model_data.get('training_data', [])
        
        # Перезагружаем SpaCy модель
        self._load_spacy_model()
        
        logger.info(f"ML model loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики модели"""
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_data),
            'classifier_type': self.config.classifier_type,
            'min_confidence': self.config.min_confidence,
            'vectorizer_features': len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }
