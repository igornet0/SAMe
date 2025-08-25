"""
Тесты для модулей предобработки текста
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

try:
    from same_clear.text_processing import (
        TextCleaner, CleaningConfig,
        Lemmatizer, LemmatizerConfig,
        TextNormalizer, NormalizerConfig,
        TextPreprocessor, PreprocessorConfig
    )
except ImportError:
    # Fallback на старый импорт
    from same.text_processing import (
        TextCleaner, CleaningConfig,
        Lemmatizer, LemmatizerConfig,
        TextNormalizer, NormalizerConfig,
        TextPreprocessor, PreprocessorConfig
    )


class TestTextCleaner:
    """Тесты для TextCleaner"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = CleaningConfig()
        self.cleaner = TextCleaner(self.config)
    
    def test_clean_html_tags(self):
        """Тест удаления HTML тегов"""
        text = "<p>Болт М10х50 <b>ГОСТ</b> 7798-70</p>"
        result = self.cleaner.clean_text(text)

        assert "<p>" not in result['html_cleaned']
        assert "<b>" not in result['html_cleaned']
        assert "Болт М10х50" in result['html_cleaned']
        # ГОСТ should be protected as a token, so check for protected token or original
        assert ("ГОСТ" in result['html_cleaned'] or "__PROTECTED_TOKEN_" in result['html_cleaned'])
    
    def test_clean_special_chars(self):
        """Тест удаления специальных символов"""
        text = "Болт М10х50 @#$% ГОСТ 7798-70"
        result = self.cleaner.clean_text(text)
        
        assert "@#$%" not in result['special_cleaned']
        assert "Болт М10х50" in result['special_cleaned']
    
    def test_normalize_spaces(self):
        """Тест нормализации пробелов"""
        text = "Болт    М10х50     ГОСТ   7798-70"
        result = self.cleaner.clean_text(text)
        
        assert "    " not in result['normalized']
        assert "Болт М10х50 ГОСТ 7798-70" in result['normalized']
    
    def test_empty_input(self):
        """Тест обработки пустого входа"""
        result = self.cleaner.clean_text("")
        
        assert result['raw'] == ""
        assert result['normalized'] == ""
    
    def test_none_input(self):
        """Тест обработки None"""
        result = self.cleaner.clean_text(None)
        
        assert result['raw'] == ""
        assert result['normalized'] == ""
    
    def test_batch_cleaning(self):
        """Тест пакетной очистки"""
        texts = [
            "<p>Болт М10х50</p>",
            "Гайка    М10",
            "Шайба@#$10"
        ]
        
        results = self.cleaner.clean_batch(texts)
        
        assert len(results) == 3
        assert all('normalized' in result for result in results)
    
    def test_cleaning_stats(self):
        """Тест статистики очистки"""
        original = "<p>Болт М10х50 @#$% ГОСТ</p>"
        cleaned = "Болт М10х50 ГОСТ"
        
        stats = self.cleaner.get_cleaning_stats(original, cleaned)
        
        assert 'original_length' in stats
        assert 'cleaned_length' in stats
        assert 'compression_ratio' in stats
        assert stats['original_length'] > stats['cleaned_length']


class TestLemmatizer:
    """Тесты для Lemmatizer"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = LemmatizerConfig()
        
    @patch('same.models.model_manager.spacy.load')
    def test_lemmatizer_initialization(self, mock_spacy_load):
        """Тест инициализации лемматизатора"""
        mock_nlp = Mock()
        mock_nlp.Defaults.stop_words = {'и', 'в', 'на'}
        mock_spacy_load.return_value = mock_nlp

        lemmatizer = Lemmatizer(self.config)

        # Проверяем, что лемматизатор создан (модель загружается лениво)
        assert lemmatizer.model_manager is not None
        assert lemmatizer._initialized is False  # Модель еще не загружена
    
    @patch('spacy.load')
    def test_lemmatize_text(self, mock_spacy_load):
        """Тест лемматизации текста"""
        # Мокаем SpaCy
        mock_token1 = Mock()
        mock_token1.text = "Болты"
        mock_token1.lemma_ = "болт"
        mock_token1.pos_ = "NOUN"
        mock_token1.is_punct = False
        mock_token1.is_space = False
        
        mock_token2 = Mock()
        mock_token2.text = "М10"
        mock_token2.lemma_ = "м10"
        mock_token2.pos_ = "NUM"
        mock_token2.is_punct = False
        mock_token2.is_space = False
        
        mock_doc = [mock_token1, mock_token2]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp
        
        lemmatizer = Lemmatizer(self.config)
        result = lemmatizer.lemmatize_text("Болты М10")
        
        assert 'lemmatized' in result
        assert 'tokens' in result
        assert 'lemmas' in result
        assert len(result['tokens']) == 2
    
    def test_empty_input(self):
        """Тест обработки пустого входа"""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = set()
            mock_spacy_load.return_value = mock_nlp
            
            lemmatizer = Lemmatizer(self.config)
            result = lemmatizer.lemmatize_text("")
            
            assert result['original'] == ""
            assert result['lemmatized'] == ""


class TestTextNormalizer:
    """Тесты для TextNormalizer"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = NormalizerConfig()
        self.normalizer = TextNormalizer(self.config)
    
    def test_normalize_units(self):
        """Тест нормализации единиц измерения"""
        text = "диаметр 10 миллиметров"
        result = self.normalizer.normalize_text(text)
        
        assert "мм" in result['units_normalized']
        assert "миллиметров" not in result['units_normalized']
    
    def test_normalize_abbreviations(self):
        """Тест нормализации аббревиатур"""
        text = "эл двигатель"
        result = self.normalizer.normalize_text(text)

        assert "электрический" in result['abbreviations_normalized']
        # Проверяем что аббревиатура заменена (не содержится как отдельное слово)
        words = result['abbreviations_normalized'].split()
        assert "эл" not in words
    
    def test_unify_technical_terms(self):
        """Тест унификации технических терминов"""
        text = "винт крепежный"
        result = self.normalizer.normalize_text(text)
        
        # Проверяем что термин унифицирован
        assert result['terms_unified'] is not None
    
    def test_extract_technical_specs(self):
        """Тест извлечения технических характеристик"""
        text = "Болт М10х50 (диаметр 10мм) ГОСТ 7798-70"
        specs = self.normalizer.extract_technical_specs(text)
        
        assert len(specs) > 0
        assert any("диаметр 10мм" in spec for spec in specs)
    
    def test_batch_normalization(self):
        """Тест пакетной нормализации"""
        texts = [
            "диаметр 10 миллиметров",
            "эл двигатель",
            "винт крепежный"
        ]
        
        results = self.normalizer.normalize_batch(texts)
        
        assert len(results) == 3
        assert all('final_normalized' in result for result in results)


class TestTextPreprocessor:
    """Тесты для TextPreprocessor"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = PreprocessorConfig()
        
    @patch('same.models.model_manager.spacy.load')
    def test_preprocess_text(self, mock_spacy_load):
        """Тест полной предобработки текста"""
        # Мокаем SpaCy
        mock_nlp = Mock()
        mock_nlp.return_value = []
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp
        
        preprocessor = TextPreprocessor(self.config)
        
        text = "<p>Болт М10х50 @#$% диаметр 10 миллиметров</p>"
        result = preprocessor.preprocess_text(text)
        
        assert 'original' in result
        assert 'cleaning' in result
        assert 'normalization' in result
        assert 'lemmatization' in result
        assert 'final_text' in result
        assert 'processing_successful' in result
        assert result['processing_successful'] is True
    
    @patch('same.models.model_manager.spacy.load')
    def test_preprocess_batch(self, mock_spacy_load):
        """Тест пакетной предобработки"""
        # Мокаем SpaCy более детально
        mock_token = Mock()
        mock_token.text = "болт"
        mock_token.lemma_ = "болт"
        mock_token.pos_ = "NOUN"
        mock_token.is_punct = False
        mock_token.is_space = False

        mock_doc = [mock_token]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe.return_value = [mock_doc, mock_doc, mock_doc]  # Для batch обработки
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp

        preprocessor = TextPreprocessor(self.config)

        texts = [
            "Болт М10х50",
            "Гайка М10",
            "Шайба 10"
        ]

        # Мокаем lemmatize_batch чтобы возвращать правильную структуру
        expected_lemma_results = [
            {
                'original': text,
                'lemmatized': 'болт',
                'tokens': ['болт'],
                'lemmas': ['болт'],
                'pos_tags': ['NOUN'],
                'filtered_lemmas': ['болт']
            } for text in texts
        ]

        with patch.object(preprocessor.lemmatizer, 'lemmatize_batch', return_value=expected_lemma_results):
            results = preprocessor.preprocess_batch(texts)

            assert len(results) == 3
            assert all(result['processing_successful'] for result in results)
    
    @patch('same.models.model_manager.spacy.load')
    def test_preprocess_dataframe(self, mock_spacy_load):
        """Тест предобработки DataFrame"""
        # Мокаем SpaCy более детально
        mock_token = Mock()
        mock_token.text = "болт"
        mock_token.lemma_ = "болт"
        mock_token.pos_ = "NOUN"
        mock_token.is_punct = False
        mock_token.is_space = False

        mock_doc = [mock_token]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe.return_value = [mock_doc, mock_doc, mock_doc]  # Для batch обработки
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp

        preprocessor = TextPreprocessor(self.config)

        df = pd.DataFrame({
            'name': ['Болт М10х50', 'Гайка М10', 'Шайба 10'],
            'id': [1, 2, 3]
        })

        # Мокаем lemmatize_batch чтобы возвращать правильную структуру
        expected_lemma_results = [
            {
                'original': text,
                'lemmatized': 'болт',
                'tokens': ['болт'],
                'lemmas': ['болт'],
                'pos_tags': ['NOUN'],
                'filtered_lemmas': ['болт']
            } for text in df['name'].tolist()
        ]

        with patch.object(preprocessor.lemmatizer, 'lemmatize_batch', return_value=expected_lemma_results):
            result_df = preprocessor.preprocess_dataframe(df, 'name')

            assert 'name_processed' in result_df.columns
            assert 'name_processing_success' in result_df.columns
            assert len(result_df) == 3
    
    @patch('same.models.model_manager.spacy.load')
    def test_empty_input(self, mock_spacy_load):
        """Тест обработки пустого входа"""
        # Мокаем SpaCy
        mock_nlp = Mock()
        mock_nlp.return_value = []
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp

        preprocessor = TextPreprocessor(self.config)
        result = preprocessor.preprocess_text("")

        assert result['original'] == ""
        assert result['processing_successful'] is False
    
    @patch('same.models.model_manager.spacy.load')
    def test_processing_summary(self, mock_spacy_load):
        """Тест получения сводной статистики"""
        # Мокаем SpaCy
        mock_nlp = Mock()
        mock_nlp.return_value = []
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp
        
        preprocessor = TextPreprocessor(self.config)
        
        texts = ["Болт М10х50", "Гайка М10"]
        results = preprocessor.preprocess_batch(texts)
        
        summary = preprocessor.get_processing_summary(results)
        
        assert 'total_texts' in summary
        assert 'successful_processing' in summary
        assert 'success_rate' in summary
        assert summary['total_texts'] == 2


# Фикстуры для тестов
@pytest.fixture
def sample_texts():
    """Образцы текстов для тестирования"""
    return [
        "Болт М10х50 ГОСТ 7798-70",
        "Гайка М10 DIN 934",
        "Шайба плоская 10 ГОСТ 11371-78",
        "<p>Винт М8х30 @#$% нержавеющая сталь</p>",
        "Труба стальная диаметр 57 миллиметров толщина 3.5мм"
    ]


@pytest.fixture
def sample_dataframe():
    """Образец DataFrame для тестирования"""
    return pd.DataFrame({
        'id': range(1, 6),
        'name': [
            "Болт М10х50 ГОСТ 7798-70",
            "Гайка М10 DIN 934", 
            "Шайба плоская 10",
            "Винт М8х30",
            "Труба стальная 57х3.5"
        ],
        'category': ['Крепеж'] * 5
    })


class TestIntegration:
    """Интеграционные тесты"""
    
    @patch('same.models.model_manager.spacy.load')
    def test_full_pipeline(self, mock_spacy_load, sample_texts):
        """Тест полного пайплайна предобработки"""
        # Мокаем SpaCy более детально
        mock_token = Mock()
        mock_token.text = "болт"
        mock_token.lemma_ = "болт"
        mock_token.pos_ = "NOUN"
        mock_token.is_punct = False
        mock_token.is_space = False

        mock_doc = [mock_token]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe.return_value = [mock_doc] * len(sample_texts)  # Для batch обработки
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp

        config = PreprocessorConfig(save_intermediate_steps=True)
        preprocessor = TextPreprocessor(config)

        # Мокаем lemmatize_batch чтобы возвращать правильную структуру
        expected_lemma_results = [
            {
                'original': text,
                'lemmatized': 'болт',
                'tokens': ['болт'],
                'lemmas': ['болт'],
                'pos_tags': ['NOUN'],
                'filtered_lemmas': ['болт']
            } for text in sample_texts
        ]

        with patch.object(preprocessor.lemmatizer, 'lemmatize_batch', return_value=expected_lemma_results):
            results = preprocessor.preprocess_batch(sample_texts)

            assert len(results) == len(sample_texts)
            assert all(result['processing_successful'] for result in results)

            # Проверяем что все этапы выполнены
            for result in results:
                assert 'cleaning' in result
                assert 'normalization' in result
                assert 'lemmatization' in result
                assert 'final_text' in result
    
    @patch('same.models.model_manager.spacy.load')
    def test_dataframe_processing(self, mock_spacy_load, sample_dataframe):
        """Тест обработки DataFrame"""
        # Мокаем SpaCy более детально
        mock_token = Mock()
        mock_token.text = "болт"
        mock_token.lemma_ = "болт"
        mock_token.pos_ = "NOUN"
        mock_token.is_punct = False
        mock_token.is_space = False

        mock_doc = [mock_token]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe.return_value = [mock_doc] * len(sample_dataframe)  # Для batch обработки
        mock_nlp.Defaults.stop_words = set()
        mock_spacy_load.return_value = mock_nlp

        preprocessor = TextPreprocessor()

        # Мокаем lemmatize_batch чтобы возвращать правильную структуру
        expected_lemma_results = [
            {
                'original': text,
                'lemmatized': 'болт',
                'tokens': ['болт'],
                'lemmas': ['болт'],
                'pos_tags': ['NOUN'],
                'filtered_lemmas': ['болт']
            } for text in sample_dataframe['name'].tolist()
        ]

        with patch.object(preprocessor.lemmatizer, 'lemmatize_batch', return_value=expected_lemma_results):
            result_df = preprocessor.preprocess_dataframe(sample_dataframe, 'name')

            assert len(result_df) == len(sample_dataframe)
            assert 'name_processed' in result_df.columns
            assert all(result_df['name_processing_success'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
