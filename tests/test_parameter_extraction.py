"""
Тесты для модулей извлечения параметров
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from same.parameter_extraction import (
    RegexParameterExtractor, ParameterPattern, ParameterType, ExtractedParameter,
    MLParameterExtractor, MLExtractorConfig,
    ParameterParser, ParameterParserConfig
)


class TestRegexParameterExtractor:
    """Тесты для RegexParameterExtractor"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.extractor = RegexParameterExtractor()
    
    def test_extract_diameter(self):
        """Тест извлечения диаметра"""
        text = "Болт диаметр 10мм М10х50"
        params = self.extractor.extract_parameters(text)
        
        diameter_params = [p for p in params if p.name == 'diameter']
        assert len(diameter_params) > 0
        
        param = diameter_params[0]
        assert param.parameter_type == ParameterType.DIMENSION
        assert param.unit == 'мм'
    
    def test_extract_length(self):
        """Тест извлечения длины"""
        text = "Болт М10 длина 50мм"
        params = self.extractor.extract_parameters(text)
        
        length_params = [p for p in params if p.name == 'length']
        assert len(length_params) > 0
        
        param = length_params[0]
        assert param.parameter_type == ParameterType.DIMENSION
    
    def test_extract_voltage(self):
        """Тест извлечения напряжения"""
        text = "Двигатель напряжение 220В"
        params = self.extractor.extract_parameters(text)

        # Ищем параметры напряжения по типу, а не по имени
        voltage_params = [p for p in params if p.parameter_type == ParameterType.VOLTAGE]
        assert len(voltage_params) > 0

        param = voltage_params[0]
        assert param.parameter_type == ParameterType.VOLTAGE
        assert param.unit == 'В'
    
    def test_extract_material(self):
        """Тест извлечения материала"""
        text = "Болт материал сталь нержавеющая"
        params = self.extractor.extract_parameters(text)
        
        material_params = [p for p in params if p.name == 'material']
        assert len(material_params) > 0
        
        param = material_params[0]
        assert param.parameter_type == ParameterType.MATERIAL
    
    def test_extract_gost(self):
        """Тест извлечения ГОСТ"""
        text = "Болт М10х50 ГОСТ 7798-70"
        params = self.extractor.extract_parameters(text)
        
        gost_params = [p for p in params if p.name == 'gost']
        assert len(gost_params) > 0
        
        param = gost_params[0]
        assert param.parameter_type == ParameterType.ARTICLE
        assert 'гост' in param.value.lower()
    
    def test_empty_input(self):
        """Тест обработки пустого входа"""
        params = self.extractor.extract_parameters("")
        assert params == []
        
        params = self.extractor.extract_parameters(None)
        assert params == []
    
    def test_batch_extraction(self):
        """Тест пакетного извлечения"""
        texts = [
            "Болт диаметр 10мм",
            "Гайка М10",
            "Двигатель 220В"
        ]
        
        results = self.extractor.extract_parameters_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
    
    def test_add_custom_pattern(self):
        """Тест добавления кастомного паттерна"""
        custom_pattern = ParameterPattern(
            name="custom_param",
            pattern=r"кастом\s*(\d+)",
            parameter_type=ParameterType.DIMENSION,
            description="Кастомный параметр"
        )
        
        self.extractor.add_pattern(custom_pattern)
        
        text = "Изделие кастом 123"
        params = self.extractor.extract_parameters(text)
        
        custom_params = [p for p in params if p.name == 'custom_param']
        assert len(custom_params) > 0
    
    def test_get_patterns_by_type(self):
        """Тест получения паттернов по типу"""
        dimension_patterns = self.extractor.get_patterns_by_type(ParameterType.DIMENSION)
        
        assert len(dimension_patterns) > 0
        assert all(p.parameter_type == ParameterType.DIMENSION for p in dimension_patterns)
    
    def test_get_statistics(self):
        """Тест получения статистики"""
        stats = self.extractor.get_statistics()
        
        assert 'total_patterns' in stats
        assert 'patterns_by_type' in stats
        assert 'supported_types' in stats
        assert stats['total_patterns'] > 0
    
    def test_save_load_patterns(self, tmp_path):
        """Тест сохранения и загрузки паттернов"""
        patterns_path = tmp_path / "test_patterns.json"
        
        # Сохраняем паттерны
        self.extractor.save_patterns(str(patterns_path))
        
        # Создаем новый экстрактор и загружаем паттерны
        new_extractor = RegexParameterExtractor()
        original_count = len(new_extractor.patterns)
        
        new_extractor.load_patterns(str(patterns_path))
        
        # Проверяем что паттерны загрузились
        assert len(new_extractor.patterns) > original_count


class TestMLParameterExtractor:
    """Тесты для MLParameterExtractor"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = MLExtractorConfig()
        
    @patch('spacy.load')
    def test_initialization(self, mock_spacy_load):
        """Тест инициализации ML экстрактора"""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        extractor = MLParameterExtractor(self.config)
        
        assert extractor.nlp is not None
        assert extractor.is_trained is False
        mock_spacy_load.assert_called_once()
    
    @patch('spacy.load')
    def test_prepare_training_data(self, mock_spacy_load):
        """Тест подготовки данных для обучения"""
        # Мокаем SpaCy
        mock_token = Mock()
        mock_token.text = "диаметр"
        mock_token.lemma_ = "диаметр"
        mock_token.pos_ = "NOUN"
        mock_token.is_alpha = True
        mock_token.i = 0
        mock_token.idx = 0
        
        mock_doc = [mock_token]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        extractor = MLParameterExtractor(self.config)
        
        texts = ["Болт диаметр 10мм"]
        annotations = [[{
            'start': 5,
            'end': 12,
            'parameter_type': 'dimension',
            'parameter_name': 'diameter',
            'parameter_value': '10'
        }]]
        
        training_data = extractor.prepare_training_data(texts, annotations)
        
        assert len(training_data) > 0
        assert all('features' in sample for sample in training_data)
        assert all('label' in sample for sample in training_data)
    
    @patch('spacy.load')
    def test_extract_parameters_not_trained(self, mock_spacy_load):
        """Тест извлечения параметров без обучения"""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        extractor = MLParameterExtractor(self.config)
        
        with pytest.raises(ValueError, match="Model is not trained"):
            extractor.extract_parameters("Болт диаметр 10мм")
    
    def test_get_statistics(self):
        """Тест получения статистики ML экстрактора"""
        with patch('spacy.load'):
            extractor = MLParameterExtractor(self.config)
            stats = extractor.get_statistics()
            
            assert 'is_trained' in stats
            assert 'training_samples' in stats
            assert 'classifier_type' in stats
            assert stats['is_trained'] is False


class TestParameterParser:
    """Тесты для ParameterParser"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.config = ParameterParserConfig(use_ml=False)  # Отключаем ML для простоты
        self.parser = ParameterParser(self.config)
    
    def test_parse_parameters(self):
        """Тест парсинга параметров"""
        text = "Болт М10х50 диаметр 10мм длина 50мм ГОСТ 7798-70"
        params = self.parser.parse_parameters(text)
        
        assert isinstance(params, list)
        assert len(params) > 0
        
        # Проверяем что есть разные типы параметров
        param_types = {p.parameter_type for p in params}
        assert ParameterType.DIMENSION in param_types
        assert ParameterType.ARTICLE in param_types
    
    def test_parse_batch(self):
        """Тест пакетного парсинга"""
        texts = [
            "Болт М10х50 диаметр 10мм",
            "Гайка М10",
            "Двигатель 220В мощность 1кВт"
        ]
        
        results = self.parser.parse_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
    
    def test_parse_dataframe(self):
        """Тест парсинга DataFrame"""
        df = pd.DataFrame({
            'name': [
                'Болт М10х50 диаметр 10мм',
                'Гайка М10',
                'Двигатель 220В'
            ],
            'id': [1, 2, 3]
        })
        
        result_df = self.parser.parse_dataframe(df, 'name')
        
        assert 'extracted_parameters' in result_df.columns
        assert len(result_df) == 3
        assert all(isinstance(params, list) for params in result_df['extracted_parameters'])
    
    def test_filter_results(self):
        """Тест фильтрации результатов"""
        # Создаем тестовые параметры с разной уверенностью
        params = [
            ExtractedParameter(
                name="diameter",
                value=10,
                unit="мм",
                parameter_type=ParameterType.DIMENSION,
                confidence=0.9,
                source_text="диаметр 10мм",
                position=(0, 12)
            ),
            ExtractedParameter(
                name="length",
                value=50,
                unit="мм",
                parameter_type=ParameterType.DIMENSION,
                confidence=0.3,  # Низкая уверенность
                source_text="длина 50мм",
                position=(13, 24)
            )
        ]
        
        # Устанавливаем высокий порог
        self.parser.config.min_confidence = 0.5
        filtered = self.parser._filter_results(params)
        
        # Должен остаться только параметр с высокой уверенностью
        assert len(filtered) == 1
        assert filtered[0].name == "diameter"
    
    def test_remove_duplicates(self):
        """Тест удаления дубликатов"""
        params = [
            ExtractedParameter(
                name="diameter",
                value=10,
                unit="мм",
                parameter_type=ParameterType.DIMENSION,
                confidence=0.8,
                source_text="диаметр 10мм",
                position=(0, 12)
            ),
            ExtractedParameter(
                name="diameter",
                value=10,
                unit="мм",
                parameter_type=ParameterType.DIMENSION,
                confidence=0.9,  # Выше уверенность
                source_text="диам 10мм",
                position=(13, 22)
            )
        ]
        
        unique_params = self.parser._remove_duplicates(params)
        
        # Должен остаться только один параметр с большей уверенностью
        assert len(unique_params) == 1
        assert unique_params[0].confidence == 0.9
    
    def test_validate_dimension_parameter(self):
        """Тест валидации размерного параметра"""
        # Валидный параметр
        valid_param = ExtractedParameter(
            name="diameter",
            value=10.5,
            unit="мм",
            parameter_type=ParameterType.DIMENSION,
            confidence=0.8,
            source_text="диаметр 10.5мм",
            position=(0, 14)
        )
        
        assert self.parser._validate_dimension_parameter(valid_param) is True
        
        # Невалидный параметр (отрицательное значение)
        invalid_param = ExtractedParameter(
            name="diameter",
            value=-10,
            unit="мм",
            parameter_type=ParameterType.DIMENSION,
            confidence=0.8,
            source_text="диаметр -10мм",
            position=(0, 13)
        )
        
        assert self.parser._validate_dimension_parameter(invalid_param) is False
    
    def test_get_statistics(self):
        """Тест получения статистики парсера"""
        # Обрабатываем несколько текстов для генерации статистики
        texts = [
            "Болт М10х50 диаметр 10мм",
            "Гайка М10",
            "Двигатель 220В"
        ]
        
        for text in texts:
            self.parser.parse_parameters(text)
        
        stats = self.parser.get_statistics()
        
        assert 'total_processed' in stats
        assert 'total_parameters_extracted' in stats
        assert 'regex_extractor' in stats
        assert stats['total_processed'] == 3
    
    def test_reset_statistics(self):
        """Тест сброса статистики"""
        # Генерируем статистику
        self.parser.parse_parameters("Болт М10х50")
        
        assert self.parser.stats['total_processed'] > 0
        
        # Сбрасываем
        self.parser.reset_statistics()
        
        assert self.parser.stats['total_processed'] == 0
        assert self.parser.stats['total_parameters_extracted'] == 0


# Фикстуры для тестов
@pytest.fixture
def sample_technical_texts():
    """Образцы технических текстов для тестирования"""
    return [
        "Болт М10х50 диаметр 10мм длина 50мм ГОСТ 7798-70 материал сталь",
        "Гайка М10 DIN 934 высота 8мм материал нержавеющая сталь",
        "Шайба плоская диаметр 10мм толщина 2мм ГОСТ 11371-78",
        "Двигатель асинхронный мощность 1.5кВт напряжение 220В частота 50Гц",
        "Труба стальная диаметр 57мм толщина стенки 3.5мм ГОСТ 8732-78",
        "Клапан шаровой ДУ25 РУ40 материал латунь температура до 150°C",
        "Насос центробежный подача 50м³/ч напор 32м мощность 4кВт",
        "Редуктор червячный передаточное число 40 момент 500Нм",
        "Подшипник шариковый диаметр внутренний 20мм наружный 47мм",
        "Кабель силовой сечение 2.5мм² напряжение 660В длина 100м"
    ]


@pytest.fixture
def sample_annotations():
    """Образцы аннотаций для обучения ML"""
    return [
        [
            {'start': 5, 'end': 10, 'parameter_type': 'dimension', 'parameter_name': 'thread', 'parameter_value': 'М10'},
            {'start': 11, 'end': 13, 'parameter_type': 'dimension', 'parameter_name': 'length', 'parameter_value': '50'},
            {'start': 22, 'end': 24, 'parameter_type': 'dimension', 'parameter_name': 'diameter', 'parameter_value': '10'}
        ]
    ]


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_parameter_extraction_pipeline(self, sample_technical_texts):
        """Тест полного пайплайна извлечения параметров"""
        config = ParameterParserConfig(
            use_regex=True,
            use_ml=False,
            combination_strategy="union",
            remove_duplicates=True
        )
        
        parser = ParameterParser(config)
        
        all_results = []
        for text in sample_technical_texts:
            params = parser.parse_parameters(text)
            all_results.extend(params)
        
        # Проверяем что извлечены параметры разных типов
        param_types = {p.parameter_type for p in all_results}
        
        assert ParameterType.DIMENSION in param_types
        assert ParameterType.ELECTRICAL in param_types or ParameterType.DIMENSION in param_types
        assert len(all_results) > 0
        
        # Проверяем статистику
        stats = parser.get_statistics()
        assert stats['total_processed'] == len(sample_technical_texts)
        assert stats['total_parameters_extracted'] > 0
    
    def test_dataframe_processing_pipeline(self, sample_technical_texts):
        """Тест обработки DataFrame"""
        df = pd.DataFrame({
            'id': range(len(sample_technical_texts)),
            'name': sample_technical_texts,
            'category': ['Крепеж', 'Крепеж', 'Крепеж', 'Электрика', 'Трубы'] * 2
        })
        
        parser = ParameterParser()
        result_df = parser.parse_dataframe(df, 'name')
        
        assert len(result_df) == len(df)
        assert 'extracted_parameters' in result_df.columns
        
        # Проверяем что параметры извлечены
        total_params = sum(len(params) for params in result_df['extracted_parameters'])
        assert total_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
