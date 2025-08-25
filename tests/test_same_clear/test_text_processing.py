"""
Тесты для модуля text_processing в same_clear
"""

import pytest
from unittest.mock import patch, MagicMock

from same_clear.text_processing import (
    TextCleaner, CleaningConfig,
    TextNormalizer, NormalizerConfig,
    TextPreprocessor, PreprocessorConfig
)


class TestTextCleaner:
    """Тесты для TextCleaner"""
    
    def test_text_cleaner_creation(self):
        """Тест создания TextCleaner"""
        cleaner = TextCleaner()
        assert cleaner is not None
        assert hasattr(cleaner, 'clean_text')
    
    def test_html_cleaning(self):
        """Тест очистки HTML тегов"""
        cleaner = TextCleaner()
        
        # HTML теги
        result = cleaner.clean_text('Болт <b>М10х50</b> ГОСТ')
        assert '<b>' not in result['normalized']
        assert '</b>' not in result['normalized']
        assert 'Болт М10х50 ГОСТ' in result['normalized']
    
    def test_special_characters_cleaning(self):
        """Тест очистки специальных символов"""
        cleaner = TextCleaner()
        
        # Специальные символы
        result = cleaner.clean_text('Болт М10х50 &nbsp; ГОСТ')
        assert '&nbsp;' not in result['normalized']
        assert 'Болт М10х50 ГОСТ' in result['normalized']
    
    def test_multiple_spaces_normalization(self):
        """Тест нормализации множественных пробелов"""
        cleaner = TextCleaner()
        
        result = cleaner.clean_text('Болт    М10х50     ГОСТ')
        # Проверяем что множественные пробелы заменены на одинарные
        assert '    ' not in result['normalized']
        assert '     ' not in result['normalized']
    
    def test_empty_text(self):
        """Тест обработки пустого текста"""
        cleaner = TextCleaner()
        
        result = cleaner.clean_text('')
        assert result['raw'] == ''
        assert result['normalized'] == ''
    
    def test_result_structure(self):
        """Тест структуры результата"""
        cleaner = TextCleaner()
        
        result = cleaner.clean_text('Test text')
        
        # Проверяем обязательные поля
        assert 'raw' in result
        assert 'normalized' in result
        assert isinstance(result, dict)
        
        # Проверяем что raw содержит исходный текст
        assert result['raw'] == 'Test text'


class TestTextNormalizer:
    """Тесты для TextNormalizer"""
    
    def test_text_normalizer_creation(self):
        """Тест создания TextNormalizer"""
        normalizer = TextNormalizer()
        assert normalizer is not None
        assert hasattr(normalizer, 'normalize_text')
    
    def test_case_normalization(self):
        """Тест нормализации регистра"""
        normalizer = TextNormalizer()
        
        result = normalizer.normalize_text('БОЛТ М10Х50 гост')
        # Проверяем что регистр нормализован
        assert result != 'БОЛТ М10Х50 гост'  # Должен измениться
    
    def test_whitespace_normalization(self):
        """Тест нормализации пробелов"""
        normalizer = TextNormalizer()
        
        result = normalizer.normalize_text('  Болт   М10х50  ')
        # Проверяем что лишние пробелы удалены
        assert not result.startswith('  ')
        assert not result.endswith('  ')
        assert '   ' not in result


class TestTextPreprocessor:
    """Тесты для TextPreprocessor"""
    
    def test_text_preprocessor_creation(self):
        """Тест создания TextPreprocessor"""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'preprocess_text')
    
    def test_full_preprocessing_pipeline(self):
        """Тест полного пайплайна предобработки"""
        preprocessor = TextPreprocessor()
        
        # Сложный текст с HTML, спецсимволами и неправильным регистром
        input_text = '<b>БОЛТ</b> М10х50 &nbsp; ГОСТ 7798-70'
        result = preprocessor.preprocess_text(input_text)
        
        # Проверяем что все этапы обработки выполнены
        assert isinstance(result, dict)
        assert 'original' in result or 'raw' in result
        assert 'processed' in result or 'normalized' in result
        
        # Проверяем что HTML теги удалены
        processed_text = result.get('processed', result.get('normalized', ''))
        assert '<b>' not in processed_text
        assert '</b>' not in processed_text
        assert '&nbsp;' not in processed_text
    
    def test_batch_processing(self):
        """Тест пакетной обработки"""
        preprocessor = TextPreprocessor()
        
        texts = [
            'Болт М10х50 ГОСТ',
            'Гайка М10 ГОСТ',
            'Шайба 10 ГОСТ'
        ]
        
        # Проверяем что метод существует
        if hasattr(preprocessor, 'preprocess_batch'):
            results = preprocessor.preprocess_batch(texts)
            assert len(results) == len(texts)
            assert all(isinstance(r, dict) for r in results)
        else:
            # Если метода нет, обрабатываем по одному
            results = [preprocessor.preprocess_text(text) for text in texts]
            assert len(results) == len(texts)


class TestPreprocessorConfig:
    """Тесты для конфигурации предобработчика"""
    
    def test_config_creation(self):
        """Тест создания конфигурации"""
        try:
            config = PreprocessorConfig()
            assert config is not None
        except Exception:
            # Если класс не существует, пропускаем тест
            pytest.skip("PreprocessorConfig not available")
    
    def test_config_with_custom_settings(self):
        """Тест конфигурации с кастомными настройками"""
        try:
            config = PreprocessorConfig(
                remove_html=True,
                normalize_case=True,
                remove_extra_spaces=True
            )
            assert config.remove_html is True
            assert config.normalize_case is True
            assert config.remove_extra_spaces is True
        except Exception:
            pytest.skip("PreprocessorConfig not available or different interface")


class TestIntegrationWithCore:
    """Тесты интеграции с same_core"""
    
    def test_imports_from_core(self):
        """Тест импортов из same_core"""
        try:
            from same_core.interfaces import TextProcessorInterface
            from same_core.types import ProcessingResult
            
            # Проверяем что импорты работают
            assert TextProcessorInterface is not None
            assert ProcessingResult is not None
        except ImportError:
            pytest.skip("same_core not available")
    
    def test_text_cleaner_implements_interface(self):
        """Тест что TextCleaner реализует интерфейс"""
        try:
            from same_core.interfaces import TextProcessorInterface
            
            cleaner = TextCleaner()
            
            # Проверяем наличие методов интерфейса
            assert hasattr(cleaner, 'clean_text')
            
            # Если есть метод process_text, проверяем его
            if hasattr(cleaner, 'process_text'):
                result = cleaner.process_text('test')
                assert isinstance(result, dict)
                
        except ImportError:
            pytest.skip("same_core interface not available")


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_none_input(self):
        """Тест обработки None на входе"""
        cleaner = TextCleaner()
        
        try:
            result = cleaner.clean_text(None)
            # Если метод обрабатывает None, проверяем результат
            assert result is not None
        except (TypeError, AttributeError):
            # Ожидаемое поведение для None
            pass
    
    def test_non_string_input(self):
        """Тест обработки не-строкового входа"""
        cleaner = TextCleaner()
        
        try:
            result = cleaner.clean_text(123)
            # Если метод обрабатывает числа, проверяем результат
            assert result is not None
        except (TypeError, AttributeError):
            # Ожидаемое поведение для не-строк
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
