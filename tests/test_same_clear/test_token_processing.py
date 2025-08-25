"""
Тесты для системы классификации и предобработки токенов.
"""

import pytest
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from same_clear.text_processing import (
    TokenClassifier, 
    create_token_classifier,
    ProductNamePreprocessor,
    create_product_preprocessor
)


class TestTokenClassifier:
    """Тесты для классификатора токенов."""
    
    def test_create_classifier(self):
        """Тест создания классификатора."""
        classifier = create_token_classifier()
        assert isinstance(classifier, TokenClassifier)
        assert len(classifier.pass_names) > 0
        assert len(classifier.parameters) > 0
    
    def test_classify_token(self):
        """Тест классификации токенов."""
        classifier = create_token_classifier()
        
        # Тестируем различные типы токенов
        assert classifier.classify_token('светильник') == ('PRODUCT_NAME', 'pass_names')
        assert classifier.classify_token('bosch') == ('BRAND_MODEL', 'names_model')
        assert classifier.classify_token('на') == ('STOP_WORD', 'stop_words')
        assert classifier.classify_token('led') == ('PARAMETER', 'TYPE')
        assert classifier.classify_token('32a') == ('PARAMETER', 'CURRENT')
        assert classifier.classify_token('неизвестный') == ('UNKNOWN', 'unknown')
    
    def test_split_complex_token(self):
        """Тест разбиения сложных токенов."""
        classifier = create_token_classifier()
        
        # Простые токены
        assert classifier.split_complex_token('led') == ['led']
        assert classifier.split_complex_token('32a') == ['32a']
        
        # Сложные токены
        assert '2' in classifier.split_complex_token('2-4(пар)')
        assert 'hp' in classifier.split_complex_token('hp-s-1047a')
        assert 'м16' in classifier.split_complex_token('м16*1,5')
    
    def test_normalize_product_name(self):
        """Тест нормализации названия товара."""
        classifier = create_token_classifier()
        
        result = classifier.normalize_product_name('светильник led потолочный белый')
        
        assert 'product_name' in result
        assert 'parameters' in result
        assert 'brands' in result
        assert result['product_name'] == ['светильник']
        assert 'led' in result['parameters']
        assert 'потолочный' in result['parameters']
        assert 'белый' in result['parameters']
    
    def test_extract_searchable_parameters(self):
        """Тест извлечения параметров для поиска."""
        classifier = create_token_classifier()
        
        params = classifier.extract_searchable_parameters('светильник led потолочный белый 32a')
        
        assert 'led' in params
        assert 'потолочный' in params
        assert 'белый' in params
        assert '32a' in params
        assert 'светильник' not in params  # Первое слово не включается
    
    def test_get_token_statistics(self):
        """Тест получения статистики токенов."""
        classifier = create_token_classifier()
        stats = classifier.get_token_statistics()
        
        assert 'product_names' in stats
        assert 'stop_words' in stats
        assert 'brands_models' in stats
        assert 'parameters_total' in stats
        assert stats['product_names'] > 0
        assert stats['parameters_total'] > 0


class TestProductNamePreprocessor:
    """Тесты для предпроцессора названий товаров."""
    
    def test_create_preprocessor(self):
        """Тест создания предпроцессора."""
        preprocessor = create_product_preprocessor()
        assert isinstance(preprocessor, ProductNamePreprocessor)
    
    def test_clean_text(self):
        """Тест очистки текста."""
        preprocessor = create_product_preprocessor()
        
        # Тест базовой очистки
        assert preprocessor.clean_text('  Светильник  LED  ') == 'светильник led'
        assert preprocessor.clean_text('Автомат 32A, 220В') == 'автомат 32a, 220в'
    
    def test_extract_numeric_values(self):
        """Тест извлечения числовых значений."""
        preprocessor = create_product_preprocessor()
        
        values = preprocessor.extract_numeric_values('светильник 32a 220в 24мм')
        
        assert len(values) >= 3
        assert ('32', 'a') in values or ('32', 'a') in values
        assert ('220', 'в') in values or ('220', 'в') in values
    
    def test_preprocess_product_name(self):
        """Тест полной предобработки названия товара."""
        preprocessor = create_product_preprocessor()
        
        result = preprocessor.preprocess_product_name('Светильник LED потолочный белый 32A 220В')
        
        assert 'original_name' in result
        assert 'cleaned_name' in result
        assert 'product_type' in result
        assert 'technical_specs' in result
        assert 'searchable_parameters' in result
        
        assert result['product_type'] == 'светильник'
        assert 'led' in result['searchable_parameters']
        assert 'потолочный' in result['searchable_parameters']
        assert 'белый' in result['searchable_parameters']
    
    def test_create_search_query(self):
        """Тест создания поискового запроса."""
        preprocessor = create_product_preprocessor()
        
        # Предобрабатываем название
        preprocessed = preprocessor.preprocess_product_name('Светильник LED потолочный белый 32A')
        
        # Создаем поисковый запрос
        query = preprocessor.create_search_query(preprocessed)
        
        assert isinstance(query, str)
        assert len(query) > 0
        assert 'светильник' in query.lower()
        assert 'led' in query.lower()
    
    def test_batch_preprocess(self):
        """Тест пакетной обработки."""
        preprocessor = create_product_preprocessor()
        
        product_names = [
            'Светильник LED потолочный',
            'Автомат дифференциальный Bosch'
        ]
        
        results = preprocessor.batch_preprocess(product_names)
        
        assert len(results) == 2
        assert all('product_type' in result for result in results)
        assert results[0]['product_type'] == 'светильник'
        assert results[1]['product_type'] == 'автомат'
    
    def test_get_preprocessing_statistics(self):
        """Тест получения статистики предобработки."""
        preprocessor = create_product_preprocessor()
        
        # Создаем тестовые данные
        test_data = [
            {'product_type': 'светильник', 'technical_specs': {'TYPE': ['led']}},
            {'product_type': 'автомат', 'technical_specs': {'TYPE': ['дифференциальный']}}
        ]
        
        stats = preprocessor.get_preprocessing_statistics(test_data)
        
        assert stats['total_processed'] == 2
        assert stats['successful'] == 2
        assert stats['success_rate'] == 1.0
        assert 'светильник' in stats['product_types_distribution']
        assert 'автомат' in stats['product_types_distribution']


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_pipeline(self):
        """Тест полного пайплайна обработки."""
        # Создаем компоненты
        classifier = create_token_classifier()
        preprocessor = create_product_preprocessor(classifier)
        
        # Тестовое название
        product_name = "Светильник LED потолочный белый 32A 220В 6500K"
        
        # Предобработка
        result = preprocessor.preprocess_product_name(product_name)
        
        # Проверяем результат
        assert result['product_type'] == 'светильник'
        assert 'led' in result['searchable_parameters']
        assert 'потолочный' in result['searchable_parameters']
        assert 'белый' in result['searchable_parameters']
        assert '32a' in result['searchable_parameters']
        assert '220в' in result['searchable_parameters']
        
        # Создаем поисковый запрос
        query = preprocessor.create_search_query(result)
        assert 'светильник' in query.lower()
        assert 'led' in query.lower()
    
    def test_config_management(self):
        """Тест управления конфигурацией."""
        classifier = create_token_classifier()
        
        # Добавляем новый токен
        original_count = len(classifier.pass_names)
        classifier.pass_names.add('тестовый_товар')
        
        assert len(classifier.pass_names) == original_count + 1
        assert 'тестовый_товар' in classifier.pass_names
        
        # Проверяем классификацию
        category, subcategory = classifier.classify_token('тестовый_товар')
        assert category == 'PRODUCT_NAME'
        assert subcategory == 'pass_names'


if __name__ == '__main__':
    pytest.main([__file__])

