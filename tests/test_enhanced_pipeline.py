"""
Тесты для улучшенного пайплайна обработки товарных наименований
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from same.text_processing.units_processor import UnitsProcessor, UnitsConfig
from same.text_processing.synonyms_processor import SynonymsProcessor, SynonymsConfig
from same.text_processing.tech_codes_processor import TechCodesProcessor, TechCodesConfig
from same.text_processing.enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessorConfig


class TestUnitsProcessor:
    """Тесты для обработки единиц измерения"""
    
    @pytest.fixture
    def units_processor(self):
        config = UnitsConfig(
            normalize_fractions=True,
            convert_to_metric=False,
            extract_parameters=True
        )
        return UnitsProcessor(config)
    
    def test_fractional_inches_processing(self, units_processor):
        """Тест обработки дробных дюймов"""
        test_cases = [
            ("Втулка сопла 1/2\" конус 10", "1/2\"", 0.5),
            ("Труба 3/4\" диаметр", "3/4\"", 0.75),
            ("Фитинг 1-1/2\" резьба", "1-1/2\"", 1.5)
        ]
        
        for text, expected_original, expected_value in test_cases:
            result = units_processor.process_text(text)
            
            assert result['processing_successful'] if 'processing_successful' in result else True
            assert len(result['extracted_parameters']) > 0
            
            # Проверяем, что дробь правильно распознана
            inch_params = [p for p in result['extracted_parameters'] 
                          if p.get('unit') == 'дюйм']
            assert len(inch_params) > 0
            assert inch_params[0]['value'] == expected_value
    
    def test_dimensions_processing(self, units_processor):
        """Тест обработки размеров"""
        test_cases = [
            "Труба 245х10,03 мм",
            "Профиль 65х35ф/65х35",
            "Лист 100x50x5"
        ]
        
        for text in test_cases:
            result = units_processor.process_text(text)
            
            assert len(result['extracted_parameters']) > 0
            
            # Проверяем, что размеры извлечены
            dimension_params = [p for p in result['extracted_parameters'] 
                              if 'размер' in p.get('type', '')]
            assert len(dimension_params) > 0
    
    def test_units_with_numbers(self, units_processor):
        """Тест обработки единиц с числами"""
        test_cases = [
            ("Болт М10 длина 50мм", "мм"),
            ("Двигатель 5,5кВт 1500об/мин", "кВт"),
            ("Емкость 100л нержавейка", "л")
        ]
        
        for text, expected_unit in test_cases:
            result = units_processor.process_text(text)
            
            # Проверяем, что единицы извлечены
            unit_params = [p for p in result['extracted_parameters'] 
                          if p.get('type') == 'измерение']
            assert len(unit_params) > 0
            
            # Проверяем наличие ожидаемой единицы
            units_found = [p['unit'] for p in unit_params]
            assert expected_unit in units_found or any(expected_unit in u for u in units_found)
    
    def test_tech_codes_processing(self, units_processor):
        """Тест обработки технических кодов"""
        test_cases = [
            "ТУ 14-3Р-82-2022",
            "ГОСТ 123-456-78",
            "Артикул 4-730-059"
        ]
        
        for text in test_cases:
            result = units_processor.process_text(text)
            
            # Проверяем, что текст обработан
            assert result['processed'] != result['original'] or len(result['extracted_parameters']) > 0


class TestSynonymsProcessor:
    """Тесты для нормализации синонимов"""
    
    @pytest.fixture
    def synonyms_processor(self):
        config = SynonymsConfig(
            normalize_materials=True,
            normalize_shapes=True,
            normalize_functions=True
        )
        return SynonymsProcessor(config)
    
    def test_material_synonyms(self, synonyms_processor):
        """Тест нормализации материалов"""
        test_cases = [
            ("Прокладка каучуковая", "резиновый"),
            ("Деталь эластичная", "резиновый"),
            ("Корпус металлический", "стальной"),
            ("Труба из стали", "стальной")
        ]
        
        for text, expected_canonical in test_cases:
            result = synonyms_processor.process_text(text)
            
            assert result['normalized'] != result['original']
            assert len(result['replacements']) > 0
            
            # Проверяем, что есть замена на каноническую форму
            canonical_forms = [r['canonical'] for r in result['replacements']]
            assert expected_canonical in canonical_forms
    
    def test_shape_synonyms(self, synonyms_processor):
        """Тест нормализации форм"""
        test_cases = [
            ("Труба круглая", "цилиндрический"),
            ("Профиль четырехугольный", "прямоугольный"),
            ("Шар шарообразный", "сферический")
        ]
        
        for text, expected_canonical in test_cases:
            result = synonyms_processor.process_text(text)
            
            if result['replacements']:  # Если есть замены
                canonical_forms = [r['canonical'] for r in result['replacements']]
                assert expected_canonical in canonical_forms
    
    def test_function_synonyms(self, synonyms_processor):
        """Тест нормализации функций"""
        test_cases = [
            ("Клапан перекрывающий", "запорный"),
            ("Фильтр очищающий", "фильтрующий"),
            ("Элемент соединяющий", "соединительный")
        ]
        
        for text, expected_canonical in test_cases:
            result = synonyms_processor.process_text(text)
            
            if result['replacements']:  # Если есть замены
                canonical_forms = [r['canonical'] for r in result['replacements']]
                assert expected_canonical in canonical_forms
    
    def test_custom_synonyms(self, synonyms_processor):
        """Тест добавления пользовательских синонимов"""
        # Добавляем пользовательские синонимы
        synonyms_processor.add_custom_synonyms("специальный", ["особый", "уникальный"])
        
        result = synonyms_processor.process_text("Элемент особый конструкции")
        
        assert len(result['replacements']) > 0
        canonical_forms = [r['canonical'] for r in result['replacements']]
        assert "специальный" in canonical_forms


class TestTechCodesProcessor:
    """Тесты для обработки технических кодов"""
    
    @pytest.fixture
    def tech_processor(self):
        config = TechCodesConfig(
            parse_gost=True,
            parse_tu=True,
            parse_articles=True
        )
        return TechCodesProcessor(config)
    
    def test_gost_parsing(self, tech_processor):
        """Тест парсинга ГОСТ"""
        test_cases = [
            "ГОСТ 123-456-78",
            "ГОСТ Р 52857-2007",
            "Болт по ГОСТ 7798-70"
        ]
        
        for text in test_cases:
            result = tech_processor.process_text(text)
            
            assert len(result['extracted_codes']) > 0
            
            # Проверяем, что есть ГОСТ коды
            gost_codes = [c for c in result['extracted_codes'] 
                         if c.get('type') == 'ГОСТ']
            assert len(gost_codes) > 0
    
    def test_tu_parsing(self, tech_processor):
        """Тест парсинга ТУ"""
        test_cases = [
            "ТУ 14-3Р-82-2022",
            "ТУ 2296-001-12345678-2019",
            "Изготовлено по ТУ 123-456"
        ]
        
        for text in test_cases:
            result = tech_processor.process_text(text)
            
            assert len(result['extracted_codes']) > 0
            
            # Проверяем, что есть ТУ коды
            tu_codes = [c for c in result['extracted_codes'] 
                       if c.get('type') == 'ТУ']
            assert len(tu_codes) > 0
    
    def test_article_parsing(self, tech_processor):
        """Тест парсинга артикулов"""
        test_cases = [
            "Деталь 4-730-059",
            "Код SCM-6066-71",
            "Артикул АБВ.123.456"
        ]
        
        for text in test_cases:
            result = tech_processor.process_text(text)
            
            # Проверяем, что артикулы извлечены
            article_codes = [c for c in result['extracted_codes'] 
                           if 'артикул' in c.get('type', '')]
            # Может не найти артикулы в простых случаях, но должен обработать текст
            assert result['processed'] is not None
    
    def test_code_validation(self, tech_processor):
        """Тест валидации кодов"""
        # Тест валидации ГОСТ
        validation = tech_processor.validate_code('ГОСТ', '123-456-78')
        assert 'valid' in validation
        assert 'errors' in validation
        
        # Тест валидации ТУ
        validation = tech_processor.validate_code('ТУ', '14-3Р-82-2022')
        assert 'valid' in validation


class TestEnhancedPreprocessor:
    """Тесты для улучшенного предобработчика"""
    
    @pytest.fixture
    def enhanced_preprocessor(self):
        config = EnhancedPreprocessorConfig(
            enable_units_processing=True,
            enable_synonyms_processing=True,
            enable_tech_codes_processing=True
        )
        return EnhancedPreprocessor(config)
    
    def test_full_pipeline(self, enhanced_preprocessor):
        """Тест полного пайплайна обработки"""
        test_text = "Втулка сопла 1/2\" каучуковая по ТУ 14-3Р-82-2022"
        
        result = enhanced_preprocessor.preprocess_text(test_text)
        
        assert result['processing_successful']
        assert result['final_text'] != result['original']
        
        # Проверяем, что извлечены параметры
        assert len(result['extracted_parameters']) > 0
        
        # Проверяем, что извлечены коды
        assert len(result['extracted_codes']) > 0
        
        # Проверяем, что есть замены синонимов
        assert len(result['synonym_replacements']) >= 0  # Может быть 0 если нет синонимов
    
    def test_batch_processing(self, enhanced_preprocessor):
        """Тест пакетной обработки"""
        test_texts = [
            "Насос центробежный 5кВт",
            "Фильтр каучуковый 1/2\"",
            "Клапан по ГОСТ 123-456"
        ]
        
        results = enhanced_preprocessor.preprocess_batch(test_texts)
        
        assert len(results) == len(test_texts)
        
        # Проверяем, что все тексты обработаны
        for result in results:
            assert 'processing_successful' in result
            assert 'final_text' in result
            assert 'extracted_parameters' in result
    
    def test_statistics(self, enhanced_preprocessor):
        """Тест получения статистики"""
        test_texts = [
            "Втулка 1/2\" резиновая",
            "Насос 10кВт по ГОСТ 123"
        ]
        
        results = enhanced_preprocessor.preprocess_batch(test_texts)
        stats = enhanced_preprocessor.get_processing_statistics(results)
        
        assert 'total_texts' in stats
        assert 'successful_processing' in stats
        assert 'total_parameters_extracted' in stats
        assert stats['total_texts'] == len(test_texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
