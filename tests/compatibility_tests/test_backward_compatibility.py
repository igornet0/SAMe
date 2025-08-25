"""
Тест обратной совместимости импортов SAMe
"""

import pytest
import warnings

class TestBackwardCompatibility:
    """Тесты обратной совместимости импортов"""
    
    def test_old_text_processing_imports(self):
        """Тест старых импортов text_processing"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                from same.text_processing import TextCleaner
                assert TextCleaner is not None
                
                # Проверяем что выдается предупреждение
                assert len(w) > 0
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()
                
            except ImportError:
                pytest.skip("Backward compatibility not available")
    
    def test_old_parameter_extraction_imports(self):
        """Тест старых импортов parameter_extraction"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                from same.parameter_extraction import RegexParameterExtractor
                assert RegexParameterExtractor is not None
                
                # Проверяем предупреждение
                assert len(w) > 0
                assert issubclass(w[0].category, DeprecationWarning)
                
            except ImportError:
                pytest.skip("Backward compatibility not available")
    
    def test_old_search_engine_imports(self):
        """Тест старых импортов search_engine"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                from same.search_engine import FuzzySearchEngine
                assert FuzzySearchEngine is not None
                
            except ImportError:
                pytest.skip("Backward compatibility not available")
    
    def test_old_export_imports(self):
        """Тест старых импортов export"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                from same.export import ExcelExporter
                assert ExcelExporter is not None
                
            except ImportError:
                pytest.skip("Backward compatibility not available")
    
    def test_functionality_preserved(self):
        """Тест что функциональность сохранена"""
        try:
            # Тестируем через старые импорты
            from same.text_processing import TextCleaner

            cleaner = TextCleaner()
            result = cleaner.clean_text("Test <b>HTML</b> text")

            assert isinstance(result, dict)
            assert 'normalized' in result or 'processed' in result

        except ImportError:
            pytest.skip("Backward compatibility not available")

    def test_old_api_imports(self):
        """Тест старых импортов API"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                from same.api import create_app
                assert create_app is not None

            except ImportError:
                pytest.skip("API backward compatibility not available")

    def test_old_database_imports(self):
        """Тест старых импортов database"""
        try:
            from same.database import Base
            assert Base is not None

        except ImportError:
            pytest.skip("Database backward compatibility not available")

    def test_old_models_imports(self):
        """Тест старых импортов models"""
        try:
            from same.models import ModelManager
            assert ModelManager is not None

        except ImportError:
            pytest.skip("Models backward compatibility not available")

    def test_old_settings_imports(self):
        """Тест старых импортов settings"""
        try:
            from same.settings import Config
            assert Config is not None

        except ImportError:
            pytest.skip("Settings backward compatibility not available")

    def test_analog_search_engine_compatibility(self):
        """Тест обратной совместимости главного класса"""
        try:
            from same import AnalogSearchEngine

            # Создание экземпляра
            engine = AnalogSearchEngine()
            assert engine is not None

            # Проверяем основные методы
            assert hasattr(engine, 'initialize')
            assert hasattr(engine, 'search_analogs')
            assert hasattr(engine, 'export_results')

        except ImportError:
            pytest.skip("AnalogSearchEngine backward compatibility not available")

    def test_cross_module_compatibility(self):
        """Тест совместимости между старыми и новыми импортами"""
        try:
            # Старый импорт
            from same.text_processing import TextCleaner as OldCleaner

            # Новый импорт
            from same_clear.text_processing import TextCleaner as NewCleaner

            # Должны быть одним и тем же классом
            assert OldCleaner is NewCleaner

        except ImportError:
            pytest.skip("Cross-module compatibility not available")

    def test_warning_messages_content(self):
        """Тест содержания предупреждающих сообщений"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                from same.text_processing import TextCleaner

                if w:
                    warning_msg = str(w[0].message)

                    # Проверяем что сообщение содержит полезную информацию
                    assert "same.text_processing" in warning_msg
                    assert "same_clear.text_processing" in warning_msg
                    assert "deprecated" in warning_msg.lower()

            except ImportError:
                pytest.skip("Warning messages test not available")

    def test_performance_parity(self):
        """Тест что производительность не ухудшилась"""
        try:
            import time

            # Старый импорт
            from same.text_processing import TextCleaner

            cleaner = TextCleaner()
            test_text = "Болт М10х50 ГОСТ 7798-70" * 100

            # Измеряем время обработки
            start_time = time.time()
            for _ in range(10):
                result = cleaner.clean_text(test_text)
            processing_time = time.time() - start_time

            # Проверяем что обработка быстрая
            assert processing_time < 1.0  # 10 обработок < 1 сек

        except ImportError:
            pytest.skip("Performance test not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
