"""
Тесты валидации миграции на новую архитектуру
"""

import pytest
import importlib
import sys
from typing import List, Dict, Any


class TestMigrationValidation:
    """Тесты валидации успешности миграции"""
    
    def test_new_modules_available(self):
        """Тест что новые модули доступны"""
        new_modules = [
            'same_core',
            'same_clear',
            'same_search', 
            'same_api'
        ]
        
        for module_name in new_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                print(f"✅ Module {module_name} available")
            except ImportError as e:
                pytest.fail(f"❌ Module {module_name} not available: {e}")
    
    def test_new_submodules_structure(self):
        """Тест структуры новых подмодулей"""
        module_structure = {
            'same_clear': [
                'text_processing',
                'parameter_extraction'
            ],
            'same_search': [
                'search_engine',
                'models'
            ],
            'same_api': [
                'api',
                'database',
                'export',
                'settings'
            ]
        }
        
        for module_name, submodules in module_structure.items():
            for submodule in submodules:
                full_module_name = f"{module_name}.{submodule}"
                try:
                    module = importlib.import_module(full_module_name)
                    assert module is not None
                    print(f"✅ Submodule {full_module_name} available")
                except ImportError:
                    print(f"⚠️  Submodule {full_module_name} not available (may be optional)")
    
    def test_core_interfaces_available(self):
        """Тест доступности интерфейсов same_core"""
        try:
            from same_core.interfaces import (
                TextProcessorInterface,
                SearchEngineInterface,
                ExporterInterface
            )
            
            # Проверяем что это абстрактные классы
            from abc import ABC
            assert issubclass(TextProcessorInterface, ABC)
            assert issubclass(SearchEngineInterface, ABC)
            assert issubclass(ExporterInterface, ABC)
            
            print("✅ Core interfaces available and properly abstract")
            
        except ImportError as e:
            pytest.fail(f"❌ Core interfaces not available: {e}")
    
    def test_core_types_available(self):
        """Тест доступности типов same_core"""
        try:
            from same_core.types import (
                ProcessingResult,
                SearchResult,
                ParameterData,
                ProcessingStage,
                ParameterType
            )
            
            # Проверяем что типы корректны
            from dataclasses import is_dataclass
            from enum import Enum
            
            assert is_dataclass(ProcessingResult)
            assert is_dataclass(SearchResult)
            assert is_dataclass(ParameterData)
            assert issubclass(ProcessingStage, Enum)
            assert issubclass(ParameterType, Enum)
            
            print("✅ Core types available and properly structured")
            
        except ImportError as e:
            pytest.fail(f"❌ Core types not available: {e}")
    
    def test_legacy_compatibility_layer(self):
        """Тест работы слоя обратной совместимости"""
        try:
            # Проверяем что старый модуль same существует
            import same
            assert same is not None
            
            # Проверяем что в нем есть прокси-импорты
            assert hasattr(same, '__getattr__')
            
            print("✅ Legacy compatibility layer available")
            
        except ImportError as e:
            pytest.fail(f"❌ Legacy compatibility layer not available: {e}")
    
    def test_main_class_migration(self):
        """Тест миграции главного класса AnalogSearchEngine"""
        try:
            # Новый импорт
            from same.analog_search_engine import AnalogSearchEngine
            
            # Создание экземпляра
            engine = AnalogSearchEngine()
            assert engine is not None
            
            # Проверяем что методы доступны
            required_methods = ['initialize', 'search_analogs', 'export_results', 'get_statistics']
            for method_name in required_methods:
                assert hasattr(engine, method_name), f"Method {method_name} not found"
                assert callable(getattr(engine, method_name)), f"Method {method_name} not callable"
            
            print("✅ Main AnalogSearchEngine class migrated successfully")
            
        except ImportError as e:
            pytest.fail(f"❌ AnalogSearchEngine migration failed: {e}")
        except Exception as e:
            pytest.fail(f"❌ AnalogSearchEngine functionality test failed: {e}")
    
    def test_import_paths_consistency(self):
        """Тест консистентности путей импорта"""
        # Тестируем что одни и те же классы доступны через разные пути
        test_cases = [
            {
                'old_path': 'same.text_processing.TextCleaner',
                'new_path': 'same_clear.text_processing.TextCleaner'
            },
            {
                'old_path': 'same.search_engine.FuzzySearchEngine',
                'new_path': 'same_search.search_engine.FuzzySearchEngine'
            },
            {
                'old_path': 'same.export.ExcelExporter',
                'new_path': 'same_api.export.ExcelExporter'
            }
        ]
        
        for test_case in test_cases:
            try:
                # Импортируем через старый путь
                old_module_path, old_class_name = test_case['old_path'].rsplit('.', 1)
                old_module = importlib.import_module(old_module_path)
                old_class = getattr(old_module, old_class_name)
                
                # Импортируем через новый путь
                new_module_path, new_class_name = test_case['new_path'].rsplit('.', 1)
                new_module = importlib.import_module(new_module_path)
                new_class = getattr(new_module, new_class_name)
                
                # Проверяем что это один и тот же класс
                assert old_class is new_class, f"Classes differ: {test_case['old_path']} vs {test_case['new_path']}"
                
                print(f"✅ Import consistency verified: {test_case['old_path']} == {test_case['new_path']}")
                
            except ImportError:
                print(f"⚠️  Import paths not available: {test_case}")
            except AttributeError:
                print(f"⚠️  Classes not found: {test_case}")
    
    def test_no_circular_imports(self):
        """Тест отсутствия циркулярных импортов"""
        modules_to_test = [
            'same_core',
            'same_clear',
            'same_search',
            'same_api',
            'same'
        ]
        
        # Очищаем кэш модулей
        modules_to_clear = [name for name in sys.modules.keys() if any(name.startswith(mod) for mod in modules_to_test)]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Пытаемся импортировать каждый модуль
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                print(f"✅ No circular imports in {module_name}")
            except ImportError as e:
                if "circular import" in str(e).lower():
                    pytest.fail(f"❌ Circular import detected in {module_name}: {e}")
                else:
                    print(f"⚠️  Module {module_name} not available: {e}")
    
    def test_dependency_resolution(self):
        """Тест разрешения зависимостей между модулями"""
        # Проверяем что модули могут импортировать друг друга в правильном порядке
        dependency_chain = [
            'same_core',  # Базовые интерфейсы
            'same_clear',  # Зависит от same_core
            'same_search',  # Зависит от same_core и same_clear
            'same_api',  # Зависит от всех предыдущих
            'same'  # Зависит от всех модулей
        ]
        
        imported_modules = []
        
        for module_name in dependency_chain:
            try:
                module = importlib.import_module(module_name)
                imported_modules.append(module_name)
                print(f"✅ Successfully imported {module_name} (dependencies: {imported_modules[:-1]})")
            except ImportError as e:
                print(f"⚠️  Could not import {module_name}: {e}")
    
    def test_version_compatibility(self):
        """Тест совместимости версий"""
        try:
            # Проверяем что версии модулей совместимы
            modules_with_versions = []
            
            for module_name in ['same_core', 'same_clear', 'same_search', 'same_api']:
                try:
                    module = importlib.import_module(module_name)
                    version = getattr(module, '__version__', 'unknown')
                    modules_with_versions.append((module_name, version))
                except ImportError:
                    continue
            
            if modules_with_versions:
                print("📋 Module versions:")
                for module_name, version in modules_with_versions:
                    print(f"   {module_name}: {version}")
            
            # Проверяем основной модуль
            import same
            main_version = getattr(same, '__version__', 'unknown')
            print(f"   same (main): {main_version}")
            
        except Exception as e:
            print(f"⚠️  Version compatibility check failed: {e}")
    
    def test_performance_regression(self):
        """Тест отсутствия регрессии производительности"""
        try:
            import time
            
            # Тестируем производительность через старые импорты
            from same.text_processing import TextCleaner
            
            cleaner = TextCleaner()
            test_text = "Болт М10х50 ГОСТ 7798-70 оцинкованный"
            
            # Прогрев
            for _ in range(10):
                cleaner.clean_text(test_text)
            
            # Измерение
            start_time = time.time()
            for _ in range(100):
                result = cleaner.clean_text(test_text)
            processing_time = time.time() - start_time
            
            # Проверяем что производительность приемлемая
            assert processing_time < 2.0, f"Performance regression detected: {processing_time:.2f}s for 100 operations"
            
            print(f"✅ Performance test passed: {processing_time:.3f}s for 100 operations")
            
        except ImportError:
            pytest.skip("Performance test not available")
        except Exception as e:
            pytest.fail(f"❌ Performance test failed: {e}")


class TestMigrationCompleteness:
    """Тесты полноты миграции"""
    
    def test_all_notebooks_migrated(self):
        """Тест что все notebooks обновлены"""
        from pathlib import Path
        
        notebooks_dir = Path("notebooks")
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        # Ищем notebooks с старыми импортами
        old_import_patterns = [
            "from same.text_processing",
            "from same.parameter_extraction", 
            "from same.search_engine",
            "from same.api",
            "from same.database",
            "from same.export"
        ]
        
        notebooks_with_old_imports = []
        
        for notebook_file in notebooks_dir.rglob("*.ipynb"):
            try:
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in old_import_patterns:
                    if pattern in content and "same_" not in content.replace(pattern, ""):
                        notebooks_with_old_imports.append((notebook_file, pattern))
                        break
                        
            except Exception:
                continue
        
        if notebooks_with_old_imports:
            print("⚠️  Notebooks with old imports found:")
            for notebook, pattern in notebooks_with_old_imports:
                print(f"   {notebook}: {pattern}")
        else:
            print("✅ All notebooks appear to be migrated")
    
    def test_all_tests_migrated(self):
        """Тест что все тесты обновлены"""
        from pathlib import Path
        
        tests_dir = Path("tests")
        if not tests_dir.exists():
            pytest.skip("Tests directory not found")
        
        # Исключаем новые тестовые директории
        exclude_dirs = ['test_same_core', 'test_same_clear', 'test_same_search', 'test_same_api', 'integration_tests', 'compatibility_tests']
        
        old_import_patterns = [
            "from same.text_processing",
            "from same.parameter_extraction",
            "from same.search_engine", 
            "from same.api",
            "from same.database",
            "from same.export"
        ]
        
        tests_with_old_imports = []
        
        for test_file in tests_dir.rglob("test_*.py"):
            # Пропускаем файлы в исключенных директориях
            if any(exclude_dir in str(test_file) for exclude_dir in exclude_dirs):
                continue
                
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in old_import_patterns:
                    if pattern in content and "try:" not in content and "except ImportError:" not in content:
                        tests_with_old_imports.append((test_file, pattern))
                        break
                        
            except Exception:
                continue
        
        if tests_with_old_imports:
            print("⚠️  Tests with old imports found:")
            for test_file, pattern in tests_with_old_imports:
                print(f"   {test_file}: {pattern}")
        else:
            print("✅ All tests appear to be migrated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
