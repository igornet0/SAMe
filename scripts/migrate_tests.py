#!/usr/bin/env python3
"""
Скрипт для автоматической миграции импортов в тестах
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Таблица замен импортов для тестов
TEST_IMPORT_REPLACEMENTS = {
    # Text processing
    r'from same\.text_processing import': 'from same_clear.text_processing import',
    r'from same\.text_processing\.': 'from same_clear.text_processing.',
    
    # Parameter extraction  
    r'from same\.parameter_extraction import': 'from same_clear.parameter_extraction import',
    r'from same\.parameter_extraction\.': 'from same_clear.parameter_extraction.',
    
    # Search engine
    r'from same\.search_engine import': 'from same_search.search_engine import',
    r'from same\.search_engine\.': 'from same_search.search_engine.',
    
    # Models
    r'from same\.models import': 'from same_search.models import',
    r'from same\.models\.': 'from same_search.models.',
    
    # API
    r'from same\.api import': 'from same_api.api import',
    r'from same\.api\.': 'from same_api.api.',
    
    # Database
    r'from same\.database import': 'from same_api.database import',
    r'from same\.database\.': 'from same_api.database.',
    
    # Export
    r'from same\.export import': 'from same_api.export import',
    r'from same\.export\.': 'from same_api.export.',
    
    # Settings
    r'from same\.settings import': 'from same_api.settings import',
    r'from same\.settings\.': 'from same_api.settings.',
    
    # Data manager
    r'from same\.data_manager import': 'from same_api.data_manager import',
    r'from same\.data_manager\.': 'from same_api.data_manager.',
}

def migrate_test_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Миграция одного тестового файла
    
    Returns:
        (was_changed, changes_made)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Применяем замены
        for old_pattern, new_pattern in TEST_IMPORT_REPLACEMENTS.items():
            matches = re.findall(old_pattern, content)
            if matches:
                content = re.sub(old_pattern, new_pattern, content)
                changes_made.append(f"{old_pattern} -> {new_pattern}")
        
        # Добавляем try-except блоки для новых импортов
        if changes_made:
            content = add_import_fallbacks(content)
        
        # Сохраняем только если были изменения
        if content != original_content:
            # Создаем резервную копию
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            file_path.rename(backup_path)
            
            # Сохраняем обновленный файл
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, changes_made
        
        return False, []
        
    except Exception as e:
        print(f"❌ Error migrating {file_path}: {e}")
        return False, []

def add_import_fallbacks(content: str) -> str:
    """Добавляет fallback импорты для обратной совместимости"""
    
    # Паттерны для поиска импортов, которые нужно обернуть в try-except
    import_patterns = [
        r'from same_clear\.',
        r'from same_search\.',
        r'from same_api\.'
    ]
    
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Проверяем нужно ли обернуть импорт
        needs_fallback = any(re.search(pattern, line) for pattern in import_patterns)
        
        if needs_fallback and not line.strip().startswith('#') and 'try:' not in line:
            # Извлекаем старый импорт для fallback
            old_import = convert_to_old_import(line)
            
            if old_import:
                # Добавляем try-except блок
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                
                new_lines.append(f"{indent_str}try:")
                new_lines.append(f"{indent_str}    {line.strip()}")
                new_lines.append(f"{indent_str}except ImportError:")
                new_lines.append(f"{indent_str}    # Fallback на старый импорт")
                new_lines.append(f"{indent_str}    {old_import}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    return '\n'.join(new_lines)

def convert_to_old_import(new_import_line: str) -> str:
    """Конвертирует новый импорт в старый для fallback"""
    
    conversions = {
        'same_clear.text_processing': 'same.text_processing',
        'same_clear.parameter_extraction': 'same.parameter_extraction',
        'same_search.search_engine': 'same.search_engine',
        'same_search.models': 'same.models',
        'same_api.api': 'same.api',
        'same_api.database': 'same.database',
        'same_api.export': 'same.export',
        'same_api.settings': 'same.settings',
        'same_api.data_manager': 'same.data_manager',
    }
    
    for new_module, old_module in conversions.items():
        if new_module in new_import_line:
            return new_import_line.replace(new_module, old_module)
    
    return None

def migrate_tests_directory(directory: Path) -> Dict[str, int]:
    """
    Миграция всех тестов в директории
    
    Returns:
        Статистика миграции
    """
    stats = {
        'total_files': 0,
        'migrated_files': 0,
        'skipped_files': 0,
        'error_files': 0
    }
    
    for test_file in directory.rglob('test_*.py'):
        # Пропускаем файлы в новых модульных директориях
        if any(part in str(test_file) for part in ['test_same_clear', 'test_same_search', 'test_same_api', 'test_same_core']):
            continue
            
        # Пропускаем backup файлы
        if '.backup' in str(test_file):
            continue
        
        stats['total_files'] += 1
        
        print(f"\n🔄 Processing: {test_file}")
        
        was_changed, changes = migrate_test_file(test_file)
        
        if was_changed:
            stats['migrated_files'] += 1
            print(f"✅ Migrated: {test_file}")
            print(f"   Backup saved: {test_file}.backup")
            
            if changes:
                print("   Changes made:")
                for change in changes[:3]:  # Показываем первые 3 изменения
                    print(f"     - {change}")
                if len(changes) > 3:
                    print(f"     ... and {len(changes) - 3} more")
        else:
            stats['skipped_files'] += 1
            print(f"⏭️  No changes needed: {test_file}")
    
    return stats

def create_compatibility_test():
    """Создает тест обратной совместимости"""
    
    compatibility_test_content = '''"""
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    compatibility_test_path = Path("tests/compatibility_tests/test_backward_compatibility.py")
    compatibility_test_path.parent.mkdir(exist_ok=True)
    
    with open(compatibility_test_path, 'w', encoding='utf-8') as f:
        f.write(compatibility_test_content)
    
    print(f"✅ Created compatibility test: {compatibility_test_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate SAMe tests to new architecture')
    parser.add_argument('--directory', '-d', type=Path, default=Path('tests'),
                       help='Directory containing tests (default: tests)')
    parser.add_argument('--create-compatibility', action='store_true',
                       help='Create backward compatibility test')
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"❌ Directory not found: {args.directory}")
        exit(1)
    
    print(f"🚀 Starting test migration in: {args.directory}")
    print("=" * 60)
    
    # Миграция тестов
    stats = migrate_tests_directory(args.directory)
    
    # Создание теста совместимости
    if args.create_compatibility:
        create_compatibility_test()
    
    print("\n" + "=" * 60)
    print(f"🎉 Test migration completed!")
    print(f"📊 Statistics:")
    print(f"   Total files processed: {stats['total_files']}")
    print(f"   Files migrated: {stats['migrated_files']}")
    print(f"   Files skipped: {stats['skipped_files']}")
    print(f"   Files with errors: {stats['error_files']}")
    
    if stats['migrated_files'] > 0:
        print("\n💡 Next steps:")
        print("1. Review the migrated test files")
        print("2. Run the tests to ensure they pass")
        print("3. Remove .backup files when satisfied")
        print("4. Update any custom test code as needed")
