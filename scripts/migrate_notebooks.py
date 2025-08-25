#!/usr/bin/env python3
"""
Скрипт для автоматической миграции импортов в Jupyter notebooks
"""

import json
import re
from pathlib import Path
from typing import Dict, List

# Таблица замен импортов для notebooks
NOTEBOOK_IMPORT_REPLACEMENTS = {
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
}

def migrate_notebook_cell(cell_source: List[str]) -> tuple[List[str], bool]:
    """
    Миграция одной ячейки notebook
    
    Returns:
        (updated_source, was_changed)
    """
    updated_source = []
    was_changed = False
    
    for line in cell_source:
        original_line = line
        
        # Применяем замены
        for old_pattern, new_pattern in NOTEBOOK_IMPORT_REPLACEMENTS.items():
            new_line = re.sub(old_pattern, new_pattern, line)
            if new_line != line:
                line = new_line
                was_changed = True
        
        updated_source.append(line)
    
    return updated_source, was_changed

def migrate_notebook(notebook_path: Path) -> bool:
    """
    Миграция одного notebook файла
    
    Returns:
        True если были изменения
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        total_changes = 0
        
        # Обрабатываем все ячейки
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, str):
                    source = source.split('\n')
                
                updated_source, was_changed = migrate_notebook_cell(source)
                
                if was_changed:
                    # Восстанавливаем формат source
                    if isinstance(cell.get('source'), str):
                        cell['source'] = '\n'.join(updated_source)
                    else:
                        cell['source'] = updated_source
                    total_changes += 1
        
        # Сохраняем только если были изменения
        if total_changes > 0:
            # Создаем резервную копию
            backup_path = notebook_path.with_suffix('.ipynb.backup')
            notebook_path.rename(backup_path)
            
            # Сохраняем обновленный notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            
            print(f"✅ Migrated: {notebook_path} ({total_changes} cells changed)")
            print(f"   Backup saved: {backup_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error migrating {notebook_path}: {e}")
        return False

def add_migration_notice(notebook_path: Path):
    """Добавляет уведомление о миграции в начало notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Создаем ячейку с уведомлением
        migration_notice = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚠️ Уведомление о миграции\n",
                "\n",
                "Этот notebook был автоматически обновлен для работы с новой модульной архитектурой SAMe:\n",
                "\n",
                "- 🧹 **same_clear** - Обработка и очистка текста\n",
                "- 🔍 **same_search** - Поиск и индексация\n",
                "- 🌐 **same_api** - API и интеграции\n",
                "\n",
                "📚 См. [MIGRATION_GUIDE.md](../../MIGRATION_GUIDE.md) для подробной информации.\n",
                "\n",
                "---\n"
            ]
        }
        
        # Добавляем уведомление в начало
        notebook['cells'].insert(0, migration_notice)
        
        # Сохраняем
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"📝 Added migration notice to: {notebook_path}")
        
    except Exception as e:
        print(f"❌ Error adding notice to {notebook_path}: {e}")

def migrate_notebooks_directory(directory: Path, add_notices: bool = True) -> int:
    """
    Миграция всех notebooks в директории
    
    Returns:
        Количество обновленных файлов
    """
    migrated_count = 0
    
    for notebook_path in directory.rglob('*.ipynb'):
        # Пропускаем backup файлы и checkpoint директории
        if '.backup' in str(notebook_path) or '.ipynb_checkpoints' in str(notebook_path):
            continue
            
        print(f"\n🔄 Processing: {notebook_path}")
        
        if migrate_notebook(notebook_path):
            migrated_count += 1
            
            if add_notices:
                add_migration_notice(notebook_path)
    
    return migrated_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate SAMe notebooks to new architecture')
    parser.add_argument('--directory', '-d', type=Path, default=Path('notebooks'),
                       help='Directory containing notebooks (default: notebooks)')
    parser.add_argument('--no-notices', action='store_true',
                       help='Skip adding migration notices')
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"❌ Directory not found: {args.directory}")
        exit(1)
    
    print(f"🚀 Starting notebook migration in: {args.directory}")
    print("=" * 60)
    
    migrated_count = migrate_notebooks_directory(
        args.directory, 
        add_notices=not args.no_notices
    )
    
    print("\n" + "=" * 60)
    print(f"🎉 Migration completed!")
    print(f"📊 Total notebooks migrated: {migrated_count}")
    
    if migrated_count > 0:
        print("\n💡 Next steps:")
        print("1. Review the migrated notebooks")
        print("2. Test the updated imports")
        print("3. Remove .backup files when satisfied")
        print("4. Update any custom code as needed")
    else:
        print("\n✅ No notebooks needed migration")
