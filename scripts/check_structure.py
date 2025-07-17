#!/usr/bin/env python3
"""
Скрипт для проверки структуры проекта SAMe
"""

import os
import sys
from pathlib import Path


def check_project_structure():
    """Проверить структуру проекта"""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src/same",
        "src/same/api",
        "src/same/text_processing",
        "src/same/search_engine", 
        "src/same/parameter_extraction",
        "src/same/export",
        "src/same/database",
        "src/same/utils",
        "tests",
        "config",
        "docs",
        "docker",
        "notebooks/demo",
        "scripts",
    ]
    
    required_files = [
        "src/same/__init__.py",
        "src/same/api/__init__.py",
        "pyproject.toml",
        "README.md",
        "Makefile",
        ".gitignore",
        ".env.example",
        ".pre-commit-config.yaml",
        "requirements.txt",
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Проверка директорий
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    # Проверка файлов
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    # Вывод результатов
    if missing_dirs or missing_files:
        print("❌ Проблемы со структурой проекта:")
        
        if missing_dirs:
            print("\n📁 Отсутствующие директории:")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
        
        if missing_files:
            print("\n📄 Отсутствующие файлы:")
            for file_path in missing_files:
                print(f"  - {file_path}")
        
        return False
    else:
        print("✅ Структура проекта корректна!")
        return True


def check_imports():
    """Проверить основные импорты"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        # Проверяем основные модули
        import same
        print(f"✅ Основной модуль same импортирован (версия: {same.__version__})")
        
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


def main():
    """Главная функция"""
    print("🔍 Проверка структуры проекта SAMe...\n")
    
    structure_ok = check_project_structure()
    imports_ok = check_imports()
    
    if structure_ok and imports_ok:
        print("\n🎉 Все проверки пройдены успешно!")
        sys.exit(0)
    else:
        print("\n💥 Обнаружены проблемы!")
        sys.exit(1)


if __name__ == "__main__":
    main()
