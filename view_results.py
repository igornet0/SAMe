#!/usr/bin/env python3
"""
Скрипт для просмотра результатов поиска дубликатов и аналогов
"""

import pandas as pd
import sys
from pathlib import Path
import argparse

def view_results(results_folder: str):
    """Просмотр результатов обработки"""
    
    folder_path = Path(results_folder)
    if not folder_path.exists():
        print(f"❌ Папка {results_folder} не найдена!")
        return
    
    print(f"📁 Просмотр результатов из папки: {folder_path}")
    print("=" * 80)
    
    # 1. Отчет об обработке
    report_file = folder_path / "processing_report.txt"
    if report_file.exists():
        print("\n📊 ОТЧЕТ ОБ ОБРАБОТКЕ:")
        print("-" * 40)
        with open(report_file, 'r', encoding='utf-8') as f:
            print(f.read())
    
    # 2. Обработанные данные
    processed_file = folder_path / "processed_data_with_duplicates.csv"
    if processed_file.exists():
        print("\n📋 ОБРАБОТАННЫЕ ДАННЫЕ (первые 10 записей):")
        print("-" * 40)
        df = pd.read_csv(processed_file)
        print(f"Всего записей (без дубликатов): {len(df)}")
        print(f"Колонки: {list(df.columns)}")
        
        # Показываем статистику по дубликатам
        if 'duplicate_count' in df.columns:
            duplicates_count = len(df[df['duplicate_count'] > 0])
            print(f"Записей с дубликатами: {duplicates_count}")
        
        print("\nПервые 10 записей:")
        print(df.head(10).to_string(index=False))
        
        # Статистика по дубликатам
        if 'duplicate_count' in df.columns:
            duplicates = df[df['duplicate_count'] > 0]
            print(f"\n🔍 Найдено записей с дубликатами: {len(duplicates)}")
            if len(duplicates) > 0:
                print("Примеры записей с дубликатами:")
                print(duplicates[['Наименование', 'duplicate_count', 'duplicate_indices']].head().to_string(index=False))
    
    # 3. Результаты поиска аналогов
    analogs_file = folder_path / "analogs_search_results.csv"
    if analogs_file.exists():
        print("\n🔗 РЕЗУЛЬТАТЫ ПОИСКА АНАЛОГОВ:")
        print("-" * 40)
        analogs_df = pd.read_csv(analogs_file)
        print(f"Всего найдено аналогов: {len(analogs_df)}")
        
        # Статистика по типам
        if 'type' in analogs_df.columns:
            type_counts = analogs_df['type'].value_counts()
            print("\nСтатистика по типам аналогов:")
            for analog_type, count in type_counts.items():
                print(f"  {analog_type}: {count}")
        
        # Топ-10 аналогов по схожести
        print("\n🏆 ТОП-10 АНАЛОГОВ ПО СХОЖЕСТИ:")
        top_analogs = analogs_df.nlargest(10, 'similarity_coefficient')
        print(top_analogs[['original_name', 'similar_name', 'similarity_coefficient', 'type']].to_string(index=False))
        
    # 4. Деревья товаров
    trees_file = folder_path / "product_trees.txt"
    if trees_file.exists():
        print("\n🌳 ДЕРЕВЬЯ ТОВАРОВ:")
        print("-" * 40)
        
        with open(trees_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Всего строк в файле деревьев: {len(lines)}")
        
        # Подсчет корневых элементов
        root_count = sum(1 for line in lines if line.startswith('- ') and '(None)' in line)
        print(f"Количество корневых деревьев: {root_count}")
        
        # Показать первые несколько деревьев
        print("\nПервые 3 дерева:")
        tree_count = 0
        for i, line in enumerate(lines):
            if line.startswith('- ') and '(None)' in line:
                tree_count += 1
                if tree_count > 3:
                    break
                print(f"\nДерево {tree_count}:")
                print(line.strip())
                
                # Показать дочерние элементы
                j = i + 1
                while j < len(lines) and lines[j].startswith('    - '):
                    print(lines[j].strip())
                    j += 1
                print()

def analyze_data_structure(input_file: str):
    """Анализ структуры входных данных"""
    print(f"\n🔍 АНАЛИЗ СТРУКТУРЫ ДАННЫХ: {input_file}")
    print("-" * 50)
    
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            print("❌ Неподдерживаемый формат файла")
            return
        
        print(f"Всего записей: {len(df)}")
        print(f"Колонки: {list(df.columns)}")
        print(f"Типы данных:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        print("\nПервые 5 записей:")
        print(df.head().to_string(index=False))
        
        # Поиск колонки с наименованиями
        name_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['наименование', 'название', 'name', 'title']):
                name_columns.append(col)
        
        print(f"\n🎯 Найденные колонки с наименованиями: {name_columns}")
        
        if name_columns:
            print(f"\nПримеры из колонки '{name_columns[0]}':")
            print(df[name_columns[0]].head(10).tolist())
        
    except Exception as e:
        print(f"❌ Ошибка при анализе файла: {e}")

def main():
    parser = argparse.ArgumentParser(description='Просмотр результатов поиска дубликатов и аналогов')
    parser.add_argument('results_folder', help='Папка с результатами (например, src/data/output/2025-08-27)')
    parser.add_argument('--analyze', help='Файл для анализа структуры данных')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_data_structure(args.analyze)
    
    view_results(args.results_folder)

if __name__ == "__main__":
    main()
