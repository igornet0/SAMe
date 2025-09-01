#!/usr/bin/env python3
"""
Демонстрация исправленной обработки на небольшом наборе реальных данных
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from data_processor import DataProcessor, ProcessingConfig

async def demo_fixed_processing():
    """Демонстрация исправленной обработки"""
    
    print("🎯 Демонстрация исправленной обработки данных")
    print("=" * 60)
    
    # Загружаем несколько записей из реального датасета
    input_file = "src/data/input/main_dataset.xlsx"
    
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return
    
    # Загружаем первые 5 записей для демонстрации
    try:
        df = pd.read_excel(input_file)
        sample_df = df.head(5).copy()
        
        print(f"📊 Взяли образец из {len(sample_df)} записей:")
        for i, row in sample_df.iterrows():
            print(f"  {i+1}. {row['Наименование'][:80]}...")
        
        # Конфигурация
        config = ProcessingConfig(
            batch_size=10,
            save_format="csv",
            include_statistics=False,
            include_metadata=False
        )
        
        # Обработка
        processor = DataProcessor(config)
        processor.original_df = sample_df
        
        print("\n🔄 Обработка...")
        processed_df = await processor.process_data()
        
        print("\n📋 Результаты (показаны ключевые различия):")
        print("=" * 80)
        
        for idx, row in processed_df.iterrows():
            print(f"\n🔸 Запись {idx + 1}:")
            print(f"   📝 Исходное название:")
            print(f"      {row['Наименование']}")
            print(f"   ✨ Processed (исходный + trim):")
            print(f"      {row['processed_name']}")
            print(f"   🧹 Cleaned (очищенный):")
            print(f"      {row['cleaned_name']}")
            print(f"   🎯 Normalized (нормализованный):")
            print(f"      {row['normalized_name']}")
            print(f"   🏆 Final (чистое наименование):")
            print(f"      {row['final_name']}")
            
            # Показываем дополнительные данные
            if row['model_brand']:
                print(f"   🏷️  Бренд: {row['model_brand']}")
            if row['extracted_parameters']:
                print(f"   ⚙️  Параметры: {row['extracted_parameters']}")
            if row['category'] != 'общие_товары':
                print(f"   📁 Категория: {row['category']} (уверенность: {row['category_confidence']:.3f})")
            
            # Анализ различий
            all_fields = [row['processed_name'], row['cleaned_name'], row['normalized_name'], row['final_name']]
            stages_different = len(set(all_fields))
            some_different = stages_different > 1
            
            if some_different:
                print(f"   ✅ ЭТАПЫ ОБРАБОТКИ РАЗЛИЧАЮТСЯ")
            else:
                print(f"   ⚠️  Все этапы одинаковы")
            
            print("-" * 80)
        
        print("\n🎉 Демонстрация завершена!")
        print("\n💡 Основные улучшения:")
        print("   • processed_name: сохраняет исходный формат")
        print("   • cleaned_name: базовая очистка")  
        print("   • normalized_name: полная нормализация с лемматизацией")
        print("   • 🆕 final_name: чистое наименование без параметров")
        print("   • Добавлены извлечение брендов и параметров")
        print("   • Добавлена классификация категорий")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(demo_fixed_processing())
