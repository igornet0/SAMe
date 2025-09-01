#!/usr/bin/env python3
"""
Быстрый тест поля final_name на реальных данных
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_processor import DataProcessor, ProcessingConfig

async def quick_test():
    """Быстрый тест final_name"""
    
    print("🚀 Быстрый тест поля final_name")
    print("=" * 50)
    
    # Загрузим 10 записей из реального датасета
    input_file = "src/data/input/main_dataset.xlsx"
    
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return
    
    df = pd.read_excel(input_file)
    sample_df = df.head(10).copy()
    
    config = ProcessingConfig(batch_size=20, include_statistics=False, include_metadata=False)
    processor = DataProcessor(config)
    processor.original_df = sample_df
    
    processed_df = await processor.process_data()
    
    print("\n📋 Результаты обработки (Исходное → Final Name):")
    print("-" * 80)
    
    for idx, row in processed_df.iterrows():
        original = row['Наименование']
        final = row['final_name']
        
        # Вычисляем степень сжатия
        compression = (len(original) - len(final)) / len(original) * 100
        
        print(f"\n{idx + 1:2d}. Исходное ({len(original)} символов):")
        print(f"    {original}")
        print(f"    🎯 Final ({len(final)} символов, сжатие {compression:.1f}%):")
        print(f"    {final}")
        
        if row['model_brand']:
            print(f"    🏷️  Бренд: {row['model_brand']}")
    
    print(f"\n📊 Статистика:")
    avg_original = processed_df['Наименование'].str.len().mean()
    avg_final = processed_df['final_name'].str.len().mean()
    avg_compression = (avg_original - avg_final) / avg_original * 100
    
    print(f"   Средняя длина исходного: {avg_original:.1f} символов")
    print(f"   Средняя длина final_name: {avg_final:.1f} символов")
    print(f"   Среднее сжатие: {avg_compression:.1f}%")
    
    print("\n✅ Тест завершен!")

if __name__ == "__main__":
    asyncio.run(quick_test())

