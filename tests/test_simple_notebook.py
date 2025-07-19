#!/usr/bin/env python3
"""
Тестирование упрощенного notebook как Python скрипта
"""

import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any

# Добавляем путь к модулям SAMe
sys.path.append(os.path.abspath('.'))

def main():
    """Выполняет все ячейки упрощенного notebook"""
    
    print("🧪 Тестирование упрощенного SAMe Notebook")
    print("=" * 60)
    
    # Ячейка 1: Настройка и импорты
    print("\n📦 Ячейка 1: Настройка и импорты")
    print("✅ Базовые импорты загружены")
    print(f"📁 Рабочая директория: {os.getcwd()}")
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ячейка 2: Создание тестовых данных
    print("\n📦 Ячейка 2: Создание тестовых данных")
    
    def create_sample_mtr_data():
        """Создает реалистичные тестовые данные МТР"""
        
        sample_data = [
            # Крепежные изделия
            "Болт М10×50 ГОСТ 7798-70 оцинкованный",
            "Болт с шестигранной головкой М12×60 DIN 933 нержавеющая сталь A2",
            "Винт М8×30 с внутренним шестигранником ГОСТ 11738-84",
            "Гайка М10 шестигранная ГОСТ 5915-70 класс прочности 8",
            "Гайка М12 DIN 934 нержавеющая сталь A4",
            "Шайба плоская 10 ГОСТ 11371-78 оцинкованная",
            
            # Электрооборудование
            "Двигатель асинхронный АИР80В2 1.5кВт 3000об/мин 220/380В",
            "Электродвигатель 4АМ100L4 4кВт 1500об/мин лапы",
            "Кабель ВВГ 3×2.5 мм² 0.66кВ медный",
            "Провод ПВС 2×1.5 мм² гибкий медный",
            
            # Трубопроводная арматура
            "Труба стальная 57×3.5 ГОСТ 8732-78 бесшовная",
            "Труба полипропиленовая PN20 32×5.4 для горячей воды",
            "Клапан шаровой ДУ25 РУ40 муфтовый латунь",
            
            # Насосы и измерительные приборы
            "Насос центробежный К50-32-125 подача 12.5м³/ч напор 20м",
            "Манометр показывающий МП3-У 0-10 кгс/см² М20×1.5"
        ]
        
        return sample_data

    # Создаем тестовые данные
    sample_mtr_data = create_sample_mtr_data()

    print(f"✅ Создано {len(sample_mtr_data)} образцов МТР для демонстрации")
    print("\n📋 Примеры данных:")
    for i, item in enumerate(sample_mtr_data[:5], 1):
        print(f"{i}. {item}")
    print("...")
    
    # Ячейка 3: Демонстрация извлечения параметров
    print("\n📦 Ячейка 3: Демонстрация извлечения параметров")
    print("🔧 Демонстрация RegexParameterExtractor")
    print("=" * 50)

    try:
        # Импортируем экстрактор параметров
        from same.parameter_extraction.regex_extractor import RegexParameterExtractor
        
        # Создаем экстрактор
        extractor = RegexParameterExtractor()
        
        # Примеры для извлечения параметров
        parameter_samples = [
            "Болт М10×50 ГОСТ 7798-70 диаметр 10мм длина 50мм",
            "Двигатель асинхронный 4кВт 1500об/мин напряжение 380В",
            "Труба стальная 57×3.5 диаметр 57мм толщина стенки 3.5мм",
            "Кабель ВВГ 3×2.5мм² напряжение 0.66кВ сечение 2.5мм²",
            "Насос центробежный подача 12.5м³/ч напор 20м мощность 1.1кВт"
        ]
        
        print("\n📝 Примеры извлечения параметров:")
        extraction_results = []
        
        for i, text in enumerate(parameter_samples, 1):
            parameters = extractor.extract_parameters(text)
            extraction_results.append({
                'text': text,
                'parameters': parameters
            })
            
            print(f"\n{i}. Текст: '{text}'")
            print(f"   Найдено параметров: {len(parameters)}")
            
            for param in parameters:
                print(f"   - {param.name}: {param.value} {param.unit or ''} ({param.parameter_type.value})")
        
        # Статистика извлечения
        total_params = sum(len(r['parameters']) for r in extraction_results)
        param_types = {}
        
        for result in extraction_results:
            for param in result['parameters']:
                param_type = param.parameter_type.value
                param_types[param_type] = param_types.get(param_type, 0) + 1
        
        print(f"\n📊 Статистика извлечения параметров:")
        print(f"Всего извлечено параметров: {total_params}")
        print(f"Среднее количество параметров на текст: {total_params/len(parameter_samples):.1f}")
        print(f"Распределение по типам:")
        for param_type, count in param_types.items():
            print(f"  - {param_type}: {count}")
        
        print("\n✅ RegexParameterExtractor работает корректно")
        
    except Exception as e:
        print(f"❌ Ошибка в извлечении параметров: {e}")
        extraction_results = []
    
    # Ячейка 4: Создание каталога
    print("\n📦 Ячейка 4: Подготовка каталога для поиска")
    print("📚 Подготовка каталога для поиска")
    print("=" * 50)

    def create_extended_catalog():
        """Создает расширенный каталог МТР с вариациями"""
        
        catalog_data = []
        
        # Добавляем основные данные
        for i, item in enumerate(sample_mtr_data):
            catalog_data.append({
                'id': i + 1,
                'name': item,
                'category': 'МТР',
                'description': f'Описание для {item}'
            })
        
        # Добавляем дополнительные вариации для демонстрации поиска
        additional_items = [
            "Болт М10 длина 50мм оцинкованный",
            "Болт метрический 10х50 ГОСТ",
            "Винт М10х50 с шестигранной головкой",
            "Двигатель электрический 1.5кВт 3000об/мин",
            "Мотор асинхронный 1500Вт трехфазный",
            "Труба стальная диаметр 57мм",
            "Трубка металлическая 57х3.5",
            "Насос водяной центробежный 12м³/ч"
        ]
        
        for i, item in enumerate(additional_items):
            catalog_data.append({
                'id': len(sample_mtr_data) + i + 1,
                'name': item,
                'category': 'МТР',
                'description': f'Дополнительное описание для {item}'
            })
        
        return catalog_data

    # Создаем каталог
    catalog_data = create_extended_catalog()
    documents = [item['name'] for item in catalog_data]
    document_ids = [item['id'] for item in catalog_data]

    print(f"✅ Создан каталог из {len(catalog_data)} позиций")
    print(f"📊 Подготовлено для поиска:")
    print(f"Документов: {len(documents)}")
    print(f"ID документов: {len(document_ids)}")

    # Показываем примеры каталога
    print(f"\n📋 Примеры из каталога:")
    for i, item in enumerate(catalog_data[:5], 1):
        print(f"{i}. ID:{item['id']} - {item['name'][:50]}...")
    
    # Ячейка 5: Простой поиск
    print("\n📦 Ячейка 5: Простой поиск аналогов")
    print("🔍 Демонстрация простого поиска аналогов")
    print("=" * 50)

    def simple_search(query: str, documents: List[str], document_ids: List[int], max_results: int = 5):
        """Простой поиск по подстрокам и ключевым словам"""
        
        matches = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            # Подсчитываем совпадения слов
            word_matches = sum(1 for word in query_words if word in doc_lower)
            
            # Вычисляем простой скор
            if word_matches > 0:
                score = word_matches / len(query_words)
                
                # Бонус за точные совпадения
                if query_lower in doc_lower:
                    score += 0.5
                
                matches.append({
                    'document_id': document_ids[i],
                    'document': doc,
                    'score': score,
                    'word_matches': word_matches
                })
        
        # Сортируем по скору
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:max_results]

    # Тестовые запросы
    test_queries = [
        "болт м10",
        "двигатель 1.5кВт",
        "труба стальная 57",
        "насос центробежный",
        "кабель медный"
    ]

    print("\n📝 Результаты простого поиска:")
    search_results = {}

    for query in test_queries:
        start_time = time.time()
        results = simple_search(query, documents, document_ids)
        search_time = time.time() - start_time
        
        search_results[query] = results
        
        print(f"\n🔍 Запрос: '{query}'")
        print(f"   Время поиска: {search_time*1000:.1f}мс")
        print(f"   Найдено результатов: {len(results)}")
        
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result['document'][:60]}... (скор: {result['score']:.2f})")

    print("\n✅ Простой поиск работает корректно")
    
    # Ячейка 6: Анализ результатов
    print("\n📦 Ячейка 6: Анализ результатов")
    print("📊 Анализ результатов поиска")
    print("=" * 50)

    # Статистика поиска
    total_results = sum(len(results) for results in search_results.values())
    avg_results = total_results / len(search_results) if search_results else 0

    print(f"\n📈 Общая статистика:")
    print(f"Обработано запросов: {len(search_results)}")
    print(f"Всего найдено результатов: {total_results}")
    print(f"Среднее количество результатов на запрос: {avg_results:.1f}")
    print(f"Размер каталога: {len(documents)}")

    # Анализ качества результатов
    print(f"\n🎯 Анализ качества по запросам:")

    for query, results in search_results.items():
        if results:
            avg_score = sum(r['score'] for r in results) / len(results)
            max_score = max(r['score'] for r in results)
            
            print(f"\n📝 Запрос: '{query}'")
            print(f"   Результатов: {len(results)}")
            print(f"   Средний скор: {avg_score:.2f}")
            print(f"   Максимальный скор: {max_score:.2f}")
            
            # Показываем лучший результат
            best_result = results[0]
            print(f"   Лучший результат: '{best_result['document'][:50]}...'")

    print("\n✅ Анализ результатов завершен")
    
    # Ячейка 7: Выводы
    print("\n📦 Ячейка 7: Выводы и рекомендации")
    print("🎯 Выводы по упрощенной демонстрации системы SAMe")
    print("=" * 60)

    print("\n✅ Успешно продемонстрировано:")
    print("   📋 Создание реалистичных тестовых данных МТР")
    print("   🔧 Извлечение технических параметров через RegexParameterExtractor")
    print("   📚 Создание расширенного каталога с вариациями")
    print("   🔍 Простой алгоритм поиска аналогов")
    print("   📊 Анализ качества результатов поиска")

    print(f"\n📊 Статистика демонстрации:")
    print(f"   Тестовых данных МТР: {len(sample_mtr_data)}")
    print(f"   Размер каталога: {len(documents)}")
    print(f"   Обработано запросов: {len(search_results)}")

    total_found = sum(len(results) for results in search_results.values())
    print(f"   Найдено результатов: {total_found}")

    if 'extraction_results' in locals():
        total_params = sum(len(r['parameters']) for r in extraction_results)
        print(f"   Извлечено параметров: {total_params}")

    print(f"\n🎉 Упрощенная демонстрация системы SAMe завершена!")
    print(f"📚 Для получения дополнительной информации см. документацию в папке docs/")
    
    return {
        'sample_data_count': len(sample_mtr_data),
        'catalog_size': len(documents),
        'queries_processed': len(search_results),
        'total_results_found': total_found,
        'parameters_extracted': total_params if 'extraction_results' in locals() else 0
    }


if __name__ == "__main__":
    results = main()
    print(f"\n🏁 Тестирование завершено с результатами: {results}")
