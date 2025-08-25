#!/usr/bin/env python3
"""
Демонстрационный скрипт для работы с классификатором токенов и предпроцессором.
Показывает, как использовать структурированные токены для предобработки названий товаров.
"""

import sys
import os
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from same_clear.text_processing.token_classifier import TokenClassifier, create_token_classifier
from same_clear.text_processing.product_name_preprocessor import ProductNamePreprocessor, create_product_preprocessor


def demo_token_classifier():
    """Демонстрация работы классификатора токенов."""
    print("=== Демонстрация классификатора токенов ===\n")
    
    # Создаем классификатор
    classifier = create_token_classifier()
    
    # Показываем статистику
    stats = classifier.get_token_statistics()
    print("Статистика токенов:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nПримеры классификации токенов:")
    test_tokens = [
        'светильник', 'led', '32a', 'белый', 'сталь', 'на', 'bosch', 'артикул'
    ]
    
    for token in test_tokens:
        category, subcategory = classifier.classify_token(token)
        print(f"  '{token}' -> {category} ({subcategory})")
    
    # Демонстрация разбиения сложных токенов
    print("\nРазбиение сложных токенов:")
    complex_tokens = ['2-4(пар)', 'hp-s-1047a', 'м16*1,5']
    
    for token in complex_tokens:
        simple_tokens = classifier.split_complex_token(token)
        print(f"  '{token}' -> {simple_tokens}")
    
    return classifier


def demo_product_preprocessor(classifier):
    """Демонстрация работы предпроцессора названий товаров."""
    print("\n=== Демонстрация предпроцессора названий товаров ===\n")
    
    # Создаем предпроцессор
    preprocessor = ProductNamePreprocessor(classifier)
    
    # Тестовые названия товаров
    test_products = [
        "Светильник LED потолочный белый 32A 220В",
        "Автомат дифференциальный Bosch 63A 30mA",
        "Строп цепной одноветвевой 6мм 2м",
        "Кольцо уплотнительное резиновое М22",
        "Панель светодиодная потолочная 24Вт 6500К"
    ]
    
    print("Предобработка названий товаров:")
    for i, product_name in enumerate(test_products, 1):
        print(f"\n{i}. Исходное название: {product_name}")
        
        # Предобрабатываем название
        preprocessed = preprocessor.preprocess_product_name(product_name)
        
        print(f"   Очищенное название: {preprocessed['cleaned_name']}")
        print(f"   Тип товара: {preprocessed['product_type']}")
        
        # Технические характеристики
        tech_specs = preprocessed['technical_specs']
        if tech_specs:
            print("   Технические характеристики:")
            for category, params in tech_specs.items():
                print(f"     {category}: {', '.join(params)}")
        
        # Бренды
        brands = preprocessed['brand_info']
        if brands:
            print(f"   Бренды: {', '.join(brands)}")
        
        # Числовые значения
        numeric_values = preprocessed['numeric_values']
        if numeric_values:
            print("   Числовые значения:")
            for value, unit in numeric_values:
                print(f"     {value} {unit}")
        
        # Параметры для поиска
        search_params = preprocessed['searchable_parameters']
        print(f"   Параметры для поиска: {', '.join(search_params)}")
        
        # Создаем поисковый запрос
        search_query = preprocessor.create_search_query(preprocessed)
        print(f"   Поисковый запрос: {search_query}")
    
    return preprocessor


def demo_batch_processing(preprocessor):
    """Демонстрация пакетной обработки."""
    print("\n=== Демонстрация пакетной обработки ===\n")
    
    # Список названий для пакетной обработки
    batch_products = [
        "Светильник LED потолочный белый 32A 220В",
        "Автомат дифференциальный Bosch 63A 30mA",
        "Строп цепной одноветвевой 6мм 2м",
        "Кольцо уплотнительное резиновое М22",
        "Панель светодиодная потолочная 24Вт 6500К",
        "Кран шаровый стальной 1/2\" 16кгс/см²",
        "Фильтр воздушный для двигателя Cummins",
        "Кабель силовой медный 3x2.5мм² 380В"
    ]
    
    print(f"Обрабатываем {len(batch_products)} названий товаров...")
    
    # Пакетная обработка
    batch_results = preprocessor.batch_preprocess(batch_products)
    
    # Статистика обработки
    stats = preprocessor.get_preprocessing_statistics(batch_results)
    
    print("\nСтатистика обработки:")
    print(f"  Всего обработано: {stats['total_processed']}")
    print(f"  Успешно: {stats['successful']}")
    print(f"  Ошибок: {stats['errors']}")
    print(f"  Процент успеха: {stats['success_rate']:.1%}")
    print(f"  Среднее количество параметров на товар: {stats['avg_parameters_per_product']:.1f}")
    
    # Распределение по типам товаров
    print("\nРаспределение по типам товаров:")
    for product_type, count in stats['product_types_distribution'].items():
        print(f"  {product_type}: {count}")
    
    # Статистика технических характеристик
    print("\nСтатистика технических характеристик:")
    for category, count in stats['technical_specs_count'].items():
        print(f"  {category}: {count}")


def demo_export_formats(preprocessor, batch_results):
    """Демонстрация экспорта в различных форматах."""
    print("\n=== Демонстрация экспорта в различных форматах ===\n")
    
    # Экспорт в JSON
    json_output = preprocessor.export_preprocessed_data(batch_results, 'json')
    print("Экспорт в JSON (первые 500 символов):")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
    
    # Экспорт в CSV
    csv_output = preprocessor.export_preprocessed_data(batch_results, 'csv')
    print("\nЭкспорт в CSV (первые 300 символов):")
    print(csv_output[:300] + "..." if len(csv_output) > 300 else csv_output)
    
    # Экспорт в текстовом формате
    text_output = preprocessor.export_preprocessed_data(batch_results, 'text')
    print("\nЭкспорт в текстовом формате:")
    print(text_output)


def demo_config_management():
    """Демонстрация управления конфигурацией токенов."""
    print("\n=== Демонстрация управления конфигурацией ===\n")
    
    # Создаем классификатор с пользовательской конфигурацией
    classifier = create_token_classifier()
    
    # Добавляем новые токены
    print("Добавляем новые токены...")
    classifier.pass_names.add('новый_товар')
    classifier.names_model.add('новый_бренд')
    classifier.parameters['TYPE'].add('новый_тип')
    
    # Сохраняем конфигурацию
    config_path = 'custom_tokens_config.json'
    classifier.save_config(config_path)
    print(f"Конфигурация сохранена в {config_path}")
    
    # Загружаем конфигурацию заново
    new_classifier = create_token_classifier(config_path)
    print("Конфигурация загружена заново")
    
    # Проверяем, что новые токены загрузились
    print(f"Новый товар в pass_names: {'новый_товар' in new_classifier.pass_names}")
    print(f"Новый бренд в names_model: {'новый_бренд' in new_classifier.names_model}")
    print(f"Новый тип в parameters: {'новый_тип' in new_classifier.parameters['TYPE']}")
    
    # Удаляем временный файл
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"Временный файл {config_path} удален")


def main():
    """Основная функция демонстрации."""
    print("Демонстрация системы классификации и предобработки токенов\n")
    print("=" * 70)
    
    try:
        # Демонстрация классификатора токенов
        classifier = demo_token_classifier()
        
        # Демонстрация предпроцессора
        preprocessor = demo_product_preprocessor(classifier)
        
        # Демонстрация пакетной обработки
        batch_results = demo_batch_processing(preprocessor)
        
        # Демонстрация экспорта
        demo_export_formats(preprocessor, batch_results)
        
        # Демонстрация управления конфигурацией
        demo_config_management()
        
        print("\n" + "=" * 70)
        print("Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"\nОшибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

