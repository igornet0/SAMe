#!/usr/bin/env python3
"""
Примеры использования продвинутых методов поиска в системе SAMe

Этот файл демонстрирует различные сценарии использования
новых алгоритмов поиска и их практическое применение.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from search_interface import SAMeSearchInterface
from same_clear.search import SearchConfig
import time


def demo_basic_usage():
    """Демонстрация базового использования новых методов поиска"""
    print("🚀 Демонстрация базового использования продвинутых методов поиска")
    print("=" * 80)
    
    # Создаем интерфейс поиска
    search_interface = SAMeSearchInterface()
    
    # Инициализируем с данными
    if not search_interface.initialize():
        print("❌ Не удалось инициализировать поисковый интерфейс")
        return
    
    # Показываем статистику
    stats = search_interface.get_stats()
    print(f"\n📊 Статистика поискового движка:")
    print(f"   Всего записей: {stats.get('total_records', 'N/A')}")
    print(f"   Уникальных токенов: {stats.get('unique_token_ids', 'N/A')}")
    print(f"   Эмбеддинги доступны: {stats.get('embeddings_available', False)}")
    
    # Показываем доступные продвинутые методы
    advanced_methods = stats.get('advanced_search_methods', {})
    print(f"\n🔧 Доступные продвинутые методы:")
    for method, available in advanced_methods.items():
        status = "✅" if available else "❌"
        print(f"   {status} {method}")
    
    # Тестовые запросы
    test_queries = [
        ("светильник LED 50W", "Поиск светодиодных светильников"),
        ("автомат дифференциальный АВДТ", "Поиск защитной автоматики"),
        ("болт М10х50 ГОСТ", "Поиск крепежных изделий"),
        ("кабель силовой ВВГ", "Поиск силовых кабелей")
    ]
    
    print(f"\n🔍 Примеры поиска с разными методами:")
    print("-" * 80)
    
    for query, description in test_queries:
        print(f"\n📝 {description}")
        print(f"Запрос: '{query}'")
        print("-" * 40)
        
        # Продвинутый гибридный поиск
        results = search_interface.search(query, method="advanced_hybrid", top_k=3)
        print(f"Advanced Hybrid ({len(results)} результатов):")
        for i, result in enumerate(results[:2], 1):
            print(f"  {i}. {result.raw_name} (оценка: {result.score:.3f})")


def demo_method_comparison():
    """Сравнение различных методов поиска"""
    print("\n🔬 Сравнение методов поиска")
    print("=" * 80)
    
    search_interface = SAMeSearchInterface()
    if not search_interface.initialize():
        return
    
    query = "светильник LED"
    methods = [
        ("token_id", "Поиск по токенам"),
        ("semantic", "Семантический поиск"),
        ("tfidf", "TF-IDF поиск"),
        ("inverted_index", "Обратный индекс"),
        ("prefix", "Префиксный поиск"),
        ("advanced_hybrid", "Продвинутый гибридный")
    ]
    
    print(f"Запрос: '{query}'")
    print("-" * 60)
    
    for method, description in methods:
        try:
            start_time = time.time()
            results = search_interface.search_by_method(query, method, top_k=5)
            end_time = time.time()
            
            execution_time = end_time - start_time
            top_score = results[0].score if results else 0.0
            
            print(f"{description:25} | {execution_time:.4f}s | {len(results):2d} результатов | лучшая оценка: {top_score:.3f}")
            
        except Exception as e:
            print(f"{description:25} | ❌ Ошибка: {str(e)[:30]}...")


def demo_specialized_searches():
    """Демонстрация специализированных поисков"""
    print("\n🎯 Специализированные поиски")
    print("=" * 80)
    
    search_interface = SAMeSearchInterface()
    if not search_interface.initialize():
        return
    
    # 1. Префиксный поиск
    print("\n1. Префиксный поиск (поиск по началу слова)")
    print("-" * 50)
    prefixes = ["свет", "авто", "кабе"]
    
    for prefix in prefixes:
        results = search_interface.search_by_method(prefix, "prefix", top_k=3)
        print(f"Префикс '{prefix}': {len(results)} результатов")
        for result in results[:2]:
            print(f"  • {result.raw_name}")
    
    # 2. TF-IDF поиск (релевантность по важности слов)
    print("\n2. TF-IDF поиск (ранжирование по релевантности)")
    print("-" * 50)
    query = "автомат защиты электрический"
    results = search_interface.search_by_method(query, "tfidf", top_k=5)
    
    print(f"Запрос: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.raw_name} (релевантность: {result.score:.3f})")
    
    # 3. LSH поиск (быстрый поиск похожих)
    print("\n3. LSH поиск (быстрый поиск похожих товаров)")
    print("-" * 50)
    query = "болт М10 DIN 912"
    results = search_interface.search_by_method(query, "lsh", top_k=5)
    
    print(f"Запрос: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.raw_name} (схожесть: {result.score:.3f})")


def demo_performance_benchmark():
    """Демонстрация бенчмарка производительности"""
    print("\n⚡ Бенчмарк производительности")
    print("=" * 80)
    
    search_interface = SAMeSearchInterface()
    if not search_interface.initialize():
        return
    
    # Запускаем бенчмарк
    query = "трансформатор понижающий"
    print(f"Тестируем производительность для запроса: '{query}'")
    print("-" * 60)
    
    benchmark_results = search_interface.benchmark_methods(query, top_k=10)
    
    # Сортируем по времени выполнения
    sorted_results = sorted(
        [(method, result) for method, result in benchmark_results.items() if result['available']],
        key=lambda x: x[1]['execution_time']
    )
    
    print(f"{'Метод':<20} {'Время (с)':<12} {'Результатов':<12} {'Лучшая оценка':<15}")
    print("-" * 65)
    
    for method, result in sorted_results:
        print(f"{method:<20} {result['execution_time']:<12.4f} {result['results_count']:<12} {result['top_score']:<15.3f}")


def demo_recommendations():
    """Демонстрация системы рекомендаций"""
    print("\n💡 Система рекомендаций методов поиска")
    print("=" * 80)
    
    search_interface = SAMeSearchInterface()
    if not search_interface.initialize():
        return
    
    query_types = [
        ("general", "Общий поиск"),
        ("technical", "Технические термины"),
        ("prefix", "Поиск по префиксу"),
        ("similar", "Поиск похожих"),
        ("fast", "Быстрый поиск")
    ]
    
    print("Рекомендации по выбору метода поиска:")
    print("-" * 50)
    
    for query_type, description in query_types:
        rec = search_interface.get_method_recommendations(query_type)
        print(f"\n{description}:")
        print(f"  Основной метод: {rec['primary']}")
        print(f"  Альтернативы: {', '.join(rec['alternatives'])}")
        print(f"  Описание: {rec['description']}")


def demo_advanced_features():
    """Демонстрация продвинутых возможностей"""
    print("\n🔬 Продвинутые возможности")
    print("=" * 80)
    
    search_interface = SAMeSearchInterface()
    if not search_interface.initialize():
        return
    
    # 1. Поиск похожих товаров по коду
    print("\n1. Поиск похожих товаров по коду")
    print("-" * 40)
    
    # Получаем первый код из данных для демонстрации
    stats = search_interface.get_stats()
    if stats.get('total_records', 0) > 0:
        # Используем семантический поиск для нахождения похожих
        sample_results = search_interface.search("светильник", method="semantic", top_k=1)
        if sample_results:
            reference_code = sample_results[0].code
            similar_items = search_interface.search_similar(reference_code, method="semantic", top_k=3)
            
            print(f"Референсный товар: {sample_results[0].raw_name}")
            print(f"Похожие товары:")
            for i, item in enumerate(similar_items, 1):
                print(f"  {i}. {item.raw_name} (схожесть: {item.score:.3f})")
    
    # 2. Детальная информация о результатах
    print("\n2. Детальная информация о результатах поиска")
    print("-" * 50)
    
    results = search_interface.search("автомат защиты", method="advanced_hybrid", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nРезультат {i}:")
        print(f"  Код: {result.code}")
        print(f"  Название: {result.raw_name}")
        print(f"  Общая оценка: {result.score:.3f}")
        print(f"  Тип совпадения: {result.match_type}")
        
        # Показываем детальные оценки если доступны
        if hasattr(result, 'method_scores') and result.method_scores:
            print(f"  Детальные оценки:")
            for method, score in result.method_scores.items():
                if score > 0:
                    print(f"    {method}: {score:.3f}")


def main():
    """Главная функция с демонстрацией всех возможностей"""
    print("🎉 Демонстрация продвинутых методов поиска SAMe")
    print("=" * 80)
    
    try:
        # Базовое использование
        demo_basic_usage()
        
        # Сравнение методов
        demo_method_comparison()
        
        # Специализированные поиски
        demo_specialized_searches()
        
        # Бенчмарк производительности
        demo_performance_benchmark()
        
        # Система рекомендаций
        demo_recommendations()
        
        # Продвинутые возможности
        demo_advanced_features()
        
        print("\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
