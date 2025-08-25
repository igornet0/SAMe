#!/usr/bin/env python3
"""
Интерфейс поиска для системы SAMe

Этот модуль предоставляет простой интерфейс для поиска по токенам
с использованием обученного векторизатора и обработанных данных.
"""

import sys
import pandas as pd
from pathlib import Path
import logging
from typing import List

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from same_clear.search import TokenSearchEngine, SearchResult, SearchConfig
# from excel_processor import ExcelProcessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAMeSearchInterface:
    """Высокоуровневый интерфейс поиска для системы SAMe"""
    
    def __init__(self, data_file: str = None, tokenizer_config: str = "vectorized_tokenizer"):
        """
        Инициализация интерфейса поиска
        
        Args:
            data_file: Путь к файлу с обработанными данными
            tokenizer_config: Конфигурация токенизатора
        """
        self.data_file = data_file
        self.tokenizer_config = tokenizer_config
        
        # Компоненты системы
        self.search_engine = None
        self.processor = None
        
        # Статус инициализации
        self.is_ready = False
        
        logger.info("SAMe Search Interface initialized")
    
    def initialize(self, data_file: str = None) -> bool:
        """
        Инициализация поискового движка
        
        Args:
            data_file: Путь к файлу с данными (опционально)
            
        Returns:
            True если инициализация прошла успешно
        """
        try:
            # Используем переданный файл или файл по умолчанию
            if data_file:
                self.data_file = data_file
            
            if not self.data_file:
                # Пробуем найти файл с улучшениями
                possible_files = [
                    "test_final_improvements_output.csv",
                    "test_vectorization_fix_output.csv", 
                    "src/data/output/proccesed2.csv"
                ]
                
                for file_path in possible_files:
                    if Path(file_path).exists():
                        self.data_file = file_path
                        logger.info(f"Using data file: {file_path}")
                        break
                
                if not self.data_file:
                    logger.error("No data file found")
                    return False
            
            # Создаем процессор для получения векторизатора
            logger.info("Initializing processor and vectorizer...")
            logger.warning("Processor not available, skipping...")
            # self.processor = ExcelProcessor(tokenizer_config_name=self.tokenizer_config)
            
            # Проверяем, что векторизатор доступен
            if not hasattr(self.processor.tokenizer, '_vectorizer') or self.processor.tokenizer._vectorizer is None:
                logger.error("Vectorizer not available in tokenizer")
                return False
            
            # Если векторизатор не обучен, пробуем загрузить или обучить
            if not self.processor.tokenizer._vectorizer.is_fitted:
                logger.info("Vectorizer not fitted, attempting to train...")
                
                # Загружаем данные для обучения
                df = pd.read_csv(self.data_file)
                success = self.processor.train_vectorizer_on_data(df, sample_size=1000)
                
                if not success:
                    logger.error("Failed to train vectorizer")
                    return False
            
            # Создаем поисковый движок
            config = SearchConfig(
                token_id_weight=0.6,
                semantic_weight=0.4,
                similarity_threshold=0.2,
                max_results=50
            )
            
            self.search_engine = TokenSearchEngine(config)
            
            # Загружаем данные в поисковый движок
            success = self.search_engine.load_data(
                self.data_file,
                self.processor.tokenizer._vectorizer,
                self.processor.tokenizer
            )
            
            if success:
                self.is_ready = True
                logger.info("✅ Search interface ready!")
                return True
            else:
                logger.error("Failed to load data into search engine")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing search interface: {e}")
            return False
    
    def search(self, query: str, method: str = "hybrid", top_k: int = 10) -> List[SearchResult]:
        """
        Поиск по запросу

        Args:
            query: Текстовый запрос
            method: Метод поиска ('token_id', 'semantic', 'hybrid', 'extended_hybrid',
                   'advanced_hybrid', 'prefix', 'inverted_index', 'tfidf', 'lsh', 'spatial')
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.is_ready:
            logger.error("Search interface not ready. Call initialize() first.")
            return []
        
        return self.search_engine.search_by_tokens(query, method, top_k)
    
    def search_similar(self, reference_code: str, method: str = "semantic", top_k: int = 10) -> List[SearchResult]:
        """
        Поиск похожих записей по коду товара
        
        Args:
            reference_code: Код товара для поиска похожих
            method: Метод поиска
            top_k: Количество результатов
            
        Returns:
            Список похожих записей
        """
        if not self.is_ready:
            logger.error("Search interface not ready")
            return []
        
        # Находим запись по коду
        df = self.search_engine.data_df
        reference_row = df[df['Код'] == reference_code]
        
        if reference_row.empty:
            logger.warning(f"Reference code not found: {reference_code}")
            return []
        
        # Используем название товара как запрос
        reference_name = reference_row.iloc[0]['Raw_Name']
        logger.info(f"Searching for items similar to: {reference_name}")
        
        # Выполняем поиск, исключая исходную запись
        results = self.search(reference_name, method, top_k + 1)
        
        # Убираем исходную запись из результатов
        filtered_results = [r for r in results if r.code != reference_code]
        
        return filtered_results[:top_k]
    
    def get_stats(self) -> dict:
        """Получение статистики поискового движка"""
        if not self.is_ready:
            return {'status': 'not_ready'}
        
        return self.search_engine.get_search_stats()

    def search_by_method(self, query: str, method: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск с использованием конкретного метода

        Args:
            query: Поисковый запрос
            method: Конкретный метод поиска
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.is_ready:
            logger.error("Search interface not ready")
            return []

        return self.search_engine.search_by_tokens(query, method, top_k)

    def benchmark_methods(self, query: str, top_k: int = 10) -> dict:
        """
        Сравнение производительности всех доступных методов поиска

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Словарь с результатами бенчмарка
        """
        if not self.is_ready:
            logger.error("Search interface not ready")
            return {}

        import time

        methods = [
            'token_id', 'semantic', 'hybrid', 'extended_hybrid', 'advanced_hybrid',
            'prefix', 'inverted_index', 'tfidf', 'lsh', 'spatial'
        ]

        benchmark_results = {}

        for method in methods:
            try:
                start_time = time.time()
                results = self.search_by_method(query, method, top_k)
                end_time = time.time()

                benchmark_results[method] = {
                    'execution_time': end_time - start_time,
                    'results_count': len(results),
                    'top_score': results[0].score if results else 0.0,
                    'available': True
                }

            except Exception as e:
                benchmark_results[method] = {
                    'execution_time': 0.0,
                    'results_count': 0,
                    'top_score': 0.0,
                    'available': False,
                    'error': str(e)
                }

        return benchmark_results

    def get_method_recommendations(self, query_type: str = "general") -> dict:
        """
        Получение рекомендаций по выбору метода поиска

        Args:
            query_type: Тип запроса ('general', 'technical', 'prefix', 'similar')

        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            'general': {
                'primary': 'advanced_hybrid',
                'alternatives': ['extended_hybrid', 'hybrid'],
                'description': 'Универсальный поиск для большинства запросов'
            },
            'technical': {
                'primary': 'token_id',
                'alternatives': ['inverted_index', 'tfidf'],
                'description': 'Поиск технических терминов и точных совпадений'
            },
            'prefix': {
                'primary': 'prefix',
                'alternatives': ['trie', 'inverted_index'],
                'description': 'Поиск по началу названия или артикула'
            },
            'similar': {
                'primary': 'semantic',
                'alternatives': ['lsh', 'spatial'],
                'description': 'Поиск семантически похожих товаров'
            },
            'fast': {
                'primary': 'spatial',
                'alternatives': ['lsh', 'inverted_index'],
                'description': 'Быстрый поиск для больших объемов данных'
            }
        }

        return recommendations.get(query_type, recommendations['general'])
    
    def print_results(self, results: List[SearchResult], show_details: bool = True):
        """
        Красивый вывод результатов поиска
        
        Args:
            results: Результаты поиска
            show_details: Показывать детали (токены, параметры)
        """
        if not results:
            print("🔍 Результаты не найдены")
            return
        
        print(f"🔍 Найдено результатов: {len(results)}")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📦 {result.code}")
            print(f"   📝 {result.raw_name}")
            print(f"   📊 Оценка: {result.score:.3f} ({result.match_type})")
            
            if show_details:
                if result.tokens:
                    print(f"   🔤 Токены: {result.tokens}")
                
                if result.token_vectors:
                    print(f"   🔢 Векторы: {result.token_vectors}")
                
                if result.parameters:
                    print(f"   ⚙️  Параметры: {result.parameters}")
                
                if hasattr(result, 'matched_tokens') and result.matched_tokens:
                    print(f"   ✅ Совпадающие токены: {result.matched_tokens}")
                
                if hasattr(result, 'similarity_score') and result.similarity_score > 0:
                    print(f"   🎯 Семантическое сходство: {result.similarity_score:.3f}")

                # Дополнительная информация для расширенного поиска
                if hasattr(result, 'method_scores') and result.method_scores:
                    print(f"   📈 Детальные оценки:")
                    for method, score in result.method_scores.items():
                        if score > 0:
                            method_name = {
                                'exact_match': 'Точное совпадение',
                                'partial_match': 'Частичное совпадение',
                                'semantic': 'Семантика',
                                'subset_match': 'Подмножества',
                            }.get(method, method)
                            print(f"      {method_name}: {score:.3f}")

                if hasattr(result, 'technical_boost') and result.technical_boost > 0:
                    print(f"   🔧 Технический бонус: {result.technical_boost:.3f}")

            print("-" * 80)


def main():
    """Демонстрация работы интерфейса поиска"""
    
    print("🚀 SAMe Search Interface Demo")
    print("=" * 50)
    
    # Создаем интерфейс
    search_interface = SAMeSearchInterface()
    
    # Инициализируем
    if not search_interface.initialize():
        print("❌ Failed to initialize search interface")
        return
    
    # Показываем статистику
    stats = search_interface.get_stats()
    print(f"\n📊 Search Engine Stats:")
    print(f"   Total records: {stats.get('total_records', 'N/A')}")
    print(f"   Unique tokens: {stats.get('unique_token_ids', 'N/A')}")
    print(f"   Embeddings available: {stats.get('embeddings_available', False)}")

    # Показываем доступные продвинутые методы
    advanced_methods = stats.get('advanced_search_methods', {})
    print(f"\n🚀 Advanced Search Methods:")
    for method, available in advanced_methods.items():
        status = "✅" if available else "❌"
        print(f"   {status} {method}")

    # Примеры поиска с разными методами
    test_queries = [
        ("светильник LED 50W", "advanced_hybrid"),
        ("автомат дифференциальный АВДТ", "token_id"),
        ("болт М10х50 ГОСТ", "tfidf"),
        ("кресло для руководителя", "semantic")
    ]

    for query, method in test_queries:
        print(f"\n🔍 Поиск: '{query}' (метод: {method})")
        print("-" * 60)

        results = search_interface.search(query, method=method, top_k=3)
        search_interface.print_results(results, show_details=False)

    # Демонстрация бенчмарка
    print(f"\n⚡ Benchmark для запроса 'светильник LED':")
    print("-" * 60)
    benchmark_results = search_interface.benchmark_methods("светильник LED", top_k=5)

    for method, result in benchmark_results.items():
        if result['available']:
            print(f"   {method:15} | {result['execution_time']:.4f}s | {result['results_count']:2d} results | score: {result['top_score']:.3f}")
        else:
            print(f"   {method:15} | ❌ Not available")

    # Показываем рекомендации
    print(f"\n💡 Рекомендации по выбору метода:")
    print("-" * 40)
    for query_type in ['general', 'technical', 'prefix', 'similar', 'fast']:
        rec = search_interface.get_method_recommendations(query_type)
        print(f"   {query_type:10}: {rec['primary']} ({rec['description']})")


if __name__ == "__main__":
    main()
