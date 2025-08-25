"""
Модуль для предобработки названий товаров перед поиском аналогов.
Использует TokenClassifier для нормализации и структурирования текста.
"""

import re
from typing import Dict, List, Tuple, Optional
from .token_classifier import TokenClassifier, create_token_classifier


class ProductNamePreprocessor:
    """
    Предпроцессор названий товаров для поиска аналогов.
    
    Основные функции:
    - Нормализация названий товаров
    - Извлечение ключевых параметров
    - Фильтрация служебных слов
    - Структурирование для поиска
    """
    
    def __init__(self, token_classifier: Optional[TokenClassifier] = None):
        """
        Инициализация предпроцессора.
        
        Args:
            token_classifier: классификатор токенов (если None, создается по умолчанию)
        """
        if token_classifier is None:
            self.token_classifier = create_token_classifier()
        else:
            self.token_classifier = token_classifier
        
        # Регулярные выражения для очистки
        self.cleanup_patterns = [
            (r'\s+', ' '),  # множественные пробелы
            (r'[^\w\s\-\.\,\/\(\)\[\]]', ''),  # удаление спецсимволов
            (r'^\s+|\s+$', ''),  # пробелы в начале и конце
        ]
        
        # Паттерны для извлечения числовых значений
        self.numeric_patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:мм|см|м|кг|г|л|мл|вт|квт|а|в|ка|к|°с|°к)',
            r'(\d+(?:[.,]\d+)?)\s*(?:mm|cm|m|kg|g|l|ml|w|kw|a|v|ka|k|c|k)',
            r'(\d+(?:[.,]\d+)?)\s*(?:дюйм|inch|дюйма)',
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Базовая очистка текста от лишних символов.
        
        Args:
            text: исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower().strip()
        
        # Применяем паттерны очистки
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def extract_numeric_values(self, text: str) -> List[Tuple[str, str]]:
        """
        Извлечение числовых значений с единицами измерения.
        
        Args:
            text: текст для анализа
            
        Returns:
            Список кортежей (значение, единица измерения)
        """
        numeric_values = []
        
        for pattern in self.numeric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                unit = match.group(0)[len(value):].strip()
                numeric_values.append((value, unit))
        
        return numeric_values
    
    def preprocess_product_name(self, product_name: str) -> Dict[str, any]:
        """
        Полная предобработка названия товара.
        
        Args:
            product_name: исходное название товара
            
        Returns:
            Словарь с обработанными данными
        """
        # Очищаем текст
        cleaned_text = self.clean_text(product_name)
        
        # Нормализуем с помощью классификатора
        normalized = self.token_classifier.normalize_product_name(cleaned_text)
        
        # Извлекаем числовые значения
        numeric_values = self.extract_numeric_values(cleaned_text)
        
        # Формируем результат
        result = {
            'original_name': product_name,
            'cleaned_name': cleaned_text,
            'normalized': normalized,
            'numeric_values': numeric_values,
            'searchable_parameters': self.token_classifier.extract_searchable_parameters(cleaned_text),
            'product_type': normalized.get('product_name', [''])[0] if normalized.get('product_name') else '',
            'technical_specs': self._extract_technical_specs(normalized),
            'brand_info': normalized.get('brands', []),
            'article_info': normalized.get('articles', [])
        }
        
        return result
    
    def _extract_technical_specs(self, normalized: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Извлечение технических характеристик по категориям.
        
        Args:
            normalized: нормализованные данные
            
        Returns:
            Словарь технических характеристик по категориям
        """
        tech_specs = {}
        parameters = normalized.get('parameters', [])
        
        # Группируем параметры по категориям
        for param in parameters:
            category, subcategory = self.token_classifier.classify_token(param)
            if category == 'PARAMETER':
                if subcategory not in tech_specs:
                    tech_specs[subcategory] = []
                tech_specs[subcategory].append(param)
        
        return tech_specs
    
    def create_search_query(self, preprocessed_data: Dict[str, any], 
                          include_brands: bool = True,
                          include_articles: bool = False,
                          min_parameter_weight: float = 0.5) -> str:
        """
        Создание поискового запроса на основе предобработанных данных.
        
        Args:
            preprocessed_data: предобработанные данные
            include_brands: включать ли бренды в поиск
            include_articles: включать ли артикулы в поиск
            min_parameter_weight: минимальный вес параметра для включения
            
        Returns:
            Поисковый запрос
        """
        query_parts = []
        
        # Добавляем тип товара
        product_type = preprocessed_data.get('product_type', '')
        if product_type:
            query_parts.append(product_type)
        
        # Добавляем технические характеристики
        tech_specs = preprocessed_data.get('technical_specs', {})
        for category, params in tech_specs.items():
            if len(params) > 0:
                # Берем наиболее значимые параметры
                significant_params = params[:3]  # Первые 3 параметра категории
                query_parts.extend(significant_params)
        
        # Добавляем бренды (если нужно)
        if include_brands:
            brands = preprocessed_data.get('brand_info', [])
            query_parts.extend(brands[:2])  # Максимум 2 бренда
        
        # Добавляем артикулы (если нужно)
        if include_articles:
            articles = preprocessed_data.get('article_info', [])
            query_parts.extend(articles[:1])  # Максимум 1 артикул
        
        # Объединяем в строку
        query = ' '.join(query_parts)
        
        # Очищаем от лишних пробелов
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def batch_preprocess(self, product_names: List[str]) -> List[Dict[str, any]]:
        """
        Пакетная предобработка списка названий товаров.
        
        Args:
            product_names: список названий товаров
            
        Returns:
            Список предобработанных данных
        """
        results = []
        
        for name in product_names:
            try:
                preprocessed = self.preprocess_product_name(name)
                results.append(preprocessed)
            except Exception as e:
                # Логируем ошибку и добавляем базовую информацию
                results.append({
                    'original_name': name,
                    'cleaned_name': self.clean_text(name),
                    'error': str(e),
                    'normalized': {},
                    'searchable_parameters': [],
                    'product_type': '',
                    'technical_specs': {},
                    'brand_info': [],
                    'article_info': []
                })
        
        return results
    
    def get_preprocessing_statistics(self, preprocessed_data: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Получение статистики по предобработке.
        
        Args:
            preprocessed_data: список предобработанных данных
            
        Returns:
            Словарь со статистикой
        """
        total = len(preprocessed_data)
        successful = sum(1 for item in preprocessed_data if 'error' not in item)
        errors = total - successful
        
        # Статистика по типам товаров
        product_types = {}
        for item in preprocessed_data:
            product_type = item.get('product_type', 'unknown')
            product_types[product_type] = product_types.get(product_type, 0) + 1
        
        # Статистика по техническим характеристикам
        tech_specs_count = {}
        for item in preprocessed_data:
            tech_specs = item.get('technical_specs', {})
            for category, params in tech_specs.items():
                tech_specs_count[category] = tech_specs_count.get(category, 0) + len(params)
        
        return {
            'total_processed': total,
            'successful': successful,
            'errors': errors,
            'success_rate': successful / total if total > 0 else 0,
            'product_types_distribution': product_types,
            'technical_specs_count': tech_specs_count,
            'avg_parameters_per_product': sum(len(item.get('searchable_parameters', [])) 
                                            for item in preprocessed_data) / total if total > 0 else 0
        }
    
    def export_preprocessed_data(self, preprocessed_data: List[Dict[str, any]], 
                               output_format: str = 'json') -> str:
        """
        Экспорт предобработанных данных в различных форматах.
        
        Args:
            preprocessed_data: предобработанные данные
            output_format: формат вывода ('json', 'csv', 'text')
            
        Returns:
            Строка с данными в указанном формате
        """
        if output_format == 'json':
            import json
            return json.dumps(preprocessed_data, ensure_ascii=False, indent=2)
        
        elif output_format == 'csv':
            import csv
            import io
            
            if not preprocessed_data:
                return ""
            
            # Определяем все возможные поля
            all_fields = set()
            for item in preprocessed_data:
                all_fields.update(item.keys())
            
            # Сортируем поля для консистентности
            fieldnames = sorted(all_fields)
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in preprocessed_data:
                # Заполняем отсутствующие поля пустыми значениями
                row = {field: item.get(field, '') for field in fieldnames}
                writer.writerow(row)
            
            return output.getvalue()
        
        elif output_format == 'text':
            lines = []
            for i, item in enumerate(preprocessed_data):
                lines.append(f"=== Товар {i+1} ===")
                lines.append(f"Оригинальное название: {item.get('original_name', '')}")
                lines.append(f"Очищенное название: {item.get('cleaned_name', '')}")
                lines.append(f"Тип товара: {item.get('product_type', '')}")
                
                tech_specs = item.get('technical_specs', {})
                if tech_specs:
                    lines.append("Технические характеристики:")
                    for category, params in tech_specs.items():
                        lines.append(f"  {category}: {', '.join(params)}")
                
                brands = item.get('brand_info', [])
                if brands:
                    lines.append(f"Бренды: {', '.join(brands)}")
                
                lines.append("")  # Пустая строка между товарами
            
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {output_format}")


def create_product_preprocessor(config_path: Optional[str] = None) -> ProductNamePreprocessor:
    """
    Фабричная функция для создания предпроцессора названий товаров.
    
    Args:
        config_path: путь к конфигурационному файлу токенов
        
    Returns:
        Экземпляр ProductNamePreprocessor
    """
    token_classifier = create_token_classifier(config_path)
    return ProductNamePreprocessor(token_classifier)
