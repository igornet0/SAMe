"""
Интеграция реального каталога товаров с системой категоризации SAMe
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict, Counter
import re

from .category_classifier import CategoryClassifier, CategoryClassifierConfig

logger = logging.getLogger(__name__)


class CatalogIntegrator:
    """Интегратор реального каталога с системой категоризации"""
    
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.df = None
        self.category_mapping = {}
        self.quality_issues = {}
        
    def load_catalog(self) -> bool:
        """Загрузка каталога товаров"""
        try:
            logger.info(f"Loading catalog from {self.catalog_path}")
            self.df = pd.read_excel(self.catalog_path)
            logger.info(f"Loaded {len(self.df):,} products with {len(self.df.columns)} columns")
            return True
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return False
    
    def analyze_category_columns(self) -> Dict[str, Dict]:
        """Анализ столбцов категорий в каталоге"""
        if self.df is None:
            return {}
        
        # Поиск столбцов категорий
        category_columns = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['группа', 'категор', 'вид', 'тип', 'класс']):
                category_columns.append(col)
        
        analysis = {}
        
        for col in category_columns:
            unique_count = self.df[col].nunique()
            non_null_count = self.df[col].notna().sum()
            top_categories = self.df[col].value_counts().head(10)
            
            analysis[col] = {
                'unique_count': unique_count,
                'non_null_count': non_null_count,
                'coverage': non_null_count / len(self.df),
                'top_categories': top_categories.to_dict(),
                'distribution': self._analyze_distribution(self.df[col])
            }
            
            logger.info(f"Column '{col}': {unique_count} unique values, "
                       f"{non_null_count:,} non-null ({non_null_count/len(self.df)*100:.1f}%)")
        
        return analysis
    
    def _analyze_distribution(self, series: pd.Series) -> Dict:
        """Анализ распределения значений в столбце"""
        value_counts = series.value_counts()
        
        return {
            'most_common': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_common': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'median_frequency': value_counts.median(),
            'categories_with_1_product': (value_counts == 1).sum(),
            'categories_with_100plus_products': (value_counts >= 100).sum()
        }
    
    def check_data_quality(self) -> Dict[str, any]:
        """Проверка качества данных для классификации"""
        if self.df is None:
            return {}
        
        # Поиск столбца с названиями
        name_columns = [col for col in self.df.columns 
                       if any(keyword in col.lower() for keyword in ['наименование', 'название', 'номенклатура'])]
        
        if not name_columns:
            logger.warning("No name column found")
            return {'error': 'No name column found'}
        
        name_col = name_columns[0]
        category_columns = [col for col in self.df.columns 
                           if any(keyword in col.lower() for keyword in ['группа', 'вид'])]
        
        quality_issues = {
            'duplicate_names': 0,
            'inconsistent_categorization': {},
            'empty_categories': 0,
            'examples': {}
        }
        
        # Анализ дубликатов названий
        duplicate_names = self.df[self.df[name_col].duplicated(keep=False)]
        quality_issues['duplicate_names'] = len(duplicate_names)
        
        # Анализ несогласованной категоризации
        for cat_col in category_columns:
            inconsistent_count = 0
            examples = []
            
            for name, group in duplicate_names.groupby(name_col):
                if group[cat_col].nunique() > 1:
                    inconsistent_count += 1
                    if len(examples) < 5:
                        categories = group[cat_col].unique()
                        examples.append((name, list(categories)))
            
            quality_issues['inconsistent_categorization'][cat_col] = {
                'count': inconsistent_count,
                'examples': examples
            }
        
        # Анализ пустых категорий
        for cat_col in category_columns:
            empty_count = self.df[cat_col].isna().sum()
            quality_issues['empty_categories'] += empty_count
        
        self.quality_issues = quality_issues
        return quality_issues
    
    def search_ice_cleats_products(self) -> Dict[str, any]:
        """Поиск товаров типа ледоходов для валидации"""
        if self.df is None:
            return {}
        
        name_columns = [col for col in self.df.columns 
                       if any(keyword in col.lower() for keyword in ['наименование', 'название'])]
        
        if not name_columns:
            return {'error': 'No name column found'}
        
        name_col = name_columns[0]
        
        # Поиск ледоходов
        ice_cleats_mask = self.df[name_col].str.contains('ледоход', case=False, na=False)
        ice_cleats_products = self.df[ice_cleats_mask]
        
        # Поиск конкретного запроса
        specific_mask = self.df[name_col].str.contains(
            r'ледоход.*проф.*10|ледоход.*10.*проф', 
            case=False, na=False, regex=True
        )
        specific_products = self.df[specific_mask]
        
        # Анализ категоризации
        category_columns = [col for col in self.df.columns 
                           if any(keyword in col.lower() for keyword in ['группа', 'вид'])]
        
        categorization = {}
        for cat_col in category_columns:
            if len(ice_cleats_products) > 0:
                categories = ice_cleats_products[cat_col].value_counts()
                categorization[cat_col] = categories.to_dict()
        
        return {
            'total_ice_cleats': len(ice_cleats_products),
            'specific_matches': len(specific_products),
            'examples': ice_cleats_products[name_col].head(10).tolist() if len(ice_cleats_products) > 0 else [],
            'specific_examples': specific_products[name_col].tolist() if len(specific_products) > 0 else [],
            'categorization': categorization
        }
    
    def map_to_same_categories(self) -> Dict[str, Dict]:
        """Сопоставление категорий каталога с системой SAMe"""
        if self.df is None:
            return {}
        
        # Категории SAMe
        same_categories = {
            'средства_защиты': ['ледоход', 'каска', 'перчатки', 'сапоги', 'жилет', 'очки', 'респиратор', 'маска', 'шлем', 'защитн'],
            'химия': ['сольвент', 'растворитель', 'краска', 'лак', 'грунтовка', 'эмаль', 'клей', 'химич'],
            'металлопрокат': ['швеллер', 'уголок', 'балка', 'лист', 'труба', 'профиль', 'арматура', 'металл'],
            'крепеж': ['болт', 'гайка', 'винт', 'саморез', 'шуруп', 'заклепка', 'дюбель', 'крепеж'],
            'электрика': ['кабель', 'провод', 'розетка', 'выключатель', 'лампа', 'светильник', 'электр'],
            'сантехника': ['кран', 'вентиль', 'фитинг', 'насос', 'фильтр', 'смеситель', 'сантехн'],
            'инструменты': ['отвертка', 'ключ', 'молоток', 'пила', 'дрель', 'болгарка', 'инструмент'],
            'текстиль': ['ткань', 'брезент', 'тент', 'полог', 'мешок', 'веревка', 'текстиль']
        }
        
        name_columns = [col for col in self.df.columns 
                       if any(keyword in col.lower() for keyword in ['наименование', 'название'])]
        category_columns = [col for col in self.df.columns 
                           if any(keyword in col.lower() for keyword in ['группа', 'вид'])]
        
        if not name_columns:
            return {}
        
        name_col = name_columns[0]
        mapping_results = {}
        
        for cat_col in category_columns:
            category_mapping = {}
            unique_categories = self.df[cat_col].dropna().unique()
            
            for dataset_category in unique_categories:
                # Получаем товары из этой категории
                category_products = self.df[self.df[cat_col] == dataset_category][name_col].dropna()
                
                if len(category_products) == 0:
                    continue
                
                # Подсчитываем совпадения с ключевыми словами
                best_match = None
                max_score = 0
                
                for same_cat, keywords in same_categories.items():
                    score = 0
                    for keyword in keywords:
                        matches = category_products.str.contains(keyword, case=False, na=False).sum()
                        score += matches
                    
                    if score > max_score:
                        max_score = score
                        best_match = same_cat
                
                if best_match and max_score > 0:
                    confidence = max_score / len(category_products)
                    category_mapping[dataset_category] = {
                        'same_category': best_match,
                        'confidence': confidence,
                        'matches': max_score,
                        'total_products': len(category_products)
                    }
            
            mapping_results[cat_col] = category_mapping
        
        self.category_mapping = mapping_results
        return mapping_results
    
    def generate_enhanced_classifier_rules(self) -> Dict[str, Set[str]]:
        """Генерация улучшенных правил классификации на основе каталога"""
        if self.df is None or not self.category_mapping:
            return {}
        
        enhanced_rules = defaultdict(set)
        
        # Извлекаем ключевые слова из реальных данных
        name_columns = [col for col in self.df.columns 
                       if any(keyword in col.lower() for keyword in ['наименование', 'название'])]
        
        if not name_columns:
            return {}
        
        name_col = name_columns[0]
        
        for cat_col, mappings in self.category_mapping.items():
            for dataset_cat, mapping_info in mappings.items():
                same_category = mapping_info['same_category']
                
                # Получаем товары из этой категории
                category_products = self.df[self.df[cat_col] == dataset_cat][name_col].dropna()
                
                # Извлекаем часто встречающиеся слова
                all_words = []
                for product_name in category_products:
                    words = re.findall(r'\b[а-яёa-z]{3,}\b', product_name.lower())
                    all_words.extend(words)
                
                # Находим наиболее частые слова (исключая общие)
                word_counts = Counter(all_words)
                common_words = {'для', 'или', 'под', 'при', 'без', 'над', 'про', 'все', 'как', 'что', 'это', 'тот'}
                
                for word, count in word_counts.most_common(10):
                    if word not in common_words and count >= 3:
                        enhanced_rules[same_category].add(word)
        
        return dict(enhanced_rules)
    
    def create_integration_report(self) -> Dict[str, any]:
        """Создание отчета об интеграции"""
        if self.df is None:
            return {'error': 'Catalog not loaded'}
        
        # Выполняем все анализы
        category_analysis = self.analyze_category_columns()
        quality_analysis = self.check_data_quality()
        ice_cleats_analysis = self.search_ice_cleats_products()
        mapping_analysis = self.map_to_same_categories()
        enhanced_rules = self.generate_enhanced_classifier_rules()
        
        # Оценка пригодности
        suitability_score = 0
        max_score = 5
        
        if len(category_analysis) > 0:
            suitability_score += 1
        
        if ice_cleats_analysis.get('total_ice_cleats', 0) > 0:
            suitability_score += 1
        
        if len(mapping_analysis) > 0:
            suitability_score += 1
        
        if len(self.df) > 1000:
            suitability_score += 1
        
        if quality_analysis.get('duplicate_names', 0) < len(self.df) * 0.1:
            suitability_score += 1
        
        return {
            'dataset_info': {
                'total_products': len(self.df),
                'columns': list(self.df.columns),
                'suitability_score': f"{suitability_score}/{max_score}"
            },
            'category_analysis': category_analysis,
            'quality_analysis': quality_analysis,
            'ice_cleats_analysis': ice_cleats_analysis,
            'mapping_analysis': mapping_analysis,
            'enhanced_rules': enhanced_rules,
            'recommendations': self._generate_recommendations(suitability_score)
        }
    
    def _generate_recommendations(self, suitability_score: int) -> List[str]:
        """Генерация рекомендаций по интеграции"""
        recommendations = []
        
        if suitability_score >= 4:
            recommendations.extend([
                "✅ Каталог отлично подходит для улучшения классификации",
                "🔧 Интегрировать enhanced_rules в CategoryClassifier",
                "📊 Использовать реальные данные для валидации точности",
                "🎯 Создать тесты на основе найденных товаров 'ледоход'"
            ])
        elif suitability_score >= 3:
            recommendations.extend([
                "⚠️ Каталог подходит с небольшими доработками",
                "🔧 Очистить данные от дубликатов и несогласованностей",
                "📊 Дополнить недостающие категории вручную",
                "🎯 Валидировать маппинг категорий"
            ])
        else:
            recommendations.extend([
                "❌ Каталог требует значительной доработки",
                "🔧 Провести глубокую очистку данных",
                "📊 Создать дополнительные правила категоризации",
                "🎯 Рассмотреть альтернативные источники данных"
            ])
        
        return recommendations
