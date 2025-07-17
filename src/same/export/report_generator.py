"""
Модуль генерации детальных отчетов по результатам поиска аналогов
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Конфигурация генератора отчетов"""
    # Типы отчетов
    include_summary: bool = True
    include_detailed_results: bool = True
    include_statistics: bool = True
    include_visualizations: bool = True
    include_quality_analysis: bool = True
    
    # Настройки визуализации
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"
    
    # Настройки HTML отчета
    template_path: Optional[str] = None
    include_css: bool = True
    
    # Пороги для анализа качества
    high_quality_threshold: float = 0.8
    medium_quality_threshold: float = 0.6
    
    # Языковые настройки
    language: str = "ru"


class ReportGenerator:
    """Генератор отчетов по результатам поиска аналогов"""
    
    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        
        # Настройка стиля графиков
        plt.style.use('default')
        sns.set_style(self.config.style)
        
        # Шаблоны отчетов
        self.html_template = self._get_html_template()
        
        logger.info("ReportGenerator initialized")
    
    def generate_comprehensive_report(self, 
                                    search_results: Dict[str, List[Dict[str, Any]]],
                                    metadata: Dict[str, Any] = None,
                                    output_path: str = None) -> str:
        """
        Генерация комплексного отчета
        
        Args:
            search_results: Результаты поиска
            metadata: Метаданные системы
            output_path: Путь к выходному файлу
            
        Returns:
            Путь к созданному отчету
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/comprehensive_report_{timestamp}.html"
        
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating comprehensive report: {output_path}")
        
        # Подготовка данных для отчета
        report_data = self._prepare_report_data(search_results, metadata)
        
        # Генерация компонентов отчета
        report_components = {}
        
        if self.config.include_summary:
            report_components['summary'] = self._generate_summary(report_data)
        
        if self.config.include_statistics:
            report_components['statistics'] = self._generate_statistics(report_data)
        
        if self.config.include_quality_analysis:
            report_components['quality_analysis'] = self._generate_quality_analysis(report_data)
        
        if self.config.include_visualizations:
            report_components['visualizations'] = self._generate_visualizations(report_data)
        
        if self.config.include_detailed_results:
            report_components['detailed_results'] = self._generate_detailed_results(report_data)
        
        # Генерация HTML отчета
        html_content = self._render_html_report(report_components, report_data)
        
        # Сохранение отчета
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report generated: {output_path}")
        
        return output_path
    
    def _prepare_report_data(self, 
                           search_results: Dict[str, List[Dict[str, Any]]], 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для отчета"""
        # Преобразуем результаты в DataFrame для анализа
        all_results = []
        
        for query, results in search_results.items():
            for i, result in enumerate(results):
                result_data = {
                    'query': query,
                    'rank': i + 1,
                    'document_id': result.get('document_id', ''),
                    'document': result.get('document', ''),
                    'similarity_score': result.get('similarity_score', result.get('combined_score', 0)),
                    'search_method': result.get('search_method', 'unknown'),
                    'processing_time': result.get('processing_time', 0)
                }
                
                # Добавляем дополнительные метрики если есть
                if 'fuzzy_score' in result:
                    result_data['fuzzy_score'] = result['fuzzy_score']
                if 'semantic_score' in result:
                    result_data['semantic_score'] = result['semantic_score']
                
                all_results.append(result_data)
        
        df_results = pd.DataFrame(all_results)
        
        return {
            'search_results': search_results,
            'results_df': df_results,
            'metadata': metadata or {},
            'generation_time': datetime.now(),
            'total_queries': len(search_results),
            'total_results': len(all_results)
        }
    
    def _generate_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки отчета"""
        df = report_data['results_df']
        
        if df.empty:
            return {'message': 'Нет данных для анализа'}
        
        summary = {
            'total_queries': report_data['total_queries'],
            'total_results': report_data['total_results'],
            'avg_results_per_query': report_data['total_results'] / report_data['total_queries'] if report_data['total_queries'] > 0 else 0,
            'avg_similarity_score': df['similarity_score'].mean(),
            'max_similarity_score': df['similarity_score'].max(),
            'min_similarity_score': df['similarity_score'].min(),
            'high_quality_results': len(df[df['similarity_score'] >= self.config.high_quality_threshold]),
            'medium_quality_results': len(df[(df['similarity_score'] >= self.config.medium_quality_threshold) & 
                                           (df['similarity_score'] < self.config.high_quality_threshold)]),
            'low_quality_results': len(df[df['similarity_score'] < self.config.medium_quality_threshold])
        }
        
        # Процентные соотношения
        total = report_data['total_results']
        if total > 0:
            summary['high_quality_percent'] = (summary['high_quality_results'] / total) * 100
            summary['medium_quality_percent'] = (summary['medium_quality_results'] / total) * 100
            summary['low_quality_percent'] = (summary['low_quality_results'] / total) * 100
        
        return summary
    
    def _generate_statistics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация статистики"""
        df = report_data['results_df']
        
        if df.empty:
            return {}
        
        statistics = {
            'similarity_stats': {
                'mean': df['similarity_score'].mean(),
                'median': df['similarity_score'].median(),
                'std': df['similarity_score'].std(),
                'min': df['similarity_score'].min(),
                'max': df['similarity_score'].max(),
                'q25': df['similarity_score'].quantile(0.25),
                'q75': df['similarity_score'].quantile(0.75)
            }
        }
        
        # Статистика по методам поиска
        if 'search_method' in df.columns:
            method_stats = df.groupby('search_method').agg({
                'similarity_score': ['count', 'mean', 'std'],
                'rank': 'mean'
            }).round(3)
            
            statistics['method_stats'] = method_stats.to_dict()
        
        # Статистика по запросам
        query_stats = df.groupby('query').agg({
            'similarity_score': ['count', 'mean', 'max'],
            'rank': 'count'
        }).round(3)
        
        statistics['query_stats'] = query_stats.to_dict()
        
        return statistics
    
    def _generate_quality_analysis(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ качества результатов"""
        df = report_data['results_df']
        
        if df.empty:
            return {}
        
        # Анализ распределения качества
        quality_distribution = {
            'high_quality': len(df[df['similarity_score'] >= self.config.high_quality_threshold]),
            'medium_quality': len(df[(df['similarity_score'] >= self.config.medium_quality_threshold) & 
                                   (df['similarity_score'] < self.config.high_quality_threshold)]),
            'low_quality': len(df[df['similarity_score'] < self.config.medium_quality_threshold])
        }
        
        # Анализ по рангам
        rank_analysis = df.groupby('rank')['similarity_score'].agg(['mean', 'count']).to_dict()
        
        # Топ и худшие запросы
        query_quality = df.groupby('query')['similarity_score'].mean().sort_values(ascending=False)
        
        quality_analysis = {
            'distribution': quality_distribution,
            'rank_analysis': rank_analysis,
            'best_queries': query_quality.head(10).to_dict(),
            'worst_queries': query_quality.tail(10).to_dict(),
            'queries_without_results': [
                query for query, results in report_data['search_results'].items() 
                if not results
            ]
        }
        
        return quality_analysis
    
    def _generate_visualizations(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Генерация визуализаций (возвращает base64 изображения)"""
        df = report_data['results_df']
        
        if df.empty:
            return {}
        
        visualizations = {}
        
        # График распределения схожести
        plt.figure(figsize=self.config.figure_size)
        plt.hist(df['similarity_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.config.high_quality_threshold, color='green', linestyle='--', 
                   label=f'Высокое качество (≥{self.config.high_quality_threshold})')
        plt.axvline(self.config.medium_quality_threshold, color='orange', linestyle='--', 
                   label=f'Среднее качество (≥{self.config.medium_quality_threshold})')
        plt.xlabel('Оценка схожести')
        plt.ylabel('Количество результатов')
        plt.title('Распределение оценок схожести')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        visualizations['similarity_distribution'] = self._fig_to_base64()
        
        # Boxplot по методам поиска
        if 'search_method' in df.columns and df['search_method'].nunique() > 1:
            plt.figure(figsize=self.config.figure_size)
            sns.boxplot(data=df, x='search_method', y='similarity_score')
            plt.title('Распределение качества по методам поиска')
            plt.xlabel('Метод поиска')
            plt.ylabel('Оценка схожести')
            plt.xticks(rotation=45)
            
            visualizations['method_comparison'] = self._fig_to_base64()
        
        # Топ-10 запросов по качеству
        query_quality = df.groupby('query')['similarity_score'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=self.config.figure_size)
        query_quality.plot(kind='barh')
        plt.title('Топ-10 запросов по среднему качеству результатов')
        plt.xlabel('Средняя оценка схожести')
        plt.ylabel('Запрос')
        plt.tight_layout()
        
        visualizations['top_queries'] = self._fig_to_base64()
        
        # Корреляция между рангом и качеством
        if len(df) > 10:
            plt.figure(figsize=self.config.figure_size)
            plt.scatter(df['rank'], df['similarity_score'], alpha=0.6)
            plt.xlabel('Ранг результата')
            plt.ylabel('Оценка схожести')
            plt.title('Зависимость качества от ранга результата')
            
            # Добавляем линию тренда
            z = np.polyfit(df['rank'], df['similarity_score'], 1)
            p = np.poly1d(z)
            plt.plot(df['rank'], p(df['rank']), "r--", alpha=0.8)
            
            visualizations['rank_quality_correlation'] = self._fig_to_base64()
        
        return visualizations
    
    def _fig_to_base64(self) -> str:
        """Конвертация matplotlib фигуры в base64"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    def _generate_detailed_results(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация детальных результатов"""
        detailed_results = {}
        
        for query, results in report_data['search_results'].items():
            query_details = {
                'query': query,
                'total_results': len(results),
                'results': []
            }
            
            for i, result in enumerate(results[:20]):  # Ограничиваем до 20 результатов
                result_detail = {
                    'rank': i + 1,
                    'document': result.get('document', ''),
                    'similarity_score': result.get('similarity_score', result.get('combined_score', 0)),
                    'search_method': result.get('search_method', 'unknown'),
                    'quality_level': self._get_quality_level(
                        result.get('similarity_score', result.get('combined_score', 0))
                    )
                }
                
                # Добавляем дополнительные метрики если есть
                if 'fuzzy_score' in result:
                    result_detail['fuzzy_score'] = result['fuzzy_score']
                if 'semantic_score' in result:
                    result_detail['semantic_score'] = result['semantic_score']
                
                query_details['results'].append(result_detail)
            
            detailed_results[query] = query_details
        
        return detailed_results
    
    def _get_quality_level(self, score: float) -> str:
        """Определение уровня качества по оценке"""
        if score >= self.config.high_quality_threshold:
            return 'Высокое'
        elif score >= self.config.medium_quality_threshold:
            return 'Среднее'
        else:
            return 'Низкое'
    
    def _get_html_template(self) -> str:
        """Получение HTML шаблона отчета"""
        if self.config.template_path and Path(self.config.template_path).exists():
            with open(self.config.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Базовый HTML шаблон
        return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет по поиску аналогов SAMe</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #e9e9e9; border-radius: 5px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; }
        .results-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .results-table th { background-color: #f2f2f2; }
        .quality-high { color: #28a745; font-weight: bold; }
        .quality-medium { color: #ffc107; font-weight: bold; }
        .quality-low { color: #dc3545; font-weight: bold; }
        .query-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Отчет по поиску аналогов SAMe</h1>
        <p>Дата генерации: {{ generation_time.strftime('%d.%m.%Y %H:%M:%S') }}</p>
        <p>Всего запросов: {{ total_queries }} | Всего результатов: {{ total_results }}</p>
    </div>

    {% if summary %}
    <div class="section">
        <h2>Сводка</h2>
        <div class="metric">Среднее количество результатов на запрос: {{ "%.2f"|format(summary.avg_results_per_query) }}</div>
        <div class="metric">Средняя оценка схожести: {{ "%.3f"|format(summary.avg_similarity_score) }}</div>
        <div class="metric">Высокое качество: {{ summary.high_quality_results }} ({{ "%.1f"|format(summary.high_quality_percent) }}%)</div>
        <div class="metric">Среднее качество: {{ summary.medium_quality_results }} ({{ "%.1f"|format(summary.medium_quality_percent) }}%)</div>
        <div class="metric">Низкое качество: {{ summary.low_quality_results }} ({{ "%.1f"|format(summary.low_quality_percent) }}%)</div>
    </div>
    {% endif %}

    {% if visualizations %}
    <div class="section">
        <h2>Визуализация</h2>
        {% for chart_name, chart_data in visualizations.items() %}
        <div class="chart">
            <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
            <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if detailed_results %}
    <div class="section">
        <h2>Детальные результаты</h2>
        {% for query, query_data in detailed_results.items() %}
        <div class="query-section">
            <h3>Запрос: "{{ query }}"</h3>
            <p>Найдено результатов: {{ query_data.total_results }}</p>
            
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ранг</th>
                        <th>Найденное наименование</th>
                        <th>Оценка схожести</th>
                        <th>Качество</th>
                        <th>Метод поиска</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in query_data.results %}
                    <tr>
                        <td>{{ result.rank }}</td>
                        <td>{{ result.document[:100] }}{% if result.document|length > 100 %}...{% endif %}</td>
                        <td>{{ "%.3f"|format(result.similarity_score) }}</td>
                        <td class="quality-{{ result.quality_level.lower() }}">{{ result.quality_level }}</td>
                        <td>{{ result.search_method }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Метаданные системы</h2>
        <pre>{{ metadata | tojson(indent=2) }}</pre>
    </div>
</body>
</html>
        """
    
    def _render_html_report(self, 
                           components: Dict[str, Any], 
                           report_data: Dict[str, Any]) -> str:
        """Рендеринг HTML отчета"""
        template = Template(self.html_template)
        
        context = {
            **components,
            **report_data,
            'generation_time': report_data['generation_time'],
            'total_queries': report_data['total_queries'],
            'total_results': report_data['total_results']
        }
        
        return template.render(**context)
    
    def generate_json_report(self, 
                           search_results: Dict[str, List[Dict[str, Any]]],
                           metadata: Dict[str, Any] = None,
                           output_path: str = None) -> str:
        """Генерация JSON отчета"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/json_report_{timestamp}.json"
        
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Подготовка данных
        report_data = self._prepare_report_data(search_results, metadata)
        
        # Генерация компонентов
        json_report = {
            'metadata': {
                'generation_time': report_data['generation_time'].isoformat(),
                'total_queries': report_data['total_queries'],
                'total_results': report_data['total_results'],
                'system_metadata': metadata or {}
            },
            'summary': self._generate_summary(report_data),
            'statistics': self._generate_statistics(report_data),
            'quality_analysis': self._generate_quality_analysis(report_data),
            'search_results': search_results
        }
        
        # Сохранение
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"JSON report generated: {output_path}")
        
        return output_path
    
    def generate_csv_summary(self, 
                           search_results: Dict[str, List[Dict[str, Any]]],
                           output_path: str = None) -> str:
        """Генерация CSV сводки"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/csv_summary_{timestamp}.csv"
        
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Подготовка данных
        report_data = self._prepare_report_data(search_results, {})
        df = report_data['results_df']
        
        if not df.empty:
            # Добавляем уровень качества
            df['quality_level'] = df['similarity_score'].apply(self._get_quality_level)
            
            # Сохраняем
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"CSV summary generated: {output_path}")
        
        return output_path
