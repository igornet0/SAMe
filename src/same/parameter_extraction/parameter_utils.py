"""
Утилиты для работы с извлеченными параметрами
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import json
from collections import defaultdict, Counter
from dataclasses import asdict

from .regex_extractor import ExtractedParameter, ParameterType

logger = logging.getLogger(__name__)


class ParameterFormatter:
    """Класс для форматирования параметров"""
    
    @staticmethod
    def format_parameter_value(param: ExtractedParameter) -> str:
        """
        Форматирование значения параметра для отображения
        
        Args:
            param: Извлеченный параметр
            
        Returns:
            Отформатированная строка
        """
        if param.unit:
            return f"{param.value} {param.unit}"
        else:
            return str(param.value)
    
    @staticmethod
    def format_parameters_list(parameters: List[ExtractedParameter]) -> str:
        """
        Форматирование списка параметров в строку
        
        Args:
            parameters: Список параметров
            
        Returns:
            Отформатированная строка
        """
        if not parameters:
            return "Параметры не найдены"
        
        formatted = []
        for param in parameters:
            value_str = ParameterFormatter.format_parameter_value(param)
            formatted.append(f"{param.name}: {value_str}")
        
        return "; ".join(formatted)
    
    @staticmethod
    def parameters_to_dict(parameters: List[ExtractedParameter]) -> Dict[str, Any]:
        """
        Преобразование параметров в словарь
        
        Args:
            parameters: Список параметров
            
        Returns:
            Словарь с параметрами
        """
        result = {}
        for param in parameters:
            result[param.name] = {
                'value': param.value,
                'unit': param.unit,
                'type': param.parameter_type.value,
                'confidence': param.confidence,
                'source': param.source_text
            }
        return result
    
    @staticmethod
    def parameters_to_json(parameters: List[ExtractedParameter]) -> str:
        """
        Преобразование параметров в JSON
        
        Args:
            parameters: Список параметров
            
        Returns:
            JSON строка
        """
        data = []
        for param in parameters:
            param_dict = asdict(param)
            param_dict['parameter_type'] = param.parameter_type.value
            data.append(param_dict)
        
        return json.dumps(data, ensure_ascii=False, indent=2)


class ParameterFilter:
    """Класс для фильтрации параметров"""
    
    @staticmethod
    def filter_by_type(parameters: List[ExtractedParameter], 
                      param_types: Union[ParameterType, List[ParameterType]]) -> List[ExtractedParameter]:
        """
        Фильтрация параметров по типу
        
        Args:
            parameters: Список параметров
            param_types: Тип или список типов для фильтрации
            
        Returns:
            Отфильтрованный список параметров
        """
        if isinstance(param_types, ParameterType):
            param_types = [param_types]
        
        return [p for p in parameters if p.parameter_type in param_types]
    
    @staticmethod
    def filter_by_confidence(parameters: List[ExtractedParameter], 
                           min_confidence: float = 0.5) -> List[ExtractedParameter]:
        """
        Фильтрация параметров по уверенности
        
        Args:
            parameters: Список параметров
            min_confidence: Минимальная уверенность
            
        Returns:
            Отфильтрованный список параметров
        """
        return [p for p in parameters if p.confidence >= min_confidence]
    
    @staticmethod
    def filter_by_value_range(parameters: List[ExtractedParameter],
                            min_value: Optional[float] = None,
                            max_value: Optional[float] = None) -> List[ExtractedParameter]:
        """
        Фильтрация параметров по диапазону значений
        
        Args:
            parameters: Список параметров
            min_value: Минимальное значение
            max_value: Максимальное значение
            
        Returns:
            Отфильтрованный список параметров
        """
        result = []
        for param in parameters:
            if isinstance(param.value, (int, float)):
                if min_value is not None and param.value < min_value:
                    continue
                if max_value is not None and param.value > max_value:
                    continue
                result.append(param)
            else:
                # Для нечисловых значений не применяем фильтр
                result.append(param)
        
        return result


class ParameterAnalyzer:
    """Класс для анализа параметров"""
    
    @staticmethod
    def analyze_parameters_batch(parameters_list: List[List[ExtractedParameter]]) -> Dict[str, Any]:
        """
        Анализ пакета параметров
        
        Args:
            parameters_list: Список списков параметров
            
        Returns:
            Статистика по параметрам
        """
        stats = {
            'total_items': len(parameters_list),
            'items_with_parameters': 0,
            'total_parameters': 0,
            'parameter_types_count': Counter(),
            'parameter_names_count': Counter(),
            'confidence_stats': {
                'min': 1.0,
                'max': 0.0,
                'avg': 0.0
            },
            'coverage_by_type': {}
        }
        
        all_confidences = []
        
        for parameters in parameters_list:
            if parameters:
                stats['items_with_parameters'] += 1
                stats['total_parameters'] += len(parameters)
                
                for param in parameters:
                    stats['parameter_types_count'][param.parameter_type.value] += 1
                    stats['parameter_names_count'][param.name] += 1
                    all_confidences.append(param.confidence)
        
        # Статистика по уверенности
        if all_confidences:
            stats['confidence_stats']['min'] = min(all_confidences)
            stats['confidence_stats']['max'] = max(all_confidences)
            stats['confidence_stats']['avg'] = sum(all_confidences) / len(all_confidences)
        
        # Покрытие по типам
        for param_type, count in stats['parameter_types_count'].items():
            stats['coverage_by_type'][param_type] = {
                'count': count,
                'coverage_percent': (count / stats['total_items']) * 100
            }
        
        return stats
    
    @staticmethod
    def compare_parameters(params1: List[ExtractedParameter], 
                         params2: List[ExtractedParameter]) -> Dict[str, Any]:
        """
        Сравнение двух наборов параметров
        
        Args:
            params1: Первый набор параметров
            params2: Второй набор параметров
            
        Returns:
            Результат сравнения
        """
        types1 = {p.parameter_type for p in params1}
        types2 = {p.parameter_type for p in params2}
        
        common_types = types1.intersection(types2)
        unique_types1 = types1 - types2
        unique_types2 = types2 - types1
        
        return {
            'common_types': list(common_types),
            'unique_types_1': list(unique_types1),
            'unique_types_2': list(unique_types2),
            'similarity_score': len(common_types) / len(types1.union(types2)) if types1.union(types2) else 0.0,
            'params_count_1': len(params1),
            'params_count_2': len(params2)
        }


class ParameterDataFrameUtils:
    """Утилиты для работы с параметрами в DataFrame"""
    
    @staticmethod
    def add_parameters_columns(df: pd.DataFrame, 
                             parameters_column: str = 'extracted_parameters') -> pd.DataFrame:
        """
        Добавление колонок с параметрами в DataFrame
        
        Args:
            df: Исходный DataFrame
            parameters_column: Название колонки с параметрами
            
        Returns:
            DataFrame с добавленными колонками
        """
        result_df = df.copy()
        
        # Добавляем общие колонки
        result_df['parameters_count'] = result_df[parameters_column].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        result_df['parameters_formatted'] = result_df[parameters_column].apply(
            lambda x: ParameterFormatter.format_parameters_list(x) if isinstance(x, list) else ""
        )
        
        # Добавляем колонки по типам параметров
        all_types = set()
        for params in result_df[parameters_column]:
            if isinstance(params, list):
                for param in params:
                    all_types.add(param.parameter_type.value)
        
        for param_type in all_types:
            col_name = f'has_{param_type}'
            result_df[col_name] = result_df[parameters_column].apply(
                lambda x: any(p.parameter_type.value == param_type for p in x) if isinstance(x, list) else False
            )
        
        return result_df
    
    @staticmethod
    def export_parameters_summary(df: pd.DataFrame, 
                                parameters_column: str = 'extracted_parameters',
                                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Создание сводки по параметрам
        
        Args:
            df: DataFrame с параметрами
            parameters_column: Название колонки с параметрами
            output_path: Путь для сохранения (опционально)
            
        Returns:
            DataFrame со сводкой
        """
        summary_data = []
        
        for idx, row in df.iterrows():
            parameters = row.get(parameters_column, [])
            if not isinstance(parameters, list):
                continue
            
            for param in parameters:
                summary_data.append({
                    'row_index': idx,
                    'parameter_name': param.name,
                    'parameter_type': param.parameter_type.value,
                    'value': param.value,
                    'unit': param.unit,
                    'confidence': param.confidence,
                    'source_text': param.source_text
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if output_path:
            summary_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Parameters summary exported to {output_path}")
        
        return summary_df
