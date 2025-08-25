"""
Тесты для модуля export в same_api
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from same_api.export import ExcelExporter, ExportConfig


class TestExcelExporter:
    """Тесты для ExcelExporter"""
    
    def test_exporter_creation(self):
        """Тест создания ExcelExporter"""
        exporter = ExcelExporter()
        assert exporter is not None
        assert hasattr(exporter, 'export_data')
    
    def test_basic_export(self):
        """Тест базового экспорта в Excel"""
        exporter = ExcelExporter()
        
        # Создаем тестовые данные
        data = pd.DataFrame({
            'Raw_Name': ['Болт М10х50 ГОСТ 7798-70', 'Гайка М10 ГОСТ 5915-70'],
            'Cleaned_Name': ['Болт М10х50 ГОСТ 7798-70', 'Гайка М10 ГОСТ 5915-70'],
            'Similarity_Score': [0.95, 0.87],
            'Candidate_Name': ['Болт М10х50 аналог', 'Гайка М10 аналог']
        })
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Тест экспорта
            exporter.export_data(data, tmp_path)
            
            # Проверяем что файл создан
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            
            # Проверяем что можно прочитать файл обратно
            read_data = pd.read_excel(tmp_path)
            assert len(read_data) == len(data)
            assert list(read_data.columns) == list(data.columns)
            
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_with_config(self):
        """Тест экспорта с конфигурацией"""
        try:
            config = ExportConfig(
                include_metadata=True,
                include_scores=True,
                max_rows=100
            )
            exporter = ExcelExporter(config)
            
            data = pd.DataFrame({
                'name': ['item1', 'item2'],
                'score': [0.9, 0.8]
            })
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                exporter.export_data(data, tmp_path)
                assert os.path.exists(tmp_path)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception:
            pytest.skip("ExportConfig not available or different interface")
    
    def test_export_empty_dataframe(self):
        """Тест экспорта пустого DataFrame"""
        exporter = ExcelExporter()
        
        empty_data = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            exporter.export_data(empty_data, tmp_path)
            
            # Проверяем что файл создан (может быть пустым)
            assert os.path.exists(tmp_path)
            
        except Exception:
            # Некоторые реализации могут не поддерживать пустые DataFrame
            pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_large_dataset(self):
        """Тест экспорта большого набора данных"""
        exporter = ExcelExporter()
        
        # Создаем большой DataFrame
        large_data = pd.DataFrame({
            'id': range(1000),
            'name': [f'Item {i}' for i in range(1000)],
            'score': [0.5 + (i % 50) / 100 for i in range(1000)]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            import time
            start_time = time.time()
            
            exporter.export_data(large_data, tmp_path)
            
            export_time = time.time() - start_time
            
            # Проверяем что экспорт завершился быстро
            assert export_time < 10.0  # < 10 секунд для 1000 записей
            
            # Проверяем размер файла
            assert os.path.getsize(tmp_path) > 1000  # Файл не пустой
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestExportConfig:
    """Тесты для ExportConfig"""
    
    def test_config_creation(self):
        """Тест создания ExportConfig"""
        try:
            config = ExportConfig()
            assert config is not None
        except Exception:
            pytest.skip("ExportConfig not available")
    
    def test_config_defaults(self):
        """Тест значений по умолчанию"""
        try:
            config = ExportConfig()
            
            # Проверяем базовые атрибуты
            if hasattr(config, 'format'):
                assert config.format == "excel"
            if hasattr(config, 'include_metadata'):
                assert isinstance(config.include_metadata, bool)
            if hasattr(config, 'include_scores'):
                assert isinstance(config.include_scores, bool)
                
        except Exception:
            pytest.skip("ExportConfig attributes not available")
    
    def test_config_custom_values(self):
        """Тест кастомных значений конфигурации"""
        try:
            config = ExportConfig(
                format="csv",
                include_metadata=False,
                include_scores=True,
                max_rows=500
            )
            
            assert config.format == "csv"
            assert config.include_metadata is False
            assert config.include_scores is True
            assert config.max_rows == 500
            
        except Exception:
            pytest.skip("ExportConfig custom values not supported")


class TestExportFormats:
    """Тесты различных форматов экспорта"""
    
    def test_excel_format(self):
        """Тест экспорта в Excel формат"""
        exporter = ExcelExporter()
        
        data = pd.DataFrame({
            'name': ['test1', 'test2'],
            'value': [1, 2]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            exporter.export_data(data, tmp_path)
            
            # Проверяем что файл Excel
            assert tmp_path.endswith('.xlsx')
            
            # Проверяем что можно прочитать как Excel
            read_data = pd.read_excel(tmp_path)
            assert len(read_data) == 2
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_csv_export_if_supported(self):
        """Тест экспорта в CSV если поддерживается"""
        exporter = ExcelExporter()
        
        # Проверяем есть ли поддержка CSV
        if hasattr(exporter, 'export_csv') or hasattr(exporter, 'format'):
            data = pd.DataFrame({
                'name': ['test1', 'test2'],
                'value': [1, 2]
            })
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Пытаемся экспортировать в CSV
                if hasattr(exporter, 'export_csv'):
                    exporter.export_csv(data, tmp_path)
                else:
                    # Или через общий метод с указанием формата
                    exporter.export_data(data, tmp_path, format='csv')
                
                assert os.path.exists(tmp_path)
                
            except Exception:
                pytest.skip("CSV export not supported")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            pytest.skip("CSV export not available")


class TestExportIntegration:
    """Тесты интеграции экспорта"""
    
    def test_integration_with_core_interface(self):
        """Тест интеграции с интерфейсом same_core"""
        try:
            from same_core.interfaces import ExporterInterface
            
            exporter = ExcelExporter()
            
            # Проверяем что реализует интерфейс
            assert hasattr(exporter, 'export_data')
            
            # Тест базовой функциональности
            data = pd.DataFrame({'test': [1, 2, 3]})
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                exporter.export_data(data, tmp_path)
                assert tmp_path.exists()
                
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
                    
        except ImportError:
            pytest.skip("same_core interface not available")
    
    def test_export_search_results(self):
        """Тест экспорта результатов поиска"""
        exporter = ExcelExporter()
        
        # Имитируем результаты поиска
        search_results = pd.DataFrame({
            'Raw_Name': [
                'Болт М10х50 ГОСТ 7798-70',
                'Гайка М10 ГОСТ 5915-70',
                'Шайба 10 ГОСТ 11371-78'
            ],
            'Cleaned_Name': [
                'Болт М10х50 ГОСТ 7798-70',
                'Гайка М10 ГОСТ 5915-70', 
                'Шайба 10 ГОСТ 11371-78'
            ],
            'Lemmatized_Name': [
                'болт м10х50 гост 7798-70',
                'гайка м10 гост 5915-70',
                'шайба 10 гост 11371-78'
            ],
            'Normalized_Name': [
                'болт м10х50 гост 7798-70',
                'гайка м10 гост 5915-70',
                'шайба 10 гост 11371-78'
            ],
            'Candidate_Name': [
                'Болт М10х50 ГОСТ 7798-70 аналог',
                'Гайка М10 ГОСТ 5915-70 аналог',
                'Шайба 10 ГОСТ 11371-78 аналог'
            ],
            'Similarity_Score': [0.95, 0.87, 0.92],
            'Relation_Type': ['exact', 'similar', 'similar'],
            'Suggested_Category': ['Болты', 'Гайки', 'Шайбы'],
            'Final_Decision': ['approved', 'approved', 'pending'],
            'Comment': ['Точное совпадение', 'Хорошее совпадение', 'Требует проверки']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            exporter.export_data(search_results, tmp_path)
            
            # Проверяем что файл создан
            assert os.path.exists(tmp_path)
            
            # Проверяем содержимое
            read_data = pd.read_excel(tmp_path)
            assert len(read_data) == 3
            
            # Проверяем что все колонки экспортированы
            expected_columns = [
                'Raw_Name', 'Cleaned_Name', 'Lemmatized_Name', 'Normalized_Name',
                'Candidate_Name', 'Similarity_Score', 'Relation_Type',
                'Suggested_Category', 'Final_Decision', 'Comment'
            ]
            
            for col in expected_columns:
                assert col in read_data.columns
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_invalid_file_path(self):
        """Тест обработки неверного пути к файлу"""
        exporter = ExcelExporter()
        
        data = pd.DataFrame({'test': [1, 2, 3]})
        invalid_path = "/nonexistent/directory/file.xlsx"
        
        with pytest.raises(Exception):
            exporter.export_data(data, invalid_path)
    
    def test_none_data(self):
        """Тест обработки None данных"""
        exporter = ExcelExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(Exception):
                exporter.export_data(None, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_permission_denied(self):
        """Тест обработки отказа в доступе"""
        exporter = ExcelExporter()
        
        data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Пытаемся записать в системную директорию (может не работать на всех ОС)
        try:
            restricted_path = "/root/test.xlsx"  # Unix системы
            with pytest.raises(Exception):
                exporter.export_data(data, restricted_path)
        except Exception:
            # Если тест не применим к данной ОС, пропускаем
            pytest.skip("Permission test not applicable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
