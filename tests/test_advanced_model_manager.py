"""
Тесты для Advanced Model Manager
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from same.models import (
    AdvancedModelManager, ModelType, ModelConfig, 
    get_model_manager, MemoryMonitor
)
from same.models.exceptions import ModelLoadError, ModelNotFoundError


class TestAdvancedModelManager:
    """Тесты для AdvancedModelManager"""
    
    @pytest.fixture
    def manager(self):
        """Создание экземпляра менеджера для тестов"""
        # Создаем новый экземпляр для каждого теста
        AdvancedModelManager._instance = None
        return AdvancedModelManager()
    
    def test_singleton_pattern(self):
        """Тест singleton pattern"""
        manager1 = AdvancedModelManager()
        manager2 = AdvancedModelManager()
        assert manager1 is manager2
    
    def test_global_instance(self):
        """Тест глобального экземпляра"""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2
    
    def test_model_config_registration(self, manager):
        """Тест регистрации конфигурации модели"""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.SPACY,
            model_path="ru_core_news_sm",
            cache_size_gb=0.1
        )
        
        manager.register_model_config(config)
        assert "test_model" in manager._model_configs
        assert manager._model_configs["test_model"] == config
    
    @pytest.mark.asyncio
    async def test_spacy_model_loading(self, manager):
        """Тест загрузки SpaCy модели"""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = set()
            mock_spacy_load.return_value = mock_nlp
            
            # Тестируем загрузку модели
            model = await manager.get_spacy_model("ru_core_news_sm")
            
            assert model is mock_nlp
            mock_spacy_load.assert_called_once_with("ru_core_news_sm")
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_loading(self, manager):
        """Тест загрузки SentenceTransformer модели"""
        with patch('same.models.model_manager.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            # Тестируем загрузку модели
            model = await manager.get_sentence_transformer()
            
            assert model is mock_model
            mock_st.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_caching(self, manager):
        """Тест кэширования моделей"""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = set()
            mock_spacy_load.return_value = mock_nlp
            
            # Первый вызов
            model1 = await manager.get_spacy_model("ru_core_news_sm")
            
            # Второй вызов должен вернуть тот же экземпляр
            model2 = await manager.get_spacy_model("ru_core_news_sm")
            
            assert model1 is model2
            # spacy.load должен быть вызван только один раз
            mock_spacy_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_loading(self, manager):
        """Тест конкурентной загрузки одной модели"""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = set()
            mock_spacy_load.return_value = mock_nlp
            
            # Запускаем несколько конкурентных загрузок
            tasks = [
                manager.get_spacy_model("ru_core_news_sm")
                for _ in range(5)
            ]
            
            models = await asyncio.gather(*tasks)
            
            # Все модели должны быть одинаковыми
            for model in models:
                assert model is mock_nlp
            
            # spacy.load должен быть вызван только один раз
            mock_spacy_load.assert_called_once()
    
    def test_memory_monitoring(self, manager):
        """Тест мониторинга памяти"""
        assert isinstance(manager.memory_monitor, MemoryMonitor)
        
        # Проверяем, что мониторинг запущен
        assert manager.memory_monitor._monitoring_thread is not None
        assert manager.memory_monitor._monitoring_thread.is_alive()
    
    def test_model_stats(self, manager):
        """Тест получения статистики моделей"""
        stats = manager.get_model_stats()
        
        assert isinstance(stats, dict)
        assert "loaded_models" in stats
        assert "total_memory_gb" in stats
        assert "models" in stats
        assert "memory_stats" in stats
    
    @pytest.mark.asyncio
    async def test_model_unloading(self, manager):
        """Тест выгрузки модели"""
        with patch('spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = set()
            mock_spacy_load.return_value = mock_nlp
            
            # Загружаем модель
            await manager.get_spacy_model("ru_core_news_sm")
            
            # Проверяем, что модель загружена
            assert "ru_core_news_sm" in manager._models
            
            # Выгружаем модель
            result = manager.unload_model("ru_core_news_sm")
            
            assert result is True
            assert "ru_core_news_sm" not in manager._models
    
    @pytest.mark.asyncio
    async def test_model_load_error(self, manager):
        """Тест обработки ошибок загрузки"""
        with patch('spacy.load', side_effect=OSError("Model not found")):
            with patch('spacy.load', side_effect=OSError("Fallback failed")):
                with pytest.raises(ModelLoadError):
                    await manager.get_spacy_model("nonexistent_model")
    
    def test_cleanup_idle_models(self, manager):
        """Тест очистки неиспользуемых моделей"""
        # Создаем mock модель с истекшим временем
        mock_instance = Mock()
        mock_instance.is_idle.return_value = True
        mock_instance.config.cache_size_gb = 0.5
        
        manager._models["idle_model"] = mock_instance
        
        # Запускаем очистку
        manager._cleanup_idle_models()
        
        # Проверяем, что модель была удалена
        assert "idle_model" not in manager._models
    
    def test_shutdown(self, manager):
        """Тест корректного завершения работы"""
        # Добавляем mock модель
        mock_instance = Mock()
        manager._models["test_model"] = mock_instance
        
        # Завершаем работу
        manager.shutdown()
        
        # Проверяем, что все модели выгружены
        assert len(manager._models) == 0
        
        # Проверяем, что мониторинг остановлен
        assert manager.memory_monitor._stop_monitoring.is_set()


class TestMemoryMonitor:
    """Тесты для MemoryMonitor"""
    
    def test_memory_stats(self):
        """Тест получения статистики памяти"""
        monitor = MemoryMonitor(memory_limit_gb=8.0)
        stats = monitor.get_memory_stats()
        
        assert stats.total_memory > 0
        assert stats.used_memory > 0
        assert stats.available_memory > 0
        assert 0 <= stats.memory_percent <= 100
    
    def test_memory_registration(self):
        """Тест регистрации использования памяти"""
        monitor = MemoryMonitor(memory_limit_gb=8.0)
        
        monitor.register_model_memory("test_model", 1.5)
        assert monitor._model_memory_estimates["test_model"] == 1.5
        
        monitor.unregister_model_memory("test_model")
        assert "test_model" not in monitor._model_memory_estimates
    
    def test_cleanup_callbacks(self):
        """Тест callback'ов очистки"""
        monitor = MemoryMonitor(memory_limit_gb=8.0)
        
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        monitor.register_cleanup_callback("test", cleanup_callback)
        monitor.force_cleanup()
        
        assert cleanup_called


if __name__ == "__main__":
    pytest.main([__file__])
