"""
Интеграционные тесты для Advanced Model Manager
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from same.models import get_model_manager, AdvancedModelManager


class TestModelManagerIntegration:
    """Интеграционные тесты для менеджера моделей"""

    def teardown_method(self):
        """Очистка после каждого теста"""
        # Останавливаем мониторинг памяти если он запущен
        if hasattr(AdvancedModelManager, '_instance') and AdvancedModelManager._instance:
            if hasattr(AdvancedModelManager._instance, 'memory_monitor'):
                AdvancedModelManager._instance.memory_monitor.stop_monitoring()
        # Сбрасываем singleton
        AdvancedModelManager._instance = None

    def test_basic_initialization(self):
        """Тест базовой инициализации"""
        # Сбрасываем singleton для чистого теста
        AdvancedModelManager._instance = None

        manager = get_model_manager()
        assert isinstance(manager, AdvancedModelManager)
        assert manager._initialized
        assert manager.memory_monitor is not None
    
    def test_singleton_behavior(self):
        """Тест поведения singleton"""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2
    
    def test_memory_monitor_running(self):
        """Тест работы мониторинга памяти"""
        manager = get_model_manager()
        assert manager.memory_monitor._monitoring_thread is not None
        assert manager.memory_monitor._monitoring_thread.is_alive()
    
    def test_model_stats_structure(self):
        """Тест структуры статистики моделей"""
        manager = get_model_manager()
        stats = manager.get_model_stats()
        
        required_keys = ["loaded_models", "total_memory_gb", "models", "memory_stats"]
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats["loaded_models"], int)
        assert isinstance(stats["total_memory_gb"], (int, float))
        assert isinstance(stats["models"], dict)
        assert isinstance(stats["memory_stats"], dict)
    
    @pytest.mark.asyncio
    async def test_spacy_model_loading_mock(self):
        """Тест загрузки SpaCy модели с mock"""
        manager = get_model_manager()

        # Очищаем кэш модели перед тестом
        manager.unload_model("ru_core_news_lg")

        with patch('same.models.model_manager.spacy.load') as mock_spacy_load:
            mock_nlp = Mock()
            mock_nlp.Defaults.stop_words = {"и", "в", "на"}
            mock_spacy_load.return_value = mock_nlp

            # Загружаем модель
            model = await manager.get_spacy_model("ru_core_news_lg")

            assert model is mock_nlp
            mock_spacy_load.assert_called_once_with("ru_core_news_lg")

            # Проверяем, что модель кэширована
            assert "ru_core_news_lg" in manager._models
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_loading_mock(self):
        """Тест загрузки SentenceTransformer модели с mock"""
        manager = get_model_manager()
        
        with patch('same.models.model_manager.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            # Загружаем модель
            model = await manager.get_sentence_transformer()
            
            assert model is mock_model
            mock_st.assert_called_once()
            
            # Проверяем, что модель кэширована
            assert "default_sentence_transformer" in manager._models
    
    def test_model_unloading(self):
        """Тест выгрузки модели"""
        manager = get_model_manager()
        
        # Добавляем mock модель
        mock_instance = Mock()
        manager._models["test_model"] = mock_instance
        
        # Выгружаем модель
        result = manager.unload_model("test_model")
        
        assert result is True
        assert "test_model" not in manager._models
        
        # Попытка выгрузить несуществующую модель
        result = manager.unload_model("nonexistent_model")
        assert result is False
    
    def test_cleanup_functionality(self):
        """Тест функциональности очистки"""
        manager = get_model_manager()
        
        # Создаем mock модель, которая считается неиспользуемой
        mock_instance = Mock()
        mock_instance.is_idle.return_value = True
        mock_instance.config.cache_size_gb = 0.5
        
        manager._models["idle_model"] = mock_instance
        
        # Запускаем очистку
        manager._cleanup_idle_models()
        
        # Проверяем, что модель была удалена
        assert "idle_model" not in manager._models
    
    def test_memory_stats_collection(self):
        """Тест сбора статистики памяти"""
        manager = get_model_manager()
        memory_stats = manager.memory_monitor.get_memory_stats()
        
        assert memory_stats.total_memory > 0
        assert memory_stats.used_memory > 0
        assert memory_stats.available_memory >= 0
        assert 0 <= memory_stats.memory_percent <= 100
        assert memory_stats.model_memory >= 0
        assert memory_stats.timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
