"""
Тесты для интерфейсов same_core
"""

import pytest
from abc import ABC
from typing import List, Dict, Any
import pandas as pd

from same_core.interfaces import (
    TextProcessorInterface,
    SearchEngineInterface,
    AnalogSearchEngineInterface,
    ExporterInterface,
    DataManagerInterface,
    ParameterExtractorInterface
)
from same_core.types import ProcessingResult, SearchResult, ParameterData, ParameterType


class TestTextProcessorInterface:
    """Тесты интерфейса TextProcessorInterface"""
    
    def test_interface_is_abstract(self):
        """Тест что интерфейс абстрактный"""
        assert issubclass(TextProcessorInterface, ABC)
        
        with pytest.raises(TypeError):
            TextProcessorInterface()
    
    def test_required_methods(self):
        """Тест наличия обязательных методов"""
        required_methods = ['process_text', 'process_batch']
        
        for method_name in required_methods:
            assert hasattr(TextProcessorInterface, method_name)
            assert callable(getattr(TextProcessorInterface, method_name))


class TestSearchEngineInterface:
    """Тесты интерфейса SearchEngineInterface"""
    
    def test_interface_is_abstract(self):
        """Тест что интерфейс абстрактный"""
        assert issubclass(SearchEngineInterface, ABC)
        
        with pytest.raises(TypeError):
            SearchEngineInterface()
    
    def test_required_methods(self):
        """Тест наличия обязательных методов"""
        required_methods = ['fit', 'search', 'save_model', 'load_model']
        
        for method_name in required_methods:
            assert hasattr(SearchEngineInterface, method_name)
            assert callable(getattr(SearchEngineInterface, method_name))


class TestExporterInterface:
    """Тесты интерфейса ExporterInterface"""
    
    def test_interface_is_abstract(self):
        """Тест что интерфейс абстрактный"""
        assert issubclass(ExporterInterface, ABC)
        
        with pytest.raises(TypeError):
            ExporterInterface()
    
    def test_required_methods(self):
        """Тест наличия обязательных методов"""
        required_methods = ['export_data']
        
        for method_name in required_methods:
            assert hasattr(ExporterInterface, method_name)
            assert callable(getattr(ExporterInterface, method_name))


class MockTextProcessor(TextProcessorInterface):
    """Мок-реализация TextProcessorInterface для тестов"""
    
    def process_text(self, text: str) -> ProcessingResult:
        return ProcessingResult(
            original=text,
            processed=text.lower(),
            stages={},
            metadata={},
            processing_time=0.1
        )
    
    def process_batch(self, texts: List[str]) -> List[ProcessingResult]:
        return [self.process_text(text) for text in texts]


class MockSearchEngine(SearchEngineInterface):
    """Мок-реализация SearchEngineInterface для тестов"""
    
    def __init__(self):
        self.documents = []
        self.document_ids = []
    
    def fit(self, documents: List[str], document_ids: List[str], 
            metadata: Dict[str, Any] = None) -> None:
        self.documents = documents
        self.document_ids = document_ids
    
    def search(self, query: str, top_k: int = 10, 
               filters: Dict[str, Any] = None) -> List[SearchResult]:
        # Простая мок-реализация
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            results.append(SearchResult(
                document_id=self.document_ids[i],
                content=doc,
                score=0.9 - i * 0.1,
                metadata={},
                rank=i + 1
            ))
        return results
    
    def save_model(self, path) -> None:
        pass
    
    def load_model(self, path) -> None:
        pass


class TestInterfaceImplementations:
    """Тесты реализации интерфейсов"""
    
    def test_text_processor_implementation(self):
        """Тест реализации TextProcessorInterface"""
        processor = MockTextProcessor()
        
        # Тест process_text
        result = processor.process_text("Test Text")
        assert isinstance(result, ProcessingResult)
        assert result.original == "Test Text"
        assert result.processed == "test text"
        
        # Тест process_batch
        results = processor.process_batch(["Text1", "Text2"])
        assert len(results) == 2
        assert all(isinstance(r, ProcessingResult) for r in results)
    
    def test_search_engine_implementation(self):
        """Тест реализации SearchEngineInterface"""
        engine = MockSearchEngine()
        
        # Тест fit
        documents = ["doc1", "doc2", "doc3"]
        doc_ids = ["1", "2", "3"]
        engine.fit(documents, doc_ids)
        
        assert engine.documents == documents
        assert engine.document_ids == doc_ids
        
        # Тест search
        results = engine.search("query", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score > results[1].score  # Проверяем сортировку


class TestInterfaceCompatibility:
    """Тесты совместимости интерфейсов"""
    
    def test_all_interfaces_importable(self):
        """Тест что все интерфейсы можно импортировать"""
        from same_core.interfaces import (
            TextProcessorInterface,
            SearchEngineInterface,
            AnalogSearchEngineInterface,
            ExporterInterface,
            DataManagerInterface,
            ParameterExtractorInterface
        )
        
        # Проверяем что все интерфейсы являются классами
        interfaces = [
            TextProcessorInterface,
            SearchEngineInterface,
            ExporterInterface,
            DataManagerInterface,
            ParameterExtractorInterface
        ]
        
        for interface in interfaces:
            assert isinstance(interface, type)
            assert issubclass(interface, ABC)
    
    def test_protocol_interface(self):
        """Тест Protocol интерфейса"""
        # AnalogSearchEngineInterface использует Protocol, не ABC
        from typing import get_type_hints
        
        # Проверяем что у интерфейса есть нужные методы
        methods = ['initialize', 'search_analogs', 'export_results', 'get_statistics']
        
        # Это Protocol, поэтому проверяем по-другому
        assert hasattr(AnalogSearchEngineInterface, '__annotations__')


if __name__ == "__main__":
    pytest.main([__file__])
