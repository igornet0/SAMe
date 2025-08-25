"""
Тесты для типов данных same_core
"""

import pytest
from dataclasses import FrozenInstanceError

from same_core.types import (
    ProcessingStage, ParameterType,
    ProcessingResult, SearchResult, ParameterData,
    SearchMethod, SearchConfig, ExportConfig
)


class TestEnums:
    """Тесты для перечислений"""
    
    def test_processing_stage_enum(self):
        """Тест ProcessingStage enum"""
        assert ProcessingStage.RAW.value == "raw"
        assert ProcessingStage.CLEANED.value == "cleaned"
        assert ProcessingStage.NORMALIZED.value == "normalized"
        assert ProcessingStage.LEMMATIZED.value == "lemmatized"
        assert ProcessingStage.ENHANCED.value == "enhanced"
        
        # Проверяем что все значения уникальны
        values = [stage.value for stage in ProcessingStage]
        assert len(values) == len(set(values))
    
    def test_parameter_type_enum(self):
        """Тест ParameterType enum"""
        assert ParameterType.NUMERIC.value == "numeric"
        assert ParameterType.UNIT.value == "unit"
        assert ParameterType.MATERIAL.value == "material"
        assert ParameterType.STANDARD.value == "standard"
        assert ParameterType.DIMENSION.value == "dimension"
        assert ParameterType.TECHNICAL_CODE.value == "technical_code"
        assert ParameterType.OTHER.value == "other"
    
    def test_search_method_enum(self):
        """Тест SearchMethod enum"""
        assert SearchMethod.FUZZY.value == "fuzzy"
        assert SearchMethod.SEMANTIC.value == "semantic"
        assert SearchMethod.HYBRID.value == "hybrid"


class TestProcessingResult:
    """Тесты для ProcessingResult"""
    
    def test_processing_result_creation(self):
        """Тест создания ProcessingResult"""
        stages = {
            ProcessingStage.RAW: "original text",
            ProcessingStage.CLEANED: "cleaned text"
        }
        
        result = ProcessingResult(
            original="original text",
            processed="processed text",
            stages=stages,
            metadata={"key": "value"},
            processing_time=0.5
        )
        
        assert result.original == "original text"
        assert result.processed == "processed text"
        assert result.stages == stages
        assert result.metadata == {"key": "value"}
        assert result.processing_time == 0.5
    
    def test_get_stage_method(self):
        """Тест метода get_stage"""
        stages = {
            ProcessingStage.RAW: "original text",
            ProcessingStage.CLEANED: "cleaned text"
        }
        
        result = ProcessingResult(
            original="original text",
            processed="processed text",
            stages=stages,
            metadata={},
            processing_time=0.1
        )
        
        # Существующий этап
        assert result.get_stage(ProcessingStage.RAW) == "original text"
        assert result.get_stage(ProcessingStage.CLEANED) == "cleaned text"
        
        # Несуществующий этап - должен вернуть processed
        assert result.get_stage(ProcessingStage.LEMMATIZED) == "processed text"


class TestSearchResult:
    """Тесты для SearchResult"""
    
    def test_search_result_creation(self):
        """Тест создания SearchResult"""
        result = SearchResult(
            document_id="doc_1",
            content="test content",
            score=0.85,
            metadata={"source": "test"},
            rank=1
        )
        
        assert result.document_id == "doc_1"
        assert result.content == "test content"
        assert result.score == 0.85
        assert result.metadata == {"source": "test"}
        assert result.rank == 1
    
    def test_search_result_validation(self):
        """Тест валидации SearchResult"""
        # Валидный score
        result = SearchResult(
            document_id="doc_1",
            content="test",
            score=0.5,
            metadata={},
            rank=1
        )
        assert result.score == 0.5
        
        # Невалидный score > 1
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                document_id="doc_1",
                content="test",
                score=1.5,
                metadata={},
                rank=1
            )
        
        # Невалидный score < 0
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                document_id="doc_1",
                content="test",
                score=-0.1,
                metadata={},
                rank=1
            )
        
        # Невалидный rank < 1
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            SearchResult(
                document_id="doc_1",
                content="test",
                score=0.5,
                metadata={},
                rank=0
            )


class TestParameterData:
    """Тесты для ParameterData"""
    
    def test_parameter_data_creation(self):
        """Тест создания ParameterData"""
        param = ParameterData(
            name="diameter",
            value="10",
            parameter_type=ParameterType.NUMERIC,
            confidence=0.9,
            position=(5, 7),
            unit="mm",
            normalized_value=10.0
        )
        
        assert param.name == "diameter"
        assert param.value == "10"
        assert param.parameter_type == ParameterType.NUMERIC
        assert param.confidence == 0.9
        assert param.position == (5, 7)
        assert param.unit == "mm"
        assert param.normalized_value == 10.0
    
    def test_parameter_data_validation(self):
        """Тест валидации ParameterData"""
        # Валидная confidence
        param = ParameterData(
            name="test",
            value="value",
            parameter_type=ParameterType.OTHER,
            confidence=0.5
        )
        assert param.confidence == 0.5
        
        # Невалидная confidence > 1
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ParameterData(
                name="test",
                value="value",
                parameter_type=ParameterType.OTHER,
                confidence=1.5
            )
        
        # Невалидная confidence < 0
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ParameterData(
                name="test",
                value="value",
                parameter_type=ParameterType.OTHER,
                confidence=-0.1
            )


class TestSearchConfig:
    """Тесты для SearchConfig"""
    
    def test_search_config_defaults(self):
        """Тест значений по умолчанию SearchConfig"""
        config = SearchConfig()
        
        assert config.method == SearchMethod.HYBRID
        assert config.similarity_threshold == 0.6
        assert config.max_results == 10
        assert config.enable_caching is True
        assert config.filters is None
    
    def test_search_config_custom(self):
        """Тест кастомных значений SearchConfig"""
        config = SearchConfig(
            method=SearchMethod.FUZZY,
            similarity_threshold=0.8,
            max_results=5,
            enable_caching=False,
            filters={"category": "bolts"}
        )
        
        assert config.method == SearchMethod.FUZZY
        assert config.similarity_threshold == 0.8
        assert config.max_results == 5
        assert config.enable_caching is False
        assert config.filters == {"category": "bolts"}
    
    def test_search_config_validation(self):
        """Тест валидации SearchConfig"""
        # Невалидный similarity_threshold > 1
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            SearchConfig(similarity_threshold=1.5)
        
        # Невалидный similarity_threshold < 0
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            SearchConfig(similarity_threshold=-0.1)
        
        # Невалидный max_results < 1
        with pytest.raises(ValueError, match="Max results must be >= 1"):
            SearchConfig(max_results=0)


class TestExportConfig:
    """Тесты для ExportConfig"""
    
    def test_export_config_defaults(self):
        """Тест значений по умолчанию ExportConfig"""
        config = ExportConfig()
        
        assert config.format == "excel"
        assert config.include_metadata is True
        assert config.include_scores is True
        assert config.max_rows is None
        assert config.custom_columns is None
    
    def test_export_config_custom(self):
        """Тест кастомных значений ExportConfig"""
        config = ExportConfig(
            format="csv",
            include_metadata=False,
            include_scores=False,
            max_rows=1000,
            custom_columns=["name", "score"]
        )
        
        assert config.format == "csv"
        assert config.include_metadata is False
        assert config.include_scores is False
        assert config.max_rows == 1000
        assert config.custom_columns == ["name", "score"]


class TestTypeAliases:
    """Тесты для алиасов типов"""
    
    def test_type_aliases_exist(self):
        """Тест что алиасы типов существуют"""
        from same_core.types import (
            TextProcessingResult,
            SearchResultItem,
            ExtractedParameter,
            CatalogData,
            ProcessingInput,
            SearchQuery
        )
        
        # Проверяем что алиасы определены
        assert TextProcessingResult is ProcessingResult
        assert SearchResultItem is SearchResult
        assert ExtractedParameter is ParameterData


if __name__ == "__main__":
    pytest.main([__file__])
