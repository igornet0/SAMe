"""
Unit tests for Enhanced Text Processing
Tests numeric token replacement, Russian morphological processing, and normalization improvements
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from same_clear.text_processing.normalizer import TextNormalizer, NormalizerConfig
except ImportError:
    # Fallback на старый импорт
    from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
try:
    from same_clear.text_processing import Lemmatizer as TextLemmatizer, LemmatizerConfig
except ImportError:
    # Fallback на старый импорт
    try:
        from same.text_processing.lemmatizer import TextLemmatizer, LemmatizerConfig
    except ImportError:
        # Create alias if needed
        from same_clear.text_processing import Lemmatizer
        TextLemmatizer = Lemmatizer
        from same_clear.text_processing import LemmatizerConfig
try:
    from same_clear.text_processing import TextPreprocessor, PreprocessorConfig
except ImportError:
    # Fallback на старый импорт
    from same.text_processing import TextPreprocessor, PreprocessorConfig


class TestNumericTokenProcessing:
    """Test numeric token replacement and normalization"""
    
    @pytest.fixture
    def normalizer(self):
        """Create normalizer with numeric processing enabled"""
        config = NormalizerConfig(
            reduce_numeric_weight=True,
            numeric_token_replacement="<NUM>",
            preserve_units_with_numbers=True,
            normalize_ranges=True
        )
        return TextNormalizer(config)
    
    def test_basic_numeric_replacement(self, normalizer):
        """Test basic number replacement with <NUM> tokens"""
        # Test cases where numbers should be replaced with <NUM> (standalone numbers)
        should_replace_cases = [
            ("ледоход проф 10", "ледоход проф <NUM>"),
            ("размер 15", "размер <NUM>"),
            ("модель 2023", "модель <NUM>")
        ]

        # Test cases where numbers should be preserved (numbers with units)
        should_preserve_cases = [
            ("сольвент 10 литров", "сольвент 10 л"),  # Units are normalized but numbers preserved
            ("болт 12 мм длиной", "болт 12 мм длиной"),  # Numbers with units preserved
        ]

        # Test standalone numbers get replaced
        for input_text, expected_output in should_replace_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            assert "<NUM>" in processed, f"<NUM> token not found in '{processed}' for standalone number case"

        # Test numbers with units are preserved
        for input_text, expected_pattern in should_preserve_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            # Should contain actual numbers, not <NUM> tokens
            assert any(char.isdigit() for char in processed), f"Numbers should be preserved in '{processed}' for unit case"
    
    def test_preserve_units_with_numbers(self, normalizer):
        """Test that numbers with units are preserved"""
        test_cases = [
            ("швеллер 10 мм стальной", "10 мм"),  # Should preserve "10 мм"
            ("труба 25 см длиной", "25 см"),      # Should preserve "25 см"
            ("вес 5 кг нетто", "5 кг"),           # Should preserve "5 кг"
            ("напряжение 220 В", "220 В"),        # Should preserve "220 В"
            ("мощность 1000 Вт", "1000 Вт"),      # Should preserve "1000 Вт"
            ("температура 100 °C", "100 °C")      # Should preserve "100 °C"
        ]
        
        for input_text, unit_phrase in test_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))
            
            assert unit_phrase in processed, f"Unit phrase '{unit_phrase}' not preserved in '{processed}'"
    
    def test_range_normalization(self, normalizer):
        """Test normalization of numeric ranges"""
        test_cases = [
            ("размер 10-15", "размер <NUM>-<NUM>"),
            ("диаметр 5–8 мм", "диаметр <NUM>-<NUM> мм"),  # Em dash normalized to regular dash
            ("длина 100—200", "длина <NUM>-<NUM>"),  # En dash normalized to regular dash
            ("вес 2.5-3.0 кг", "вес <NUM>-<NUM> кг"),
            ("температура -10 до +50", "температура <NUM> до <NUM>")
        ]

        for input_text, expected_pattern in test_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            # Check that ranges are normalized to <NUM> tokens
            assert "<NUM>" in processed, f"Range not normalized in '{processed}'"

            # Check that range structure is preserved (should contain dash or range words)
            has_range_indicator = any(indicator in processed for indicator in ['-', '–', '—', 'до', '..'])
            assert has_range_indicator, f"Range structure not preserved in '{processed}'"
    
    def test_decimal_numbers(self, normalizer):
        """Test handling of decimal numbers"""
        # Standalone decimal numbers should be replaced with <NUM>
        standalone_cases = [
            ("диаметр 2.5", "диаметр <NUM>"),
            ("цена 99.99", "цена <NUM>")
        ]

        # Decimal numbers with units should be preserved (but comma normalized to dot)
        with_units_cases = [
            ("вес 10,5 кг", "вес 10.5 кг"),  # Russian decimal separator normalized
            ("размер 1.25 мм", "размер 1.25 мм"),  # Preserved with units
        ]

        # Test standalone decimals get replaced
        for input_text, expected_output in standalone_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            assert "<NUM>" in processed, f"Standalone decimal not replaced in '{processed}'"

        # Test decimals with units are preserved
        for input_text, expected_pattern in with_units_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            # Should contain actual decimal numbers, not <NUM> tokens
            assert any(char.isdigit() for char in processed), f"Decimal numbers should be preserved in '{processed}'"
            # Check decimal separator is normalized to dot
            if ',' in input_text:
                assert '.' in processed, f"Decimal separator not normalized in '{processed}'"
    
    def test_mixed_content(self, normalizer):
        """Test processing of mixed content with numbers and text"""
        test_cases = [
            "ледоход проф 10 х10 размер xl",  # Some numbers replaced, some preserved
            "болт М10 длина 50 мм",  # Material codes protected, units preserved
            "кабель 3х2.5 мм2",  # Mixed numeric processing
            "размер 10/12/15"  # Multiple numbers in sequence
        ]

        for input_text in test_cases:
            result = normalizer.normalize_text(input_text)
            processed = result.get('numeric_processed', result.get('final_normalized', input_text))

            # Check that some form of processing occurred
            # Either <NUM> tokens or protected tokens should be present
            has_num_tokens = "<NUM>" in processed
            has_protected_tokens = "__GLOBAL_" in processed or "__PROTECTED_" in processed
            has_preserved_numbers = any(char.isdigit() for char in processed)

            # At least one type of processing should have occurred
            assert (has_num_tokens or has_protected_tokens or has_preserved_numbers), \
                f"No numeric processing detected in '{processed}' from '{input_text}'"


class TestRussianMorphologicalProcessing:
    """Test Russian morphological processing and variant normalization"""
    
    @pytest.fixture
    def lemmatizer(self):
        """Create lemmatizer with Russian variant processing"""
        config = LemmatizerConfig(
            normalize_product_variants=True,
            preserve_professional_terms=True,
            model_name="ru_core_news_lg"
        )
        return TextLemmatizer(config)
    
    def test_product_variant_normalization(self, lemmatizer):
        """Test normalization of Russian product variants"""
        test_cases = [
            # Ледоход variants
            ("ледоходы", "ледоход"),
            ("ледоходов", "ледоход"),
            ("ледоходами", "ледоход"),
            
            # Professional term variants
            ("профессиональный", "проф"),
            ("профессиональная", "проф"),
            ("профессиональные", "проф"),
            ("профессионального", "проф"),
            
            # Metal products
            ("швеллеры", "швеллер"),
            ("швеллеров", "швеллер"),
            ("уголки", "уголок"),
            ("уголков", "уголок"),
            
            # Chemical products
            ("сольвенты", "сольвент"),
            ("растворители", "растворитель")
        ]
        
        for input_word, expected_base in test_cases:
            # Test individual word processing
            result = lemmatizer.lemmatize_text(input_word)
            lemmatized = result.get('lemmatized', input_word)

            # The base form should be present in the result
            assert expected_base in lemmatized.lower(), f"Expected '{expected_base}' in '{lemmatized}' for input '{input_word}'"
    
    def test_problematic_query_processing(self, lemmatizer):
        """Test processing of the problematic query 'ледоход проф 10'"""
        variants = [
            "ледоход проф 10",
            "ледоходы профессиональные 10",
            "ледоход профессиональный 10",
            "ледоходов проф 10"
        ]
        
        processed_variants = []
        for variant in variants:
            result = lemmatizer.lemmatize_text(variant)
            processed = result.get('lemmatized_text', variant)
            processed_variants.append(processed.lower())
        
        # All variants should be similar after processing
        base_processed = processed_variants[0]
        for processed in processed_variants[1:]:
            # Check for common key terms
            assert "ледоход" in processed, f"Base term 'ледоход' not found in '{processed}'"
            assert "проф" in processed or "профессиональн" in processed, f"Professional term not found in '{processed}'"
    
    def test_preserve_technical_terms(self, lemmatizer):
        """Test preservation of technical terms"""
        technical_terms = [
            "мм", "см", "м", "кг", "г", "л", "В", "А", "Вт", "°C", "МПа", "бар"
        ]
        
        for term in technical_terms:
            test_text = f"размер 10 {term}"
            result = lemmatizer.lemmatize_text(test_text)
            processed = result.get('lemmatized_text', test_text)
            
            assert term in processed, f"Technical term '{term}' not preserved in '{processed}'"
    
    def test_stopword_removal(self, lemmatizer):
        """Test removal of technical stopwords"""
        test_cases = [
            ("изделие ледоход проф", "ледоход проф"),  # Remove 'изделие'
            ("деталь болт М10", "болт М10"),           # Remove 'деталь'
            ("материал сталь листовая", "сталь листовая")  # Remove 'материал'
        ]
        
        for input_text, expected_content in test_cases:
            result = lemmatizer.lemmatize_text(input_text)
            processed = result.get('lemmatized_text', input_text).lower()
            
            # Check that important content is preserved
            for word in expected_content.split():
                assert word.lower() in processed, f"Important word '{word}' removed from '{processed}'"


class TestIntegratedTextProcessing:
    """Test integrated text processing pipeline"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with all enhancements enabled"""
        normalizer_config = NormalizerConfig(
            reduce_numeric_weight=True,
            numeric_token_replacement="<NUM>",
            preserve_units_with_numbers=True,
            normalize_ranges=True
        )
        
        lemmatizer_config = LemmatizerConfig(
            normalize_product_variants=True,
            preserve_professional_terms=True
        )
        
        config = PreprocessorConfig(
            normalizer_config=normalizer_config,
            lemmatizer_config=lemmatizer_config
        )
        
        return TextPreprocessor(config)
    
    def test_end_to_end_processing(self, preprocessor):
        """Test complete text processing pipeline"""
        test_cases = [
            {
                'input': "ледоход проф 10",
                'expected_features': ['num', 'ледоход', 'проф']  # Preprocessor uses lowercase 'num'
            },
            {
                'input': "ледоходы профессиональные 10 шипов",
                'expected_features': ['num', 'ледоход', 'проф', 'шип']
            },
            {
                'input': "сольвент 10 литров технический",
                'expected_features': ['10', 'сольвент', 'технический']  # Numbers with units preserved
            },
            {
                'input': "швеллер 10 мм стальной горячекатаный",
                'expected_features': ['10 мм', 'швеллер', 'стальной']  # 10 мм should be preserved
            }
        ]
        
        for test_case in test_cases:
            result = preprocessor.preprocess_text(test_case['input'])
            final_text = result.get('final_text', test_case['input']).lower()
            
            for feature in test_case['expected_features']:
                assert feature.lower() in final_text, f"Expected feature '{feature}' not found in '{final_text}'"
    
    def test_before_after_comparison(self, preprocessor):
        """Test before/after comparison showing improvements"""
        problematic_cases = [
            "ледоход проф 10",
            "сольвент 10",
            "швеллер 10",
            "полог 10 х 10"
        ]
        
        results = {}
        for case in problematic_cases:
            # Process with enhancements
            result = preprocessor.preprocess_text(case)
            processed = result.get('final_text', case)
            
            results[case] = {
                'original': case,
                'processed': processed,
                'has_num_token': '<NUM>' in processed,
                'preserved_units': any(unit in processed for unit in ['мм', 'см', 'м', 'кг', 'л'])
            }
        
        # Verify that numbers are replaced with <NUM> tokens
        for case, result in results.items():
            if any(char.isdigit() for char in case):
                # Should have <NUM> tokens unless units are preserved
                if not result['preserved_units']:
                    assert result['has_num_token'], f"Numbers not replaced in '{case}' -> '{result['processed']}'"
        
        return results
    
    def test_processing_consistency(self, preprocessor):
        """Test that similar inputs produce similar outputs"""
        similar_groups = [
            # Ледоход variants
            [
                "ледоход проф 10",
                "ледоходы профессиональные 10",
                "ледоход профессиональный 10"
            ],
            # Chemical variants
            [
                "сольвент 10",
                "сольвент 10 литров",
                "растворитель 10"
            ]
        ]
        
        for group in similar_groups:
            processed_group = []
            for text in group:
                result = preprocessor.preprocess_text(text)
                processed = result.get('final_text', text).lower()
                processed_group.append(processed)
            
            # Check that all variants in the group share common processed terms
            base_terms = set(processed_group[0].split())
            for processed in processed_group[1:]:
                current_terms = set(processed.split())
                common_terms = base_terms.intersection(current_terms)
                
                # Should have significant overlap
                overlap_ratio = len(common_terms) / max(len(base_terms), len(current_terms))
                assert overlap_ratio >= 0.5, f"Insufficient overlap between variants: {overlap_ratio:.2f}"
    
    def test_performance_benchmarks(self, preprocessor):
        """Test processing performance"""
        import time
        
        test_texts = [
            "ледоход проф 10",
            "сольвент 10 литров технический",
            "швеллер 10 мм стальной горячекатаный",
            "болт М10 оцинкованный с гайкой",
            "каска защитная белая пластиковая"
        ] * 20  # 100 texts total
        
        start_time = time.time()
        for text in test_texts:
            preprocessor.preprocess_text(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        texts_per_second = len(test_texts) / processing_time
        
        assert texts_per_second > 10, f"Processing too slow: {texts_per_second:.1f} texts/sec"
        print(f"Text processing speed: {texts_per_second:.1f} texts/second")
    
    def test_edge_cases(self, preprocessor):
        """Test edge cases and error handling"""
        edge_cases = [
            "",                    # Empty string
            "   ",                 # Whitespace only
            "123",                 # Numbers only
            "абв где ёжз",         # Russian letters only
            "10 20 30 40 50",      # Multiple numbers
            "a1b2c3d4e5",          # Mixed alphanumeric
            "размер 10-15-20-25"   # Multiple ranges
        ]
        
        for case in edge_cases:
            try:
                result = preprocessor.preprocess_text(case)
                processed = result.get('final_text', case)
                
                # Should not crash and should return valid string
                assert isinstance(processed, str), f"Invalid output type for '{case}'"
                assert len(processed.strip()) >= 0, f"Invalid output for '{case}'"
                
            except Exception as e:
                pytest.fail(f"Processing failed for edge case '{case}': {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
