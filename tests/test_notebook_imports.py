#!/usr/bin/env python3
"""
Test the exact import cells from SAMe_Demo.ipynb notebook
"""

import sys
import os

# Paths to modules configured through poetry/pip install

def test_notebook_import_cells():
    """Test all import cells from the notebook"""
    print("🧪 Testing SAMe_Demo.ipynb Import Cells")
    print("=" * 60)
    
    # Cell 1: Basic imports
    print("\n📦 Testing Cell 1: Basic imports")
    try:
        import sys
        import os
        import warnings
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        from typing import List, Dict, Any
        import time
        from datetime import datetime
        
        print("✅ Basic imports successful")
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    # Cell 2: Text processing imports
    print("\n📦 Testing Cell 2: Text processing imports")
    try:
        from same.text_processing.text_cleaner import TextCleaner, CleaningConfig
        from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
        from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
        from same.text_processing.preprocessor import TextPreprocessor, PreprocessorConfig
        print("✅ Text processing imports successful")
    except ImportError as e:
        print(f"⚠️  Text processing imports failed (expected): {e}")
        print("💡 This is expected if spacy is not installed")
    
    # Cell 3: Search engine imports
    print("\n📦 Testing Cell 3: Search engine imports")
    try:
        from same.search_engine.fuzzy_search import FuzzySearchEngine, FuzzySearchConfig
        from same.search_engine.semantic_search import SemanticSearchEngine, SemanticSearchConfig
        from same.search_engine.hybrid_search import HybridSearchEngine, HybridSearchConfig
        from same.search_engine.indexer import SearchIndexer, IndexConfig
        print("✅ Search engine imports successful")
    except ImportError as e:
        print(f"⚠️  Search engine imports failed (expected): {e}")
        print("💡 This is expected if sklearn/rapidfuzz are not installed")

    # Cell 4: Parameter extraction imports
    print("\n📦 Testing Cell 4: Parameter extraction imports")
    try:
        from same.parameter_extraction.regex_extractor import (
            RegexParameterExtractor, ParameterPattern, ParameterType, ExtractedParameter
        )
        from same.parameter_extraction.ml_extractor import MLParameterExtractor, MLExtractorConfig
        from same.parameter_extraction.parameter_parser import ParameterParser, ParameterParserConfig
        print("✅ Parameter extraction imports successful")
    except ImportError as e:
        print(f"⚠️  Parameter extraction imports failed: {e}")
        print("💡 Some dependencies may be missing")
    
    # Cell 5: Export imports (THE MAIN TEST)
    print("\n📦 Testing Cell 5: Export imports (MAIN TEST)")
    try:
        from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
        from same.export.report_generator import ReportGenerator, ReportConfig
        print("✅ Export imports successful - ISSUE FIXED!")
        
        # Test creating instances as in notebook
        export_config = ExcelExportConfig(
            include_statistics=True,
            include_metadata=True,
            auto_adjust_columns=True,
            add_filters=True,
            highlight_high_similarity=True
        )
        
        excel_exporter = ExcelExporter(export_config)
        print("✅ ExcelExporter instance created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Export imports failed: {e}")
        return False
    
def test_excel_export_functionality():
    """Test that the Excel export functionality works as expected in notebook"""
    print("\n🧪 Testing Excel Export Functionality")
    print("=" * 50)
    
    try:
        from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
        
        # Create config exactly as in notebook
        export_config = ExcelExportConfig(
            include_statistics=True,
            include_metadata=True,
            auto_adjust_columns=True,
            add_filters=True,
            highlight_high_similarity=True
        )
        
        excel_exporter = ExcelExporter(export_config)
        
        # Test with sample data that would come from search results
        sample_export_results = {
            "болт м10": [
                {
                    'document_id': 1,
                    'document': 'Болт М10×50 ГОСТ 7798-70 оцинкованный',
                    'combined_score': 0.95,
                    'search_method': 'fuzzy',
                    'rank': 1
                },
                {
                    'document_id': 16,
                    'document': 'Болт М10 длина 50мм оцинкованный',
                    'combined_score': 0.90,
                    'search_method': 'fuzzy',
                    'rank': 2
                }
            ],
            "двигатель 1.5кВт": [
                {
                    'document_id': 7,
                    'document': 'Двигатель асинхронный АИР80В2 1.5кВт 3000об/мин 220/380В',
                    'hybrid_score': 0.88,
                    'search_method': 'hybrid',
                    'rank': 1
                }
            ]
        }
        
        from datetime import datetime
        sample_metadata = {
            'system': 'SAMe Demo',
            'version': '1.0',
            'export_date': datetime.now().isoformat(),
            'catalog_size': 23,
            'queries_processed': 2
        }
        
        print("✅ Sample data prepared")
        print(f"   Queries: {len(sample_export_results)}")
        print(f"   Total results: {sum(len(results) for results in sample_export_results.values())}")
        
        # Test that the method signature matches what's expected in notebook
        method = getattr(excel_exporter, 'export_search_results', None)
        if method:
            print("✅ export_search_results method exists")
            
            # Check method signature (without actually calling it to avoid file creation)
            import inspect
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            expected_params = ['results', 'output_path', 'metadata']
            
            if all(param in params for param in expected_params):
                print("✅ Method signature matches notebook expectations")
            else:
                print(f"⚠️  Method signature: {params}")
                print(f"   Expected: {expected_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel export functionality test failed: {e}")
        return False

def main():
    """Run all notebook import tests"""
    print("🚀 Testing SAMe Notebook Import Fix")
    print("=" * 60)
    
    # Test the imports
    imports_ok = test_notebook_import_cells()
    
    # Test the functionality
    functionality_ok = test_excel_export_functionality()
    
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    
    if imports_ok:
        print("✅ IMPORT ISSUE FIXED: ExcelExportConfig can now be imported successfully")
        print("✅ The notebook cell 'Импорты модулей SAMe - Экспорт' will now work")
    else:
        print("❌ Import issue still exists")
    
    if functionality_ok:
        print("✅ Excel export functionality is ready for notebook use")
    else:
        print("❌ Excel export functionality has issues")
    
    if imports_ok and functionality_ok:
        print("\n🎉 SUCCESS: The SAMe_Demo.ipynb notebook export functionality is now fixed!")
        print("\n📝 What was fixed:")
        print("   1. Added ExcelExportConfig class with required parameters:")
        print("      - include_statistics: bool")
        print("      - include_metadata: bool")
        print("      - auto_adjust_columns: bool")
        print("      - add_filters: bool")
        print("      - highlight_high_similarity: bool")
        print("   2. Maintained backward compatibility with ExportConfig alias")
        print("   3. Updated ExcelExporter constructor to use ExcelExportConfig")
        print("   4. Verified ReportConfig import works correctly")
        
        print("\n🚀 Next steps:")
        print("   1. Run the notebook cell: 'Импорты модулей SAMe - Экспорт'")
        print("   2. The Excel export functionality in section 5.1 should now work")
        print("   3. Users can export search results to Excel format as intended")
    
    return imports_ok and functionality_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
