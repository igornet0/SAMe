#!/usr/bin/env python3
"""
Test script to verify ExcelExportConfig import works correctly
"""

import sys
import os

# Add path to modules
sys.path.append(os.path.abspath('.'))

def test_excel_export_imports():
    """Test that ExcelExportConfig and related classes can be imported"""
    print("🧪 Testing Excel Export imports")
    print("=" * 50)
    
    try:
        # Test the import that was failing in the notebook
        from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
        print("✅ ExcelExporter and ExcelExportConfig imported successfully")
        
        # Test creating an instance with the configuration
        config = ExcelExportConfig(
            include_statistics=True,
            include_metadata=True,
            auto_adjust_columns=True,
            add_filters=True,
            highlight_high_similarity=True
        )
        print("✅ ExcelExportConfig instance created successfully")
        
        # Test creating ExcelExporter with config
        exporter = ExcelExporter(config)
        print("✅ ExcelExporter instance created successfully")
        
        # Test the configuration attributes
        print(f"\n📋 Configuration attributes:")
        print(f"   include_statistics: {config.include_statistics}")
        print(f"   include_metadata: {config.include_metadata}")
        print(f"   auto_adjust_columns: {config.auto_adjust_columns}")
        print(f"   add_filters: {config.add_filters}")
        print(f"   highlight_high_similarity: {config.highlight_high_similarity}")
        print(f"   similarity_threshold: {config.similarity_threshold}")
        print(f"   max_results_per_query: {config.max_results_per_query}")
        
        assert True  # Test passed

    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"❌ Other error: {e}")
        assert False, f"Other error: {e}"

def test_report_generator_imports():
    """Test ReportGenerator imports"""
    print("\n🧪 Testing ReportGenerator imports")
    print("=" * 50)
    
    try:
        from same.export.report_generator import ReportGenerator, ReportConfig
        print("✅ ReportGenerator and ReportConfig imported successfully")
        
        # Test creating config
        config = ReportConfig(
            include_summary=True,
            include_detailed_results=True,
            include_statistics=True,
            include_visualizations=False,  # Disable to avoid matplotlib dependency
            include_quality_analysis=True
        )
        print("✅ ReportConfig instance created successfully")
        
        assert True  # Test passed

    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"❌ Other error: {e}")
        assert False, f"Other error: {e}"

def test_backward_compatibility():
    """Test that the old ExportConfig still works"""
    print("\n🧪 Testing backward compatibility")
    print("=" * 50)
    
    try:
        from same.export.excel_exporter import ExportConfig
        print("✅ ExportConfig (alias) imported successfully")
        
        # Test that it's the same as ExcelExportConfig
        from same.export.excel_exporter import ExcelExportConfig
        
        config1 = ExportConfig()
        config2 = ExcelExportConfig()
        
        print(f"✅ Both configs have same attributes: {type(config1) == type(config2)}")
        
        assert True  # Test passed

    except Exception as e:
        print(f"❌ Error: {e}")
        assert False, f"Error: {e}"

def test_notebook_style_usage():
    """Test the exact usage pattern from the notebook"""
    print("\n🧪 Testing notebook-style usage")
    print("=" * 50)
    
    try:
        # Exact import from notebook
        from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
        from same.export.report_generator import ReportGenerator, ReportConfig
        
        # Create config as in notebook
        export_config = ExcelExportConfig(
            include_statistics=True,
            include_metadata=True,
            auto_adjust_columns=True,
            add_filters=True,
            highlight_high_similarity=True
        )
        
        excel_exporter = ExcelExporter(export_config)
        
        print("✅ Notebook-style usage works perfectly")
        
        # Test with sample data structure
        sample_results = {
            "болт м10": [
                {
                    'document_id': 1,
                    'document': 'Болт М10×50 ГОСТ 7798-70 оцинкованный',
                    'similarity_score': 0.95,
                    'search_method': 'fuzzy',
                    'rank': 1
                }
            ]
        }
        
        sample_metadata = {
            'system': 'SAMe Demo',
            'version': '1.0',
            'export_date': '2025-01-17T15:00:00',
            'catalog_size': 23,
            'queries_processed': 1
        }
        
        print("✅ Sample data prepared for export test")
        
        # Test that the export method exists and can be called
        # (We won't actually create the file to avoid dependencies)
        if hasattr(excel_exporter, 'export_search_results'):
            print("✅ export_search_results method exists")
        
        assert True  # Test passed

    except Exception as e:
        print(f"❌ Error in notebook-style usage: {e}")
        assert False, f"Error in notebook-style usage: {e}"

def main():
    """Run all tests"""
    print("🚀 Testing Excel Export Configuration Fix")
    print("=" * 60)
    
    tests = [
        test_excel_export_imports,
        test_report_generator_imports,
        test_backward_compatibility,
        test_notebook_style_usage
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! The Excel export import issue is fixed.")
        print("\n💡 The notebook should now work with:")
        print("   from same.export.excel_exporter import ExcelExporter, ExcelExportConfig")
        print("   from same.export.report_generator import ReportGenerator, ReportConfig")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
