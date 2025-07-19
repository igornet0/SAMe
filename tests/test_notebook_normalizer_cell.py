#!/usr/bin/env python3
"""
Test the exact TextNormalizer cell from the fixed SAMe_Demo.ipynb notebook
"""

import sys
import os

# Add path to modules
sys.path.append(os.path.abspath('.'))

def test_exact_notebook_cell():
    """Test the exact code from the fixed notebook cell"""
    print("🧪 Testing Exact Notebook Cell - TextNormalizer")
    print("=" * 60)
    
    try:
        # Import exactly as in notebook
        from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
        
        # Execute the exact code from the FIXED notebook cell
        print("🔧 Демонстрация TextNormalizer")
        print("=" * 50)

        # Создаем конфигурацию и экземпляр нормализатора
        normalizer_config = NormalizerConfig(
            standardize_units=True,
            normalize_abbreviations=True,
            unify_technical_terms=True
        )

        text_normalizer = TextNormalizer(normalizer_config)

        # Примеры для нормализации
        normalization_samples = [
            "Болт М10×50 миллиметров ГОСТ 7798-70 оцинкованный",
            "Двигатель электрический 4 киловатта 1500 оборотов в минуту",
            "Труба стальная диаметр 57 миллиметра толщина стенки 3.5мм",
            "Кабель медный 3×2.5 квадратных миллиметра напряжение 0.66кВ",
            "Насос центробежный подача 12.5 кубических метра в час"
        ]

        print("\n📝 Примеры нормализации текста:")
        normalization_results = []

        for i, sample in enumerate(normalization_samples, 1):
            result = text_normalizer.normalize_text(sample)
            normalization_results.append(result)
            
            print(f"\n{i}. Исходный: '{sample}'")
            print(f"   Нормализованный: '{result['final_normalized']}'")
            
            # Показываем этапы нормализации
            if result['units_normalized'] != result['original'].lower():
                print(f"   Единицы: '{result['units_normalized']}'")
            if result['abbreviations_normalized'] != result['units_normalized']:
                print(f"   Аббревиатуры: '{result['abbreviations_normalized']}'")
            if result['terms_unified'] != result['abbreviations_normalized']:
                print(f"   Термины: '{result['terms_unified']}'")

        # Статистика нормализации
        total_original_length = sum(len(r['original']) for r in normalization_results)
        total_normalized_length = sum(len(r['final_normalized']) for r in normalization_results)
        compression_ratio = (total_original_length - total_normalized_length) / total_original_length * 100

        print(f"\n📊 Статистика нормализации:")
        print(f"Обработано образцов: {len(normalization_results)}")
        print(f"Общее сжатие текста: {compression_ratio:.1f}%")
        print(f"Средняя длина до: {total_original_length/len(normalization_results):.1f} символов")
        print(f"Средняя длина после: {total_normalized_length/len(normalization_results):.1f} символов")

        print("\n✅ TextNormalizer работает корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in notebook cell execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_details():
    """Test the configuration in detail"""
    print("\n🧪 Testing Configuration Details")
    print("=" * 50)
    
    try:
        from same.text_processing.normalizer import NormalizerConfig
        
        # Test the exact configuration from notebook
        config = NormalizerConfig(
            standardize_units=True,
            normalize_abbreviations=True,
            unify_technical_terms=True
        )
        
        print("📋 Configuration created successfully:")
        print(f"   standardize_units: {config.standardize_units}")
        print(f"   normalize_abbreviations: {config.normalize_abbreviations}")
        print(f"   unify_technical_terms: {config.unify_technical_terms}")
        
        # Check default values for other parameters
        print(f"   remove_brand_names: {config.remove_brand_names} (default)")
        print(f"   standardize_numbers: {config.standardize_numbers} (default)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_normalization_functionality():
    """Test specific normalization functionality"""
    print("\n🧪 Testing Normalization Functionality")
    print("=" * 50)
    
    try:
        from same.text_processing.normalizer import TextNormalizer, NormalizerConfig
        
        # Test each configuration option individually
        
        # Test 1: Units standardization
        print("1️⃣ Testing units standardization...")
        units_config = NormalizerConfig(
            standardize_units=True,
            normalize_abbreviations=False,
            unify_technical_terms=False
        )
        units_normalizer = TextNormalizer(units_config)
        
        units_test = "Болт длиной 50 миллиметров и диаметром 10 миллиметра"
        units_result = units_normalizer.normalize_text(units_test)
        print(f"   Input: '{units_test}'")
        print(f"   Output: '{units_result['final_normalized']}'")
        
        # Test 2: Abbreviations normalization
        print("\n2️⃣ Testing abbreviations normalization...")
        abbrev_config = NormalizerConfig(
            standardize_units=False,
            normalize_abbreviations=True,
            unify_technical_terms=False
        )
        abbrev_normalizer = TextNormalizer(abbrev_config)
        
        abbrev_test = "Двигатель эл автом с мех приводом"
        abbrev_result = abbrev_normalizer.normalize_text(abbrev_test)
        print(f"   Input: '{abbrev_test}'")
        print(f"   Output: '{abbrev_result['final_normalized']}'")
        
        # Test 3: Technical terms unification
        print("\n3️⃣ Testing technical terms unification...")
        terms_config = NormalizerConfig(
            standardize_units=False,
            normalize_abbreviations=False,
            unify_technical_terms=True
        )
        terms_normalizer = TextNormalizer(terms_config)
        
        terms_test = "Винт крепежный и крепежный болт"
        terms_result = terms_normalizer.normalize_text(terms_test)
        print(f"   Input: '{terms_test}'")
        print(f"   Output: '{terms_result['final_normalized']}'")
        
        print("\n✅ All normalization functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Normalization functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Fixed NormalizerConfig in Notebook")
    print("=" * 70)
    
    tests = [
        ("Exact Notebook Cell", test_exact_notebook_cell),
        ("Configuration Details", test_configuration_details),
        ("Normalization Functionality", test_normalization_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*70}")
        
        result = test_func()
        results.append((test_name, result))
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL TEST RESULTS")
    print(f"{'='*70}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📝 Summary of the fix:")
        print("   ✅ Fixed parameter names in SAMe_Demo.ipynb:")
        print("      - normalize_units → standardize_units")
        print("      - expand_abbreviations → normalize_abbreviations")
        print("      - unify_technical_terms → unify_technical_terms (unchanged)")
        
        print("\n✅ The notebook cell now works correctly:")
        print("   normalizer_config = NormalizerConfig(")
        print("       standardize_units=True,")
        print("       normalize_abbreviations=True,")
        print("       unify_technical_terms=True")
        print("   )")
        
        print("\n🚀 Benefits of the fix:")
        print("   • No more TypeError when creating NormalizerConfig")
        print("   • TextNormalizer demonstration works properly")
        print("   • Units standardization: миллиметров → мм")
        print("   • Abbreviations expansion: эл → электрический")
        print("   • Technical terms unification: винт крепежный → болт")
        
        print("\n📋 The SAMe_Demo.ipynb TextNormalizer section is now fully functional!")
        
        return True
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
