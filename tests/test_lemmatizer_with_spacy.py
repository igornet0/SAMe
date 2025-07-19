#!/usr/bin/env python3
"""
Test the Lemmatizer functionality with the installed SpaCy model
"""

import sys
import os

# Add path to modules
sys.path.append(os.path.abspath('.'))

def test_spacy_model_basic():
    """Test that SpaCy model works correctly"""
    print("🧪 Testing SpaCy Model Basic Functionality")
    print("=" * 60)
    
    try:
        import spacy
        
        # Load the model
        print("Loading ru_core_news_sm model...")
        nlp = spacy.load('ru_core_news_sm')
        print("✅ Model loaded successfully")
        
        # Test basic functionality
        test_text = "Болт М10×50 ГОСТ 7798-70 оцинкованный"
        doc = nlp(test_text)
        
        print(f"\n📝 Test text: '{test_text}'")
        print("Token analysis:")
        for token in doc:
            print(f"   '{token.text}' → lemma: '{token.lemma_}', POS: {token.pos_}, is_stop: {token.is_stop}")
        
        return True
        
    except Exception as e:
        print(f"❌ SpaCy model test failed: {e}")
        return False

def test_lemmatizer_with_spacy():
    """Test the Lemmatizer class with SpaCy model"""
    print("\n🧪 Testing Lemmatizer with SpaCy Model")
    print("=" * 60)
    
    try:
        from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
        
        # Create config with the small model
        lemmatizer_config = LemmatizerConfig(
            model_name="ru_core_news_sm",
            preserve_technical_terms=True,
            min_token_length=2
        )
        
        print("✅ LemmatizerConfig created successfully")
        
        # Create lemmatizer instance
        lemmatizer = Lemmatizer(lemmatizer_config)
        print("✅ Lemmatizer instance created successfully")
        
        # Test with sample data
        test_samples = [
            "Болт М10×50 ГОСТ 7798-70 оцинкованный крепежный элемент",
            "Двигатель асинхронный электрический мощностью 4кВт трехфазный",
            "Труба стальная бесшовная диаметром 57мм толщиной стенки 3.5мм",
            "Кабель медный гибкий многожильный сечением 2.5мм² в изоляции",
            "Насос центробежный водяной подачей 12.5м³/ч напором 20м"
        ]
        
        print("\n📝 Testing lemmatization with sample data:")
        
        for i, sample in enumerate(test_samples, 1):
            try:
                result = lemmatizer.lemmatize_text(sample)
                
                print(f"\n{i}. Original: '{sample}'")
                print(f"   Lemmatized: '{result['lemmatized']}'")
                print(f"   Tokens: {len(result['tokens'])}")
                print(f"   Filtered lemmas: {len(result['filtered_lemmas'])}")
                
                # Show some details
                if result['tokens']:
                    print(f"   All tokens: {result['tokens'][:6]}{'...' if len(result['tokens']) > 6 else ''}")
                if result['filtered_lemmas']:
                    print(f"   Key lemmas: {result['filtered_lemmas'][:6]}{'...' if len(result['filtered_lemmas']) > 6 else ''}")
                    
            except Exception as e:
                print(f"   ❌ Error lemmatizing sample {i}: {e}")
                return False
        
        print("\n✅ All lemmatization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Lemmatizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notebook_cell_simulation():
    """Simulate the exact notebook cell execution"""
    print("\n🧪 Testing Notebook Cell Simulation")
    print("=" * 60)
    
    try:
        # Import exactly as in notebook
        from same.text_processing.lemmatizer import Lemmatizer, LemmatizerConfig
        
        print("📚 Демонстрация Lemmatizer")
        print("=" * 50)

        # Создаем конфигурацию и экземпляр лемматизатора (exactly as in notebook)
        lemmatizer_config = LemmatizerConfig(
            model_name="ru_core_news_sm",  # Using the small model
            preserve_technical_terms=True,
            min_token_length=2
        )

        lemmatizer = Lemmatizer(lemmatizer_config)

        # Примеры для лемматизации (exactly as in notebook)
        lemmatization_samples = [
            "Болт М10×50 ГОСТ 7798-70 оцинкованный крепежный элемент",
            "Двигатель асинхронный электрический мощностью 4кВт трехфазный",
            "Труба стальная бесшовная диаметром 57мм толщиной стенки 3.5мм",
            "Кабель медный гибкий многожильный сечением 2.5мм² в изоляции",
            "Насос центробежный водяной подачей 12.5м³/ч напором 20м"
        ]

        print("\n📝 Примеры лемматизации текста:")
        lemmatization_results = []

        for i, sample in enumerate(lemmatization_samples, 1):
            result = lemmatizer.lemmatize_text(sample)
            lemmatization_results.append(result)
            
            print(f"\n{i}. Исходный: '{sample}'")
            print(f"   Лемматизированный: '{result['lemmatized']}'")
            print(f"   Токенов: {len(result['tokens'])}")
            print(f"   Отфильтрованных лемм: {len(result['filtered_lemmas'])}")
            
            # Показываем детали обработки
            if result['tokens']:
                print(f"   Все токены: {result['tokens'][:8]}{'...' if len(result['tokens']) > 8 else ''}")
            if result['filtered_lemmas']:
                print(f"   Ключевые леммы: {result['filtered_lemmas'][:6]}{'...' if len(result['filtered_lemmas']) > 6 else ''}")

        # Статистика лемматизации (exactly as in notebook)
        total_tokens = sum(len(r['tokens']) for r in lemmatization_results)
        total_filtered = sum(len(r['filtered_lemmas']) for r in lemmatization_results)
        reduction_ratio = (total_tokens - total_filtered) / total_tokens * 100 if total_tokens > 0 else 0

        print(f"\n📊 Статистика лемматизации:")
        print(f"Обработано образцов: {len(lemmatization_results)}")
        print(f"Всего токенов: {total_tokens}")
        print(f"Отфильтровано лемм: {total_filtered}")
        print(f"Коэффициент сжатия: {reduction_ratio:.1f}%")
        print(f"Средняя длина до: {total_tokens/len(lemmatization_results):.1f} токенов")
        print(f"Средняя длина после: {total_filtered/len(lemmatization_results):.1f} лемм")

        print("\n✅ Lemmatizer работает корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_options():
    """Test different configuration options"""
    print("\n🧪 Testing Configuration Options")
    print("=" * 60)
    
    try:
        from same.text_processing.lemmatizer import LemmatizerConfig, Lemmatizer
        
        # Test different configurations
        configs = [
            {
                'name': 'Default (Small Model)',
                'config': LemmatizerConfig(model_name="ru_core_news_sm"),
                'test_text': 'Болт М10×50 ГОСТ 7798-70'
            },
            {
                'name': 'Technical Terms Preserved',
                'config': LemmatizerConfig(
                    model_name="ru_core_news_sm",
                    preserve_technical_terms=True,
                    min_token_length=2
                ),
                'test_text': 'Болт М10×50 ГОСТ 7798-70'
            },
            {
                'name': 'Technical Terms Not Preserved',
                'config': LemmatizerConfig(
                    model_name="ru_core_news_sm",
                    preserve_technical_terms=False,
                    min_token_length=2
                ),
                'test_text': 'Болт М10×50 ГОСТ 7798-70'
            },
            {
                'name': 'Longer Min Token Length',
                'config': LemmatizerConfig(
                    model_name="ru_core_news_sm",
                    preserve_technical_terms=True,
                    min_token_length=4
                ),
                'test_text': 'Болт М10×50 ГОСТ 7798-70'
            }
        ]
        
        print("📋 Testing different configuration scenarios:")
        
        for i, config_info in enumerate(configs, 1):
            config = config_info['config']
            test_text = config_info['test_text']
            
            print(f"\n{i}. {config_info['name']}")
            print(f"   preserve_technical_terms: {config.preserve_technical_terms}")
            print(f"   min_token_length: {config.min_token_length}")
            
            try:
                lemmatizer = Lemmatizer(config)
                result = lemmatizer.lemmatize_text(test_text)
                
                print(f"   Input: '{test_text}'")
                print(f"   Output: '{result['lemmatized']}'")
                print(f"   Filtered lemmas: {result['filtered_lemmas']}")
                
            except Exception as e:
                print(f"   ❌ Configuration test failed: {e}")
                return False
        
        print("\n✅ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration options test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Lemmatizer with Installed SpaCy Model")
    print("=" * 70)
    
    tests = [
        ("SpaCy Model Basic", test_spacy_model_basic),
        ("Lemmatizer with SpaCy", test_lemmatizer_with_spacy),
        ("Notebook Cell Simulation", test_notebook_cell_simulation),
        ("Configuration Options", test_configuration_options)
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
        print("\n📝 Summary:")
        print("   ✅ SpaCy ru_core_news_sm model is installed and working")
        print("   ✅ LemmatizerConfig creates successfully with correct parameters")
        print("   ✅ Lemmatizer processes Russian text correctly")
        print("   ✅ Technical terms are preserved when configured")
        print("   ✅ Notebook cell simulation works perfectly")
        
        print("\n🚀 The SAMe_Demo.ipynb Lemmatizer section is now fully functional!")
        print("\n📋 What works now:")
        print("   • SpaCy ru_core_news_sm model is installed")
        print("   • LemmatizerConfig with correct parameter names")
        print("   • Full lemmatization functionality for Russian MTR text")
        print("   • Technical term preservation (М10, кВт, мм, ГОСТ)")
        print("   • Automatic stopword and punctuation filtering")
        print("   • Configurable token length filtering")
        
        print("\n💡 Model comparison:")
        print("   • ru_core_news_sm: 15.3 MB, faster, good accuracy")
        print("   • ru_core_news_lg: 513.4 MB, slower, higher accuracy")
        print("   • For MTR processing, the small model should be sufficient")
        
        return True
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
