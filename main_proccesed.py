#!/usr/bin/env python3
"""
Расширенный процессор Excel файлов для проекта SAMe
Включает улучшенную токенизацию, машинное обучение и расширенные возможности

Новые возможности:
- Увеличенный размер словаря до 50,000 токенов
- Субсловная токенизация (BPE/WordPiece)
- Предобученные эмбеддинги для русского языка
- Расширенные технические паттерны
- ML-классификация параметров vs артикулов
- Дополнительные поля анализа
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import warnings
import asyncio
warnings.filterwarnings('ignore')

# Добавляем путь к модулям SAMe
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импорты SAMe модулей
try:
    from src.same_clear.text_processing.text_cleaner import TextCleaner
    from src.same_clear.text_processing.preprocessor import TextPreprocessor
    from src.same_clear.text_processing.tokenizer import Tokenizer
    from src.same_clear.parameter_extraction.regex_extractor import RegexParameterExtractor
    from src.data_manager import data_helper
    SAME_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMe modules not available. Missing: {e}")
    SAME_MODULES_AVAILABLE = False

# Импорт основного процессора
from src.same.excel_processor import AdvancedExcelProcessor

# Новые импорты для расширенных возможностей
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available. Missing: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Настройка логирования
log_dir = Path("src/logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'excel_processor_advanced.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Advanced Excel Processor for SAMe project')
    parser.add_argument('input_file', help='Input Excel file path')
    parser.add_argument('--output-file', help='Output CSV file path (optional, will use src/data/output/ by default)')
    parser.add_argument('--tokenizer-config', default='advanced_tokenizer',
                       help='Tokenizer configuration name')
    parser.add_argument('--max-rows', type=int, help='Maximum number of rows to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Проверка входного файла
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1

    # Определение выходного файла
    if args.output_file:
        output_file = args.output_file
    else:
        # Автоматическое определение выходного файла в src/data/output/
        input_path = Path(args.input_file)
        output_dir = Path("src/data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_path.stem}_processed.csv"

    # Создание выходной директории
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {output_file}")

    # Определение типа файла по расширению
    input_path = Path(args.input_file)
    file_extension = input_path.suffix.lower()
    
    if file_extension in ['.xlsx', '.xls']:
        file_type = 'excel'
    elif file_extension == '.csv':
        file_type = 'csv'
    else:
        logger.error(f"Unsupported file format: {file_extension}. Supported formats: .xlsx, .xls, .csv")
        return 1

    logger.info(f"Detected file type: {file_type}")

    # Инициализация процессора
    try:
        processor = AdvancedExcelProcessor(args.tokenizer_config)
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return 1

    # Обработка файла
    if file_type == 'excel':
        success = processor.process_excel_file(args.input_file, str(output_file), args.max_rows)
    else:  # csv
        success = processor.process_csv_file(args.input_file, str(output_file), args.max_rows)

    if success:
        logger.info("🎉 Advanced processing completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        return 0
    else:
        logger.error("❌ Processing failed")
        return 1

if __name__ == '__main__':
    exit(main())
