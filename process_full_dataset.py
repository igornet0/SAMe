#!/usr/bin/env python3
"""
Скрипт для оптимальной обработки полного датасета main_dataset_processed.csv
Автоматически выбирает лучшую стратегию на основе доступных ресурсов
Сохраняет результаты в Excel формате с использованием модулей экспорта SAMe
"""

import os
import sys
import psutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd

# Добавляем путь к модулям SAMe
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импорты модулей экспорта SAMe
try:
    from src.same_api.export import ExcelExporter, ExcelExportConfig
    EXPORT_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAMe export modules not available. Missing: {e}")
    EXPORT_MODULES_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_system_resources():
    """Проверка системных ресурсов"""
    # Память
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    total_memory_gb = memory.total / (1024**3)
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # Диск
    disk = psutil.disk_usage('.')
    available_disk_gb = disk.free / (1024**3)
    
    logger.info(f"System resources:")
    logger.info(f"  Memory: {available_memory_gb:.1f}GB available / {total_memory_gb:.1f}GB total")
    logger.info(f"  CPU: {cpu_count} cores ({cpu_count_logical} logical)")
    logger.info(f"  Disk: {available_disk_gb:.1f}GB available")
    
    return {
        'available_memory_gb': available_memory_gb,
        'total_memory_gb': total_memory_gb,
        'cpu_count': cpu_count,
        'cpu_count_logical': cpu_count_logical,
        'available_disk_gb': available_disk_gb
    }


def recommend_strategy(resources, dataset_size_mb):
    """Рекомендация стратегии обработки"""
    memory_gb = resources['available_memory_gb']
    cpu_count = resources['cpu_count']
    
    logger.info(f"Dataset size: {dataset_size_mb:.1f}MB")
    
    if memory_gb >= 8 and cpu_count >= 4:
        strategy = "parallel"
        chunk_size = 8000
        workers = min(cpu_count, 8)
        logger.info(f"🚀 Recommended strategy: PARALLEL processing")
        logger.info(f"   Chunk size: {chunk_size}, Workers: {workers}")
    elif memory_gb >= 4:
        strategy = "large_dataset"
        chunk_size = 10000
        workers = 1
        logger.info(f"🔄 Recommended strategy: LARGE DATASET processing")
        logger.info(f"   Chunk size: {chunk_size}")
    else:
        strategy = "emergency"
        chunk_size = 5000
        workers = 1
        logger.info(f"⚡ Recommended strategy: EMERGENCY processing")
        logger.info(f"   Chunk size: {chunk_size}")
    
    return strategy, chunk_size, workers


def convert_csv_to_excel(csv_file: str, excel_file: str, metadata: dict = None) -> bool:
    """Конвертация CSV в Excel с использованием улучшенного конвертера"""
    try:
        logger.info(f"Converting {csv_file} to Excel format...")

        # Используем наш улучшенный конвертер
        cmd = [
            sys.executable, "csv_to_excel_converter.py",
            csv_file,
            "-o", excel_file
        ]

        logger.info(f"Running Excel converter: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Excel conversion completed successfully!")
        logger.info(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Excel conversion failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")

        # Fallback к простому pandas
        try:
            logger.info("Trying fallback conversion with pandas...")
            df = pd.read_csv(csv_file)

            # Проверяем размер данных
            if len(df) > 1000000:
                logger.warning(f"Dataset too large ({len(df)} rows) for simple Excel export")
                return False

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Результаты поиска', index=False)

            logger.info(f"✅ Fallback Excel file created: {excel_file}")
            return True

        except Exception as fallback_error:
            logger.error(f"Fallback conversion also failed: {fallback_error}")
            return False

    except Exception as e:
        logger.error(f"Error converting to Excel: {e}")
        return False


def run_processing(input_file, output_csv, strategy, chunk_size, workers, threshold):
    """Запуск обработки с выбранной стратегией"""

    if strategy == "parallel":
        cmd = [
            sys.executable, "parallel_dbscan_processor.py",
            input_file,
            "-o", output_csv,
            "-c", str(chunk_size),
            "-t", str(threshold),
            "-j", str(workers),
            "--overlap", "1000"
        ]
    elif strategy == "large_dataset":
        cmd = [
            sys.executable, "large_dataset_processor.py",
            input_file,
            "-o", output_csv,
            "-c", str(chunk_size),
            "-t", str(threshold),
            "--max-features", "3000",
            "--n-components", "50"
        ]
    else:  # emergency
        cmd = [
            sys.executable, "emergency_search.py",
            input_file,
            "-o", output_csv,
            "-t", str(threshold),
            "-l", str(chunk_size * 10),  # Для emergency используем больший лимит
            "-b", str(chunk_size)
        ]

    logger.info(f"Executing command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Processing completed successfully!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Processing failed with error: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process full main_dataset_processed.csv optimally and save to Excel')
    parser.add_argument('input_file', nargs='?',
                       default='src/data/output/main_dataset_processed.csv',
                       help='Input CSV file (default: src/data/output/main_dataset_processed.csv)')
    parser.add_argument('-o', '--output', help='Output Excel file (.xlsx) or CSV file (.csv)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Similarity threshold (default: 0.5)')
    parser.add_argument('-s', '--strategy', choices=['auto', 'parallel', 'large_dataset', 'emergency'],
                       default='auto', help='Processing strategy (default: auto)')
    parser.add_argument('--chunk-size', type=int, help='Override chunk size')
    parser.add_argument('--workers', type=int, help='Override number of workers')
    parser.add_argument('--keep-csv', action='store_true', help='Keep intermediate CSV file')
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    input_file = args.input_file
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Определяем выходные файлы
    if args.output:
        if args.output.endswith('.xlsx'):
            excel_file = args.output
            csv_file = args.output.replace('.xlsx', '.csv')
        else:
            excel_file = args.output.replace('.csv', '.xlsx')
            csv_file = args.output
    else:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = f"full_dataset_results_{timestamp}.xlsx"
        csv_file = f"full_dataset_results_{timestamp}.csv"
    
    logger.info(f"🎯 Processing full dataset: {input_file}")
    logger.info(f"📄 CSV output: {csv_file}")
    logger.info(f"📊 Excel output: {excel_file}")
    logger.info(f"🎚️  Similarity threshold: {args.threshold}")
    
    # Проверяем системные ресурсы
    resources = check_system_resources()
    
    # Получаем размер файла
    file_size_mb = os.path.getsize(input_file) / (1024**2)
    
    # Определяем стратегию
    if args.strategy == 'auto':
        strategy, chunk_size, workers = recommend_strategy(resources, file_size_mb)
    else:
        strategy = args.strategy
        chunk_size = args.chunk_size or 8000
        workers = args.workers or min(4, psutil.cpu_count())
        logger.info(f"Using manual strategy: {strategy}")
    
    # Переопределяем параметры если заданы вручную
    if args.chunk_size:
        chunk_size = args.chunk_size
    if args.workers:
        workers = args.workers
    
    logger.info(f"Final parameters:")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Threshold: {args.threshold}")
    
    # Запускаем обработку
    start_time = datetime.now()
    success = run_processing(input_file, csv_file, strategy, chunk_size, workers, args.threshold)
    processing_end_time = datetime.now()

    processing_duration = processing_end_time - start_time

    if success:
        logger.info(f"✅ Data processing completed successfully!")
        logger.info(f"⏱️  Processing time: {processing_duration}")

        # Создаем метаданные для Excel
        metadata = {
            'Дата обработки': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'Время обработки': str(processing_duration),
            'Входной файл': input_file,
            'Стратегия обработки': strategy,
            'Размер чанка': chunk_size,
            'Количество воркеров': workers,
            'Порог схожести': args.threshold,
            'Версия SAMe': '2.0'
        }

        # Конвертируем в Excel
        logger.info("🔄 Converting results to Excel format...")
        excel_success = convert_csv_to_excel(csv_file, excel_file, metadata)

        end_time = datetime.now()
        total_duration = end_time - start_time

        if excel_success:
            logger.info(f"✅ Full processing completed successfully!")
            logger.info(f"⏱️  Total time: {total_duration}")
            logger.info(f"📄 CSV results: {csv_file}")
            logger.info(f"📊 Excel results: {excel_file}")

            # Показываем статистику результатов
            if os.path.exists(csv_file):
                try:
                    results_df = pd.read_csv(csv_file)
                    logger.info(f"📊 Results summary:")
                    logger.info(f"   Total relationships found: {len(results_df)}")

                    if 'Relation_Type' in results_df.columns:
                        relation_counts = results_df['Relation_Type'].value_counts()
                        for relation_type, count in relation_counts.items():
                            logger.info(f"   {relation_type}: {count}")

                    if 'Similarity_Score' in results_df.columns:
                        logger.info(f"   Average similarity: {results_df['Similarity_Score'].mean():.3f}")
                        logger.info(f"   Max similarity: {results_df['Similarity_Score'].max():.3f}")

                except Exception as e:
                    logger.warning(f"Could not read results for statistics: {e}")

            # Удаляем промежуточный CSV файл если Excel создан успешно и не установлен флаг keep-csv
            if not args.keep_csv:
                try:
                    if os.path.exists(csv_file) and csv_file != excel_file:
                        os.remove(csv_file)
                        logger.info(f"🗑️  Removed intermediate CSV file: {csv_file}")
                except Exception as e:
                    logger.warning(f"Could not remove CSV file: {e}")
            else:
                logger.info(f"📄 CSV file kept: {csv_file}")

            return 0
        else:
            logger.warning(f"⚠️  Processing completed but Excel conversion failed")
            logger.info(f"📄 CSV results available: {csv_file}")
            return 0
    else:
        logger.error(f"❌ Processing failed!")
        return 1


if __name__ == "__main__":
    exit(main())
