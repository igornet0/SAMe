#!/usr/bin/env python3
"""
Быстрый скрипт для обработки csv в Excel
Предустановленные оптимальные параметры для разных сценариев
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import subprocess


from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_quick_processing(scenario: str, input_file: str, output_file: str = None, threshold: float = 0.5):
    """Быстрый запуск обработки с предустановленными параметрами"""
    
    # Предустановленные сценарии
    scenarios = {
        'fast': {
            'strategy': 'emergency',
            'chunk_size': 10000,
            'workers': 1,
            'description': 'Быстрая обработка (5-10 минут, среднее качество)'
        },
        'balanced': {
            'strategy': 'auto',
            'chunk_size': 8000,
            'workers': 4,
            'description': 'Сбалансированная обработка (15-30 минут, высокое качество)'
        },
        'quality': {
            'strategy': 'parallel',
            'chunk_size': 5000,
            'workers': 8,
            'description': 'Качественная обработка (20-45 минут, максимальное качество)'
        },
        'memory_safe': {
            'strategy': 'large_dataset',
            'chunk_size': 15000,
            'workers': 1,
            'description': 'Безопасная для памяти обработка (30-60 минут, низкое потребление RAM)'
        }
    }
    
    if scenario not in scenarios:
        logger.error(f"Unknown scenario: {scenario}")
        logger.info(f"Available scenarios: {', '.join(scenarios.keys())}")
        return False
    
    config = scenarios[scenario]
    logger.info(f"🚀 Running scenario: {scenario}")
    logger.info(f"📝 Description: {config['description']}")
    
    # Определяем выходной файл
    if not output_file:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{scenario}_{timestamp}.xlsx"
    
    # Формируем команду
    cmd = [
        sys.executable, "process_full_dataset.py",
        input_file,
        "-o", output_file,
        "-s", config['strategy'],
        "-t", str(threshold),
        "--chunk-size", str(config['chunk_size']),
        "--workers", str(config['workers'])
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✅ Processing completed successfully!")
        logger.info(f"📊 Results saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Processing failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick processing of main_dataset_processed.csv to Excel')
    parser.add_argument('scenario', choices=['fast', 'balanced', 'quality', 'memory_safe'],
                       help='Processing scenario')
    parser.add_argument('input_file', nargs='?',
                       default='src/data/output/main_dataset_processed.csv',
                       help='Input CSV file (default: src/data/output/main_dataset_processed.csv)')
    parser.add_argument('-o', '--output', help='Output Excel file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Similarity threshold')
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Показываем информацию о сценариях
    scenarios_info = {
        'fast': '⚡ Быстро (5-10 мин, среднее качество)',
        'balanced': '⚖️  Сбалансированно (15-30 мин, высокое качество)', 
        'quality': '🎯 Качественно (20-45 мин, максимальное качество)',
        'memory_safe': '🛡️  Безопасно для памяти (30-60 мин, низкое RAM)'
    }
    
    logger.info(f"📋 Available scenarios:")
    for scenario, description in scenarios_info.items():
        marker = "👉 " if scenario == args.scenario else "   "
        logger.info(f"{marker}{scenario}: {description}")
    
    # Запускаем обработку
    success = run_quick_processing(args.scenario, args.input_file, args.output, args.threshold)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
