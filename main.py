#!/usr/bin/env python3
"""
Проект SAMe - Главная точка входа конвейера

Этот скрипт реализует полный сквозной конвейер для проекта SAMe (Search Analog Model Engine):

ЭТАПЫ РАБОТЫ:
1. Загрузка данных: Загрузка номенклатурных данных из data/input/main_dataset.xlsx
2. Обработка данных: Обработка загруженных данных и сохранение в data/processed/
3. Обучение моделей: Обучение моделей поиска по сходству и сохранение в src/models/
4. Интерактивный режим запросов: Позволяет пользователям искать похожие элементы
5. Сохранение моделей: Загрузка существующих моделей при наличии (пропускает обучение)

ВОЗМОЖНОСТИ:
- Автоматическое сохранение и загрузка моделей
- Интерактивный интерфейс поиска
- Экспорт в Excel со стандартными колонками (Raw_Name, Cleaned_Name, Lemmatized_Name,
  Normalized_Name, Candidate_Name, Similarity_Score, Relation_Type,
  Suggested_Category, Final_Decision, Comment)
- Обратная совместимость с существующей структурой API SAMe
- Использует текущую структуру импорта модулей same.*
- Комплексное логирование и обработка ошибок

ИСПОЛЬЗОВАНИЕ:
    python main.py

ТРЕБОВАНИЯ:
- Должен существовать файл data/input/main_dataset.xlsx
- Все зависимости SAMe должны быть установлены
- Достаточно места на диске для обработанных данных и моделей

ВЫХОДНЫЕ ДИРЕКТОРИИ:
- data/processed/: Файлы обработанных данных (формат parquet)
- src/models/: Артефакты обученных моделей и конфигурация
- data/output/: Файлы экспорта Excel из результатов поиска
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# SAMe imports using current module structure
from same.analog_search_engine import AnalogSearchEngine, AnalogSearchConfig
from same.export.excel_exporter import ExcelExporter, ExcelExportConfig
from same.data_manager import data_helper
from same.settings import settings

# Configure logging
project_root = Path(__file__).parent.resolve()
logs_dir = project_root / "src" / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)-45s:%(lineno)-3d - %(levelname)-7s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / 'main_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class SAMePipeline:
    """Main pipeline class for SAMe project"""

    def __init__(self):
        self.engine: Optional[AnalogSearchEngine] = None
        self.processed_data: Optional[pd.DataFrame] = None

        # Get the project root directory (where main.py is located)
        self.project_root = Path(__file__).parent.resolve()

        # Define paths relative to src directory (data and logs now in src/)
        self.models_dir = (self.project_root / "src" / "models").resolve()
        self.data_input_dir = (self.project_root / "src" / "data" / "input").resolve()
        self.data_processed_dir = (self.project_root / "src" / "data" / "processed").resolve()
        self.data_output_dir = (self.project_root / "src" / "data" / "output").resolve()

        # Ensure directories exist
        for directory in [self.models_dir, self.data_processed_dir, self.data_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Pipeline directories initialized:")
        logger.info(f"   Input: {self.data_input_dir}")
        logger.info(f"   Processed: {self.data_processed_dir}")
        logger.info(f"   Models: {self.models_dir}")
        logger.info(f"   Output: {self.data_output_dir}")

    def _serialize_parameters(self, parameters):
        """Convert ExtractedParameter objects to serializable format for Parquet"""
        if parameters is None or len(parameters) == 0:
            return None

        try:
            # Convert list of ExtractedParameter objects to list of dictionaries
            if isinstance(parameters, list):
                serialized = []
                for param in parameters:
                    if hasattr(param, '__dict__'):
                        # Convert ExtractedParameter to dict
                        param_dict = {
                            'name': param.name,
                            'value': param.value,
                            'unit': param.unit,
                            'parameter_type': param.parameter_type.value if hasattr(param.parameter_type, 'value') else str(param.parameter_type),
                            'confidence': param.confidence,
                            'source_text': param.source_text,
                            'position': param.position
                        }
                        serialized.append(param_dict)
                    else:
                        serialized.append(str(param))

                # Return JSON string for Parquet compatibility
                return json.dumps(serialized, ensure_ascii=False)
            else:
                return str(parameters)
        except Exception as e:
            logger.warning(f"Failed to serialize parameters: {e}")
            return str(parameters) if parameters else None

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")

        # Check if input dataset exists
        dataset_path = self.data_input_dir / "main_dataset.xlsx"
        if not dataset_path.exists():
            logger.error(f"❌ Required dataset not found: {dataset_path}")
            return False

        # Check if we can write to directories
        for directory in [self.data_processed_dir, self.models_dir, self.data_output_dir]:
            if not directory.exists() or not directory.is_dir():
                logger.error(f"❌ Directory not accessible: {directory}")
                return False

        logger.info("✅ All prerequisites met")
        return True
    
    async def load_data(self) -> pd.DataFrame:
        """Step 1: Load nomenclature data from data/input/main_dataset.xlsx"""
        logger.info("🔄 Step 1: Loading nomenclature data...")
        
        dataset_path = self.data_input_dir / "main_dataset.xlsx"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        try:
            data = pd.read_excel(dataset_path)
            logger.info(f"✅ Dataset loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            logger.info(f"📊 Columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    async def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Process the loaded data and save to data/processed/"""
        logger.info("🔄 Step 2: Processing data...")

        try:
            # Create enhanced search engine configuration
            config = AnalogSearchConfig(
                search_method="hybrid",
                similarity_threshold=0.6,
                max_results_per_query=50,
                enable_parameter_extraction=True,
                data_dir=self.project_root / "src" / "data",
                models_dir=self.models_dir,
                output_dir=self.data_output_dir
            )

            logger.info("🔧 Enhanced search features enabled:")
            logger.info("  - Categorical pre-filtering for better relevance")
            logger.info("  - Reduced numeric token weight to prevent false matches")
            logger.info("  - Improved Russian morphological processing")
            logger.info("  - Multi-metric scoring (semantic + lexical + key terms)")

            # Initialize search engine for processing
            self.engine = AnalogSearchEngine(config)

            # Initialize with data (this will process the data)
            await self.engine.initialize(data)

            # Get processed data
            processed_data = self.engine.processed_catalog.copy()

            # Convert ExtractedParameter objects to serializable format
            if 'extracted_parameters' in processed_data.columns:
                logger.info("🔄 Converting extracted parameters to serializable format...")
                processed_data['extracted_parameters'] = processed_data['extracted_parameters'].apply(
                    self._serialize_parameters
                )

            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Try to save as Parquet first, fallback to CSV if it fails
            try:
                processed_file = self.data_processed_dir / f"processed_data_{timestamp}.parquet"
                await data_helper.write_file(processed_data, processed_file, format="parquet")
                logger.info(f"✅ Data processed and saved to: {processed_file}")
            except Exception as parquet_error:
                logger.warning(f"Failed to save as Parquet: {parquet_error}")
                logger.info("🔄 Falling back to CSV format...")
                processed_file = self.data_processed_dir / f"processed_data_{timestamp}.csv"
                await data_helper.write_file(processed_data, processed_file, format="csv")
                logger.info(f"✅ Data processed and saved to: {processed_file}")

            logger.info(f"📊 Processed data shape: {processed_data.shape}")

            self.processed_data = processed_data
            return processed_data

        except Exception as e:
            logger.error(f"❌ Error processing data: {e}")
            raise
    
    async def train_models(self) -> bool:
        """Step 3: Train similarity search models and save to src/models/"""
        logger.info("🔄 Step 3: Training similarity search models...")
        
        if self.engine is None:
            logger.error("❌ Search engine not initialized")
            return False
        
        try:
            # Models are already trained during initialization
            # Save the trained models
            await self.engine.save_models(str(self.models_dir))
            
            # Save configuration and metadata
            config_file = self.models_dir / "model_config.json"
            metadata = {
                "training_timestamp": datetime.now().isoformat(),
                "data_shape": list(self.processed_data.shape) if self.processed_data is not None else None,
                "search_method": self.engine.config.search_method,
                "similarity_threshold": self.engine.config.similarity_threshold,
                "model_version": "1.0.0"
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Models trained and saved to: {self.models_dir}")
            logger.info(f"📊 Configuration saved to: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            return False
    
    async def load_existing_models(self) -> bool:
        """Step 5: Load existing trained models if available"""
        logger.info("🔍 Checking for existing trained models...")
        
        # Check for model files
        fuzzy_model = self.models_dir / "fuzzy_search_model.pkl"
        semantic_model = self.models_dir / "semantic_search_model.pkl"
        config_file = self.models_dir / "model_config.json"
        
        if not (fuzzy_model.exists() or semantic_model.exists()):
            logger.info("📝 No existing models found")
            return False
        
        try:
            # Create enhanced search engine with default config
            config = AnalogSearchConfig(
                search_method="hybrid",
                similarity_threshold=0.6,
                data_dir=self.project_root / "src" / "data",
                models_dir=self.models_dir,
                output_dir=self.data_output_dir
            )

            logger.info("🚀 Loading models with enhanced search capabilities")

            self.engine = AnalogSearchEngine(config)

            # Try to load existing models
            try:
                await self.engine.load_models(str(self.models_dir))
                logger.info("✅ Existing models loaded successfully")
            except Exception as model_error:
                logger.warning(f"Failed to load models: {model_error}")
                # If model loading fails, we'll need to retrain
                return False

            # Load processed data if available
            processed_files = list(self.data_processed_dir.glob("processed_data_*.parquet"))
            processed_files.extend(list(self.data_processed_dir.glob("processed_data_*.csv")))

            if processed_files:
                # Sort files by modification time (newest first) and find the first non-empty file
                processed_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                latest_file = None
                for file_path in processed_files:
                    if file_path.stat().st_size > 0:  # Skip empty files
                        latest_file = file_path
                        break

                if latest_file is None:
                    logger.warning("All processed data files are empty")
                    return False

                file_format = "parquet" if latest_file.suffix == ".parquet" else "csv"
                self.processed_data = await data_helper.read_file(latest_file, format=file_format)
                logger.info(f"📊 Loaded processed data: {latest_file}")

                # CRITICAL FIX: Set the processed_catalog in the search engine
                self.engine.processed_catalog = self.processed_data
                self.engine.catalog_data = self.processed_data  # Also set the original catalog data
                logger.info(f"✅ Search engine initialized with processed catalog: {len(self.processed_data)} items")
            else:
                logger.warning("No processed data files found")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error loading existing models: {str(e)[:40]}")
            return False
    
    async def interactive_query_mode(self):
        """Step 4: Interactive query mode for searching similar items"""
        logger.info("🔄 Step 4: Starting interactive query mode...")
        
        if self.engine is None or not self.engine.is_ready:
            logger.error("❌ Search engine not ready")
            return
        
        print("\n" + "="*60)
        print("🔍 SAMe Interactive Query Mode")
        print("="*60)
        print("Enter search queries to find similar items.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'stats' to see system statistics.")
        print("-"*60)
        
        while True:
            try:
                query = input("\n🔍 Enter search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = self.engine.get_statistics()
                    print(f"\n📊 System Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue
                
                if not query:
                    continue
                
                # Search for analogs
                print(f"🔄 Searching for: '{query}'...")
                try:
                    results = await self.engine.search_analogs([query])
                except Exception as search_error:
                    logger.error(f"Error during search_analogs: {search_error}")
                    results = None

                # Check if results exist and are not None
                if results and query in results and results[query] is not None and len(results[query]) > 0:
                    print(f"\n✅ Found {len(results[query])} similar items:")
                    for i, result in enumerate(results[query][:10], 1):
                        score = result.get('similarity_score', result.get('combined_score', 0))
                        document = result.get('document', 'Unknown')
                        print(f"   {i}. {document} (Score: {score:.3f})")

                    # Ask if user wants to export results
                    export = input(f"\n💾 Export results to Excel? (y/n): ").strip().lower()
                    if export in ['y', 'yes']:
                        await self.export_results(results, query)
                else:
                    print("❌ No similar items found")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in query processing: {e}")
                print(f"❌ Error: {e}")
    
    async def export_results(self, results: Dict[str, Any], query: str):
        """Export search results to Excel with standard columns"""
        try:
            # Create export configuration
            export_config = ExcelExportConfig(
                include_statistics=True,
                include_metadata=True,
                auto_adjust_columns=True,
                add_filters=True,
                highlight_high_similarity=True,
                similarity_threshold=0.8
            )
            
            exporter = ExcelExporter(export_config)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()[:20]
            filename = f"search_results_{safe_query}_{timestamp}.xlsx"
            output_path = self.data_output_dir / filename
            
            # Export results
            exporter.export_search_results(
                results=results,
                output_path=str(output_path),
                metadata={
                    'system': 'SAMe Pipeline',
                    'version': '1.0.0',
                    'export_date': datetime.now().isoformat(),
                    'query': query,
                    'total_results': len(results.get(query, []))
                }
            )
            
            print(f"✅ Results exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            print(f"❌ Export error: {e}")


async def main():
    """Main pipeline execution"""
    print("🚀 Starting SAMe Pipeline...")
    print("="*60)

    pipeline = SAMePipeline()

    try:
        # Check prerequisites first
        if not pipeline.check_prerequisites():
            print("❌ Prerequisites not met. Please check the logs for details.")
            sys.exit(1)

        # Step 5: Check for existing models first
        models_exist = await pipeline.load_existing_models()

        if models_exist:
            print("✅ Using existing trained models")
            print("🔄 Skipping data processing and model training...")
            # Skip to interactive mode
            await pipeline.interactive_query_mode()
        else:
            print("📝 No existing models found, starting full pipeline...")

            # Step 1: Load data
            data = await pipeline.load_data()

            # Step 2: Process data
            await pipeline.process_data(data)

            # Step 3: Train models
            training_success = await pipeline.train_models()

            if training_success:
                # Step 4: Interactive query mode
                await pipeline.interactive_query_mode()
            else:
                logger.error("❌ Model training failed, cannot proceed to query mode")
                print("❌ Model training failed. Check logs for details.")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
