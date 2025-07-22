#!/usr/bin/env python3
"""
SAMe Project - Main Pipeline Entry Point

This script implements a complete end-to-end pipeline for the SAMe (Search Analog Model Engine) project:

WORKFLOW STEPS:
1. Data Loading: Load nomenclature data from data/input/main_dataset.xlsx
2. Data Processing: Process the loaded data and save to data/processed/
3. Model Training: Train similarity search models and save to src/models/
4. Interactive Query Mode: Allow users to search for similar items
5. Model Persistence: Load existing models if available (skips training if found)

FEATURES:
- Automatic model persistence and loading
- Interactive search interface
- Excel export with standard columns (Raw_Name, Cleaned_Name, Lemmatized_Name,
  Normalized_Name, Candidate_Name, Similarity_Score, Relation_Type,
  Suggested_Category, Final_Decision, Comment)
- Backward compatibility with existing SAMe API structure
- Uses current same.* module import structure
- Comprehensive logging and error handling

USAGE:
    python main.py

REQUIREMENTS:
- data/input/main_dataset.xlsx must exist
- All SAMe dependencies must be installed
- Sufficient disk space for processed data and models

OUTPUT DIRECTORIES:
- data/processed/: Processed data files (parquet format)
- src/models/: Trained model artifacts and configuration
- data/output/: Excel export files from search results
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/main_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class SAMePipeline:
    """Main pipeline class for SAMe project"""

    def __init__(self):
        self.engine: Optional[AnalogSearchEngine] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.models_dir = Path("src/models")
        self.data_input_dir = Path("data/input")
        self.data_processed_dir = Path("data/processed")
        self.data_output_dir = Path("data/output")

        # Ensure directories exist
        for directory in [self.models_dir, self.data_processed_dir, self.data_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Pipeline directories initialized:")
        logger.info(f"   Input: {self.data_input_dir}")
        logger.info(f"   Processed: {self.data_processed_dir}")
        logger.info(f"   Models: {self.models_dir}")
        logger.info(f"   Output: {self.data_output_dir}")

    def _serialize_parameters(self, parameters):
        """Convert ExtractedParameter objects to serializable format"""
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
                return serialized
            else:
                return str(parameters)
        except Exception as e:
            logger.warning(f"Failed to serialize parameters: {e}")
            return str(parameters)

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
            # Create search engine configuration
            config = AnalogSearchConfig(
                search_method="hybrid",
                similarity_threshold=0.6,
                max_results_per_query=50,
                enable_parameter_extraction=True,
                data_dir=Path("data"),
                models_dir=self.models_dir,
                output_dir=self.data_output_dir
            )

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
            # Create search engine with default config
            config = AnalogSearchConfig(
                search_method="hybrid",
                similarity_threshold=0.6,
                models_dir=self.models_dir,
                output_dir=self.data_output_dir
            )

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
                latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
                file_format = "parquet" if latest_file.suffix == ".parquet" else "csv"
                self.processed_data = await data_helper.read_file(latest_file, format=file_format)
                logger.info(f"📊 Loaded processed data: {latest_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Error loading existing models: {e}")
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
                results = await self.engine.search_analogs([query])
                
                if query in results and results[query]:
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
