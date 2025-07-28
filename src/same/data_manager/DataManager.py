from pydantic_settings import BaseSettings
from pathlib import Path

from typing import Optional, Literal, Any, Dict, Union, List
import pandas as pd
import aiofiles
import json
from io import StringIO
import zipfile
import shutil
from datetime import datetime

import logging

# Импорты для работы с параметрами
try:
    from ..parameter_extraction.regex_extractor import ExtractedParameter
    from ..parameter_extraction.parameter_utils import ParameterFormatter, ParameterAnalyzer
except ImportError:
    # Fallback если модули параметров не доступны
    ExtractedParameter = None
    ParameterFormatter = None
    ParameterAnalyzer = None

logger = logging.getLogger("DataManager")

class SettingsTrade(BaseSettings):

    # Пути к данным - перемещаем data и logs в src/
    # __file__ = src/same/data_manager/DataManager.py
    # .parent.parent.parent.parent = корень проекта
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "src" / "data"

    LOG_PATH: Path = BASE_DIR / "src" / "logs"

    # Модели ML - в src/models
    MODELS_DIR: Path = BASE_DIR / "src" / "models"
    MODELS_CONFIGS_PATH: Path = MODELS_DIR / "configs"
    MODELS_LOGS_PATH: Path = MODELS_DIR / "logs"

class DataManager:

    _settings = SettingsTrade()

    def __init__(self):
        self.required_dirs = {
            "data": self.settings.DATA_DIR,
            "log": self.settings.LOG_PATH,
            "models": self.settings.MODELS_DIR,
            "models logs": self.settings.MODELS_LOGS_PATH,
            "models configs": self.settings.MODELS_CONFIGS_PATH,
        }

        self._ensure_directories_exist()
        self._setup_logging()

    @property
    def settings(self) -> SettingsTrade:
        return self._settings

    def _ensure_directories_exist(self):
        """Создает все необходимые директории при инициализации"""

        for directory in self.required_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Настройка логирования для DataManager"""
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.settings.LOG_PATH / "data_manager.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    def __getitem__(self, key):
        if not self.required_dirs.get(key):
            raise AttributeError(f"SettingsTrade has no attribute '{key}'")
        
        return self.required_dirs[key]
    
    @classmethod
    def get_df(cls, path: Path) -> pd.DataFrame:
        """Кэшированный список монет"""
        return pd.read_csv(path)

    async def read_file(self, path: Path, format: str = "csv") -> pd.DataFrame:
        """
        Асинхронное чтение файлов данных
        Поддерживаемые форматы: csv, parquet, json
        """
        try:
            if format == "csv":
                async with aiofiles.open(path, mode="r") as f:
                    return pd.read_csv(await f.read())
            elif format == "parquet":
                try:
                    return pd.read_parquet(path, engine='pyarrow')
                except ImportError:
                    logger.warning("PyArrow not available, trying fastparquet...")
                    try:
                        return pd.read_parquet(path, engine='fastparquet')
                    except ImportError:
                        logger.error("Neither pyarrow nor fastparquet available for parquet support")
                        raise ValueError("Parquet support requires pyarrow or fastparquet. Install with: pip install pyarrow")
            elif format == "json":
                async with aiofiles.open(path, mode="r") as f:
                    json_content = await f.read()
                    return pd.read_json(StringIO(json_content))
            else:
                raise ValueError(f"Unsupported format: {format}")
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)[:40]}")
            raise

    async def write_file(
        self,
        data: Union[pd.DataFrame, Dict],
        path: Path,
        format: str = "csv",
        mode: str = "w"
    ) -> None:
        """
        Асинхронная запись данных в файл
        """
        try:
            if format == "csv":
                async with aiofiles.open(path, mode=mode) as f:
                    await f.write(data.to_csv(index=False))

            elif format == "parquet":
                try:
                    data.to_parquet(path, engine='pyarrow')
                except ImportError:
                    logger.warning("PyArrow not available, trying fastparquet...")
                    try:
                        data.to_parquet(path, engine='fastparquet')
                    except ImportError:
                        logger.error("Neither pyarrow nor fastparquet available for parquet support")
                        raise ValueError("Parquet support requires pyarrow or fastparquet. Install with: pip install pyarrow")

            elif format == "json":
                async with aiofiles.open(path, mode=mode) as f:
                    await f.write(json.dumps(data))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error writing to {path}: {str(e)}")
            raise

    def cache_data(self, data: Any, key: str) -> None:
        """
        Кэширование данных в памяти и на диске
        """
        try:
            cache_path = self.settings.CACHED_DATA_PATH / f"{key}.pkl"
            pd.to_pickle(data, cache_path)
            logger.info(f"Data cached: {key}")
        except Exception as e:
            logger.error(f"Cache error for {key}: {str(e)}")

    def load_cache(self, key: str) -> Any:
        """
        Загрузка данных из кэша
        """
        try:
            cache_path = self.settings.CACHED_DATA_PATH / f"{key}.pkl"
            return pd.read_pickle(cache_path)
        except FileNotFoundError:
            logger.warning(f"Cache not found: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache load error for {key}: {str(e)}")
            return None

    def backup_data(self, paths: list[Path], backup_name: str = None) -> Path:
        """
        Создание резервной копии данных
        """
        backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = self.settings.BACKUP_DATA_PATH / backup_name
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in paths:
                    if path.is_file():
                        zipf.write(path, arcname=path.name)
                        
                    elif path.is_dir():
                        for file in path.rglob('*'):
                            if file.is_file():
                                zipf.write(file, arcname=file.relative_to(path.parent))
            
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise

    def validate_dataset(self, df: pd.DataFrame, expected_columns: list) -> bool:
        """
        Валидация структуры датасета
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid data type. Expected DataFrame")
            return False
        
        missing_cols = set(expected_columns) - set(df.columns)

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.warning("Empty dataset")
            return False
        
        return True

    def create_dir(self, type_dir: Literal["raw", "processed", "cached", "backup"], name_of_dir: str) -> Path:
        """
        Создание директории
        """
        try:
            path = self.required_dirs[type_dir] / name_of_dir
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {path}")
            return path
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            raise
        
    def get_latest_processed_data(self, coin: str, type_data: str = "processed") -> Optional[pd.DataFrame]:
        """
        Получение последней версии обработанных данных для указанной монеты
        """
        try:
            coin_dir = self.required_dirs[type_data] / coin
            if not coin_dir.exists():
                return None

            latest_file = max(
                coin_dir.glob("*.parquet"),
                key=lambda x: x.stat().st_mtime,
                default=None
            )

            return pd.read_parquet(latest_file) if latest_file else None
        except Exception as e:
            logger.error(f"Error getting latest data for {coin}: {str(e)}")
            return None

    async def cleanup_trach(self, max_age_days: int = 30) -> None:
        """
        Очистка устаревших файлов в директории trach
        """
        try:
            now = datetime.now().timestamp()
            for file in self.settings.TRACH_PATH.glob('*'):
                if (now - file.stat().st_mtime) > max_age_days * 86400:
                    if file.is_dir():
                        shutil.rmtree(file)
                    else:
                        file.unlink()
                    logger.info(f"Removed old file: {file}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            raise

    def get_model_config(self, agent_type: str, model_name: str) -> dict:
        """
        Загрузка конфигурации модели
        """

        config_path = self.settings.MODELS_CONFIGS_PATH / agent_type / f"{model_name}.json"
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Model config not found: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model config {model_name}: {str(e)}")
        
        return {}

    async def save_processed_data(
        self,
        coin: str,
        data: pd.DataFrame,
        dataset_type: str = "clear",
        version: str = None
    ) -> Path:
        """
        Сохранение обработанных данных с версионированием
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = self.get_path(
            "processed",
            coin=coin,
            dataset_type=dataset_type,
            timetravel=self.settings.TIMETRAVEL
        )
        
        filename = f"{coin}_{dataset_type}_{self.settings.TIMETRAVEL}_{version}.parquet"
        save_path = save_dir / filename
        
        await self.write_file(data, save_path, format="parquet")
        
        return save_path

    async def save_parameters_data(
        self,
        data: pd.DataFrame,
        parameters_column: str = 'extracted_parameters',
        dataset_name: str = "parameters",
        version: str = None
    ) -> Path:
        """
        Сохранение данных с извлеченными параметрами

        Args:
            data: DataFrame с параметрами
            parameters_column: Название колонки с параметрами
            dataset_name: Название датасета
            version: Версия (опционально)

        Returns:
            Путь к сохраненному файлу
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = self.settings.DATA_DIR / "processed" / "parameters"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Создаем копию данных для сериализации
        data_to_save = data.copy()

        # Сериализуем ExtractedParameter объекты в колонке параметров
        if parameters_column in data_to_save.columns:
            logger.info(f"Serializing {parameters_column} column for Parquet compatibility...")
            data_to_save[parameters_column] = data_to_save[parameters_column].apply(
                self._serialize_parameters_for_parquet
            )

        # Сохраняем основные данные
        main_filename = f"{dataset_name}_{version}.parquet"
        main_path = save_dir / main_filename

        await self.write_file(data_to_save, main_path, format="parquet")

        # Создаем сводку по параметрам если доступны утилиты
        if ParameterAnalyzer and parameters_column in data.columns:
            try:
                # Анализ параметров
                parameters_list = data[parameters_column].tolist()
                stats = ParameterAnalyzer.analyze_parameters_batch(parameters_list)

                # Сохраняем статистику
                stats_filename = f"{dataset_name}_stats_{version}.json"
                stats_path = save_dir / stats_filename

                async with aiofiles.open(stats_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(stats, ensure_ascii=False, indent=2))

                logger.info(f"Parameters statistics saved to {stats_path}")

            except Exception as e:
                logger.warning(f"Failed to create parameters statistics: {e}")

        logger.info(f"Parameters data saved to {main_path}")
        return main_path

    def _serialize_parameters_for_parquet(self, parameters):
        """
        Сериализация ExtractedParameter объектов для совместимости с Parquet

        Args:
            parameters: Список ExtractedParameter объектов или другие данные

        Returns:
            Сериализованные данные (список словарей или строка)
        """
        if not parameters:
            return None

        try:
            # Если это список ExtractedParameter объектов
            if isinstance(parameters, list):
                serialized = []
                for param in parameters:
                    if hasattr(param, '__dict__') and hasattr(param, 'name'):
                        # Конвертируем ExtractedParameter в словарь
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
                        # Если это уже словарь или другой сериализуемый объект
                        serialized.append(param if isinstance(param, dict) else str(param))

                # Возвращаем JSON строку для совместимости с Parquet
                return json.dumps(serialized, ensure_ascii=False)
            else:
                # Если это не список, возвращаем как строку
                return str(parameters)

        except Exception as e:
            logger.warning(f"Failed to serialize parameters: {e}")
            return str(parameters) if parameters else None

    def _deserialize_parameters_from_parquet(self, serialized_data):
        """
        Десериализация параметров из Parquet формата

        Args:
            serialized_data: JSON строка с сериализованными параметрами

        Returns:
            Список словарей с параметрами или None
        """
        if not serialized_data:
            return None

        try:
            # Если это JSON строка, парсим её
            if isinstance(serialized_data, str):
                return json.loads(serialized_data)
            else:
                # Если это уже список или другой объект, возвращаем как есть
                return serialized_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to deserialize parameters: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error deserializing parameters: {e}")
            return None

    def load_parameters_data(
        self,
        dataset_name: str,
        version: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Загрузка данных с параметрами

        Args:
            dataset_name: Название датасета
            version: Версия (если не указана, загружается последняя)

        Returns:
            DataFrame с параметрами или None
        """
        try:
            save_dir = self.settings.DATA_DIR / "processed" / "parameters"

            if version:
                filename = f"{dataset_name}_{version}.parquet"
                file_path = save_dir / filename
            else:
                # Ищем последнюю версию
                pattern = f"{dataset_name}_*.parquet"
                files = list(save_dir.glob(pattern))
                if not files:
                    logger.warning(f"No parameters data found for {dataset_name}")
                    return None

                file_path = max(files, key=lambda x: x.stat().st_mtime)

            if file_path.exists():
                data = pd.read_parquet(file_path)

                # Deserialize parameters if they exist
                if 'extracted_parameters' in data.columns:
                    logger.info("Deserializing extracted_parameters column...")
                    data['extracted_parameters'] = data['extracted_parameters'].apply(
                        self._deserialize_parameters_from_parquet
                    )

                logger.info(f"Parameters data loaded from {file_path}")
                return data
            else:
                logger.warning(f"Parameters data file not found: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading parameters data: {e}")
            return None

    def get_parameters_statistics(
        self,
        dataset_name: str,
        version: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Загрузка статистики по параметрам

        Args:
            dataset_name: Название датасета
            version: Версия (если не указана, загружается последняя)

        Returns:
            Словарь со статистикой или None
        """
        try:
            save_dir = self.settings.DATA_DIR / "processed" / "parameters"

            if version:
                filename = f"{dataset_name}_stats_{version}.json"
                file_path = save_dir / filename
            else:
                # Ищем последнюю версию
                pattern = f"{dataset_name}_stats_*.json"
                files = list(save_dir.glob(pattern))
                if not files:
                    logger.warning(f"No parameters statistics found for {dataset_name}")
                    return None

                file_path = max(files, key=lambda x: x.stat().st_mtime)

            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                logger.info(f"Parameters statistics loaded from {file_path}")
                return stats
            else:
                logger.warning(f"Parameters statistics file not found: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading parameters statistics: {e}")
            return None

data_helper = DataManager()