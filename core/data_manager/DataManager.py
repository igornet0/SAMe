from pydantic_settings import BaseSettings
from pathlib import Path

from typing import Optional, Literal, Any, Dict, Union
import pandas as pd
import aiofiles
import json
import zipfile
import shutil
from datetime import datetime

import logging

logger = logging.getLogger("DataManager")

class SettingsTrade(BaseSettings):

    # Пути к данным
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    DATASET_DATA_PATH: Path = DATA_DIR / "dataset"

    LOG_PATH: Path = DATA_DIR / "log"

    # Модели ML
    MODELS_DIR: Path = BASE_DIR / "models"
    MODELS_CONFIGS_PATH: Path = MODELS_DIR / "configs"
    MODELS_LOGS_PATH: Path = MODELS_DIR / "logs"
    MODEL_PTH_PATH: Path = MODELS_DIR / "models_pth"

class DataManager:

    _settings = SettingsTrade()

    def __init__(self):
        self.required_dirs = {
            "data": self.settings.DATA_DIR,
            "dataset": self.settings.DATASET_DATA_PATH,
            "log": self.settings.LOG_PATH,
            "models": self.settings.MODELS_DIR,
            "models logs": self.settings.MODELS_LOGS_PATH,
            "models configs": self.settings.MODELS_CONFIGS_PATH,
            "models pth": self.settings.MODEL_PTH_PATH,
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
                return pd.read_parquet(path)
            elif format == "json":
                async with aiofiles.open(path, mode="r") as f:
                    return pd.read_json(await f.read())
            else:
                raise ValueError(f"Unsupported format: {format}")
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
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
                data.to_parquet(path)

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
    
data_helper = DataManager()