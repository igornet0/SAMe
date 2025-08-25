"""
Setup script for same_search module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Версия модуля
__version__ = "1.0.0"

setup(
    name="same-search",
    version=__version__,
    author="SAMe Development Team",
    author_email="dev@same-project.com",
    description="Search engine and ML models module for SAMe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/same-project/same-search",
    
    packages=find_packages(),
    
    # Основные зависимости
    install_requires=[
        "same-core>=1.0.0",
        "same-clear>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.12.0",
        "faiss-cpu>=1.7.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
    ],
    
    # Дополнительные зависимости
    extras_require={
        "gpu": [
            "faiss-gpu>=1.7.0",
            "torch>=1.12.0+cu116",
        ],
        "advanced": [
            "elasticsearch>=8.0.0",
            "redis>=4.0.0",
            "pymongo>=4.0.0",
        ],
        "monitoring": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
            "mlflow>=1.28.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "performance": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
            "openvino>=2022.3.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ]
    },
    
    # Классификаторы
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
    ],
    
    # Требования к Python
    python_requires=">=3.8",
    
    # Включение файлов данных
    include_package_data=True,
    package_data={
        "same_search": [
            "models/configs/*.json",
            "models/pretrained/*.bin",
            "data/embeddings/*.npy",
            "data/indices/*.faiss",
        ],
    },
    
    # Точки входа
    entry_points={
        "console_scripts": [
            "same-search=same_search.cli:search_command",
            "same-index=same_search.cli:index_command",
            "same-model=same_search.cli:model_command",
        ],
    },
    
    # Метаданные для поиска
    keywords="semantic search, machine learning, embeddings, faiss, transformers, information retrieval",
    
    # Информация о проекте
    project_urls={
        "Bug Reports": "https://github.com/same-project/same-search/issues",
        "Source": "https://github.com/same-project/same-search",
        "Documentation": "https://same-search.readthedocs.io/",
        "Model Hub": "https://huggingface.co/same-project",
    },
    
    # Лицензия
    license="MIT",
    
    # Поддержка zip-safe
    zip_safe=False,
    
    # Системные требования
    # install_requires дополнительно может включать системные зависимости
    # для FAISS, PyTorch и других библиотек
)
