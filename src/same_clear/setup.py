"""
Setup script for same_clear module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Версия модуля
__version__ = "1.0.0"

setup(
    name="same-clear",
    version=__version__,
    author="SAMe Development Team",
    author_email="dev@same-project.com",
    description="Text processing and parameter extraction module for SAMe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/same-project/same-clear",
    
    packages=find_packages(),
    
    # Основные зависимости
    install_requires=[
        "same-core>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "spacy>=3.4.0",
        "regex>=2022.1.18",
        "beautifulsoup4>=4.10.0",
        "lxml>=4.6.0",
        "nltk>=3.7",
    ],
    
    # Дополнительные зависимости
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "enhanced": [
            "pymorphy2>=0.9.1",
            "pymorphy2-dicts-ru>=2.4.0",
            "langdetect>=1.0.9",
        ],
        "performance": [
            "numba>=0.56.0",
            "cython>=0.29.0",
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
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Text Processing :: Filters",
    ],
    
    # Требования к Python
    python_requires=">=3.8",
    
    # Включение файлов данных
    include_package_data=True,
    package_data={
        "same_clear": [
            "data/*.json",
            "data/*.txt",
            "data/dictionaries/*.json",
            "data/patterns/*.json",
        ],
    },
    
    # Точки входа
    entry_points={
        "console_scripts": [
            "same-clean=same_clear.cli:clean_command",
            "same-extract=same_clear.cli:extract_command",
        ],
    },
    
    # Метаданные для поиска
    keywords="text processing, text cleaning, parameter extraction, nlp, preprocessing",
    
    # Информация о проекте
    project_urls={
        "Bug Reports": "https://github.com/same-project/same-clear/issues",
        "Source": "https://github.com/same-project/same-clear",
        "Documentation": "https://same-clear.readthedocs.io/",
    },
    
    # Лицензия
    license="MIT",
    
    # Поддержка zip-safe
    zip_safe=False,
    
    # Пост-установочные скрипты
    # Для загрузки моделей spaCy
    # cmdclass={
    #     'install': PostInstallCommand,
    # },
)
