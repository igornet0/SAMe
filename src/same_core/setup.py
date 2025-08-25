"""
Setup script for same_core module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Версия модуля
__version__ = "1.0.0"

setup(
    name="same-core",
    version=__version__,
    author="SAMe Development Team",
    author_email="dev@same-project.com",
    description="Core interfaces and types for SAMe (Semantic Analog Matching Engine)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/same-project/same-core",
    
    packages=find_packages(),
    
    # Минимальные зависимости - только стандартная библиотека и typing
    install_requires=[
        "typing-extensions>=4.0.0",
        "dataclasses>=0.6; python_version<'3.7'",
    ],
    
    # Дополнительные зависимости для разработки
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
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
    ],
    
    # Требования к Python
    python_requires=">=3.8",
    
    # Включение файлов данных
    include_package_data=True,
    
    # Точки входа (если нужны)
    entry_points={
        "console_scripts": [
            # "same-core=same_core.cli:main",
        ],
    },
    
    # Метаданные для поиска
    keywords="semantic search, text processing, machine learning, nlp, analog matching",
    
    # Информация о проекте
    project_urls={
        "Bug Reports": "https://github.com/same-project/same-core/issues",
        "Source": "https://github.com/same-project/same-core",
        "Documentation": "https://same-core.readthedocs.io/",
    },
    
    # Лицензия
    license="MIT",
    
    # Поддержка zip-safe
    zip_safe=False,
)
