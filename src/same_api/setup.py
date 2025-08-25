"""
Setup script for same_api module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Версия модуля
__version__ = "1.0.0"

setup(
    name="same-api",
    version=__version__,
    author="SAMe Development Team",
    author_email="dev@same-project.com",
    description="API, database and export module for SAMe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/same-project/same-api",
    
    packages=find_packages(),
    
    # Основные зависимости
    install_requires=[
        "same-core>=1.0.0",
        "same-clear>=1.0.0",
        "same-search>=1.0.0",
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.20.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.10.0",
        "asyncpg>=0.27.0",  # PostgreSQL async driver
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.0",
    ],
    
    # Дополнительные зависимости
    extras_require={
        "postgresql": [
            "psycopg2-binary>=2.9.0",
            "asyncpg>=0.27.0",
        ],
        "mysql": [
            "pymysql>=1.0.0",
            "aiomysql>=0.1.1",
        ],
        "sqlite": [
            "aiosqlite>=0.18.0",
        ],
        "redis": [
            "redis>=4.0.0",
            "aioredis>=2.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
            "structlog>=22.0.0",
        ],
        "security": [
            "cryptography>=40.0.0",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
        ],
        "websocket": [
            "websockets>=11.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "httpx>=0.24.0",  # Для тестирования FastAPI
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
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Database",
        "Topic :: Office/Business",
        "Framework :: FastAPI",
    ],
    
    # Требования к Python
    python_requires=">=3.8",
    
    # Включение файлов данных
    include_package_data=True,
    package_data={
        "same_api": [
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
            "migrations/*.py",
            "migrations/versions/*.py",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    
    # Точки входа
    entry_points={
        "console_scripts": [
            "same-api=same_api.cli:api_command",
            "same-server=same_api.cli:server_command",
            "same-migrate=same_api.cli:migrate_command",
            "same-export=same_api.cli:export_command",
        ],
    },
    
    # Метаданные для поиска
    keywords="api, fastapi, database, export, web service, rest api, sqlalchemy",
    
    # Информация о проекте
    project_urls={
        "Bug Reports": "https://github.com/same-project/same-api/issues",
        "Source": "https://github.com/same-project/same-api",
        "Documentation": "https://same-api.readthedocs.io/",
        "API Docs": "https://api.same-project.com/docs",
    },
    
    # Лицензия
    license="MIT",
    
    # Поддержка zip-safe
    zip_safe=False,
    
    # Дополнительные файлы для включения
    data_files=[
        ('config', ['config/default.yaml']),
    ] if Path('config/default.yaml').exists() else [],
)
