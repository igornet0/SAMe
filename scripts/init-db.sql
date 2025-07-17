-- Инициализация базы данных для SAMe системы

-- Создание расширений
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Создание схем
CREATE SCHEMA IF NOT EXISTS same_core;
CREATE SCHEMA IF NOT EXISTS same_search;
CREATE SCHEMA IF NOT EXISTS same_analytics;

-- Настройка поиска по тексту для русского языка
CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS russian_unaccent (COPY = russian);
ALTER TEXT SEARCH CONFIGURATION russian_unaccent
    ALTER MAPPING FOR word, asciiword WITH unaccent, russian_stem;

-- Создание пользователей и ролей
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'same_app') THEN
        CREATE ROLE same_app LOGIN PASSWORD 'same_app_password';
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'same_readonly') THEN
        CREATE ROLE same_readonly LOGIN PASSWORD 'same_readonly_password';
    END IF;
END
$$;

-- Предоставление прав
GRANT USAGE ON SCHEMA same_core TO same_app, same_readonly;
GRANT USAGE ON SCHEMA same_search TO same_app, same_readonly;
GRANT USAGE ON SCHEMA same_analytics TO same_app, same_readonly;

GRANT ALL PRIVILEGES ON SCHEMA same_core TO same_app;
GRANT ALL PRIVILEGES ON SCHEMA same_search TO same_app;
GRANT ALL PRIVILEGES ON SCHEMA same_analytics TO same_app;

GRANT SELECT ON ALL TABLES IN SCHEMA same_core TO same_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA same_search TO same_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA same_analytics TO same_readonly;

-- Настройка по умолчанию для новых таблиц
ALTER DEFAULT PRIVILEGES IN SCHEMA same_core GRANT SELECT ON TABLES TO same_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA same_search GRANT SELECT ON TABLES TO same_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA same_analytics GRANT SELECT ON TABLES TO same_readonly;

-- Создание функций для аудита
CREATE OR REPLACE FUNCTION same_core.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Создание функции для полнотекстового поиска
CREATE OR REPLACE FUNCTION same_search.create_search_vector(text_data TEXT)
RETURNS tsvector AS $$
BEGIN
    RETURN to_tsvector('russian_unaccent', COALESCE(text_data, ''));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Настройка логирования
SET log_statement = 'mod';
SET log_min_duration_statement = 1000;

-- Создание индексов для производительности
-- (будут созданы автоматически через Alembic миграции)

COMMENT ON DATABASE same_db IS 'SAMe - Search Analog Model Engine Database';
COMMENT ON SCHEMA same_core IS 'Основные таблицы системы';
COMMENT ON SCHEMA same_search IS 'Таблицы для поискового движка';
COMMENT ON SCHEMA same_analytics IS 'Таблицы для аналитики и метрик';
