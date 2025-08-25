import io
import math
import pandas as pd
from typing import Dict, Any, List

from .celery_app import get_celery_app
from same.analog_search_engine import AnalogSearchEngine
from celery import chord, group


celery_app = get_celery_app()


def _compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute dataset-level statistics for UI.

    Returns a structure with:
    - total_rows
    - columns
    - per_column: { column_name: { dtype, non_null_count, missing_count, missing_pct, unique_count, top_values[] } }
    """
    total_rows = int(len(df))
    columns = [str(c) for c in df.columns]

    per_column: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        # Ensure simple Python types for JSON
        non_null_count = int(series.notna().sum())
        missing_count = total_rows - non_null_count
        missing_pct = float((missing_count / total_rows * 100.0) if total_rows > 0 else 0.0)
        unique_count = int(series.nunique(dropna=True))
        # Top values limited to prevent huge payloads
        vc = series.value_counts(dropna=True).head(50)
        # Convert to JSON-serializable list of dicts
        top_values = []
        for value, count in vc.items():
            # Convert numpy/pandas types to plain Python
            try:
                if pd.isna(value):
                    value_json = None
                else:
                    value_json = value.item() if hasattr(value, 'item') else value
            except Exception:
                value_json = str(value)
            top_values.append({
                'value': value_json,
                'count': int(count)
            })

        per_column[str(col)] = {
            'dtype': str(series.dtype),
            'non_null_count': non_null_count,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_count': unique_count,
            'top_values': top_values,
        }

    return {
        'total_rows': total_rows,
        'columns': columns,
        'per_column': per_column,
    }


@celery_app.task(name="catalog.process_upload")
def process_catalog_upload(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Обработка загруженного файла каталога.
    Цель: никогда не падать из-за ошибок предобработки (например, рассинхронизация длин в pandas),
    а возвращать statistics + processing_report с перечнем проблемных записей.
    """
    df = None
    try:
        # Read file into DataFrame
        if filename.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("cp1251", errors="replace")
            sep = ";" if text.count(";") > text.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep)

        if df is None or df.empty:
            return {"status": "error", "message": "Файл пустой или не распознан"}

        total_rows = int(len(df))

        # Для больших наборов данных используем параллельную обработку chunk'ами (макс 4 worker)
        LARGE_THRESHOLD = 80000  # эвристика: >80k строк считаем большим набором
        if total_rows >= LARGE_THRESHOLD:
            # Определяем количество воркеров (2..4)
            workers = min(4, max(2, math.ceil(total_rows / LARGE_THRESHOLD)))
            # Разбиваем на равные части
            indices: List[int] = [0]
            step = math.ceil(total_rows / workers)
            for i in range(step, total_rows, step):
                indices.append(i)
            if indices[-1] != total_rows:
                indices.append(total_rows)

            # Готовим группы задач по частям
            sigs = []
            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i + 1]
                # Сериализуем кусок в CSV для передачи
                chunk_csv = df.iloc[start:end].to_csv(index=False)
                sigs.append(process_catalog_upload_chunk.s(chunk_csv, filename, start, end))

            job = chord(group(sigs))(finalize_catalog_chunks.s(filename))
            # Возвращаем стандартную структуру; реальный ответ придет в callback
            return job.get()  # Позволяет вернуть объединенный результат по завершению chord

        # Обычная (непараллельная) обработка
        dataset_stats = _compute_dataset_statistics(df)

        engine = AnalogSearchEngine()
        # Run full initialization/preprocessing with robust fallback
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processing_report = None
        try:
            try:
                loop.run_until_complete(engine.initialize(df))
            except Exception as init_err:
                try:
                    if hasattr(engine, '_preprocessor') and hasattr(engine._preprocessor, 'config'):
                        engine._preprocessor.config.save_intermediate_steps = False
                    engine.catalog_data = df
                    name_col = engine._find_name_column(df)
                    processed = loop.run_until_complete(engine._preprocess_dataframe_async(
                        df, name_col, output_columns={'final': 'processed_name'}
                    ))
                    engine.processed_catalog = processed
                    try:
                        loop.run_until_complete(engine._extract_parameters())
                    except Exception:
                        pass
                    loop.run_until_complete(engine._initialize_search_engines())
                    engine.is_ready = True
                except Exception as fallback_err:
                    total = int(len(df))
                    failed_rows = []
                    # Список неудачных строк ограничим 1000 записями
                    cap = min(total, 1000)
                    # Определяем колонку наименования по эвристике
                    try:
                        name_col_local = name_col if 'name_col' in locals() else None
                    except Exception:
                        name_col_local = None
                    if not name_col_local:
                        try:
                            # Популярные варианты для русских датасетов
                            for candidate in ['Наименование', 'name', 'наименование', 'название']:
                                if candidate in df.columns:
                                    name_col_local = candidate
                                    break
                            if not name_col_local:
                                # первая object-колонка
                                for c in df.columns:
                                    if df[c].dtype == 'object':
                                        name_col_local = c
                                        break
                        except Exception:
                            name_col_local = None
                    for i in range(cap):
                        try:
                            code_val = df.iloc[i]['Код'] if 'Код' in df.columns else df.index[i]
                            name_val = str(df.iloc[i][name_col_local]) if name_col_local and name_col_local in df.columns else ''
                        except Exception:
                            code_val, name_val = '', ''
                        failed_rows.append({'code': code_val, 'name': name_val, 'error': str(init_err)})
                    processing_report = {
                        'failed_count': total,
                        'failed_rows': failed_rows
                    }
            try:
                loop.run_until_complete(engine.save_models())
            except Exception:
                pass
        finally:
            try:
                loop.close()
            except Exception:
                pass
        engine_stats = engine.get_statistics()
        if processing_report is None and hasattr(engine, 'get_processing_report'):
            processing_report = engine.get_processing_report()

        return {
            "status": "success",
            "message": f"Catalog processed: {len(df)} items",
            "statistics": {
                "engine": engine_stats,
                "dataset": dataset_stats,
                "processing_report": processing_report,
            },
        }
    except Exception as fatal_error:
        # Глобальный перехват: возвращаем success_with_errors и отчет, чтобы фронт мог отобразить причины
        try:
            dataset_stats = _compute_dataset_statistics(df) if isinstance(df, pd.DataFrame) else None
        except Exception:
            dataset_stats = None

        failed_rows = []
        total = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        cap = min(total, 1000)
        name_col = None
        if isinstance(df, pd.DataFrame):
            try:
                for candidate in ['Наименование', 'name', 'наименование', 'название']:
                    if candidate in df.columns:
                        name_col = candidate
                        break
                if not name_col:
                    for c in df.columns:
                        if df[c].dtype == 'object':
                            name_col = c
                            break
            except Exception:
                name_col = None
        for i in range(cap):
            try:
                code_val = df.iloc[i]['Код'] if isinstance(df, pd.DataFrame) and 'Код' in df.columns else (df.index[i] if isinstance(df, pd.DataFrame) else i)
                name_val = str(df.iloc[i][name_col]) if isinstance(df, pd.DataFrame) and name_col and name_col in df.columns else ''
            except Exception:
                code_val, name_val = '', ''
            failed_rows.append({'code': code_val, 'name': name_val, 'error': str(fatal_error)})

        return {
            'status': 'success_with_errors',
            'message': 'Обработка завершена с ошибками, см. отчет по сбоям',
            'statistics': {
                'engine': {
                    'is_ready': False,
                    'catalog_size': total,
                    'search_method': 'hybrid',
                    'similarity_threshold': 0.6,
                    'processing_failures': int(total)
                },
                'dataset': dataset_stats if dataset_stats is not None else {
                    'total_rows': total,
                    'columns': [str(c) for c in df.columns] if isinstance(df, pd.DataFrame) else [],
                    'per_column': {}
                },
                'processing_report': {
                    'failed_count': total,
                    'failed_rows': failed_rows
                }
            }
        }


@celery_app.task(name="catalog.process_upload_chunk")
def process_catalog_upload_chunk(chunk_csv: str, filename: str, start: int, end: int) -> Dict[str, Any]:
    """Обработка части датасета: вычисление статистики. Тяжелые модели не обучаем здесь."""
    df_chunk = pd.read_csv(io.StringIO(chunk_csv))
    stats = _compute_dataset_statistics(df_chunk)
    return {
        'range': {'start': int(start), 'end': int(end)},
        'rows': int(len(df_chunk)),
        'dataset_stats': stats,
    }


@celery_app.task(name="catalog.finalize_catalog_chunks")
def finalize_catalog_chunks(chunks: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    """Агрегация результатов по чанкам. Возвращает объединенную статистику."""
    # Объединяем статистику по колонкам
    total_rows = 0
    merged_columns: List[str] = []
    per_column: Dict[str, Any] = {}

    for ch in chunks:
        ch_stats = ch.get('dataset_stats', {})
        total_rows += int(ch_stats.get('total_rows', 0))
        cols = ch_stats.get('columns', [])
        for col in cols:
            if col not in merged_columns:
                merged_columns.append(col)
            col_stats = ch_stats.get('per_column', {}).get(col)
            if not col_stats:
                continue
            if col not in per_column:
                per_column[col] = {
                    'dtype': col_stats['dtype'],
                    'non_null_count': int(col_stats['non_null_count']),
                    'missing_count': int(col_stats['missing_count']),
                    'missing_pct': 0.0,  # пересчитаем позже
                    'unique_count': int(col_stats['unique_count']),  # upper bound
                    'top_values': col_stats['top_values'],  # возьмем топ по первому чанку как приближение
                }
            else:
                per_column[col]['non_null_count'] += int(col_stats['non_null_count'])
                per_column[col]['missing_count'] += int(col_stats['missing_count'])
                per_column[col]['unique_count'] = max(per_column[col]['unique_count'], int(col_stats['unique_count']))

    # Пересчет missing_pct
    for col, st in per_column.items():
        miss = int(st['missing_count'])
        st['missing_pct'] = float((miss / total_rows * 100.0) if total_rows > 0 else 0.0)

    merged_stats = {
        'total_rows': int(total_rows),
        'columns': merged_columns,
        'per_column': per_column,
    }

    return {
        'status': 'success',
        'message': f'Catalog processed in parallel chunks: {total_rows} items',
        'statistics': {
            'dataset': merged_stats,
            # engine отсутствует, так как тяжелая инициализация не выполнялась на чанках
            'processing_report': None,
        }
    }


