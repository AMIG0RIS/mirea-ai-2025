from __future__ import annotations
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class QualityDetailResponse(BaseModel):
    """Ответ с детализированной информацией о качестве данных."""
    flags_bool: Dict[str, bool] = Field(
        ...,
        description="Булевы флаги качества (все 4 новые эвристики из HW03)"
    )
    flags_metrics: Dict[str, float] = Field(
        ...,
        description="Числовые метрики качества (counts, shares и т.д.)"
    )
    problematic_columns: Dict[str, List[Union[str, Dict[str, Any]]]] = Field(
        ...,
        description="Конкретные проблемные колонки по каждой эвристике"
    )
    dataset_info: Dict[str, Any] = Field(
        ...,
        description="Базовая информация о датасете"
    )

# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


@app.post(
    "/quality-flags-from-csv",
    response_model=QualityDetailResponse,
    tags=["quality"],
    summary="Детализированные флаги качества с новыми эвристиками из HW03"
)
async def quality_flags_from_csv(
    file: UploadFile = File(...),
    high_cardinality_threshold: int = 50,
    zero_threshold: float = 0.5
) -> QualityDetailResponse:
    """
    Возвращает ДЕТАЛИЗИРОВАННЫЕ флаги качества с использованием всех новых эвристик из HW03.
    
    Основные отличия от /quality-from-csv:
    - Возвращает не только булевы флаги, но и конкретные проблемные колонки
    - Включает дополнительные числовые метрики
    - Позволяет настраивать пороги для проверок
    - Не возвращает интегральную оценку quality_score
    
    Параметры:
    - high_cardinality_threshold: порог уникальных значений для категориальных колонок (по умолчанию 50)
    - zero_threshold: порог доли нулей в числовых колонок (по умолчанию 0.5 = 50%)
    
    Именно этот эндпоинт демонстрирует ВСЕ новые эвристики из HW03 с подробностями.
    """
    
    start = perf_counter()
    
    # Проверка типа файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")
    
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пуст")
    
    # 1. Используем ядро из HW03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # 2. Вызываем compute_quality_flags с df (третьим параметром для новых эвристик)
    flags_all = compute_quality_flags(summary, missing_df, df)
    
    # 3. ВРУЧНУЮ собираем ДЕТАЛИЗИРОВАННУЮ информацию по новым эвристикам
    problematic_columns: Dict[str, List[Union[str, Dict[str, Any]]]] = {
        "constant_columns": [],
        "high_cardinality_columns": [],
        "id_columns_with_duplicates": [],
        "columns_with_many_zeros": []
    }
    
    # Анализируем каждую колонку подробно
    constant_cols_count = 0
    high_cardinality_count = 0
    many_zeros_count = 0
    
    for col_summary in summary.columns:
        col_name = col_summary.name
        
        # Проверка на константные колонки
        if col_summary.unique == 1 and col_summary.missing == 0:
            constant_cols_count += 1
            problematic_columns["constant_columns"].append(col_name)
        
        # Проверка на высокую кардинальность (категориальные)
        if not col_summary.is_numeric and col_summary.unique > high_cardinality_threshold:
            high_cardinality_count += 1
            problematic_columns["high_cardinality_columns"].append({
                "column": col_name,
                "unique_values": col_summary.unique,
                "threshold": high_cardinality_threshold
            })
        
        # Проверка на много нулей (числовые)
        if col_summary.is_numeric and col_summary.non_null > 0:
            try:
                col_data = df[col_name]
                zero_count = (col_data == 0).sum()
                zero_share = zero_count / col_summary.non_null
                if zero_share > zero_threshold:
                    many_zeros_count += 1
                    problematic_columns["columns_with_many_zeros"].append({
                        "column": col_name,
                        "zero_share": round(zero_share, 3),
                        "zero_count": int(zero_count),
                        "threshold": zero_threshold
                    })
            except (KeyError, TypeError):
                continue
    
    # Проверка дубликатов в ID-колонках
    id_duplicates_found = False
    for col_name in df.columns:
        if 'id' in col_name.lower():
            if df[col_name].duplicated().any():
                id_duplicates_found = True
                problematic_columns["id_columns_with_duplicates"].append(col_name)
    
    # 4. Формируем булевы флаги (все 4 новые эвристики из HW03)
    flags_bool = {
        "has_constant_columns": constant_cols_count > 0,
        "has_high_cardinality_categoricals": high_cardinality_count > 0,
        "has_suspicious_id_duplicates": id_duplicates_found,
        "has_many_zero_values": many_zeros_count > 0,
        # Базовые флаги из S03 тоже включаем
        "too_few_rows": summary.n_rows < 100,
        "too_many_columns": summary.n_cols > 100,
        "too_many_missing": not missing_df.empty and missing_df["missing_share"].max() > 0.5
    }
    
    # 5. Формируем числовые метрики
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    
    flags_metrics = {
        "constant_columns_count": constant_cols_count,
        "high_cardinality_count": high_cardinality_count,
        "many_zero_values_count": many_zeros_count,
        "max_missing_share": max_missing_share,
        "total_rows": summary.n_rows,
        "total_columns": summary.n_cols,
        "numeric_columns": sum(1 for col in summary.columns if col.is_numeric),
        "categorical_columns": sum(1 for col in summary.columns if not col.is_numeric),
    }
    
    # 6. Информация о датасете
    dataset_info = {
        "filename": file.filename,
        "parameters_used": {
            "high_cardinality_threshold": high_cardinality_threshold,
            "zero_threshold": zero_threshold
        },
        "total_problematic_columns": sum(len(cols) for cols in problematic_columns.values())
    }
    
    latency_ms = (perf_counter() - start) * 1000.0
    
    # Логирование для отладки
    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"new_flags_found={constant_cols_count + high_cardinality_count + many_zeros_count} "
        f"latency_ms={latency_ms:.1f} ms"
    )
    
    return QualityDetailResponse(
        flags_bool=flags_bool,
        flags_metrics=flags_metrics,
        problematic_columns=problematic_columns,
        dataset_info=dataset_info
    )