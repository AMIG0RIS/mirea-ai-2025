from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    # Передаем df как третий параметр
    flags = compute_quality_flags(summary, missing_df, df)

    assert 0.0 <= flags["quality_score"] <= 1.0
    # Проверяем, что новые флаги существуют
    assert "has_constant_columns" in flags
    assert "has_high_cardinality_categoricals" in flags
    assert "has_suspicious_id_duplicates" in flags
    assert "has_many_zero_values" in flags


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_new_heuristics():
    """Тест новых эвристик качества данных."""
    
    # Тест 1: Константная колонка
    df_const = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [5, 5, 5, 5],  # Все значения одинаковые
        "normal_col": [1, 2, 3, 4]
    })
    
    summary = summarize_dataset(df_const)
    missing_df = missing_table(df_const)
    flags = compute_quality_flags(summary, missing_df, df_const)
    
    assert flags["has_constant_columns"] is True
    assert flags["constant_columns_count"] == 1
    
    # Тест 2: Дубликаты ID
    df_duplicates = pd.DataFrame({
        "user_id": [1, 2, 3, 1],  # Дубликат ID
        "name": ["A", "B", "C", "D"]
    })
    
    summary = summarize_dataset(df_duplicates)
    missing_df = missing_table(df_duplicates)
    flags = compute_quality_flags(summary, missing_df, df_duplicates)
    
    assert flags["has_suspicious_id_duplicates"] is True
    
    # Тест 3: Много нулей
    df_zeros = pd.DataFrame({
        "col1": [0, 0, 0, 10],  # 75% нулей
        "col2": [1, 2, 3, 4]
    })
    
    summary = summarize_dataset(df_zeros)
    missing_df = missing_table(df_zeros)
    flags = compute_quality_flags(summary, missing_df, df_zeros)
    
    assert flags["has_many_zero_values"] is True
    assert flags["many_zero_values_count"] >= 1
    
    # Тест 4: Высокая кардинальность
    df_high_card = pd.DataFrame({
        "category": [f"cat_{i}" for i in range(60)],  # 60 уникальных значений
        "value": list(range(60))
    })
    
    summary = summarize_dataset(df_high_card)
    missing_df = missing_table(df_high_card)
    flags = compute_quality_flags(summary, missing_df, df_high_card)
    
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["high_cardinality_count"] >= 1


def test_quality_score_with_new_heuristics():
    """Тест корректности скоринга с новыми эвристиками."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 1],  # Дубликат
        "constant": [10, 10, 10, 10],  # Константа
        "zeros": [0, 0, 0, 5],  # Много нулей
        "normal": [1, 2, 3, 4]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что все флаги установлены правильно
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is True
    assert flags["has_many_zero_values"] is True
    
    # Проверяем, что quality_score в диапазоне 0-1
    assert 0.0 <= flags["quality_score"] <= 1.0
    # Проверяем, что штрафы применились (скоринг должен быть меньше 1)
    assert flags["quality_score"] < 1.0