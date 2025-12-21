from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    # НОВЫЕ ПАРАМЕТРЫ:
    top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категориальных признаков."),
    report_title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.5, help="Порог доли пропусков для проблемных колонок."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор с новыми параметрами
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, max_columns=10, top_k=top_k_categories)  # Используем параметр

    # 2. Качество в целом с новым параметром
    quality_flags = compute_quality_flags(summary, missing_df, df)
    
    # 3. Определяем проблемные колонки по пропускам
    problematic_columns = []
    if not missing_df.empty:
        problematic_columns = missing_df[missing_df["missing_share"] > min_missing_share].index.tolist()

    # 4. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт с использованием новых параметров
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {report_title}\n\n")  # Используем параметр заголовка
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")
        
        f.write("## Параметры генерации отчёта\n\n")
        f.write(f"- Top-K категорий: **{top_k_categories}**\n")
        f.write(f"- Порог проблемных пропусков: **{min_missing_share:.0%}**\n")
        f.write(f"- Максимум гистограмм: **{max_hist_columns}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.3f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        
        # Выводим новые эвристики
        f.write(f"- Константные колонки: **{quality_flags['has_constant_columns']}**")
        if quality_flags.get('constant_columns_count', 0) > 0:
            f.write(f" (найдено: {quality_flags['constant_columns_count']})")
        f.write("\n")
        
        f.write(f"- Высокая кардинальность категорий: **{quality_flags['has_high_cardinality_categoricals']}**")
        if quality_flags.get('high_cardinality_count', 0) > 0:
            f.write(f" (найдено: {quality_flags['high_cardinality_count']})")
        f.write("\n")
        
        f.write(f"- Подозрительные дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n")
        f.write(f"- Много нулевых значений: **{quality_flags['has_many_zero_values']}**")
        if quality_flags.get('many_zero_values_count', 0) > 0:
            f.write(f" (найдено: {quality_flags['many_zero_values_count']})")
        f.write("\n\n")

        f.write("## Проблемные колонки\n\n")
        if problematic_columns:
            f.write(f"Колонки с долей пропусков > {min_missing_share:.0%}:\n")
            for col in problematic_columns:
                missing_share = missing_df.loc[col, "missing_share"]
                f.write(f"- `{col}`: {missing_share:.1%}\n")
            f.write("\n")
        else:
            f.write(f"Нет колонок с долей пропусков > {min_missing_share:.0%}.\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"См. файлы в папке `top_categories/` (Top-{top_k_categories} значений).\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"См. файлы `hist_*.png` (не более {max_hist_columns} гистограмм).\n")

    # 6. Картинки с использованием параметра
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    typer.echo(f"\nПараметры отчёта:")
    typer.echo(f"- Top-K категорий: {top_k_categories}")
    typer.echo(f"- Порог проблемных пропусков: {min_missing_share:.0%}")
    typer.echo(f"- Заголовок: {report_title}")


@app.command()
def head(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, help="Количество строк для вывода."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Вывести первые N строк датасета.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    
    if df.empty:
        typer.echo("Файл пуст.")
        return
    
    typer.echo(f"Первые {min(n, len(df))} строк из {len(df)}:")
    typer.echo("-" * 50)
    typer.echo(df.head(n).to_string(index=False))
    typer.echo("-" * 50)

    
if __name__ == "__main__":
    app()
