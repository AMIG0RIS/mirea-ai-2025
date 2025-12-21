# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта `eda-cli` из Семинара 03.

К существующему CLI-приложению для EDA добавлен **HTTP-сервис на FastAPI** с эндпоинтами `/health`, `/quality` и `/quality-from-csv`.  
Используется в рамках Семинара 04 курса «Инженерия ИИ».

---

## Связь с S03

Проект в S04 основан на том же пакете `eda_cli`, что и в S03:

- сохраняется структура `src/eda_cli/` и CLI-команда `eda-cli`;
- добавлен модуль `api.py` с FastAPI-приложением;
- в зависимости добавлены `fastapi` и `uvicorn[standard]`.

Цель S04 – показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис.

---

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему
- Браузер (для Swagger UI `/docs`) или любой HTTP-клиент:
  - `curl` / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

---

## Инициализация проекта

В корне проекта (каталог S04/eda-cli):

```bash
uv sync
```

Команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml` (включая FastAPI и Uvicorn);
- установит сам проект `eda-cli` в окружение.

---

## Запуск CLI (как в S03)

CLI остаётся доступным и в S04.

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` - разделитель (по умолчанию `,`);
- `--encoding` - кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` - основной отчёт в Markdown;
- `summary.csv` - таблица по колонкам;
- `missing.csv` - пропуски по колонкам;
- `correlation.csv` - корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` - top-k категорий по строковым признакам;
- `hist_*.png` - гистограммы числовых колонок;
- `missing_matrix.png` - визуализация пропусков;
- `correlation_heatmap.png` - тепловая карта корреляций.

---

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Пояснения:

- `eda_cli.api:app` - путь до объекта FastAPI `app` в модуле `eda_cli.api`;
- `--reload` - автоматический перезапуск сервера при изменении кода (удобно для разработки);
- `--port 8000` - порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

---

## Эндпоинты сервиса

### 1. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /health`;
- вызывать `POST /quality` (форма для JSON);
- вызывать `POST /quality-from-csv` (форма для загрузки файла).

---

### 3. `POST /quality` – заглушка по агрегированным признакам

Эндпоинт принимает **агрегированные признаки датасета** (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

**Пример запроса:**

```http
POST /quality
Content-Type: application/json
```

Тело:

```json
{
  "n_rows": 10000,
  "n_cols": 12,
  "max_missing_share": 0.15,
  "numeric_cols": 8,
  "categorical_cols": 4
}
```

**Пример ответа `200 OK`:**

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 3.2,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 10000,
    "n_cols": 12
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

---

### 4. `POST /quality-from-csv` – оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл, внутри:

- читает его в `pandas.DataFrame`;
- вызывает функции из `eda_cli.core`:

  - `summarize_dataset`,
  - `missing_table`,
  - `compute_quality_flags`;
- возвращает оценку качества датасета в том же формате, что `/quality`.

**Запрос:**

```http
POST /quality-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в `/docs` открыть `POST /quality-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

Ответ будет содержать:

- `ok_for_model` - результат по эвристикам;
- `quality_score` - интегральный скор качества;
- `flags` - булевы флаги из `compute_quality_flags`;
- `dataset_shape` - реальные размеры датасета (`n_rows`, `n_cols`);
- `latency_ms` - время обработки запроса.

---


### 5. `POST /quality-flags-from-csv` – Детализированные флаги качества с новыми эвристиками

**Новый эндпоинт, добавленный в рамках HW04.**

Возвращает детализированную информацию о качестве данных, используя все новые эвристики из HW03. Не возвращает интегральную оценку `quality_score`, вместо этого предоставляет подробную диагностику с конкретными проблемными колонками.

**Параметры запроса (query parameters):**
- `high_cardinality_threshold` (int, по умолчанию 50) – порог уникальных значений для категориальных колонок
- `zero_threshold` (float, по умолчанию 0.5) – порог доли нулей в числовых колонках

**Запрос через curl:**
```bash
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv?high_cardinality_threshold=50&zero_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.csv;type=text/csv"
```

**Пример ответа 200 OK:**
```json
{
  "flags_bool": {
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_suspicious_id_duplicates": true,
    "has_many_zero_values": true,
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false
  },
  "flags_metrics": {
    "constant_columns_count": 0,
    "high_cardinality_count": 0,
    "many_zero_values_count": 1,
    "max_missing_share": 0.056,
    "total_rows": 36,
    "total_columns": 14,
    "numeric_columns": 9,
    "categorical_columns": 5
  },
  "problematic_columns": {
    "constant_columns": [],
    "high_cardinality_columns": [],
    "id_columns_with_duplicates": ["user_id"],
    "columns_with_many_zeros": [
      {
        "column": "churned",
        "zero_share": 0.667,
        "zero_count": 24,
        "threshold": 0.5
      }
    ]
  },
  "dataset_info": {
    "filename": "example.csv",
    "parameters_used": {
      "high_cardinality_threshold": 50,
      "zero_threshold": 0.5
    },
    "total_problematic_columns": 2
  }
}
```

**Ключевые поля ответа:**
- `flags_bool` – булевы флаги всех эвристик (включая новые из HW03)
- `flags_metrics` – числовые показатели качества
- `problematic_columns` – конкретные проблемные колонки по каждой эвристике
- `dataset_info` – базовая информация о датасете и использованных параметрах

**Новые эвристики из HW03, отображаемые в ответе:**
- `has_constant_columns` – есть ли колонки, где все значения одинаковые
- `has_high_cardinality_categoricals` – категориальные признаки с очень большим числом уникальных значений
- `has_suspicious_id_duplicates` – дубликаты в колонках, содержащих "id" в названии
- `has_many_zero_values` – числовые колонки с высокой долей нулевых значений

**Отличие от `/quality-from-csv`:**
- Не возвращает интегральную оценку `quality_score`
- Не возвращает итоговое решение `ok_for_model`
- Предоставляет детализированную диагностику вместо общей оценки
- Позволяет настраивать пороги срабатывания эвристик через параметры запроса

---

## Структура проекта (упрощённо)

```text
S04/
  eda-cli/
    pyproject.toml
    README.md                # этот файл
    src/
      eda_cli/
        __init__.py
        core.py              # EDA-логика, эвристики качества
        viz.py               # визуализации
        cli.py               # CLI (overview/report)
        api.py               # HTTP-сервис (FastAPI)
    tests/
      test_core.py           # тесты ядра
    data/
      example.csv            # учебный CSV для экспериментов
```

---

## Тесты

Запуск тестов (как и в S03):

```bash
uv run pytest -q
```

Рекомендуется перед любыми изменениями в логике качества данных и API:

1. Запустить тесты `pytest`;
2. Проверить работу CLI (`uv run eda-cli ...`);
3. Проверить работу HTTP-сервиса (`uv run uvicorn ...`, затем `/health` и `/quality`/`/quality-from-csv` через `/docs` или HTTP-клиент).



