#!/usr/bin/env python3
"""Подготовить данные и последовательно выполнить неизменяемые шаблоны тетрадок."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import nbformat
import polars as pl
import yaml
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "notebooks"
RUNS_DIR = ROOT / "artifacts" / "runs"
NOTEBOOKS = [
    "1_prepare_dataset.ipynb",
    "2_create_train_test_datasets_V2_MAIN.ipynb",
    "3_feature_selection.ipynb",
    "4_tuning.ipynb",
    "5_exact_optuna_best_full_test_batched.ipynb",
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def require_columns(path: Path, required: set[str], label: str) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"{label}: отсутствуют колонки {sorted(missing)} в {path}")


def stage_events(event_files: list[Path], data_dir: Path) -> list[int]:
    """Собрать все новые parquet-файлы в ext-journal-<год>.parquet для шаблона."""
    required = {"ид_события", "ид_канала_данных", "дата", "время", "значение_датчика", "тревожное"}
    for file in event_files:
        require_columns(file, required, "Журнал событий")

    events = pl.scan_parquet(event_files).with_columns(
        pl.col("дата").cast(pl.Date, strict=False).alias("дата")
    )
    years = (
        events.select(pl.col("дата").dt.year().alias("year"))
        .drop_nulls()
        .unique()
        .sort("year")
        .collect()["year"]
        .to_list()
    )
    if not years:
        raise ValueError("В новых журналах не найдено корректных значений в колонке «дата».")

    for year in years:
        events.filter(pl.col("дата").dt.year() == year).sink_parquet(
            data_dir / f"ext-journal-{year}.parquet"
        )
    return [int(year) for year in years]


def copy_reference_data(config: dict, data_dir: Path) -> None:
    input_config = config["input"]
    references = {
        "справочник_каналов_датчиков.parquet": input_config["sensors_file"],
        "справочник_объектов_диспетчер.parquet": input_config["territory_file"],
        "open-meteo-55.75N37.63E140m.csv": input_config["weather_file"],
    }
    for destination, source_value in references.items():
        source = resolve(ROOT, source_value)
        if not source.is_file():
            raise FileNotFoundError(f"Не найден обязательный справочник: {source}")
        shutil.copy2(source, data_dir / destination)


def patch_notebook(source: Path, destination: Path, replacements: dict[str, str]) -> None:
    """Создать рабочую копию, не изменяя шаблон в notebooks/."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        for old, new in replacements.items():
            code = code.replace(old, new)
        # В исходной тетрадке есть случайный символ, делающий ячейку невалидной Python.
        code = code.replace("import polars as plА", "import polars as pl")
        cell["source"] = code.splitlines(keepends=True)
        cell["execution_count"] = None
        cell["outputs"] = []
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def execute_notebook(notebook: Path, run_dir: Path, kernel: str, timeout: int) -> None:
    print(f"\n>>> Выполняется {notebook.name}")
    node = nbformat.read(notebook, as_version=4)
    preprocessor = ExecutePreprocessor(
        kernel_name=kernel,
        timeout=None if timeout == 0 else timeout,
    )
    try:
        preprocessor.preprocess(node, {"metadata": {"path": str(run_dir)}})
    finally:
        nbformat.write(node, notebook)


def chunk_count(run_dir: Path, split: str) -> int:
    return len(list((run_dir / "data" / f"final_{split}").glob("chunk_*.parquet")))


def chunks(limit: int | str, available: int, split: str) -> list[int]:
    if available == 0:
        raise RuntimeError(f"После разбиения не найдено чанков {split}.")
    if limit == "all":
        return list(range(available))
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"Неверный лимит чанков {split}: {limit!r}")
    return list(range(min(limit, available)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный пайплайн обучения модели")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--run-id", help="Имя каталога прогона; по умолчанию текущие дата и время")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        raise FileExistsError(f"Прогон уже существует: {run_dir}. Укажите другой --run-id.")

    incoming_dir = resolve(ROOT, config["input"]["events_dir"])
    event_files = sorted(incoming_dir.glob(config["input"]["events_glob"]))
    if not event_files:
        raise FileNotFoundError(f"В {incoming_dir} нет файлов {config['input']['events_glob']}")
    if len(event_files) > 1 and any(file.name == "журнал_событий_пример.parquet" for file in event_files):
        raise ValueError("Исключите журнал_событий_пример.parquet из полного прогона через events_dir или events_glob")

    run_dir.mkdir(parents=True)
    data_dir = run_dir / "data"
    data_dir.mkdir()
    copy_reference_data(config, data_dir)
    years = stage_events(event_files, data_dir)
    print(f"Добавлено файлов журналов: {len(event_files)}; годы: {years}")

    notebooks_dir = run_dir / "notebooks"
    notebooks_dir.mkdir()
    first_replacements = {"for year in [2024, 2025, 2026]": f"for year in {years}"}
    patch_notebook(TEMPLATES_DIR / NOTEBOOKS[0], notebooks_dir / NOTEBOOKS[0], first_replacements)
    kernel = config["execution"]["jupyter_kernel"]
    timeout = int(config["execution"]["timeout_seconds"])
    execute_notebook(notebooks_dir / NOTEBOOKS[0], run_dir, kernel, timeout)

    patch_notebook(TEMPLATES_DIR / NOTEBOOKS[1], notebooks_dir / NOTEBOOKS[1], {})
    execute_notebook(notebooks_dir / NOTEBOOKS[1], run_dir, kernel, timeout)

    limits = config["notebook_limits"]
    available = {split: chunk_count(run_dir, split) for split in ("train", "val", "test")}
    fs = limits["feature_selection"]
    fs_replacements = {
        "TRAIN_CHUNKS = [*range(4)]": f"TRAIN_CHUNKS = {chunks(fs['train_chunks'], available['train'], 'train')}",
        "VAL_CHUNKS = [*range(20)]": f"VAL_CHUNKS = {chunks(fs['val_chunks'], available['val'], 'val')}",
        "TEST_CHUNKS = [*range(20)]": f"TEST_CHUNKS = {chunks(fs['test_chunks'], available['test'], 'test')}",
    }
    patch_notebook(TEMPLATES_DIR / NOTEBOOKS[2], notebooks_dir / NOTEBOOKS[2], fs_replacements)
    execute_notebook(notebooks_dir / NOTEBOOKS[2], run_dir, kernel, timeout)

    tuning = limits["tuning"]
    tuning_replacements = {
        "TRAIN_CHUNKS = [*range(8)]": f"TRAIN_CHUNKS = {chunks(tuning['train_chunks'], available['train'], 'train')}",
        "VAL_CHUNKS = [*range(80)]": f"VAL_CHUNKS = {chunks(tuning['val_chunks'], available['val'], 'val')}",
        "TEST_CHUNKS = [*range(20)]": f"TEST_CHUNKS = {chunks(tuning['test_chunks'], available['test'], 'test')}",
    }
    patch_notebook(TEMPLATES_DIR / NOTEBOOKS[3], notebooks_dir / NOTEBOOKS[3], tuning_replacements)
    execute_notebook(notebooks_dir / NOTEBOOKS[3], run_dir, kernel, timeout)

    final = limits.get("final_training", tuning)
    shutil.copy2(ROOT / "disk_pool.py", run_dir / "disk_pool.py")
    full_pool = '''train_pool, y_train_binary, n_train = make_pool(
    "train",
    TRAIN_CHUNKS,
)

val_pool, y_val_binary, n_val = make_pool(
    "val",
    VAL_CHUNKS,
)'''
    disk_pool = '''from disk_pool import binary_target, quantized_pool

quantized_dir = Path("data/quantized")
train_frame = scan_split("train", TRAIN_CHUNKS)
val_frame = scan_split("val", VAL_CHUNKS)
train_pool = quantized_pool(
    train_frame, selected_features, TARGET_COLS, quantized_dir, "train"
)
n_train = train_pool.num_row()
borders_path = quantized_dir / "borders.txt"
train_pool.save_quantization_borders(str(borders_path))
val_pool = quantized_pool(
    val_frame, selected_features, TARGET_COLS, quantized_dir, "val", borders_path
)
n_val = val_pool.num_row()
y_val_binary = binary_target(val_frame, HORIZON)'''
    final_replacements = {
        "TRAIN_CHUNKS = [*range(8)]": f"TRAIN_CHUNKS = {chunks(final['train_chunks'], available['train'], 'train')}",
        "VAL_CHUNKS = [*range(80)]": f"VAL_CHUNKS = {chunks(final['val_chunks'], available['val'], 'val')}",
        "# Те же чанки, что были в 4_tuning.ipynb": "# Финальное обучение использует лимиты из config.yaml",
        full_pool: disk_pool,
        'print(f"Model saved to: {BEST_MODEL_PATH}")': 'print(f"Model saved to: {BEST_MODEL_PATH}")\n\ndel train_pool\ngc.collect()',
        'del p_val\n\ngc.collect()': 'del p_val\ndel val_pool\ndel y_val_binary\n\ngc.collect()',
    }
    patch_notebook(TEMPLATES_DIR / NOTEBOOKS[4], notebooks_dir / NOTEBOOKS[4], final_replacements)
    execute_notebook(notebooks_dir / NOTEBOOKS[4], run_dir, kernel, timeout)

    print(f"\nГотово. Модель: {run_dir / 'survival_optuna/best_model.cbm'}")
    print(f"Метрики: {run_dir / 'survival_optuna/scores_final.csv'}")


if __name__ == "__main__":
    main()
