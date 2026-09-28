#!/bin/bash
# incident4_v4: nb1 -> nb2 -> nb3 -> обучение сегментами -> слепая оценка (один раз).
# Каталог прогона — аргумент или RUN_DIR, по умолчанию retraining/artifacts/runs/incident4_v4.
# В нём заранее лежит data/ (журналы, справочники, погода); см. README.md.
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${1:-${RUN_DIR:-$HERE/../../artifacts/runs/incident4_v4}}
mkdir -p "$RUN/survival_optuna" && RUN=$(cd "$RUN" && pwd) && cd "$RUN" || exit 1
PY=${PY:-python}
cp "$HERE"/[123]_*.ipynb .
[ -f survival_optuna/best_params.json ] || cp "$HERE/best_params.json" survival_optuna/
run_nbs() {
  for nb in "$@"; do
    echo ">>> $nb"
    $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=None "$nb" || return 1
  done
}
run_nbs 1_prepare_dataset.ipynb 2_create_train_test_datasets_V2_MAIN.ipynb 3_feature_selection.ipynb \
  && $PY "$HERE/train_v4.py" "$RUN" --smoke \
  && rm -f survival_optuna/model_*.cbm survival_optuna/val_by_trees.csv survival_optuna/best_model.cbm \
  && $PY "$HERE/train_v4.py" "$RUN" \
  && $PY "$HERE/eval_v4.py" "$RUN"
echo "PIPELINE_EXIT $?"
