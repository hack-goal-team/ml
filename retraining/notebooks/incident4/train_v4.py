"""incident4_v4: весь прореженный train (80% каналов, до июня 2026), признаки выбранного шага nb3, параметры v1.
Обучаем сегментами по SEG деревьев (init_model), после каждого — macro PR-AUC на val; стоп без улучшения PATIENCE сегментов."""
import gc, json, sys, time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostRegressor, Pool, FeaturesData
from scipy.special import ndtr
from sklearn.metrics import average_precision_score

RUN = Path(sys.argv[1]).resolve()
SMOKE = "--smoke" in sys.argv
H, SEG, MAX_TREES, PATIENCE = 24.0, 100, 1500, 3
if SMOKE:
    SEG, MAX_TREES = 10, 30
TO_DROP = ["ид_события", "ид_канала_данных", "дата", "время", "datetime"]


def selected():
    # как в nb5: все признаки минус удалённые до выбранного шага nb3 включительно
    step = json.loads((RUN / "feature_selection_shap_rfe/selected_step.json").read_text())["step"]
    hist = pl.read_csv(RUN / "feature_selection_shap_rfe/history.csv")
    removed = {f for r in hist.filter(pl.col("step") <= step)["removed_features"].drop_nulls() for f in r.split(" | ") if f}
    sch = pl.scan_parquet(RUN / "data/final_train/chunk_00000.parquet").drop(TO_DROP).collect_schema()
    fs = [c for c in sch.names() if c not in ("target_lower", "target_upper") and c not in removed]
    assert len(fs) == hist.filter(pl.col("step") == step)["n_features"].item()
    out = {"step": step, "features": fs, "cat_features": [c for c in fs if sch[c] in (pl.String, pl.Categorical)]}
    (RUN / "selected_features.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


feat = selected()
FEATS, CATS = feat["features"], feat["cat_features"]
NUMS = [f for f in FEATS if f not in CATS]
inc = json.load(open(RUN / "data/incident_by_sensor.json"))
CLS = sorted(set(inc.values()))
params = json.load(open(RUN / "survival_optuna/best_params.json"))
scale = float(params.pop("scale")); params.pop("iterations", None)
OUT = RUN / "survival_optuna"
t0 = time.time()
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')} t={time.time()-t0:.0f}s]", *a, flush=True)


def load(files):
    # числа — сразу в заранее выделенный float32, категории — строками (null -> "null", как в v2)
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    X = np.empty((n, len(NUMS)), dtype=np.float32)
    C = np.empty((n, len(CATS)), dtype=object)
    y = np.empty((n, 2), dtype=np.float64); sensor = np.empty(n, dtype=object); i = 0
    for f in files:
        d = pl.read_parquet(f, columns=[*dict.fromkeys(FEATS + ["target_lower", "target_upper", "тип_датчика"])])
        k = d.height
        if k == 0:
            continue
        X[i:i + k] = d.select(pl.col(NUMS).cast(pl.Float32)).to_numpy()
        for j, c in enumerate(CATS):
            s = d[c].cast(pl.String).fill_null("null").cast(pl.Categorical)
            C[i:i + k, j] = np.array(s.cat.get_categories().to_list(), dtype=object)[s.to_physical().to_numpy()]
        y[i:i + k] = d.select("target_lower", "target_upper").to_numpy()
        sensor[i:i + k] = d["тип_датчика"].to_numpy(); i += k
        del d
    assert i == n
    pool = Pool(FeaturesData(num_feature_data=X, cat_feature_data=C, num_feature_names=NUMS, cat_feature_names=CATS), label=y)
    ybin = ((y[:, 1] != -1) & (y[:, 0] <= H)).astype(np.int8)
    del X, C, y; gc.collect()
    return pool, ybin, sensor


def macro(y, p, sensor):
    cls = pl.Series(sensor).replace_strict(inc, default=None).to_numpy()
    per = {c: float(average_precision_score(y[cls == c], p[cls == c])) for c in CLS if y[cls == c].any()}
    return float(np.mean(list(per.values()))), per


train_files = sorted((RUN / "data/final_train").glob("chunk_*.parquet"))
val_files = sorted((RUN / "data/final_val").glob("chunk_*.parquet"))
if SMOKE:
    train_files, val_files = train_files[:3], val_files[:40]
train_pool, ytr, _ = load(train_files)
log("train rows", train_pool.num_row(), "pos", int(ytr.sum()), "files", len(train_files))
train_pool.quantize(); gc.collect()
log("quantized")
val_pool, yval, sval = load(val_files)
log("val rows", val_pool.num_row(), "pos", int(yval.sum()))

model, hist, best = None, [], (-1.0, 0)
while True:
    m = CatBoostRegressor(loss_function=f"SurvivalAft:dist=Normal;scale={scale}", iterations=SEG, **params,
                          random_seed=42, verbose=max(SEG // 4, 1), allow_writing_files=False)
    m.fit(train_pool, init_model=model)
    model = m; n = model.tree_count_
    p = ndtr((np.log(H) - model.predict(val_pool)) / scale)
    mac, per = macro(yval, p, sval)
    hist.append(dict(trees=n, val_macro=mac, val_all=float(average_precision_score(yval, p)), **per))
    pl.DataFrame(hist).write_csv(OUT / "val_by_trees.csv")
    model.save_model(str(OUT / f"model_{n}.cbm"))
    log("trees", n, "val_macro", round(mac, 4), {k: round(v, 3) for k, v in per.items()})
    if mac > best[0]:
        best = (mac, n)
    if n - best[1] >= PATIENCE * SEG or n >= MAX_TREES:
        break
log("best trees", best[1], "val_macro", round(best[0], 4))
bm = CatBoostRegressor(); bm.load_model(str(OUT / f"model_{best[1]}.cbm"))
bm.save_model(str(OUT / "best_model.cbm"))
json.dump({"best_trees": best[1], "val_macro": best[0], "scale": scale}, open(OUT / "best_trees.json", "w"))
print("TRAIN_DONE", flush=True)
