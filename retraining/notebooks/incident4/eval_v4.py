"""Слепая оценка incident4_v4 (один раз): new vs v3 vs prod(old) vs правила; (a) слепые каналы, (b) июнь 2026."""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np, polars as pl
from catboost import CatBoostRegressor, Pool
from scipy.special import ndtr
from sklearn.metrics import average_precision_score

RUN = Path(sys.argv[1]).resolve()
H = 24.0
# Модели и прогоны для сравнения — вне репозитория, пути через env (по умолчанию RUN/compare/)
CMP = Path(os.environ.get("COMPARE_DIR", RUN / "compare"))
MODELS = {"new": RUN / "survival_optuna/best_model.cbm",
          "v3": Path(os.environ.get("V3_MODEL", CMP / "incident4_v3_500.cbm")),
          "prod": Path(os.environ.get("PROD_MODEL", CMP / "old_main.cbm"))}
V3_CHUNK_MAP = Path(os.environ.get("V3_CHUNK_MAP", CMP / "run_v3/data/chunk_map.parquet"))
V2_CHUNK_MAP = Path(os.environ.get("V2_CHUNK_MAP", CMP / "run_v2/data/chunk_map.parquet"))
t0 = time.time()
inc = json.load(open(RUN / "data/incident_by_sensor.json"))
models = {}
for k, p in MODELS.items():
    m = CatBoostRegressor(); m.load_model(str(p))
    scale = float(m.get_all_params()["loss_function"].split("scale=")[1])
    models[k] = (m, scale, m.feature_names_, [m.feature_names_[i] for i in m.get_cat_feature_indices()])
    print(k, m.tree_count_, len(m.feature_names_), round(scale, 4), flush=True)


def predict(split):
    files = sorted((RUN / f"data/final_{split}").glob("chunk_*.parquet"))
    need = sorted({f for _, _, fs, _ in models.values() for f in fs} | {"target_lower", "target_upper", "тип_датчика", "ид_канала_данных", "hours_since_prev_target"})
    extra = ["__blind_channel"] if split == "june" else []
    parts = []
    for b in np.array_split(np.array(files, dtype=object), 16):
        if len(b) == 0:
            continue
        df = pl.scan_parquet(list(b)).select(need + extra).collect(engine="streaming")
        if df.height == 0:
            continue
        out = {"ch": df["ид_канала_данных"].cast(pl.Int64), "sensor": df["тип_датчика"],
               "y": ((df["target_upper"] != -1) & (df["target_lower"] <= H)).cast(pl.Int8),
               "hprev": df["hours_since_prev_target"].cast(pl.Float64)}
        if extra:
            out["blind_ch"] = df["__blind_channel"]
        for k, (m, scale, fs, cats) in models.items():
            X = df.select(fs).with_columns(pl.col(cats).cast(pl.String).fill_null("null"))
            out[f"p_{k}"] = ndtr((np.log(H) - m.predict(Pool(X, cat_features=cats))) / scale)
        parts.append(pl.DataFrame(out)); del df; gc.collect()
    r = pl.concat(parts).with_columns(
        pl.col("sensor").replace_strict(inc, default=None).alias("cls"),
        pl.when(pl.col("hprev").is_not_null()).then(1.0 / (1.0 + pl.col("hprev"))).otherwise(0.0).alias("p_rule"),
        (pl.col("hprev") <= H).fill_null(False).cast(pl.Float64).alias("p_rule24"))
    r.write_parquet(RUN / f"eval_preds_{split}.parquet")
    print(split, "rows", r.height, "channels", r["ch"].n_unique(), "pos", int(r["y"].sum()), f"t={time.time()-t0:.0f}s", flush=True)
    return r


def ap(d, c):
    return round(float(average_precision_score(d["y"].to_numpy(), d[c].to_numpy())), 3) if d.height and d["y"].sum() else None


# Каналы, которые v3/prod видели при обучении или выборе (для честного сравнения)
v3_seen = pl.concat([pl.read_parquet(V3_CHUNK_MAP).select("ид_канала_данных"),
                     pl.read_parquet(V2_CHUNK_MAP).filter(pl.col("split") != "test").select("ид_канала_данных")])
ids = pl.concat(
    pl.scan_parquet(RUN / f"data/ext-journal-{y}.parquet").select(pl.col("ид_канала_данных").cast(pl.Int64, strict=False)).unique().collect()
    for y in (2024, 2025, 2026)
).drop_nulls().unique().sort("ид_канала_данных").sample(fraction=1.0, shuffle=True, seed=69)
prod_seen = ids["ид_канала_данных"][: int(ids.height * 0.9)]  # сплит prod: seed 69, 90% train (как tests/backtest)
SEEN = set(v3_seen["ид_канала_данных"].cast(pl.Int64).to_list()) | set(prod_seen.to_list())

K = ["new", "v3", "prod", "rule", "rule24"]
CLS = ["FIRE", "FLOOD", "GAS", "INTRUSION"]
blind, june = predict("blind"), predict("june")
rows = []
unseen = pl.col("ch").is_in(list(SEEN)).not_()
for subset, d0 in [("a_blind_channels", blind), ("a_blind_unseen_by_v3_prod", blind.filter(unseen)),
                   ("b_june_all", june), ("b_june_unseen_by_v3_prod", june.filter(unseen)),
                   ("b_june_blind_channels", june.filter(pl.col("blind_ch")))]:
    for n, d in [(c, d0.filter(pl.col("cls") == c)) for c in CLS] + [("ALL_rows", d0)]:
        rows.append(dict(subset=subset, cls=n, rows=d.height, channels=d["ch"].n_unique(), pos=int(d["y"].sum()), **{k: ap(d, f"p_{k}") for k in K}))
    per = [x for x in rows if x["subset"] == subset and x["cls"] in CLS]
    rows.append(dict(subset=subset, cls="MACRO", rows=None, channels=None, pos=None,
                     **{k: round(float(np.mean([x[k] for x in per if x[k] is not None])), 3) for k in K}))
res = pl.DataFrame(rows)
res.write_csv(RUN / "eval_blind.csv")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
print(res)
print("EVAL_DONE", round(time.time() - t0), flush=True)
