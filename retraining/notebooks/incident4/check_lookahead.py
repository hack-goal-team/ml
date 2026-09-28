"""Проверка фикса look-ahead: sum_1h = число строк категории в (t-1h, t] до текущей строки включительно (brute force)."""
import sys
import numpy as np, polars as pl
run = sys.argv[1]
d = pl.read_parquet(f"{run}/data/preprocessed_dataset/chunk_00000.parquet")
col = "value_cat__Норма"
f1h = f"{col}__sum_1h"
d = d.with_columns((pl.col("дата").dt.combine(pl.col("время"))).alias("dt")).sort("ид_канала_данных", "dt", "ид_события")
bad = checked = ties = 0
for ch, g in d.group_by("ид_канала_данных", maintain_order=True):
    g = g.head(3000)
    # исходного индикатора в выходе нет: восстанавливаем из sum_current
    ind = g[f"{col}__sum_current"].to_numpy().astype(int)
    t = g["dt"].to_numpy()
    for i in range(len(g)):
        lo = t[i] - np.timedelta64(3600, "s")
        exp = int(ind[: i + 1][(t[: i + 1] > lo)].sum())
        checked += 1
        ties += int(i + 1 < len(g) and t[i + 1] == t[i])
        bad += int(exp != g[f1h][i])
print("checked", checked, "rows with same-second successor", ties, "mismatch", bad)
