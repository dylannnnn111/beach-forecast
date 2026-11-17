"""Train the two models on every labelled row and save what the daily job needs.

Writes model/artifacts/: y200.txt and y40.txt (LightGBM), sites.csv (each site's
cell, region and record so far) and meta.json. Small and committed, so the daily
job on GitHub Actions needs no raw data.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from model.backtest import FEATURES, PARAMS

TARGETS = ("y200", "y40")
CELL_COLS = ["site_key", "cell_id", "cell_lat", "cell_lon"]


def site_table(df: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("site_key", observed=True)
    t = pd.DataFrame({
        "site_name": g["site_name"].first(),
        "region": g["region"].first(),
        "hist_n": g.size().astype(float),
        "hist_rate200": g["y200"].mean(),
        "hist_rate40": g["y40"].mean(),
    }).reset_index()
    t["site_key"] = t["site_key"].astype(str)
    return t.merge(cells[CELL_COLS], on="site_key", how="left", validate="one_to_one")


def train(df: pd.DataFrame, cells: pd.DataFrame, out: str) -> dict:
    df = df.copy()
    cats = {c: sorted(df[c].astype(str).unique()) for c in ("site_key", "region")}
    for c, v in cats.items():
        df[c] = df[c].astype(pd.CategoricalDtype(v))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for t in TARGETS:
        m = lgb.LGBMClassifier(**PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df[FEATURES], df[t])
        models[t] = m.booster_
    sites = site_table(df, cells)  # build everything before writing anything
    meta = {"trained_through": str(df["date"].max().date()), "rows": int(len(df)), "features": FEATURES,
            "categories": cats, "params": PARAMS, "base_rate": {t: float(df[t].mean()) for t in TARGETS}}
    for t, b in models.items():
        b.save_model(str(out / f"{t}.txt"))
    sites.to_csv(out / "sites.csv", index=False)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--cells", default="model/artifacts/sites.csv")
    ap.add_argument("--out", default="model/artifacts")
    a = ap.parse_args(argv)
    meta = train(pd.read_csv(a.features, parse_dates=["date"]), pd.read_csv(a.cells), a.out)
    print(f"trained on {meta['rows']} rows through {meta['trained_through']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
