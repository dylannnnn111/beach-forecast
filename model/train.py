"""Train the two models on every sample we have and save what the daily job needs.

Writes model/artifacts/: y200.txt and y40.txt (LightGBM), sites.csv (each site's
cell, region and past record) and meta.json. These are small and committed, so
the daily job on GitHub Actions needs no raw data.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.backtest import FEATURES, PARAMS

TARGETS = ("y200", "y40")


def site_table(df: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("site_key", observed=True)
    t = pd.DataFrame({
        "site_name": g["site_name"].first(),
        "region": g["region"].first(),
        "hist_n": g.size().astype(float),
        "hist_rate200": g["y200"].mean(),
        "hist_rate40": g["y40"].mean(),
        "hist_logmean": g["enterococci"].apply(lambda e: float(np.log1p(e).mean())),
    }).reset_index()
    return t.merge(cells, on="site_key", how="left", validate="one_to_one")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--cells", default="data/processed/site_cells.csv")
    ap.add_argument("--out", default="model/artifacts")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features, parse_dates=["date"])
    cats = {c: sorted(df[c].unique()) for c in ("site_key", "region")}
    for c, v in cats.items():
        df[c] = df[c].astype(pd.CategoricalDtype(v))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        m = lgb.LGBMClassifier(**PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df[FEATURES], df[t])
        m.booster_.save_model(str(out / f"{t}.txt"))
    site_table(df, pd.read_csv(a.cells)).to_csv(out / "sites.csv", index=False)
    meta = {"trained_through": str(df["date"].max().date()), "rows": int(len(df)), "features": FEATURES,
            "categories": cats, "params": PARAMS, "base_rate": {t: float(df[t].mean()) for t in TARGETS}}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"trained on {len(df)} samples through {meta['trained_through']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
