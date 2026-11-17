"""Rolling-origin backtest: for each test year, train on every earlier year.

Compares three things on the same samples:
  site_history - each site's past rate of bad results (no weather at all)
  rain_rule    - logistic regression on recent rain only (a stand-in for a rainfall-based forecast)
  model        - gradient boosting on rain, site, season and site history

Writes out/backtest.json. Scores: average precision (how well bad days are ranked
first; a random guess scores the base rate), Brier score (lower is better) and ROC AUC.
"""
from __future__ import annotations

import argparse
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RAIN = ["rain_6h", "rain_12h", "rain_24h", "rain_48h", "rain_72h", "rain_7d", "rain_max1h_24h", "hours_since_1mm"]
HIST = ["hist_n", "hist_rate200", "hist_rate40"]
FEATURES = RAIN + HIST + ["doy_sin", "doy_cos", "site_key", "region"]
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=80,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)


def scores(y, p) -> dict:
    return {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
            "avg_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p))}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    y = train[target].to_numpy()
    base = y.mean()
    rate = "hist_rate200" if target == "y200" else "hist_rate40"
    out = {"site_history": test[rate].fillna(base).clip(0.001, 0.999).to_numpy()}
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(np.log1p(train[RAIN]), y)
    out["rain_rule"] = lr.predict_proba(np.log1p(test[RAIN]))[:, 1]
    m = lgb.LGBMClassifier(**PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train[FEATURES], y)
        out["model"] = m.predict_proba(test[FEATURES])[:, 1]
    return out


def run(df: pd.DataFrame, first_test_year: int, targets=("y200", "y40")) -> tuple[dict, pd.DataFrame]:
    df = df.copy()
    for c in ("site_key", "region"):
        df[c] = df[c].astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    preds = []
    for year in range(first_test_year, int(df["year"].max()) + 1):
        train, test = df[df["year"] < year], df[df["year"] == year]
        if test.empty:
            continue
        row = test[["site_key", "date", "year", "y200", "y40", "rain_24h"]].copy()
        for t in targets:
            for name, p in fit_predict(train, test, t).items():
                row[f"{t}_{name}"] = p
        preds.append(row)
    preds = pd.concat(preds)
    result = {"first_test_year": first_test_year, "last_test_year": int(preds["year"].max()), "targets": {}}
    for t in targets:
        names = ["site_history", "rain_rule", "model"]
        result["targets"][t] = {
            "overall": {n: scores(preds[t].to_numpy(), preds[f"{t}_{n}"].to_numpy()) for n in names},
            "by_year": {str(y): {n: scores(g[t].to_numpy(), g[f"{t}_{n}"].to_numpy()) for n in names}
                        for y, g in preds.groupby("year")},
        }
    return result, preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--out", default="out/backtest.json")
    ap.add_argument("--preds", default="data/processed/backtest_preds.csv")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features, parse_dates=["date"])
    result, preds = run(df, a.first_test_year)
    result["rows"] = int(len(df))
    result["sites"] = int(df["site_key"].nunique())
    result["features"] = FEATURES
    json.dump(result, open(a.out, "w"), indent=1)
    preds.to_csv(a.preds, index=False)
    for t, r in result["targets"].items():
        print(t, {n: {k: round(v, 4) for k, v in s.items() if k in ("avg_precision", "brier", "roc_auc")} for n, s in r["overall"].items()},
              "base", round(r["overall"]["model"]["base_rate"], 4), "n", r["overall"]["model"]["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
