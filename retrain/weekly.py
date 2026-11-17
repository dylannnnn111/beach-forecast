"""Weekly retrain: pull in the new lab results and rain, check them, retrain, record it.

1. Top up the rain files from Open-Meteo's archive.
2. Labels = the base download (data/training/labels.csv.gz) + every newer sample the
   forecast logger has seen.
3. Rebuild the features, then run the checks. The new model is only promoted (written
   to model/artifacts/) if they pass:
     - the training table did not shrink and no site lost its rain cell
     - on the most recent 12 months (trained on everything before them), average
       precision for 'above 200' is at least 80% of what the last promoted model scored
4. Append the outcome to model/registry.json either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from features.build import build, load_rain
from ingest import rain as rain_mod
from ingest.log_labels import labels as log_labels, read_log
from model import backtest
from model.train import train

AP_FLOOR = 0.8


def all_labels(base_csv: str, log_dir: str, sites: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    base = pd.read_csv(base_csv, parse_dates=["date"])
    new = log_labels(read_log(log_dir))
    new = new[new["date"] > base["date"].max()]
    new = new.merge(sites[["site_key", "site_name", "region"]], on="site_key", how="inner")
    out = base if new.empty else pd.concat([base, new[base.columns]], ignore_index=True)
    out = out.drop_duplicates(["site_key", "date"], keep="first").astype({"y200": int, "y40": int})
    return out, int(len(out) - len(base))


def recent_year_check(df: pd.DataFrame) -> dict:
    split = df["date"].max() - pd.Timedelta(days=365)
    train_df, test = df[df["date"] <= split], df[df["date"] > split]
    for c in ("site_key", "region"):
        dtype = pd.CategoricalDtype(sorted(df[c].astype(str).unique()))
        train_df, test = train_df.assign(**{c: train_df[c].astype(dtype)}), test.assign(**{c: test[c].astype(dtype)})
    out = {"from": str((split + pd.Timedelta(days=1)).date()), "to": str(df["date"].max().date())}
    for t in ("y200", "y40"):
        p = backtest.fit_predict(train_df, test, t)
        out[t] = {n: backtest.scores(test[t].to_numpy(), v) for n, v in p.items()}
    return out


def main(argv=None, rain_fetcher=rain_mod.fetch, today=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/training/labels.csv.gz")
    ap.add_argument("--log", default="data/forecasts")
    ap.add_argument("--rain", default="data/rain")
    ap.add_argument("--artifacts", default="model/artifacts")
    ap.add_argument("--registry", default="model/registry.json")
    ap.add_argument("--skip-rain", action="store_true", help="don't call Open-Meteo (use the rain already on disk)")
    a = ap.parse_args(argv)
    today = today or datetime.now(timezone.utc).date()
    sites = pd.read_csv(Path(a.artifacts) / "sites.csv")
    reg_path = Path(a.registry)
    registry = json.load(open(reg_path)) if reg_path.exists() else []
    last = next((r for r in reversed(registry) if r["promoted"]), None)

    hours = 0
    if not a.skip_rain:
        hours = rain_mod.extend(a.rain, (today - timedelta(days=6)).isoformat(), fetcher=rain_fetcher)
    labels, n_new = all_labels(a.base, a.log, sites)
    df = build(labels, sites, load_rain(a.rain))
    check = recent_year_check(df)
    ap200 = check["y200"]["model"]["avg_precision"]

    problems = []
    if last and len(df) < last["rows"]:
        problems.append(f"training table shrank: {len(df)} rows, was {last['rows']}")
    missing = sorted(set(labels["site_key"]) - set(sites["site_key"]))
    if missing:
        problems.append(f"sites with no rain cell: {missing[:5]}")
    if last and ap200 < AP_FLOOR * last["recent_year"]["y200"]["model"]["avg_precision"]:
        problems.append(f"recent-year average precision fell to {ap200:.3f} (last promoted {last['recent_year']['y200']['model']['avg_precision']:.3f})")

    entry = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": int(len(df)),
             "new_labels_from_log": n_new, "rain_hours_added": int(hours),
             "trained_through": str(df["date"].max().date()), "recent_year": check,
             "promoted": not problems, "problems": problems}
    if not problems:
        train(df, sites, a.artifacts)
    registry.append(entry)
    json.dump(registry, open(reg_path, "w"), indent=1)
    print(f"{len(df)} rows ({n_new} new from the log, {hours} rain hours added), recent-year AP {ap200:.3f}, "
          + ("promoted" if not problems else "NOT promoted: " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
