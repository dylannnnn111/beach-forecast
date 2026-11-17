"""Bundle everything the project page needs into one small file: out/site.json.

Today's call per site, the official forecast for the same morning, a what-if table
(each site's risk under a range of overnight rain), the backtest summary, the track
record and the latest registry entry. The page reads this file straight from GitHub.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingest.log_labels import official, read_log
from score.daily import LEVELS, level
from score.track_record import build as build_record, read_predictions

SCENARIOS_MM = [0, 2, 5, 10, 20, 40, 80]  # rain spread evenly over the 12 hours before 6am, after a dry week


def scenario_features(total: float) -> dict:
    per_hour = total / 12
    return {"rain_6h": total / 2, "rain_12h": total, "rain_24h": total, "rain_48h": total, "rain_72h": total,
            "rain_7d": total, "rain_max1h_24h": per_hour, "hours_since_1mm": 0 if per_hour >= 1 else 720}


def what_if(sites: pd.DataFrame, meta: dict, booster: lgb.Booster, day) -> np.ndarray:
    doy = pd.Timestamp(day).dayofyear
    cols = []
    for mm in SCENARIOS_MM:
        x = sites.assign(**scenario_features(mm), doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25))
        for c, v in meta["categories"].items():
            x[c] = x[c].astype(pd.CategoricalDtype(v))
        cols.append(booster.predict(x[meta["features"]]))
    return np.column_stack(cols)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="model/artifacts")
    ap.add_argument("--log", default="data/forecasts")
    ap.add_argument("--predictions", default="data/predictions")
    ap.add_argument("--backtest", default="out/backtest.json")
    ap.add_argument("--registry", default="model/registry.json")
    ap.add_argument("--out", default="out/site.json")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    meta = json.load(open(art / "meta.json"))
    sites = pd.read_csv(art / "sites.csv")
    preds = read_predictions(a.predictions)
    if preds.empty:
        print("no predictions yet - nothing to export")
        return 1
    day = preds["date"].max()
    today = preds[preds["date"] == day].set_index("site_key")
    log = read_log(a.log)
    off = official(log)
    off = off[off["date"] == day].set_index("site_key")["official"] if len(off) else pd.Series(dtype=str)
    wi = what_if(sites, meta, lgb.Booster(model_file=str(art / "y200.txt")), day)
    coords = {}
    if not log.empty:
        coords = log.drop_duplicates("site_key").set_index("site_key")[["lat", "lon"]].to_dict("index")
    rows = []
    for i, s in enumerate(sites.itertuples()):
        if s.site_key not in today.index:
            continue
        t = today.loc[s.site_key]
        c = coords.get(s.site_key, {})
        rows.append({"key": s.site_key, "name": s.site_name, "region": s.region,
                     "lat": c.get("lat", s.cell_lat), "lon": c.get("lon", s.cell_lon), "exact": s.site_key in coords,
                     "p200": float(t["p200"]), "p40": float(t["p40"]), "level": t["level"],
                     "rain_24h": float(t["rain_24h"]), "rain_72h": float(t["rain_72h"]),
                     "official": off.get(s.site_key), "samples": int(s.hist_n), "past_rate200": round(float(s.hist_rate200), 4),
                     "what_if": [round(float(v), 4) for v in wi[i]]})
    bt = json.load(open(a.backtest))
    reg = json.load(open(a.registry)) if Path(a.registry).exists() else []
    record = build_record(log, preds)
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": str(day.date()),
        "scored_at": str(today["scored_at"].iloc[0]),
        "trained_through": meta["trained_through"],
        "training_rows": meta["rows"],
        "levels": {name: cut for name, cut in LEVELS},
        "scenarios_mm": SCENARIOS_MM,
        "sites": rows,
        "days_scored": int(preds["date"].nunique()),
        "first_day_scored": str(preds["date"].min().date()),
        "official_logged_today": int(sum(r["official"] is not None for r in rows)),
        "backtest": {"first_test_year": bt["first_test_year"], "last_test_year": bt["last_test_year"],
                     "overall": bt["targets"]["y200"]["overall"],
                     "by_year": {y: {n: {"avg_precision": s["avg_precision"], "n": s["n"], "positives": s["positives"]} for n, s in v.items()}
                                 for y, v in bt["targets"]["y200"]["by_year"].items()},
                     "rain_bands": bt.get("rain_bands", []), "calibration": bt.get("calibration_y200", []), "sites": bt["sites"]},
        "track_record": {k: record[k] for k in ("first_day", "last_day", "all_calls", "head_to_head")},
        "track_rows": record["rows"][-400:],
        "registry": [{"run_at": r["run_at"], "rows": r["rows"], "new_labels_from_log": r["new_labels_from_log"],
                      "trained_through": r["trained_through"], "promoted": r["promoted"],
                      "recent_year_ap": r["recent_year"]["y200"]["model"]["avg_precision"]} for r in reg[-12:]],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(f"site.json: {out['date']}, {len(rows)} sites, {out['official_logged_today']} with an official forecast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
