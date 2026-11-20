"""Score every site for this morning and append the result to data/predictions/.

Runs on GitHub Actions after 6am Sydney time. Uses rain up to 6am only, from
Open-Meteo's forecast API (which also serves the last few weeks of hourly rain),
and the committed models in model/artifacts/. One JSON line per site per day.

Weather data by Open-Meteo.com (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.http import get_json
from features.build import CUTOFF_HOUR, TZ, rain_features

API = "https://api.open-meteo.com/v1/forecast"
PAST_DAYS = 31  # hours_since_1mm looks back up to 720 hours
# Fixed on 17 Nov 2026 from the 2016-2026 backtest, before any live scoring.
LEVELS = (("Likely", 0.15), ("Possible", 0.04))


def level(p200: float) -> str:
    for name, cut in LEVELS:
        if p200 >= cut:
            return name
    return "Unlikely"


def fetch_rain(cells: pd.DataFrame) -> list[dict]:
    q = {"latitude": ",".join(f"{v:.2f}" for v in cells["cell_lat"]),
         "longitude": ",".join(f"{v:.2f}" for v in cells["cell_lon"]),
         "hourly": "precipitation", "timezone": "GMT", "past_days": PAST_DAYS, "forecast_days": 1}
    data = get_json(API + "?" + urllib.parse.urlencode(q, safe=","), "beach-forecast (github.com/dylannnnn111/beach-forecast)")
    return data if isinstance(data, list) else [data]


def cutoff_utc(day) -> pd.Timestamp:
    local = datetime(day.year, day.month, day.day, CUTOFF_HOUR, tzinfo=ZoneInfo(TZ))
    return pd.Timestamp(local.astimezone(timezone.utc).replace(tzinfo=None))


def score(sites: pd.DataFrame, rain_by_cell: dict, meta: dict, boosters: dict, day) -> pd.DataFrame:
    cut = cutoff_utc(day)
    parts = []
    for cell, g in sites.groupby("cell_id", sort=False):
        start, precip = rain_by_cell[cell]
        if start + pd.Timedelta(hours=len(precip) - 1) < cut:
            raise ValueError(f"rain for {cell} stops before the 6am cutoff")
        if np.isnan(precip[: int((cut - start) / pd.Timedelta(hours=1)) + 1]).any():
            raise ValueError(f"rain for {cell} has gaps before the cutoff")
        f = rain_features(start, np.nan_to_num(precip), pd.DatetimeIndex([cut] * len(g)))
        f.index = g.index
        parts.append(f)
    x = pd.concat([sites, pd.concat(parts).sort_index()], axis=1)
    doy = pd.Timestamp(day).dayofyear
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c, v in meta["categories"].items():
        x[c] = x[c].astype(pd.CategoricalDtype(v))
    out = x[["site_key", "site_name", "rain_24h", "rain_72h"]].copy()
    for t, b in boosters.items():
        out[f"p{t[1:]}"] = b.predict(x[meta["features"]])
    out["level"] = out["p200"].map(level)
    return out


def main(argv=None, fetcher=fetch_rain, now=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="model/artifacts")
    ap.add_argument("--out", default="data/predictions")
    ap.add_argument("--strict", action="store_true", help="the day's last attempt: fail if today still has no predictions")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    meta = json.load(open(art / "meta.json"))
    sites = pd.read_csv(art / "sites.csv")
    boosters = {t: lgb.Booster(model_file=str(art / f"{t}.txt")) for t in ("y200", "y40")}
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(ZoneInfo(TZ)).date()
    path = Path(a.out) / f"{day:%Y-%m}.jsonl"
    done = set()
    if path.exists():
        done = {(r["date"], r["site_key"]) for r in map(json.loads, path.read_text().splitlines()) if r}
    if all((str(day), k) in done for k in sites["site_key"]):
        print(f"{day}: already scored")
        return 0
    if pd.Timestamp(now.replace(tzinfo=None)) < cutoff_utc(day):
        print("it is not 6am in Sydney yet - nothing to score")
        return 0
    cells = sites.drop_duplicates("cell_id")[["cell_id", "cell_lat", "cell_lon"]].reset_index(drop=True)
    try:
        payload = fetcher(cells)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        # The job runs several times a morning, so one bad attempt is not a failure - unless it is the last.
        print(f"could not get rain from Open-Meteo: {e}", file=sys.stderr)
        return 1 if a.strict else 0
    if len(payload) != len(cells):
        print(f"asked for {len(cells)} cells, got {len(payload)}", file=sys.stderr)
        return 1
    rain = {c: (pd.Timestamp(p["hourly"]["time"][0]),
                np.array([np.nan if v is None else v for v in p["hourly"]["precipitation"]], dtype=float))
            for c, p in zip(cells["cell_id"], payload)}
    res = score(sites, rain, meta, boosters, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0
    with path.open("a") as fh:
        for r in res.itertuples():
            if (str(day), r.site_key) in done:
                continue
            fh.write(json.dumps({"date": str(day), "site_key": r.site_key, "site_name": r.site_name,
                                 "p200": round(float(r.p200), 4), "p40": round(float(r.p40), 4), "level": r.level,
                                 "rain_24h": round(float(r.rain_24h), 1), "rain_72h": round(float(r.rain_72h), 1),
                                 "trained_through": meta["trained_through"], "scored_at": stamp}, sort_keys=True) + "\n")
            added += 1
    print(f"{day}: {len(res)} sites scored, {added} new rows, levels {res['level'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
