"""Turn the forecast log into new training labels and the official forecast per day.

The public Beachwatch feed gives each site's latest result as a category, and the
categories line up with the count thresholds (Good <= 40, Fair 41-200, Poor 201-500,
Bad > 500 - checked against 106 samples that are in both the log and the full
download). So every new sample the logger sees becomes a label with no manual download.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TZ = "Australia/Sydney"
CATEGORY = {"Good": (0, 0), "Fair": (0, 1), "Poor": (1, 1), "Bad": (1, 1)}  # (y200, y40)
FORECASTS = ("Unlikely", "Possible", "Likely")


def site_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def read_log(log_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(log_dir).glob("*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["site_key"] = df["site_name"].map(site_key)
    return df


def labels(log: pd.DataFrame) -> pd.DataFrame:
    """One row per site per sample date seen in the log: site_key, date, y200, y40."""
    cols = ["site_key", "date", "y200", "y40"]
    if log.empty:
        return pd.DataFrame(columns=cols)
    d = log[log["latestResult"].isin(CATEGORY) & log["latestResultObservationDate"].notna()].copy()
    d["date"] = pd.to_datetime(d["latestResultObservationDate"].str[:10])  # already Sydney local time
    d["y200"] = d["latestResult"].map(lambda c: CATEGORY[c][0])
    d["y40"] = d["latestResult"].map(lambda c: CATEGORY[c][1])
    return d.groupby(["site_key", "date"], as_index=False)[["y200", "y40"]].max()[cols]


def official(log: pd.DataFrame) -> pd.DataFrame:
    """The first official forecast issued for each site on each Sydney date."""
    cols = ["site_key", "date", "official"]
    if log.empty:
        return pd.DataFrame(columns=cols)
    d = log[log["pollutionForecast"].isin(FORECASTS) & log["pollutionForecastTimeStamp"].notna()].copy()
    ts = pd.to_datetime(d["pollutionForecastTimeStamp"], utc=True, format="ISO8601")
    d["issued"] = ts
    d["date"] = ts.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    d = d.sort_values("issued").drop_duplicates(["site_key", "date"], keep="first")
    return d.rename(columns={"pollutionForecast": "official"})[cols].reset_index(drop=True)
