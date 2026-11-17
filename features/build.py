"""Join each water sample to the rain that fell before it.

Everything is measured as of 6am Sydney time on the sample day - the time a
morning forecast would be made - so nothing here uses rain that fell after that,
the sample's own water temperature or conductivity, or lab results that would
not have been published yet (results older than RESULT_LAG_DAYS only).

Output: data/processed/features.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Australia/Sydney"
CUTOFF_HOUR = 6
WINDOWS = {"rain_6h": 6, "rain_12h": 12, "rain_24h": 24, "rain_48h": 48, "rain_72h": 72, "rain_7d": 168}
RESULT_LAG_DAYS = 7
DRY_CAP = 720


def load_rain(rain_dir: str) -> dict:
    out = {}
    for p in sorted(Path(rain_dir).glob("*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        out[p.name[:-8]] = (pd.Timestamp(d["start"]), np.asarray(d["precipitation"], dtype=float))
    return out


def rain_features(start: pd.Timestamp, precip: np.ndarray, cutoff_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Open-Meteo stamps each hour's rain at the END of the hour, so the value at
    time T is rain that fell in (T-1h, T]. A window of n hours ending at the
    cutoff is the n values stamped (cutoff-n, cutoff]."""
    cs = np.concatenate([[0.0], np.cumsum(precip)])
    i = ((cutoff_utc - start) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    if (i < max(WINDOWS.values())).any() or (i >= len(precip)).any():
        raise ValueError("a sample falls outside the rain record")
    out = {name: cs[i + 1] - cs[i + 1 - n] for name, n in WINDOWS.items()}
    # biggest single hour in the last 24
    win = np.lib.stride_tricks.sliding_window_view(precip, 24).max(axis=1)
    out["rain_max1h_24h"] = win[i - 23]
    # hours since the last hour with at least 1 mm
    last_wet = np.where(precip >= 1.0, np.arange(len(precip)), -10**9)
    last_wet = np.maximum.accumulate(last_wet)
    out["hours_since_1mm"] = np.minimum(i - last_wet[i], DRY_CAP)
    return pd.DataFrame(out)


def site_history(df: pd.DataFrame) -> pd.DataFrame:
    """Each site's record using only results at least RESULT_LAG_DAYS old."""
    df = df.sort_values(["site_key", "date"]).reset_index(drop=True)
    cols = {"hist_n": [], "hist_rate200": [], "hist_rate40": [], "hist_logmean": []}
    for _, g in df.groupby("site_key", sort=False):
        d = g["date"].to_numpy()
        known = np.searchsorted(d, d - np.timedelta64(RESULT_LAG_DAYS, "D"), side="right")  # samples dated <= d-lag
        e = g["enterococci"].to_numpy()
        c200 = np.concatenate([[0], np.cumsum(e > 200)])
        c40 = np.concatenate([[0], np.cumsum(e > 40)])
        clog = np.concatenate([[0], np.cumsum(np.log1p(e))])
        n = known.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols["hist_n"].extend(n)
            cols["hist_rate200"].extend(np.where(n > 0, c200[known] / n, np.nan))
            cols["hist_rate40"].extend(np.where(n > 0, c40[known] / n, np.nan))
            cols["hist_logmean"].extend(np.where(n > 0, clog[known] / n, np.nan))
    for k, v in cols.items():
        df[k] = v
    return df


def build(samples_csv: str, cells_csv: str, rain_dir: str) -> pd.DataFrame:
    s = pd.read_csv(samples_csv, parse_dates=["date"])
    cells = pd.read_csv(cells_csv)[["site_key", "cell_id"]]
    s = s.merge(cells, on="site_key", how="left", validate="many_to_one")
    rain = load_rain(rain_dir)
    last = min(st + pd.Timedelta(hours=len(p) - 1) for st, p in rain.values())
    cutoff_local = (s["date"] + pd.Timedelta(hours=CUTOFF_HOUR)).dt.tz_localize(TZ)
    s["cutoff_utc"] = cutoff_local.dt.tz_convert("UTC").dt.tz_localize(None)
    s = s[s["cutoff_utc"] <= last].reset_index(drop=True)
    parts = []
    for cell, g in s.groupby("cell_id", sort=False):
        start, precip = rain[cell]
        f = rain_features(start, precip, pd.DatetimeIndex(g["cutoff_utc"]))
        f.index = g.index
        parts.append(f)
    s = pd.concat([s, pd.concat(parts).sort_index()], axis=1)
    s = site_history(s)
    doy = s["date"].dt.dayofyear
    s["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    s["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    s["year"] = s["date"].dt.year
    s["y200"] = (s["enterococci"] > 200).astype(int)
    s["y40"] = (s["enterococci"] > 40).astype(int)
    keep = ["site_key", "site_name", "region", "cell_id", "date", "year", "enterococci", "y200", "y40",
            *WINDOWS, "rain_max1h_24h", "hours_since_1mm", "hist_n", "hist_rate200", "hist_rate40", "hist_logmean",
            "doy_sin", "doy_cos"]
    return s[keep].sort_values(["date", "site_key"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/processed/samples.csv")
    ap.add_argument("--cells", default="data/processed/site_cells.csv")
    ap.add_argument("--rain", default="data/raw/rain")
    ap.add_argument("--out", default="data/processed/features.csv")
    a = ap.parse_args(argv)
    df = build(a.samples, a.cells, a.rain)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows, {df['site_key'].nunique()} sites, {df['date'].min().date()} to {df['date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
