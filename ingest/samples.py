"""Clean the Beachwatch water quality download into one tidy sample table.

Input: the CSV from beachwatch.nsw.gov.au/waterMonitoring/waterQualityData
(Category "Water quality", all three measures). Output: data/processed/samples.csv
with ISO dates, numeric columns and a stable site key. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

COLS = {
    "Region": "region",
    "Council": "council",
    "Swim site": "site_name",
    "Date": "date",
    "Time": "time",
    "Enterococci (cfu/100mL)": "enterococci",
    "Water temperature (°C)": "water_temp_c",
    "Conductivity (µS/cm)": "conductivity",
    "Latitude": "lat",
    "Longitude": "lon",
}
OUT_FIELDS = ["site_key", "site_name", "region", "council", "lat", "lon", "date", "time",
              "enterococci", "enterococci_censored", "water_temp_c", "conductivity"]


def site_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def number(text: str):
    """'12' -> (12.0, ''), '<1' -> (1.0, '<'), '>2000' -> (2000.0, '>'), '' -> (None, '')."""
    t = text.strip().replace(",", "")
    if not t:
        return None, ""
    flag = t[0] if t[0] in "<>" else ""
    try:
        return float(t.lstrip("<>~ ")), flag
    except ValueError:
        return None, "?"


def clean_row(raw: dict):
    row = {new: (raw.get(old) or "").strip() for old, new in COLS.items()}
    ent, flag = number(row["enterococci"])
    if ent is None or not row["site_name"]:
        return None  # no result to learn from
    try:
        date = datetime.strptime(row["date"], "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None
    temp, _ = number(row["water_temp_c"])
    cond, _ = number(row["conductivity"])
    return {
        "site_key": site_key(row["site_name"]),
        "site_name": row["site_name"],
        "region": row["region"],
        "council": row["council"],
        "lat": row["lat"],
        "lon": row["lon"],
        "date": date,
        "time": row["time"][:5],
        "enterococci": ent,
        "enterococci_censored": flag,
        "water_temp_c": "" if temp is None else temp,
        "conductivity": "" if cond is None else cond,
    }


def clean(rows):
    seen, out, dropped = set(), [], 0
    for raw in rows:
        row = clean_row(raw)
        if row is None:
            dropped += 1
            continue
        k = (row["site_key"], row["date"], row["time"], row["enterococci"])
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        out.append(row)
    out.sort(key=lambda r: (r["date"], r["site_key"], r["time"]))
    return out, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="data/raw/beachwatch-sydney-2010-2026.csv")
    ap.add_argument("--out", default="data/processed/samples.csv")
    a = ap.parse_args(argv)
    with open(a.raw, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"columns missing from the download: {missing}", file=sys.stderr)
            return 1
        rows, dropped = clean(reader)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    sites = {r["site_key"] for r in rows}
    print(f"{len(rows)} samples, {len(sites)} sites, {rows[0]['date']} to {rows[-1]['date']}, {dropped} dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
