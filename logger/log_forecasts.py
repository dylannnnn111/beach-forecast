"""Log the official Beachwatch pollution forecast for every site.

Beachwatch only publishes the latest forecast, not a history, so the only
way to score a model against it later is to record it every day.
One JSON line per site per forecast issue, in data/forecasts/YYYY-MM.jsonl.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from common.http import get_json

URL = "https://api.beachwatch.nsw.gov.au/public/sites/geojson"
FIELDS = (
    "pollutionForecast",
    "pollutionForecastTimeStamp",
    "latestResult",
    "latestResultRating",
    "latestResultObservationDate",
)


def fetch(url: str = URL) -> dict:
    return get_json(url, "beach-forecast-logger")


def rows(geojson: dict, fetched_at: str) -> list[dict]:
    out = []
    for f in geojson.get("features", []):
        p = f.get("properties") or {}
        if not p.get("id"):
            continue
        lon, lat = (f.get("geometry") or {}).get("coordinates", [None, None])[:2]
        row = {"fetched_at": fetched_at, "site_id": p["id"], "site_name": p.get("siteName"), "lat": lat, "lon": lon}
        row.update({k: p.get(k) for k in FIELDS})
        out.append(row)
    return out


def key(row: dict) -> tuple:
    # A row is new when the forecast was reissued or a new sample result came in.
    return (row["site_id"], row["pollutionForecastTimeStamp"], row["latestResultObservationDate"])


def append_new(new_rows: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path in out_dir.glob("*.jsonl"):
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    seen.add(key(json.loads(line)))
    added = 0
    for row in new_rows:
        if key(row) in seen:
            continue
        seen.add(key(row))
        month = row["fetched_at"][:7]
        with (out_dir / f"{month}.jsonl").open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        added += 1
    return added


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/forecasts")
    ap.add_argument("--from-file", help="read a saved geojson instead of calling the API")
    a = ap.parse_args(argv)
    data = json.load(open(a.from_file)) if a.from_file else fetch()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    got = rows(data, now)
    if len(got) < 100:
        print(f"only {len(got)} sites returned - expected 200+, not logging", file=sys.stderr)
        return 1
    added = append_new(got, Path(a.out))
    print(f"{len(got)} sites, {added} new rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
