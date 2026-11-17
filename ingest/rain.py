"""Download hourly rainfall for every sample site from Open-Meteo's historical API.

Sites are grouped into 0.05 degree cells (about 5 km) so beaches that share a
weather-model grid square share one request. One gzipped JSON per cell in
data/rain/, plus data/processed/site_cells.csv mapping each site to its cell.
Resumable: cells already on disk are skipped. Times are UTC. Standard library only.

Weather data by Open-Meteo.com (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://archive-api.open-meteo.com/v1/archive"
CELL = 0.05


def cell_of(lat: float, lon: float) -> tuple[float, float]:
    return (round(round(lat / CELL) * CELL, 2), round(round(lon / CELL) * CELL, 2))


def cell_id(cell: tuple[float, float]) -> str:
    return f"{cell[0]:.2f}_{cell[1]:.2f}"


def site_cells(samples_csv: str) -> dict[str, tuple[float, float]]:
    out = {}
    with open(samples_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["site_key"] not in out:
                out[r["site_key"]] = cell_of(float(r["lat"]), float(r["lon"]))
    return out


def url_for(cell, start: str, end: str) -> str:
    q = {"latitude": cell[0], "longitude": cell[1], "start_date": start, "end_date": end,
         "hourly": "precipitation", "timezone": "GMT"}
    return API + "?" + urllib.parse.urlencode(q)


def fetch(url: str, tries: int = 5, pause: float = 60.0) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "beach-forecast (github.com/dylannnnn111/beach-forecast)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                print(f"  rate limited, waiting {pause:.0f}s", file=sys.stderr)
                time.sleep(pause)
                continue
            raise
    raise RuntimeError("unreachable")


def compact(payload: dict) -> dict:
    h = payload["hourly"]
    if len(h["time"]) != len(h["precipitation"]):
        raise ValueError("time and precipitation lengths differ")
    return {
        "grid_lat": payload.get("latitude"),
        "grid_lon": payload.get("longitude"),
        "start": h["time"][0],
        "end": h["time"][-1],
        "unit": payload.get("hourly_units", {}).get("precipitation"),
        "precipitation": h["precipitation"],
    }


def extend(rain_dir: str, end: str, fetcher=fetch, sleep=time.sleep, pause: float = 5.0) -> int:
    """Top up every cell file in rain_dir to `end` (YYYY-MM-DD). Returns hours added."""
    from datetime import datetime, timedelta
    added = 0
    files = sorted(Path(rain_dir).glob("*.json.gz"))
    for n, path in enumerate(files):
        with gzip.open(path, "rt") as fh:
            d = json.load(fh)
        last = datetime.fromisoformat(d["end"])
        if last.hour != 23:
            raise ValueError(f"{path.name} does not end on a full day")
        start = (last + timedelta(days=1)).date().isoformat()
        if start > end:
            continue
        lat, lon = (float(x) for x in path.name[:-8].split("_"))
        new = compact(fetcher(url_for((lat, lon), start, end)))
        vals = new["precipitation"]
        while vals and vals[-1] is None:  # the newest hours can be unpublished - leave them for next time
            vals = vals[:-1]
        vals = vals[: len(vals) // 24 * 24]
        if any(v is None for v in vals):
            raise ValueError(f"{path.name}: gap in the new rain data")
        if new["start"] != start + "T00:00" or not vals:
            continue
        d["precipitation"].extend(vals)
        d["end"] = (last + timedelta(hours=len(vals))).strftime("%Y-%m-%dT%H:%M")
        with gzip.open(path, "wt") as fh:
            json.dump(d, fh)
        added += len(vals)
        if n < len(files) - 1:
            sleep(pause)
    return added


def main(argv=None, fetcher=fetch, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/processed/samples.csv")
    ap.add_argument("--out", default="data/rain")
    ap.add_argument("--map", default="data/processed/site_cells.csv")
    ap.add_argument("--start", default="2009-12-25")
    ap.add_argument("--end", required=True, help="last date to fetch, YYYY-MM-DD (the archive runs about 5 days behind)")
    ap.add_argument("--limit", type=int, help="only fetch this many cells (for a first test)")
    ap.add_argument("--pause", type=float, default=20.0, help="seconds between requests")
    a = ap.parse_args(argv)

    sites = site_cells(a.samples)
    Path(a.map).parent.mkdir(parents=True, exist_ok=True)
    with open(a.map, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site_key", "cell_id", "cell_lat", "cell_lon"])
        for s, c in sorted(sites.items()):
            w.writerow([s, cell_id(c), c[0], c[1]])

    cells = sorted(set(sites.values()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells if not (out / f"{cell_id(c)}.json.gz").exists()]
    print(f"{len(sites)} sites in {len(cells)} cells, {len(todo)} to fetch")
    if a.limit:
        todo = todo[: a.limit]
    for i, c in enumerate(todo):
        data = compact(fetcher(url_for(c, a.start, a.end)))
        with gzip.open(out / f"{cell_id(c)}.json.gz", "wt") as fh:
            json.dump(data, fh)
        wet = sum(1 for v in data["precipitation"] if v)
        print(f"  {cell_id(c)}: {len(data['precipitation'])} hours {data['start']} to {data['end']}, {wet} wet, grid {data['grid_lat']},{data['grid_lon']}")
        if i < len(todo) - 1:
            sleep(a.pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())
