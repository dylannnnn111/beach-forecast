import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from ingest import rain


def write_samples(path, sites):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site_key", "lat", "lon"])
        for s in sites:
            w.writerow(s)
            w.writerow(s)  # repeated samples for a site must not matter


def fake_payload(url):
    return {"latitude": -33.9, "longitude": 151.25, "hourly_units": {"precipitation": "mm"},
            "hourly": {"time": ["2020-01-01T00:00", "2020-01-01T01:00"], "precipitation": [0.0, 1.2]}}


class RainTest(unittest.TestCase):
    def test_cell_rounding_groups_neighbours(self):
        self.assertEqual(rain.cell_of(-33.891157, 151.278267), (-33.9, 151.3))
        self.assertEqual(rain.cell_of(-33.914578, 151.267659), (-33.9, 151.25))
        self.assertEqual(rain.cell_id((-33.9, 151.3)), "-33.90_151.30")

    def test_url_has_the_fields_the_api_needs(self):
        u = rain.url_for((-33.9, 151.3), "2010-01-01", "2010-01-31")
        for part in ("latitude=-33.9", "longitude=151.3", "start_date=2010-01-01", "end_date=2010-01-31", "hourly=precipitation", "timezone=GMT"):
            self.assertIn(part, u)

    def test_compact_rejects_mismatched_lengths(self):
        bad = fake_payload("")
        bad["hourly"]["precipitation"] = [0.0]
        with self.assertRaises(ValueError):
            rain.compact(bad)

    def test_main_fetches_each_cell_once_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_samples(d / "s.csv", [("bondi", -33.891157, 151.278267), ("bronte", -33.9040, 151.2790), ("palm", -33.5995, 151.3247)])
            calls = []
            f = lambda u: (calls.append(u), fake_payload(u))[1]
            args = ["--samples", str(d / "s.csv"), "--out", str(d / "rain"), "--map", str(d / "map.csv"), "--end", "2020-01-01"]
            self.assertEqual(rain.main(args, fetcher=f, sleep=lambda s: None), 0)
            self.assertEqual(len(calls), 2)  # bondi and bronte share a cell
            self.assertEqual(rain.main(args, fetcher=f, sleep=lambda s: None), 0)
            self.assertEqual(len(calls), 2)  # nothing refetched
            rows = list(csv.DictReader(open(d / "map.csv")))
            self.assertEqual({r["site_key"]: r["cell_id"] for r in rows}["bondi"], "-33.90_151.30")
            saved = json.load(gzip.open(d / "rain" / "-33.90_151.30.json.gz", "rt"))
            self.assertEqual(saved["precipitation"], [0.0, 1.2])


if __name__ == "__main__":
    unittest.main()
