import json
import tempfile
import unittest
from pathlib import Path

from logger.log_forecasts import append_new, rows


def feature(i, forecast="Unlikely", ts="2026-11-16T20:15:50+00:00", obs="2026-11-11T11:00:00+11:00"):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [151.0 + i / 100, -33.0]},
        "properties": {
            "id": f"site-{i}",
            "siteName": f"Beach {i}",
            "pollutionForecast": forecast,
            "pollutionForecastTimeStamp": ts,
            "latestResult": "Good",
            "latestResultRating": 4,
            "latestResultObservationDate": obs,
        },
    }


class LoggerTest(unittest.TestCase):
    def test_rows_keep_fields_and_coords(self):
        r = rows({"features": [feature(1)]}, "2026-11-17T00:00:00Z")[0]
        self.assertEqual(r["site_id"], "site-1")
        self.assertEqual(r["pollutionForecast"], "Unlikely")
        self.assertEqual((r["lon"], r["lat"]), (151.01, -33.0))

    def test_skips_features_without_id(self):
        bad = {"type": "Feature", "geometry": None, "properties": {}}
        self.assertEqual(rows({"features": [bad]}, "x"), [])

    def test_same_forecast_is_not_logged_twice(self):
        with tempfile.TemporaryDirectory() as d:
            first = rows({"features": [feature(1), feature(2)]}, "2026-11-17T00:00:00Z")
            self.assertEqual(append_new(first, Path(d)), 2)
            again = rows({"features": [feature(1), feature(2)]}, "2026-11-17T06:00:00Z")
            self.assertEqual(append_new(again, Path(d)), 0)

    def test_reissued_forecast_is_logged_and_split_by_month(self):
        with tempfile.TemporaryDirectory() as d:
            append_new(rows({"features": [feature(1)]}, "2026-11-30T21:00:00Z"), Path(d))
            new = rows({"features": [feature(1, "Likely", "2026-12-01T20:15:00+00:00")]}, "2026-12-01T21:00:00Z")
            self.assertEqual(append_new(new, Path(d)), 1)
            self.assertEqual(sorted(p.name for p in Path(d).glob("*.jsonl")), ["2026-11.jsonl", "2026-12.jsonl"])
            line = json.loads((Path(d) / "2026-12.jsonl").read_text())
            self.assertEqual(line["pollutionForecast"], "Likely")


if __name__ == "__main__":
    unittest.main()
