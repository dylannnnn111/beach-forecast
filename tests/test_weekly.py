import gzip
import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from ingest import rain
from retrain import weekly


def cell_file(d, name="-33.90_151.30", hours=48):
    p = Path(d) / f"{name}.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump({"grid_lat": -33.9, "grid_lon": 151.3, "start": "2026-11-01T00:00", "end": "2026-11-02T23:00",
                   "unit": "mm", "precipitation": [0.0] * hours}, fh)
    return p


def api(values):
    def fetch(url):
        start = url.split("start_date=")[1][:10]
        times = [t.strftime("%Y-%m-%dT%H:%M") for t in pd.date_range(start + "T00:00", periods=len(values), freq="h")]
        return {"latitude": -33.9, "longitude": 151.3, "hourly_units": {"precipitation": "mm"},
                "hourly": {"time": times, "precipitation": list(values)}}
    return fetch


class ExtendTest(unittest.TestCase):
    def test_appends_whole_days_and_drops_unpublished_hours(self):
        with tempfile.TemporaryDirectory() as d:
            p = cell_file(d)
            added = rain.extend(d, "2026-11-04", fetcher=api([1.0] * 30 + [None] * 18), sleep=lambda s: None)
            self.assertEqual(added, 24)
            saved = json.load(gzip.open(p, "rt"))
            self.assertEqual((len(saved["precipitation"]), saved["end"]), (72, "2026-11-03T23:00"))

    def test_nothing_to_do_when_up_to_date(self):
        with tempfile.TemporaryDirectory() as d:
            cell_file(d)
            self.assertEqual(rain.extend(d, "2026-11-02", fetcher=api([]), sleep=lambda s: None), 0)

    def test_gap_in_the_middle_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            cell_file(d)
            with self.assertRaises(ValueError):
                rain.extend(d, "2026-11-03", fetcher=api([1.0, None] + [1.0] * 22), sleep=lambda s: None)


@unittest.skipUnless(Path("data/training/labels.csv.gz").exists() and Path("model/artifacts/sites.csv").exists(), "training data not present")
class WeeklyTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        shutil.copytree("model/artifacts", self.d / "artifacts")
        (self.d / "log").mkdir()
        self.args = ["--skip-rain", "--artifacts", str(self.d / "artifacts"), "--registry", str(self.d / "registry.json"),
                     "--log", str(self.d / "log")]

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_first_run_promotes_then_a_shrunken_table_is_refused(self):
        self.assertEqual(weekly.main(self.args, today=date(2026, 11, 17)), 0)
        reg = json.load(open(self.d / "registry.json"))
        self.assertTrue(reg[0]["promoted"])
        self.assertGreater(reg[0]["recent_year"]["y200"]["model"]["avg_precision"], reg[0]["recent_year"]["y200"]["rain_rule"]["avg_precision"])
        before = (self.d / "artifacts" / "y200.txt").read_bytes()
        # break the input: half the labels go missing
        small = self.d / "labels.csv.gz"
        pd.read_csv("data/training/labels.csv.gz").iloc[::2].to_csv(small, index=False)
        self.assertEqual(weekly.main(self.args + ["--base", str(small)], today=date(2026, 11, 24)), 1)
        reg = json.load(open(self.d / "registry.json"))
        self.assertFalse(reg[1]["promoted"])
        self.assertIn("shrank", reg[1]["problems"][0])
        self.assertEqual((self.d / "artifacts" / "y200.txt").read_bytes(), before)  # the live model was not touched


if __name__ == "__main__":
    unittest.main()
