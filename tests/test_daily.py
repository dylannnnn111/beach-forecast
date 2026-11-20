import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from score import daily

ART = Path("model/artifacts")
NOW = datetime(2026, 11, 16, 20, 10, tzinfo=timezone.utc)  # 7:10am Tue 17 Nov in Sydney


def payload(cells, wet_hours=()):
    """32 days of hourly rain ending at the end of the UTC day, like the real API."""
    times = pd.date_range("2026-10-16T00:00", "2026-11-16T23:00", freq="h")
    out = []
    for _ in range(len(cells)):
        rain = [0.0] * len(times)
        for t, mm in wet_hours:
            rain[times.get_loc(pd.Timestamp(t))] = mm
        out.append({"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "precipitation": rain}})
    return out


@unittest.skipUnless((ART / "meta.json").exists(), "model artifacts not built")
class DailyTest(unittest.TestCase):
    def run_daily(self, d, wet=(), now=NOW):
        return daily.main(["--out", d], fetcher=lambda cells: payload(cells, wet), now=now)

    def rows(self, d):
        return [json.loads(l) for l in (Path(d) / "2026-11.jsonl").read_text().splitlines()]

    def test_cutoff_is_6am_sydney(self):
        self.assertEqual(daily.cutoff_utc(datetime(2026, 11, 17).date()), pd.Timestamp("2026-11-16T19:00"))  # daylight saving
        self.assertEqual(daily.cutoff_utc(datetime(2026, 7, 1).date()), pd.Timestamp("2026-06-30T20:00"))

    def test_scores_every_site_once_for_the_sydney_date(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_daily(d), 0)
            self.assertEqual(self.run_daily(d), 0)  # second run adds nothing
            rows = self.rows(d)
            sites = pd.read_csv(ART / "sites.csv")
            self.assertEqual(len(rows), len(sites))
            self.assertEqual({r["date"] for r in rows}, {"2026-11-17"})
            self.assertTrue(all(0 <= r["p200"] <= 1 and r["rain_24h"] == 0 for r in rows))

    def test_heavy_rain_before_6am_raises_the_risk_and_rain_after_does_not(self):
        with tempfile.TemporaryDirectory() as dry, tempfile.TemporaryDirectory() as wet, tempfile.TemporaryDirectory() as late:
            self.run_daily(dry)
            self.run_daily(wet, wet=[(f"2026-11-16T{h:02d}:00", 8.0) for h in range(10, 19)])
            self.run_daily(late, wet=[("2026-11-16T20:00", 50.0)])  # 7am Sydney, after the cutoff
            mean = lambda d: sum(r["p200"] for r in self.rows(d)) / len(self.rows(d))
            self.assertGreater(mean(wet), 3 * mean(dry))
            self.assertEqual([r["p200"] for r in self.rows(late)], [r["p200"] for r in self.rows(dry)])
            self.assertEqual(self.rows(wet)[0]["rain_24h"], 72.0)

    def test_before_6am_it_skips_without_failing_or_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_daily(d, now=datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)), 0)
            self.assertFalse((Path(d) / "2026-11.jsonl").exists())

    def test_a_failed_fetch_is_quiet_until_the_last_attempt_and_a_later_attempt_recovers(self):
        import urllib.error

        def down(cells):
            raise urllib.error.URLError("handshake timed out")

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(daily.main(["--out", d], fetcher=down, now=NOW), 0)
            self.assertEqual(daily.main(["--out", d, "--strict"], fetcher=down, now=NOW), 1)
            self.assertFalse((Path(d) / "2026-11.jsonl").exists())
            self.assertEqual(self.run_daily(d), 0)
            self.assertEqual(len(self.rows(d)), len(pd.read_csv(ART / "sites.csv")))

    def test_once_scored_it_does_not_call_the_api_again(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_daily(d)

            def boom(cells):
                raise AssertionError("should not fetch")

            self.assertEqual(daily.main(["--out", d, "--strict"], fetcher=boom, now=NOW), 0)

    def test_levels(self):
        self.assertEqual([daily.level(p) for p in (0.01, 0.04, 0.149, 0.15, 0.9)], ["Unlikely", "Possible", "Possible", "Likely", "Likely"])


if __name__ == "__main__":
    unittest.main()
