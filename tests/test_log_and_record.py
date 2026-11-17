import unittest

import pandas as pd

from ingest.log_labels import labels, official, site_key
from score.track_record import build


def log_row(name, forecast, issued, result, observed):
    return {"site_name": name, "site_key": site_key(name), "pollutionForecast": forecast,
            "pollutionForecastTimeStamp": issued, "latestResult": result, "latestResultObservationDate": observed}


LOG = pd.DataFrame([
    # morning of Wed 18 Nov (Sydney): forecasts issued 7:15am local = 20:15 UTC the day before
    log_row("Bondi Beach", "Likely", "2026-11-17T20:15:50.1866667+00:00", "Good", "2026-11-13T07:34:00+11:00"),
    log_row("Coogee Beach", "Unlikely", "2026-11-17T20:15:50+00:00", "Good", "2026-11-13T08:00:00+11:00"),
    log_row("Megalong Creek", "Forecast not available", "2026-11-17T20:15:50+00:00", "Good", "2026-11-13T12:20:00+11:00"),
    # a later reissue the same morning must not replace the first call
    log_row("Bondi Beach", "Possible", "2026-11-17T23:00:00+00:00", "Good", "2026-11-13T07:34:00+11:00"),
    # a week on, the 18 Nov lab results are out
    log_row("Bondi Beach", "Unlikely", "2026-11-24T20:15:00+00:00", "Poor", "2026-11-18T07:30:00+11:00"),
    log_row("Coogee Beach", "Unlikely", "2026-11-24T20:15:00+00:00", "Fair", "2026-11-18T08:00:00+11:00"),
    log_row("Megalong Creek", "Forecast not available", "2026-11-24T20:15:00+00:00", "Bad", "2026-11-18T12:00:00+11:00"),
])
PREDS = pd.DataFrame([
    {"date": "2026-11-18", "site_key": "bondi-beach", "site_name": "Bondi Beach", "p200": 0.3, "level": "Likely"},
    {"date": "2026-11-18", "site_key": "coogee-beach", "site_name": "Coogee Beach", "p200": 0.01, "level": "Unlikely"},
    {"date": "2026-11-18", "site_key": "megalong-creek", "site_name": "Megalong Creek", "p200": 0.5, "level": "Likely"},
    {"date": "2026-11-19", "site_key": "bondi-beach", "site_name": "Bondi Beach", "p200": 0.02, "level": "Unlikely"},
])
PREDS["date"] = pd.to_datetime(PREDS["date"])


class LogTest(unittest.TestCase):
    def test_categories_become_labels(self):
        lab = labels(LOG).set_index(["site_key", "date"])
        self.assertEqual(tuple(lab.loc[("bondi-beach", pd.Timestamp("2026-11-18"))]), (1, 1))   # Poor
        self.assertEqual(tuple(lab.loc[("coogee-beach", pd.Timestamp("2026-11-18"))]), (0, 1))  # Fair
        self.assertEqual(tuple(lab.loc[("bondi-beach", pd.Timestamp("2026-11-13"))]), (0, 0))   # Good
        self.assertEqual(len(lab), 6)

    def test_official_uses_sydney_date_and_first_issue(self):
        off = official(LOG).set_index(["site_key", "date"])["official"]
        self.assertEqual(off[("bondi-beach", pd.Timestamp("2026-11-18"))], "Likely")
        self.assertNotIn(("megalong-creek", pd.Timestamp("2026-11-18")), off.index)  # no forecast is not a call


class TrackRecordTest(unittest.TestCase):
    def test_only_days_with_a_prediction_and_a_result_count(self):
        r = build(LOG, PREDS)
        self.assertEqual((r["first_day"], r["last_day"]), ("2026-11-18", "2026-11-18"))
        m = r["all_calls"]["model"]
        self.assertEqual((m["n"], m["above_200"], m["flagged"], m["caught_above_200"]), (3, 2, 2, 2))

    def test_head_to_head_only_where_the_official_forecast_made_a_call(self):
        h = build(LOG, PREDS)["head_to_head"]
        self.assertEqual(h["model"]["n"], 2)
        self.assertEqual(h["official"]["by_level"]["Likely"], {"n": 1, "above_200": 1, "above_40": 1})
        self.assertEqual(h["official"]["caught_above_40"], 1)  # missed Coogee's Fair result
        self.assertEqual(h["model"]["caught_above_200"], 1)

    def test_empty_inputs(self):
        r = build(pd.DataFrame(), PREDS)
        self.assertIsNone(r["all_calls"])
        self.assertEqual(r["rows"], [])


if __name__ == "__main__":
    unittest.main()


class ExportTest(unittest.TestCase):
    def test_scenarios_get_riskier_with_rain_and_the_file_has_what_the_page_reads(self):
        import json
        import tempfile
        from pathlib import Path

        from score import export_site

        if not Path("model/artifacts/meta.json").exists():
            self.skipTest("model artifacts not built")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "log").mkdir()
            (d / "pred").mkdir()
            sites = pd.read_csv("model/artifacts/sites.csv")
            rows = [{"date": "2026-11-18", "site_key": s.site_key, "site_name": s.site_name, "p200": 0.05, "p40": 0.2, "level": "Possible",
                     "rain_24h": 3.0, "rain_72h": 9.0, "trained_through": "2026-11-12", "scored_at": "2026-11-17T20:12:00Z"} for s in sites.itertuples()]
            (d / "pred" / "2026-11.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (d / "log" / "2026-11.jsonl").write_text("\n".join(json.dumps({**r, "lat": -33.9, "lon": 151.2}) for r in LOG.drop(columns="site_key").to_dict("records")) + "\n")
            self.assertEqual(export_site.main(["--log", str(d / "log"), "--predictions", str(d / "pred"), "--out", str(d / "site.json")]), 0)
            out = json.load(open(d / "site.json"))
            self.assertEqual(len(out["sites"]), len(sites))
            bondi = next(s for s in out["sites"] if s["key"] == "bondi-beach")
            self.assertEqual((bondi["official"], bondi["exact"]), ("Likely", True))
            dry = sum(s["what_if"][0] for s in out["sites"])
            wet = sum(s["what_if"][-1] for s in out["sites"])
            self.assertGreater(wet, 5 * dry)
            for key in ("date", "levels", "scenarios_mm", "backtest", "track_record", "registry", "days_scored"):
                self.assertIn(key, out)
