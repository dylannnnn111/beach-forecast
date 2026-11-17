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
