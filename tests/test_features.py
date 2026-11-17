import unittest

import numpy as np
import pandas as pd

from features.build import rain_features, site_history


class RainFeatureTest(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2020-01-01T00:00")
        self.p = np.zeros(400)

    def feats(self, hour):
        cut = pd.DatetimeIndex([self.start + pd.Timedelta(hours=hour)])
        return rain_features(self.start, self.p, cut).iloc[0]

    def test_window_includes_the_hour_ending_at_the_cutoff(self):
        self.p[200] = 5.0  # rain that fell 199:00-200:00
        f = self.feats(200)
        self.assertEqual(f["rain_6h"], 5.0)
        self.assertEqual(f["hours_since_1mm"], 0)

    def test_rain_after_the_cutoff_is_never_used(self):
        self.p[201] = 9.0
        f = self.feats(200)
        self.assertEqual(f["rain_7d"], 0.0)
        self.assertEqual(f["rain_max1h_24h"], 0.0)

    def test_window_edges(self):
        self.p[176] = 2.0  # 24 hours before the cutoff stamp: outside 24h, inside 48h
        self.p[177] = 3.0  # inside 24h
        f = self.feats(200)
        self.assertEqual((f["rain_24h"], f["rain_48h"], f["rain_max1h_24h"]), (3.0, 5.0, 3.0))
        self.assertEqual(f["hours_since_1mm"], 23)

    def test_sample_before_the_record_starts_is_an_error(self):
        with self.assertRaises(ValueError):
            self.feats(10)


class HistoryTest(unittest.TestCase):
    def test_history_only_uses_results_at_least_a_week_old(self):
        df = pd.DataFrame({
            "site_key": ["a"] * 4,
            "date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-08", "2020-01-20"]),
            "y200": [1, 0, 0, 0],
            "y40": [1, 0, 0, 0],
        })
        h = site_history(df)
        self.assertTrue(np.isnan(h.loc[0, "hist_rate200"]))
        self.assertEqual(h.loc[1, "hist_n"], 0)            # 1 Jan is only 4 days old
        self.assertEqual(h.loc[2, "hist_n"], 1)            # 1 Jan is exactly 7 days old
        self.assertEqual(h.loc[2, "hist_rate200"], 1.0)
        self.assertEqual(h.loc[3, "hist_n"], 3)
        self.assertAlmostEqual(h.loc[3, "hist_rate200"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
