import unittest

from ingest.samples import clean, number, site_key

RAW = {
    "Region": "Sydney City", "Council": "Waverley Council", "Swim site": "Bondi Beach",
    "Date": "13/11/2026", "Time": "07:34:00", "Enterococci (cfu/100mL)": "3",
    "Water temperature (°C)": "20", "Conductivity (µS/cm)": "53600",
    "Latitude": "-33.891157000", "Longitude": "151.278267000",
}


class SamplesTest(unittest.TestCase):
    def test_number_handles_censored_and_blank(self):
        self.assertEqual(number("12"), (12.0, ""))
        self.assertEqual(number("<1"), (1.0, "<"))
        self.assertEqual(number(">2000"), (2000.0, ">"))
        self.assertEqual(number(""), (None, ""))
        self.assertEqual(number("n/a"), (None, "?"))

    def test_site_key(self):
        self.assertEqual(site_key("Shelly Beach (Sutherland)"), "shelly-beach-sutherland")

    def test_clean_parses_date_and_numbers(self):
        rows, dropped = clean([RAW])
        self.assertEqual(dropped, 0)
        r = rows[0]
        self.assertEqual((r["date"], r["time"], r["enterococci"], r["water_temp_c"]), ("2026-11-13", "07:34", 3.0, 20.0))

    def test_drops_blank_results_duplicates_and_bad_dates(self):
        blank = dict(RAW, **{"Enterococci (cfu/100mL)": ""})
        bad_date = dict(RAW, Date="2026-13-45")
        rows, dropped = clean([RAW, RAW, blank, bad_date])
        self.assertEqual((len(rows), dropped), (1, 3))

    def test_missing_optional_measures_stay_blank(self):
        rows, _ = clean([dict(RAW, **{"Water temperature (°C)": "", "Conductivity (µS/cm)": ""})])
        self.assertEqual((rows[0]["water_temp_c"], rows[0]["conductivity"]), ("", ""))

    def test_sorted_by_date(self):
        rows, _ = clean([RAW, dict(RAW, Date="01/01/2010")])
        self.assertEqual([r["date"] for r in rows], ["2010-01-01", "2026-11-13"])


if __name__ == "__main__":
    unittest.main()
