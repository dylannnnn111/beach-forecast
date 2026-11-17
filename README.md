# Beach Forecast

Which Sydney beaches are safe to swim at each morning? A model trained on NSW Beachwatch water samples and hourly rainfall, scored every day against the official pollution forecast.

Status: in progress. The pipeline runs end to end and scores every site each morning. The head-to-head with the official forecast started on 17 Nov 2026, because Beachwatch publishes only its latest forecast and nobody keeps the history - so this repo records it.

## How it works
1. `logger/` records the official Beachwatch forecast for every site twice a day -> `data/forecasts/`.
2. `ingest/samples.py` cleans the Beachwatch sample download (enterococci results, 112 Sydney sites, 2010 on).
3. `ingest/rain.py` downloads hourly rainfall for each site from Open-Meteo's historical API.
4. `features/build.py` joins each sample to the rain before it, as of 6am on the sample day. No rain after 6am, no measurements taken with the sample, no lab results less than a week old.
5. `model/backtest.py` tests year by year: train on every earlier year, test on the next. Results in `out/backtest.json`.
6. `model/train.py` fits the final models -> `model/artifacts/`.
7. `score/daily.py` runs each morning on GitHub Actions and writes a risk for every site -> `data/predictions/`.
8. `score/track_record.py` lines up each morning's call, the official forecast and the lab result that came in later -> `out/track_record.json`.
9. `retrain/weekly.py` runs every week: tops up the rain, turns the new lab results the logger has seen into labels, rebuilds the features and retrains. A new model only replaces the live one if the checks pass; every run is recorded in `model/registry.json`.

## Backtest (test years 2016-2026, 50,701 site-days)
Target: a sample above 200 enterococci per 100 mL (3.9% of site-days). Average precision, where a random guess scores 0.039:

| Method | Average precision |
|---|---|
| Site history only | 0.132 |
| Rain-only rule | 0.183 |
| Model | 0.316 |

The rain-only rule is this project's own stand-in for a rainfall-based forecast. It is not the official forecast, which has no public history. Settings are near default and untuned.

## Known limits
- Samples are taken about every six days per site, so the model can only be checked on those days.
- The rainfall is modelled on a grid, not measured at a gauge: the 112 sites share 15 grid points, and big local storms are smoothed out.
- New labels come from the public feed as categories (Good / Fair / Poor / Bad), which line up with the count thresholds; checked on 106 samples that appear in both.
- The daily job reads recent rain from Open-Meteo's forecast API while training used its historical archive. The two can differ; that gap has not been measured yet.

## Run it
`pip install -r requirements.txt`, then `python -m unittest discover tests`. The raw sample download is not in the repo (`data/raw/` is ignored), but the labels built from it (`data/training/`) and the rainfall (`data/rain/`) are, so the backtest and retrain run from a clone: `python -m features.build && python -m model.backtest`.

## Data and credit
- Beachwatch data: © State Government of NSW and NSW Department of Climate Change, Energy, the Environment and Water, Creative Commons Attribution.
- Weather data by Open-Meteo.com (CC BY 4.0).

This project is not affiliated with Beachwatch and is not swimming advice.
