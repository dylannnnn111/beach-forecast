# Beach Forecast

Which Sydney beaches are safe to swim at each morning? A model trained on NSW Beachwatch water samples and rainfall, scored every day against the official pollution forecast.

Status: early. The first piece is the forecast logger - Beachwatch publishes only its latest forecast, so this repo records it twice a day to build the track record the model gets scored against.

## Forecast logger
- `logger/log_forecasts.py` calls the public Beachwatch API and appends one line per site per forecast issue to `data/forecasts/YYYY-MM.jsonl`.
- `.github/workflows/log-forecasts.yml` runs it at about 7:37am and 1:37pm Sydney time and commits new rows.
- Run locally: `python -m logger.log_forecasts`. Tests: `python -m unittest discover tests`.

## Data
Beachwatch data: © State Government of NSW and NSW Department of Climate Change, Energy, the Environment and Water, Creative Commons Attribution. This project is not affiliated with Beachwatch and is not swimming advice.
