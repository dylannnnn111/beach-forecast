"""Score the model and the official forecast against the lab results that came in later.

For every sample the logger has seen since scoring started: what did the model say
that morning, what did the official forecast say, and what did the lab find?
Writes out/track_record.json. Only days with a model call are counted, and the
head-to-head only uses days where the official forecast also made a call.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingest.log_labels import labels, official, read_log

LEVELS = ("Unlikely", "Possible", "Likely")


def read_predictions(pred_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(pred_dir).glob("*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def summary(calls: pd.Series, y200: pd.Series, y40: pd.Series) -> dict:
    out = {"n": int(len(calls)), "above_200": int(y200.sum()), "above_40": int(y40.sum()), "by_level": {}}
    for lv in LEVELS:
        m = calls == lv
        out["by_level"][lv] = {"n": int(m.sum()), "above_200": int(y200[m].sum()), "above_40": int(y40[m].sum())}
    flagged = calls != "Unlikely"
    out["flagged"] = int(flagged.sum())
    out["caught_above_200"] = int(y200[flagged].sum())
    out["caught_above_40"] = int(y40[flagged].sum())
    return out


def build(log: pd.DataFrame, preds: pd.DataFrame) -> dict:
    result = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "first_day": None, "last_day": None, "all_calls": None, "head_to_head": None, "rows": []}
    if log.empty or preds.empty:
        return result
    lab = labels(log)
    j = preds.merge(lab, on=["site_key", "date"], how="inner").merge(official(log), on=["site_key", "date"], how="left")
    if j.empty:
        return result
    j = j.sort_values(["date", "site_key"])
    result["first_day"], result["last_day"] = str(j["date"].min().date()), str(j["date"].max().date())
    result["all_calls"] = {"model": summary(j["level"], j["y200"], j["y40"])}
    h = j[j["official"].notna()]
    if len(h):
        result["head_to_head"] = {"model": summary(h["level"], h["y200"], h["y40"]),
                                  "official": summary(h["official"], h["y200"], h["y40"])}
    result["rows"] = [{"date": str(r.date.date()), "site_key": r.site_key, "site_name": r.site_name, "model": r.level,
                       "p200": r.p200, "official": None if pd.isna(r.official) else r.official,
                       "above_200": int(r.y200), "above_40": int(r.y40)} for r in j.itertuples()]
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/forecasts")
    ap.add_argument("--predictions", default="data/predictions")
    ap.add_argument("--out", default="out/track_record.json")
    a = ap.parse_args(argv)
    res = build(read_log(a.log), read_predictions(a.predictions))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    n = res["all_calls"]["model"]["n"] if res["all_calls"] else 0
    print(f"track record: {n} scored sample-days, {res['first_day']} to {res['last_day']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
