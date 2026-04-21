# DATATHON Part 3 Forecasting

## Part 3 pipeline

This repository now includes an end-to-end Part 3 script:

- `part3_forecast.py`

It builds a daily feature mart from all provided CSV files, forecasts exogenous driver series for test horizon, then predicts `Revenue` and `COGS` using a two-model blend (LightGBM + CatBoost when available, ridge fallback otherwise).

## Run

From project root:

```bash
.venv/bin/python part3_forecast.py --n-trials 20 --out-dir outputs/part3
```

Arguments:

- `--base-dir`: folder containing all competition CSV files (default: current repo root)
- `--out-dir`: output folder for submission and reports (default: `outputs/part3`)
- `--n-trials`: random-search trials per model/target (default: `20`)
- `--seed`: random seed (default: `42`)

## Outputs

The script writes:

- `outputs/part3/submission.csv`
- `outputs/part3/driver_model_selection.json`
- `outputs/part3/target_model_report.json`
- `outputs/part3/submission_checks.json`

`submission.csv` is aligned to `sample_submission.csv` date order and contains:

- `Date`
- `Revenue`
- `COGS`
