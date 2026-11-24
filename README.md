# Perpetual-Funding-Cost-VaR

Code accompanying **"Funding Is Not Free: Deterministic Bounds, a Collateral-Erosion
Identity, and Out-of-Sample Risk Forecasts for Cumulative Perpetual-Futures Funding
Costs"** (Mesias Alfeus, Tina Prodromou), submitted to *Frontiers of Mathematical
Finance* (AIMS) as a Rapid Communication.

## What this is

A reproducible pipeline for:

- building an hourly Deribit BTC-PERPETUAL panel from raw ticker/trade data, with an
  explicit missing-hour convention (unobserved, never interpolated, never silently
  dropped);
- eleven out-of-sample forecasting models for the 95% quantile of cumulative funding
  cost at 72h and 168h horizons (historical simulation, quantile regression, CAViaR,
  HAR-QR, ARX-GARCH-t, GAS-t, MS-GARCH-t, SV-t, SVJ-t, a fractional-kernel quantile
  model, and a stationary block bootstrap);
- backtest diagnostics (Kupiec, Christoffersen, Diebold-Mariano with HAC variance and
  Benjamini-Hochberg FDR control, adaptive conformal recalibration);
- a Stage 1 test of whether forecast funding-cost VaR predicts price-only VaR
  failures (logit and Firth-penalised);
- a collateral-erosion / peak-additional-collateral-requirement (PACR) analysis and a
  reserve-policy comparison (McNemar's exact test).

## What is not included

Raw Deribit/Binance ticker and trade data are not redistributed here (see the paper's
Data paragraph for provider and sample details). `analysis/output/` (run artifacts)
is likewise not included; running `run_pipeline.py` and the `run_*.py` scripts
against your own data reproduces it.

## Layout

```
analysis/
  config.py              paths, horizons, leverage grid, estimation windows
  data_loading.py         raw file readers
  data_validation.py      funding-convention checks (Deribit continuous accrual vs.
                          Binance's discrete settlement)
  panel.py                hourly panel construction (missing-hour handling)
  funding_cost.py         C^g, C^s, L^P, L^N, collateral drain
  forecast_set.py         predictor construction
  backtesting.py          Kupiec, Christoffersen, quantile loss, Diebold-Mariano
  recalibration.py        adaptive conformal recalibration (Gibbs & Candes, 2021)
  stage1_omission.py      Stage 1: does funding-cost VaR predict price-only VaR failures?
  omission_gap.py         omission-gap (G_t) construction and regressions
  collateral_erosion.py   PACR identity, reserve-policy shortfall tests
  models/                 the eleven forecasting models
  tests/                  automated tests, incl. missing-hour/lag-alignment adversarial tests
  run_pipeline.py         main entry point (python -m analysis.run_pipeline --horizon 72 ...)
  run_*.py                auxiliary analyses (collateral erosion, recalibration, cash timing, ...)
```

## Reproducing a run

```bash
pip install -r analysis/requirements.txt
python -m analysis.run_pipeline --start 2020-07-22 --end 2025-06-30 --horizon 72 \
    --models historical_simulation direct_quantile caviar arx_garch_t sv_t svj_t \
             rough_vol har_quantile gas_t ms_garch_t stationary_bootstrap \
    --out analysis/output/your_run/H72
python -m pytest analysis/tests
```

## Citation

If you use this code, please cite the paper above (details to follow on acceptance).
