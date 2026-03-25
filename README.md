# MarketTensor

MarketTensor is a research-first framework for reproducible directional forecasting experiments in crypto perpetuals and futures markets. The repository is designed as benchmark infrastructure, not a trading product, and emphasizes leak-free preprocessing, strict time-series evaluation, and paper-ready experiment management.

The project is inspired by Rahul Gupta's December 2025 arXiv submission, "S&P 500 Stock's Movement Prediction using CNN", but it deliberately raises the scientific bar. MarketTensor focuses on chronological evaluation, explicit leakage controls, richer crypto-perpetual market signals, stronger baseline models, and experiment structures that can support a publishable benchmark study.

## Why this repository exists

Many financial forecasting repositories mix exploratory code, loosely defined preprocessing, and time-series leakage. MarketTensor exists to provide a more defensible alternative:

- No random shuffling across temporal splits.
- Train, validation, and test segmentation by time.
- Train-only scaling and deterministic preprocessing artifacts.
- Config-driven experiments and ablations.
- Clear separation between prediction metrics and trading metrics.
- Open-source research structure that can evolve into a paper and later into lightweight browser inference.

## Supported signals

Initial data families:

- OHLCV
- Funding rate
- Open-interest-style exchange metrics
- Liquidation feature hooks with explicit source errors until a reproducible historical source is added

Initial target markets:

- BTCUSDT perpetual
- ETHUSDT perpetual
- SOLUSDT perpetual

## Benchmark scope

Implemented or scaffolded experiment dimensions:

- Feature sets: `ohlcv`, `ohlcv_funding`, `ohlcv_open_interest`, `ohlcv_liquidation`, combined variants
- Labels: next-bar direction, k-bar direction, return-threshold classification
- Horizons: 1, 4, 12, 24 bars
- Models: logistic regression, histogram gradient boosting, MLP, 1D CNN, TCN, LSTM
- Evaluation: chronological split, walk-forward evaluation, expanding and rolling windows, pooled and per-symbol analysis

## Repository layout

```text
MarketTensor/
├── configs/
├── data/
├── docs/
├── notebooks/exploration/
├── scripts/
├── src/markettensor/
└── tests/
```

The `.lab/` directory is reserved for gitignored daily research notes and experiment journal entries.

## Quickstart

1. Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

2. Download raw futures archives:

```bash
python scripts/download_data.py --symbol BTCUSDT --interval 1h
```

3. Build a processed dataset:

```bash
python scripts/build_dataset.py --config-name cnn_ohlcv_funding
```

4. Train a model:

```bash
python scripts/train.py --config-name cnn_ohlcv_funding
```

5. Evaluate a run:

```bash
python scripts/evaluate.py --run-id latest
python scripts/backtest.py --run-id latest
python scripts/export_onnx.py --run-id latest
```

## Example experiment configs

- `cnn_ohlcv`
- `cnn_ohlcv_funding`
- `cnn_ohlcv_funding_open_interest`
- `cnn_ohlcv_liquidation`
- `cnn_combined_all`
- `lstm_ohlcv`
- `logistic_ohlcv`
- `mlp_ohlcv_open_interest`

Each experiment stores the resolved Hydra configuration, feature manifest, scaler parameters, metrics, predictions, and model artifacts for reproducibility.

## Corrected Benchmark Snapshot

The repository now includes a longer-history walk-forward benchmark over raw Binance USD-M futures archives downloaded for `2023-01-01` through `2025-12-31`, using pooled `BTCUSDT`, `ETHUSDT`, and `SOLUSDT` data. The current README-facing results use the corrected trading evaluator:

- positions are taken from saved binary predictions, not re-thresholded probabilities
- turnover is computed per symbol rather than across pooled assets
- multi-bar trading metrics use non-overlapping holding periods

A raw-archive integrity sweep after download verified zero missing checksum files and zero checksum mismatches across:

- `165` kline archives per symbol
- `36` funding-rate archives per symbol
- `1,096` metrics archives per symbol

### Naive baselines vs learned models on `1h`

The main `1h` comparison in [docs/results/long_history_naive_vs_best_summary.csv](docs/results/long_history_naive_vs_best_summary.csv) uses four expanding folds and compares naïve baselines against the current strongest learned candidates.

Headline results:

- Best mean accuracy: `LogReg` at `0.5123`
- Best mean ROC-AUC: `LogReg` at `0.5163`
- Best learned trading result: `1D CNN (OHLCV + Open Interest)` with cumulative return `0.0838` and Sharpe `0.0828`
- Naïve `Majority` is still very close, with cumulative return `0.0728` and Sharpe `0.0822`

This is the current honest takeaway for `1h`: the learned models are only marginally better than trivial baselines, and most of the apparent edge comes from small trading improvements rather than strong classification separation.

![Long-history naive vs best summary](docs/figures/long_history_naive_vs_best_summary.png)

![Long-history naive vs best fold traces](docs/figures/long_history_naive_vs_best_fold_traces.png)

Per-symbol and multi-horizon tables are also available:

- [docs/results/long_history_naive_vs_best_per_symbol_summary.csv](docs/results/long_history_naive_vs_best_per_symbol_summary.csv)
- [docs/results/long_history_naive_vs_best_horizons_summary.csv](docs/results/long_history_naive_vs_best_horizons_summary.csv)

### Timeframe comparison

The same pooled four-fold protocol was then run across `15m`, `1h`, and `4h` bars for `Majority`, `LogReg`, `LSTM`, and `1D CNN (OHLCV + Open Interest)`. The combined result table is in [docs/results/long_history_timeframe_models_summary.csv](docs/results/long_history_timeframe_models_summary.csv).

Current timeframe takeaways:

- `15m`: no useful learned edge; `LSTM` has the best classification (`0.5150` accuracy, `0.5202` ROC-AUC) but negative trading metrics, while `Majority` is the best trading baseline with Sharpe `0.0441`
- `1h`: `LSTM` remains the best classifier, but `1D CNN (OHLCV + Open Interest)` is the best trading model with Sharpe `0.0828`
- `4h`: this is the strongest setup so far; `LSTM` has the best classification (`0.5219` accuracy), while `1D CNN (OHLCV + Open Interest)` has the best trading result with cumulative return `0.3140` and Sharpe `0.8068`

The current research interpretation is therefore cautious but useful: lower timeframes are still close to noise, `1h` contains only weak signal, and `4h` is the first interval where the learned models separate more meaningfully from the trivial baselines under the corrected backtest.

![Long-history timeframe comparison](docs/figures/long_history_timeframe_models_summary.png)

### Recreate the current snapshot

```bash
python scripts/run_walk_forward.py \
  --suite-name long_history_naive_vs_best \
  --config-name majority_ohlcv \
  --config-name random_ohlcv \
  --config-name persistence_ohlcv \
  --config-name logistic_ohlcv \
  --config-name cnn_ohlcv_open_interest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --override experiment.eval.walk_forward.n_splits=4
```

```bash
python scripts/generate_walk_forward_figures.py \
  --summary-csv docs/results/long_history_naive_vs_best_summary.csv \
  --folds-csv docs/results/long_history_naive_vs_best_folds.csv \
  --prefix long_history_naive_vs_best \
  --title "Long-History Walk-Forward Naive Baselines vs Best Learned Models"
```

```bash
python scripts/run_walk_forward.py \
  --suite-name long_history_tf_4h_models \
  --config-name majority_ohlcv \
  --config-name logistic_ohlcv \
  --config-name lstm_ohlcv \
  --config-name cnn_ohlcv_open_interest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --override experiment.eval.walk_forward.n_splits=4 \
  --override experiment.data.interval=4h
```

```bash
python scripts/run_walk_forward.py \
  --suite-name long_history_tf_1h_models \
  --config-name majority_ohlcv \
  --config-name logistic_ohlcv \
  --config-name lstm_ohlcv \
  --config-name cnn_ohlcv_open_interest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --override experiment.eval.walk_forward.n_splits=4 \
  --override experiment.data.interval=1h
```

```bash
python scripts/run_walk_forward.py \
  --suite-name long_history_tf_15m_models \
  --config-name majority_ohlcv \
  --config-name logistic_ohlcv \
  --config-name lstm_ohlcv \
  --config-name cnn_ohlcv_open_interest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --override experiment.eval.walk_forward.n_splits=4 \
  --override experiment.data.interval=15m
```

```bash
python scripts/generate_timeframe_tables.py \
  --suite long_history_tf_15m_models:15m:docs/results/long_history_tf_15m_models_summary.csv \
  --suite long_history_tf_1h_models:1h:docs/results/long_history_tf_1h_models_summary.csv \
  --suite long_history_tf_4h_models:4h:docs/results/long_history_tf_4h_models_summary.csv \
  --output-prefix docs/results/long_history_timeframe_models_summary
```

```bash
python scripts/generate_timeframe_figures.py \
  --summary-csv docs/results/long_history_timeframe_models_summary.csv \
  --prefix long_history_timeframe_models_summary \
  --title "Long-History Walk-Forward Comparison Across Timeframes"
```

## Methodology principles

- Chronological train/validation/test splits only
- Optional purge and embargo support for overlapping labels
- Lagged feature alignment to avoid contemporaneous leakage
- Deterministic seeds
- Clear distinction between statistical performance and simulated trading outcomes
- Reproducible experiment registry and artifact layout

See [docs/methodology.md](docs/methodology.md), [docs/reproducibility.md](docs/reproducibility.md), and [docs/paper_notes.md](docs/paper_notes.md) for details.

## Roadmap

- Add reproducible liquidation data integration
- Extend exchange adapters beyond Binance USD-M futures
- Add richer microstructure features such as basis and mark-index spread
- Expand experiment registry outputs into paper tables and publication figures
- Export lightweight models for browser-based inference and chart-indicator workflows

## Disclaimer

MarketTensor is research infrastructure for forecasting experiments. It is not investment advice, not a trading recommendation system, and not a production execution stack.
