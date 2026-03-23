# Paper Notes

## Motivation

Recent CNN-based financial forecasting claims often rely on weak evaluation discipline, unclear preprocessing, or market settings that do not generalize to crypto perpetuals. MarketTensor is designed to reproduce the attractive parts of those claims while correcting the methodological weaknesses.

## Relation to the inspiration paper

The immediate inspiration is Rahul Gupta's December 2025 arXiv paper on CNN-based S&P 500 movement prediction. MarketTensor does not attempt a direct dataset reproduction. Instead, it borrows the idea of a CNN-centered benchmark and relocates it to crypto perpetuals with stronger time-series safeguards.

## Shortcomings in prior work

- Randomized or weakly controlled splits
- Insufficient leakage analysis
- Underdeveloped ablations
- OHLCV-only focus despite richer market microstructure in perpetuals
- Incomplete reproducibility artifacts

## Proposed contributions

- Leak-free benchmark framework for crypto perpetual direction forecasting
- Transparent OHLCV versus microstructure-signal ablations
- Strong baseline suite beyond a single deep model
- Experiment registry designed for publication and replication
- Future path to lightweight browser inference

## Experiment matrix

- Assets: BTCUSDT, ETHUSDT, SOLUSDT
- Horizons: 1, 4, 12, 24 bars
- Feature families: OHLCV, funding, open interest, liquidation hooks, combined
- Training modes: single-asset and pooled-symbol
- Models: logistic, boosting, MLP, CNN1D, TCN, LSTM

## Threats to validity

- Exchange-specific biases in public futures data
- Regime dependence across crypto market cycles
- Imperfect proxy quality in public metrics features
- Missing high-quality historical liquidation data in the initial open-source stack
