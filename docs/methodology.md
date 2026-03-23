# Methodology

## Research objective

MarketTensor studies directional classification in crypto perpetuals and futures markets under strict time-series controls. The initial benchmark compares OHLCV-only models against variants that include funding and exchange metrics that proxy market microstructure state.

## Evaluation rules

- No random shuffling across train, validation, and test partitions.
- All splits are chronological and configurable.
- Walk-forward evaluation is a first-class evaluation path.
- Preprocessing parameters such as scaling are fit on the training partition only.
- Feature alignment uses lagged or backward-looking joins to avoid contemporaneous leakage.
- Prediction metrics and trading metrics are reported separately.

## Feature families

- OHLCV bar features
- Funding-rate features
- Open-interest-style metrics from exchange archives
- Liquidation feature hooks pending a reproducible historical source

## Label families

- Next-bar direction
- K-bar direction
- Return-threshold classification
- Triple-barrier labeling is reserved for later work

## Baselines and primary models

The baseline suite includes logistic regression, histogram gradient boosting, and MLP. The first deep benchmark family includes 1D CNN, TCN, and LSTM models. The 1D CNN is the primary architecture for reproducing and extending CNN-based market forecasting claims.

## Reporting

Every run should save:

- resolved configuration
- seed
- feature manifest
- scaler statistics
- fold definitions
- predictions and targets
- classification metrics
- cost-adjusted trading metrics
