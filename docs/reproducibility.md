# Reproducibility

## Checklist

- Fixed random seeds
- Chronological data splits
- Train-only scaler fitting
- Versioned raw-data archive source URLs
- Saved resolved Hydra config per run
- Saved feature manifest and label definition
- Saved artifact metadata for inference parity
- Deterministic export paths for metrics and predictions

## Data provenance

The initial data source is Binance USD-M futures public archives hosted under `data.binance.vision`. Raw archives should be downloaded into `data/raw/` without manual modification.

## Experiment registry

Each run directory should contain:

- `config.yaml`
- `metadata.json`
- `metrics.json`
- `predictions.csv`
- `artifact.pt` or `artifact.pkl`
- `scaler.json`
- optional `model.onnx`

## Known limitations

- Liquidation features are scaffolded but not source-complete.
- Public exchange metrics may change schema over time and should be validated on ingestion.
- Browser export is supported at the artifact level, not yet as a production deployment stack.
