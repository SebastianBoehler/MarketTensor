# Browser Deployment

## Goal

The long-term deployment target is a lightweight inference path suitable for browser-side use as a chart indicator or overlay. The training code therefore stores deterministic preprocessing metadata and supports ONNX export.

## Constraints

- Browser inference must use a compact model with bounded memory and latency.
- Live environments may not provide every offline research feature in real time.
- Preprocessing must exactly match the training-time feature order, scaling, and lag policy.
- Any missing live feature should fail explicitly rather than silently degrade into a different model input contract.

## Recommended v1 export target

- Small CNN1D or MLP variants
- ONNX artifact plus JSON feature manifest
- Deterministic scaler parameters

## Chart-indicator use case

Potential browser usage:

- render the latest directional probability beside price bars
- overlay regime or confidence state on a trading chart
- compare model outputs across horizons

## Live feature availability

- OHLCV is usually available live
- Funding is delayed and lower-frequency
- Exchange metrics may have cadence or endpoint restrictions
- Liquidation data remains source-dependent and is not yet production ready

## Latency targets

For browser-side inference, a practical target is low tens of milliseconds per inference on a single latest window. Model export and preprocessing should therefore avoid large transformer-style architectures until the feature contract is stable.
