"""Training and evaluation orchestration."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from markettensor.evaluation.classification import classification_metrics
from markettensor.evaluation.splits import apply_purge, chronological_split
from markettensor.evaluation.trading import trading_metrics
from markettensor.inference.export import InferenceArtifact, save_artifact_metadata
from markettensor.models.registry import build_model, is_torch_model
from markettensor.training.callbacks import LossTracker
from markettensor.training.dataset import (
    SequenceDataset,
    build_windowed_arrays,
    reshape_features_for_model,
)
from markettensor.training.losses import binary_classification_loss
from markettensor.training.preprocess import TrainOnlyScaler
from markettensor.utils.config import ensure_dir, save_json, save_yaml
from markettensor.utils.seed import seed_everything


@dataclass
class TrainingResult:
    """Artifacts produced by a training run."""

    run_dir: Path
    metrics: dict[str, float]
    predictions: pd.DataFrame


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _save_model(model: Any, path: Path) -> None:
    if is_torch_model(model):
        torch.save(model.state_dict(), path)
        return
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def train_experiment(
    config: dict[str, Any], dataset: pd.DataFrame, feature_columns: list[str]
) -> TrainingResult:
    """Train a configured experiment and persist a run directory."""

    seed_everything(int(config["train"]["seed"]))
    split = chronological_split(
        dataset,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )
    train_index = apply_purge(split.train, purge_bars=int(config["eval"]["purge_bars"]))
    train_frame = dataset.loc[train_index].sort_values(["symbol", "timestamp"])
    val_frame = dataset.loc[split.val].sort_values(["symbol", "timestamp"])
    test_frame = dataset.loc[split.test].sort_values(["symbol", "timestamp"])

    scaler = TrainOnlyScaler.fit(train_frame, feature_columns)
    artifact_root = ensure_dir(Path(config["train"]["output_dir"]))
    run_id = f"{config['run_name']}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(artifact_root / run_id)

    if config["model"]["family"] == "sklearn":
        predictions = _train_sklearn_model(
            config, train_frame, test_frame, feature_columns, scaler, run_dir
        )
    else:
        predictions = _train_torch_model(
            config,
            train_frame,
            val_frame,
            test_frame,
            feature_columns,
            scaler,
            run_dir,
        )

    prediction_metrics = classification_metrics(
        targets=predictions["target"].to_numpy(),
        predictions=predictions["prediction"].to_numpy(),
        probabilities=predictions["probability"].to_numpy(),
    )
    strategy_metrics = trading_metrics(
        probabilities=predictions["probability"].to_numpy(),
        future_returns=predictions["future_return"].to_numpy(),
        fee_bps=float(config["eval"]["fee_bps"]),
        slippage_bps=float(config["eval"]["slippage_bps"]),
    )
    metrics = {**prediction_metrics, **strategy_metrics}

    save_yaml(config, run_dir / "config.yaml")
    save_json(metrics, run_dir / "metrics.json")
    save_json(scaler.state_dict(), run_dir / "scaler.json")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    artifact = InferenceArtifact(
        model_name=config["model"]["name"],
        feature_names=feature_columns,
        label_name=config["labels"]["name"],
        lookback=int(config["model"]["lookback"]),
        scaler_state=scaler.state_dict(),
    )
    save_artifact_metadata(artifact, run_dir / "artifact_metadata.json")
    save_json({"run_id": run_id, "feature_columns": feature_columns}, run_dir / "metadata.json")
    return TrainingResult(run_dir=run_dir, metrics=metrics, predictions=predictions)


def _train_sklearn_model(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: TrainOnlyScaler,
    run_dir: Path,
) -> pd.DataFrame:
    model = build_model(config["model"], input_dim=len(feature_columns))
    x_train = scaler.transform(train_frame)
    y_train = train_frame["target"].astype(int).to_numpy()
    x_test = scaler.transform(test_frame)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    _save_model(model, run_dir / "model.pkl")
    return test_frame.loc[:, ["timestamp", "symbol", "target", "future_return"]].assign(
        probability=probability,
        prediction=prediction,
    )


def _train_torch_model(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: TrainOnlyScaler,
    run_dir: Path,
) -> pd.DataFrame:
    scaled_train = train_frame.copy()
    scaled_val = val_frame.copy()
    scaled_test = test_frame.copy()
    scaled_train.loc[:, feature_columns] = scaler.transform(train_frame)
    scaled_val.loc[:, feature_columns] = scaler.transform(val_frame)
    scaled_test.loc[:, feature_columns] = scaler.transform(test_frame)
    lookback = int(config["model"]["lookback"])

    train_arrays = build_windowed_arrays(scaled_train, feature_columns, lookback=lookback)
    val_arrays = build_windowed_arrays(scaled_val, feature_columns, lookback=lookback)
    test_arrays = build_windowed_arrays(scaled_test, feature_columns, lookback=lookback)
    model_name = config["model"]["name"]
    train_features = reshape_features_for_model(train_arrays.features, model_name=model_name)
    val_features = reshape_features_for_model(val_arrays.features, model_name=model_name)
    test_features = reshape_features_for_model(test_arrays.features, model_name=model_name)
    if len(train_features) == 0:
        raise ValueError("Training split produced no usable windows.")
    if len(val_features) == 0:
        raise ValueError("Validation split produced no usable windows.")
    if len(test_features) == 0:
        raise ValueError("Test split produced no usable windows.")
    input_dim = train_features.shape[-1]
    model = build_model(config["model"], input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["model"]["learning_rate"]))
    loss_fn = binary_classification_loss()
    tracker = LossTracker(losses=[])
    loader = DataLoader(
        SequenceDataset(train_features, train_arrays.targets),
        batch_size=int(config["model"]["batch_size"]),
        shuffle=False,
    )
    patience = int(config["model"].get("early_stopping_patience", 3))
    min_delta = float(config["model"].get("early_stopping_min_delta", 0.0))
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    val_loss_history: list[float] = []

    model.train()
    for epoch in range(int(config["model"]["epochs"])):
        epoch_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        tracker.log(epoch_loss / max(len(loader), 1))
        val_loss = _evaluate_torch_loss(model, val_features, val_arrays.targets, loss_fn)
        val_loss_history.append(val_loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state_dict is None:
        raise RuntimeError("Deep training did not produce a valid validation checkpoint.")
    model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_features)).numpy()
    probability = _sigmoid(logits)
    prediction = (probability >= 0.5).astype(int)
    test_targets = (
        scaled_test.loc[test_arrays.row_indices, ["timestamp", "symbol", "target", "future_return"]]
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    _save_model(model, run_dir / "model.pt")
    save_json(
        {
            "train_loss_history": tracker.losses,
            "val_loss_history": val_loss_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        run_dir / "training_history.json",
    )
    return test_targets.assign(probability=probability, prediction=prediction)


def _evaluate_torch_loss(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    loss_fn: torch.nn.Module,
) -> float:
    """Evaluate binary loss on a validation slice."""

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        loss = loss_fn(logits, torch.from_numpy(targets.astype(np.float32)))
    model.train()
    return float(loss.item())


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load saved predictions for a run."""

    return pd.read_csv(run_dir / "predictions.csv", parse_dates=["timestamp"])


def latest_run_path(root: Path) -> Path:
    """Return the most recent run directory."""

    runs = sorted(root.iterdir())
    if not runs:
        raise FileNotFoundError(f"No runs found in {root}.")
    return runs[-1]
