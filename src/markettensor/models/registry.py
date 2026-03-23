"""Model registry."""

from __future__ import annotations

from typing import Any

from torch import nn

from markettensor.models.cnn1d import CNN1DClassifier
from markettensor.models.logistic import build_logistic_regression
from markettensor.models.lstm import LSTMClassifier
from markettensor.models.mlp import MLPClassifier
from markettensor.models.tcn import TCNClassifier
from markettensor.models.tree import build_hist_gradient_boosting


def build_model(model_config: dict[str, Any], input_dim: int) -> Any:
    """Instantiate a model from config."""

    name = model_config["name"]
    if name == "logistic":
        return build_logistic_regression(max_iter=model_config.get("max_iter", 500))
    if name == "hgbt":
        return build_hist_gradient_boosting(
            max_depth=model_config.get("max_depth", 6),
            learning_rate=model_config.get("learning_rate", 0.1),
        )
    if name == "mlp":
        return MLPClassifier(input_dim=input_dim, hidden_dim=model_config.get("hidden_dim", 64))
    if name == "cnn1d":
        return CNN1DClassifier(input_dim=input_dim, hidden_dim=model_config.get("hidden_dim", 64))
    if name == "tcn":
        return TCNClassifier(input_dim=input_dim, hidden_dim=model_config.get("hidden_dim", 64))
    if name == "lstm":
        return LSTMClassifier(input_dim=input_dim, hidden_dim=model_config.get("hidden_dim", 64))
    raise KeyError(f"Unsupported model config: {name}")


def is_torch_model(model: Any) -> bool:
    """Return whether a model is a PyTorch module."""

    return isinstance(model, nn.Module)
