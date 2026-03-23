from __future__ import annotations

import torch

from markettensor.models.cnn1d import CNN1DClassifier
from markettensor.models.lstm import LSTMClassifier
from markettensor.models.mlp import MLPClassifier
from markettensor.models.tcn import TCNClassifier


def test_cnn_shape():
    model = CNN1DClassifier(input_dim=6)
    logits = model(torch.randn(4, 16, 6))
    assert logits.shape == (4,)


def test_tcn_shape():
    model = TCNClassifier(input_dim=6)
    logits = model(torch.randn(4, 16, 6))
    assert logits.shape == (4,)


def test_lstm_shape():
    model = LSTMClassifier(input_dim=6)
    logits = model(torch.randn(4, 16, 6))
    assert logits.shape == (4,)


def test_mlp_shape():
    model = MLPClassifier(input_dim=12)
    logits = model(torch.randn(4, 12))
    assert logits.shape == (4,)
