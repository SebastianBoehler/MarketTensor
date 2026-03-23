PYTHON ?= python3

.PHONY: install format lint test download build train evaluate backtest export

install:
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

download:
	$(PYTHON) scripts/download_data.py --symbol BTCUSDT --interval 1h

build:
	$(PYTHON) scripts/build_dataset.py --config-name cnn_ohlcv

train:
	$(PYTHON) scripts/train.py --config-name cnn_ohlcv

evaluate:
	$(PYTHON) scripts/evaluate.py --run-id latest

backtest:
	$(PYTHON) scripts/backtest.py --run-id latest

export:
	$(PYTHON) scripts/export_onnx.py --run-id latest
