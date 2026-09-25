from __future__ import annotations
import json, os
from historical_engine import build_high_precision_models

if __name__ == "__main__":
    data_dir = os.getenv("HISTORICAL_DATA_DIR", "data/BTCUSDT")
    model_dir = os.getenv("MODEL_DIR", "models")
    result = build_high_precision_models(data_dir, model_dir, folds=int(os.getenv("BACKTEST_FOLDS", "5")))
    print(json.dumps(result, indent=2))
