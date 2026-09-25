from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def run(cmd: list[str], env: dict | None = None):
    print("[BOOTSTRAP]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env or os.environ.copy(), check=True)


def build_symbol(symbol: str, start: str, end: str):
    data_dir = ROOT / "data" / symbol
    model_dir = MODELS if symbol == "BTCUSDT" else MODELS / symbol
    model_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "download_binance.py",
        "--market", "futures",
        "--symbol", symbol,
        "--start", start,
        "--end", end,
        "--out", "data",
    ])

    env = os.environ.copy()
    env["HISTORICAL_DATA_DIR"] = str(data_dir)
    env["MODEL_DIR"] = str(model_dir)
    env.setdefault("RR_FLOOR", "2.0")
    env.setdefault("PRECISION_TP_R", "2.0")
    env.setdefault("WIN_RATE_FLOOR", "0.70")
    env.setdefault("MODEL_PROBABILITY_FLOOR", "0.70")
    env.setdefault("REQUIRE_EVERY_FOLD", "true")
    env.setdefault("MIN_OOS_SIGNALS", "60")
    env.setdefault("FINAL_HOLDOUT_FRACTION", "0.10")

    print(f"[BOOTSTRAP] Building strict historical model for {symbol}...", flush=True)
    run([sys.executable, "build_models.py"], env=env)

    sel = model_dir / "selection.json"
    if not sel.exists():
        raise SystemExit(f"{symbol}: selection.json missing")

    obj = json.loads(sel.read_text(encoding="utf-8"))
    summary = {
        "symbol": symbol,
        "eligible": obj.get("eligible"),
        "selected": obj.get("selected", {}),
        "deployment_gate": obj.get("deployment_gate", {}),
        "data": obj.get("data", {}),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not obj.get("eligible"):
        raise SystemExit(f"{symbol}: no strategy passed strict historical gate")

    selected = obj.get("selected", {})
    holdout = selected.get("holdout", {})
    oos = float(selected.get("pooled_oos_win_rate") or 0)
    holdwr = float(holdout.get("win_rate") or 0)
    if oos < 0.70 or holdwr < 0.70 or not holdout.get("passed"):
        raise SystemExit(f"{symbol}: historical gate failed")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    start = os.getenv("HIST_START", "2020-01-01")
    end = os.getenv("HIST_END", "2026-08-31")

    for symbol in SYMBOLS:
        build_symbol(symbol, start, end)

    print("[BOOTSTRAP] BTCUSDT + ETHUSDT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
