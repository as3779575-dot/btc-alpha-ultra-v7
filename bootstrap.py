from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "BTCUSDT"
MODELS = ROOT / "models"


def run(cmd: list[str]):
    print("[BOOTSTRAP]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    start = os.getenv("HIST_START", "2020-01-01")
    end = os.getenv("HIST_END", "2026-08-31")
    run([sys.executable, "download_binance.py", "--market", "futures", "--symbol", "BTCUSDT", "--start", start, "--end", end, "--out", "data"])
    env = os.environ.copy()
    env["HISTORICAL_DATA_DIR"] = str(DATA)
    env["MODEL_DIR"] = str(MODELS)
    env.setdefault("RR_FLOOR", "2.0")
    env.setdefault("PRECISION_TP_R", "2.0")
    env.setdefault("WIN_RATE_FLOOR", "0.70")
    env.setdefault("MODEL_PROBABILITY_FLOOR", "0.70")
    env.setdefault("REQUIRE_EVERY_FOLD", "true")
    env.setdefault("MIN_OOS_SIGNALS", "60")
    env.setdefault("FINAL_HOLDOUT_FRACTION", "0.10")
    print("[BOOTSTRAP] Running strict historical model selection...", flush=True)
    subprocess.run([sys.executable, "build_models.py"], cwd=ROOT, env=env, check=True)
    sel = MODELS / "selection.json"
    if sel.exists():
        obj = json.loads(sel.read_text(encoding="utf-8"))
        print(json.dumps({
            "eligible": obj.get("eligible"),
            "selected": obj.get("selected", {}),
            "deployment_gate": obj.get("deployment_gate", {}),
            "data": obj.get("data", {}),
        }, indent=2), flush=True)
    print("[BOOTSTRAP] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
