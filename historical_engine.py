from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

COLS6 = ["timestamp", "open", "high", "low", "close", "volume"]
COLS12 = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
]
TF_RULES = {"1W": "W-SUN", "1D": "D", "4H": "4h", "1H": "1h", "15M": "15min", "5M": "5min"}
DAILY_TFS = ["1D", "4H", "1H", "15M", "5M"]
WEEKLY_TFS = ["1W"] + DAILY_TFS


@dataclass
class ResearchConfig:
    rr_floor: float = 2.0
    target_r: float = 2.0
    signal_win_rate_floor: float = 0.70
    model_probability_floor: float = 0.70
    min_calibration_signals: int = 30
    min_oos_signals: int = 60
    min_fold_signals: int = 10
    min_fold_win_rate: float = 0.70
    require_every_fold: bool = True
    final_holdout_fraction: float = 0.10
    folds: int = 5
    cooldown_bars: int = 12
    fee_per_side: float = 0.0004
    slippage_per_side: float = 0.0001
    purge_bars: int = 48
    random_state: int = 17
    # Fixed 2R families. Search is deliberately finite to keep OOS selection auditable.
    horizon_minutes: tuple[int, ...] = (60, 120, 240)
    stop_atr: tuple[float, ...] = (0.8, 1.2)
    strategies: tuple[str, ...] = ("trend_alignment", "trend_flow", "breakout_flow")
    # Probability gate is searched only above the user's floor.
    probability_grid: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    margin_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

    @classmethod
    def from_env(cls, folds: int = 5) -> "ResearchConfig":
        return cls(
            rr_floor=float(os.getenv("RR_FLOOR", "2.0")),
            target_r=max(float(os.getenv("PRECISION_TP_R", "2.0")), float(os.getenv("RR_FLOOR", "2.0"))),
            signal_win_rate_floor=float(os.getenv("WIN_RATE_FLOOR", "0.70")),
            model_probability_floor=float(os.getenv("MODEL_PROBABILITY_FLOOR", "0.70")),
            min_calibration_signals=int(os.getenv("MIN_CALIBRATION_SIGNALS", "30")),
            min_oos_signals=int(os.getenv("MIN_OOS_SIGNALS", "60")),
            min_fold_signals=int(os.getenv("MIN_FOLD_SIGNALS", "10")),
            min_fold_win_rate=float(os.getenv("MIN_FOLD_WIN_RATE", os.getenv("WIN_RATE_FLOOR", "0.70"))),
            require_every_fold=os.getenv("REQUIRE_EVERY_FOLD", "true").lower() in {"1", "true", "yes", "y"},
            final_holdout_fraction=float(os.getenv("FINAL_HOLDOUT_FRACTION", "0.10")),
            folds=folds,
            cooldown_bars=int(os.getenv("PRECISION_COOLDOWN_BARS", "12")),
            fee_per_side=float(os.getenv("BACKTEST_FEE_PER_SIDE", "0.0004")),
            slippage_per_side=float(os.getenv("BACKTEST_SLIPPAGE_PER_SIDE", "0.0001")),
            purge_bars=int(os.getenv("PURGE_BARS", "48")),
        )


@dataclass
class BarrierSpec:
    horizon_minutes: int
    stop_atr: float
    target_r: float


def load_1m(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files under {folder}")
    frames: list[pd.DataFrame] = []
    for f in files:
        raw = pd.read_csv(f, header=None)
        if raw.shape[1] >= 12:
            raw = raw.iloc[:, :12]
            raw.columns = COLS12
            if str(raw.iloc[0]["timestamp"]).lower() in {"timestamp", "open_time", "date"}:
                raw = raw.iloc[1:]
            df = raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote"]].copy()
        else:
            if raw.shape[1] < 6:
                continue
            raw = raw.iloc[:, :6]
            raw.columns = COLS6
            df = raw.copy()
            df["quote_volume"] = df["close"] * df["volume"]
            df["trades"] = 0
            df["taker_base"] = df["volume"] * 0.5
            df["taker_quote"] = df["quote_volume"] * 0.5

        tsnum = pd.to_numeric(df.timestamp, errors="coerce")
        if tsnum.notna().mean() > 0.9:
            df.timestamp = pd.to_datetime(tsnum, unit="ms", utc=True)
        else:
            df.timestamp = pd.to_datetime(df.timestamp, utc=True, errors="coerce")
        for c in [x for x in df.columns if x != "timestamp"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]))

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return raw


def resample(raw: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = raw.set_index("timestamp")
    out = x.resample(rule, label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
        taker_base=("taker_base", "sum"), taker_quote=("taker_quote", "sum"), trades=("trades", "sum")
    )
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def feat(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = df.copy()
    c = x.close
    tr = pd.concat([(x.high - x.low), (x.high - c.shift()).abs(), (x.low - c.shift()).abs()], axis=1).max(axis=1)
    atr_abs = tr.rolling(14, min_periods=14).mean()
    atr = (atr_abs / c).replace([np.inf, -np.inf], np.nan)
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    vm = x.volume.rolling(30, min_periods=30)
    volz = (x.volume - vm.mean()) / (vm.std() + 1e-9)
    prev_h = x.high.rolling(20, min_periods=20).max().shift(1)
    prev_l = x.low.rolling(20, min_periods=20).min().shift(1)
    taker_delta = ((2 * x.taker_base / x.volume.replace(0, np.nan)) - 1).clip(-1, 1)
    quote_delta = ((2 * x.taker_quote / x.quote_volume.replace(0, np.nan)) - 1).clip(-1, 1)
    body = (x.close - x.open) / c
    upper_wick = (x.high - x[["open", "close"]].max(axis=1)) / c
    lower_wick = (x[["open", "close"]].min(axis=1) - x.low) / c
    range20 = ((x.high.rolling(20, min_periods=20).max() - x.low.rolling(20, min_periods=20).min()) / c)
    ret1 = c.pct_change()
    ret4 = c.pct_change(4)
    ret12 = c.pct_change(12)
    slope = e20.pct_change(5)
    return pd.DataFrame({
        "timestamp": x.timestamp,
        f"{prefix}_ret1": ret1,
        f"{prefix}_ret4": ret4,
        f"{prefix}_ret12": ret12,
        f"{prefix}_ema20": c / e20 - 1,
        f"{prefix}_ema50": c / e50 - 1,
        f"{prefix}_ema_slope": slope,
        f"{prefix}_atr": atr,
        f"{prefix}_rsi": _rsi(c),
        f"{prefix}_volz": volz,
        f"{prefix}_range20": range20,
        f"{prefix}_taker_delta": taker_delta,
        f"{prefix}_quote_delta": quote_delta,
        f"{prefix}_body": body,
        f"{prefix}_upper_wick": upper_wick,
        f"{prefix}_lower_wick": lower_wick,
        f"{prefix}_breakout_up": (c > prev_h).astype(float),
        f"{prefix}_breakout_down": (c < prev_l).astype(float),
    })


def build_feature_table(raw: pd.DataFrame, weekly: bool) -> pd.DataFrame:
    base = resample(raw, TF_RULES["5M"])[["timestamp"]].copy()
    order = WEEKLY_TFS if weekly else DAILY_TFS
    for tf in order:
        ftf = feat(resample(raw, TF_RULES[tf]), tf.lower())
        base = pd.merge_asof(base.sort_values("timestamp"), ftf.sort_values("timestamp"), on="timestamp", direction="backward", allow_exact_matches=True)
    base["hour"] = base.timestamp.dt.hour.astype(float)
    base["dow"] = base.timestamp.dt.dayofweek.astype(float)
    base["hour_sin"] = np.sin(2 * np.pi * base.hour / 24.0)
    base["hour_cos"] = np.cos(2 * np.pi * base.hour / 24.0)
    base["dow_sin"] = np.sin(2 * np.pi * base.dow / 7.0)
    base["dow_cos"] = np.cos(2 * np.pi * base.dow / 7.0)
    # Cross-timeframe state descriptors.
    trend_cols = [f"{tf.lower()}_ema20" for tf in DAILY_TFS]
    mom_cols = [f"{tf.lower()}_ret4" for tf in DAILY_TFS]
    flow_cols = ["15m_taker_delta", "5m_taker_delta"]
    base["trend_votes"] = sum(np.sign(base[c].fillna(0)) for c in trend_cols)
    base["momentum_votes"] = sum(np.sign(base[c].fillna(0)) for c in mom_cols)
    base["flow_score"] = base[flow_cols].fillna(0).sum(axis=1)
    base["vol_regime"] = base["5m_atr"].rolling(60, min_periods=20).rank(pct=True)
    return base


def primary_side(df: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    trend = df["trend_votes"].fillna(0).to_numpy(float)
    mom = df["momentum_votes"].fillna(0).to_numpy(float)
    flow = df["flow_score"].fillna(0).to_numpy(float)
    bu = df["5m_breakout_up"].fillna(0).to_numpy(float)
    bd = df["5m_breakout_down"].fillna(0).to_numpy(float)
    score = np.zeros(len(df), dtype=float)
    side = np.zeros(len(df), dtype=np.int8)

    if strategy == "trend_alignment":
        score = trend + 0.5 * mom
        side[score >= 3.0] = 1
        side[score <= -3.0] = -1
    elif strategy == "trend_flow":
        score = trend + 0.5 * mom + 2.0 * flow
        side[score >= 3.5] = 1
        side[score <= -3.5] = -1
    elif strategy == "breakout_flow":
        score = trend + 2.0 * flow + 3.0 * (bu - bd)
        side[(bu > 0) & (flow >= 0.10) & (trend >= 1)] = 1
        side[(bd > 0) & (flow <= -0.10) & (trend <= -1)] = -1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return side, score


def _atr_decision(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    bars = resample(raw, TF_RULES["5M"])
    tr = pd.concat([(bars.high - bars.low), (bars.high - bars.close.shift()).abs(), (bars.low - bars.close.shift()).abs()], axis=1).max(axis=1)
    atr_abs = tr.rolling(14, min_periods=14).mean().to_numpy(float)
    return bars, atr_abs


def barrier_labels(raw: pd.DataFrame, spec: BarrierSpec) -> pd.DataFrame:
    bars, atr_abs = _atr_decision(raw)
    raw_ts = raw.timestamp.to_numpy(dtype="datetime64[ns]")
    hi = raw.high.to_numpy(float)
    lo = raw.low.to_numpy(float)
    bts = bars.timestamp.to_numpy(dtype="datetime64[ns]")
    close = bars.close.to_numpy(float)
    raw_pos = np.searchsorted(raw_ts, bts, side="right")
    horizon = int(spec.horizon_minutes)
    long_win = np.full(len(bars), np.nan)
    short_win = np.full(len(bars), np.nan)
    for i in range(len(bars)):
        if not np.isfinite(atr_abs[i]) or raw_pos[i] >= len(raw):
            continue
        stop_dist = atr_abs[i] * float(spec.stop_atr)
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue
        entry = close[i]
        up = entry + float(spec.target_r) * stop_dist
        dn = entry - stop_dist
        start = int(raw_pos[i])
        end = min(start + horizon, len(raw))
        if end <= start:
            continue
        fh = hi[start:end]
        fl = lo[start:end]
        up_idx = np.flatnonzero(fh >= up)
        dn_idx = np.flatnonzero(fl <= dn)
        fu = int(up_idx[0]) if len(up_idx) else horizon + 1
        fd = int(dn_idx[0]) if len(dn_idx) else horizon + 1
        if fu == horizon + 1 and fd == horizon + 1:
            continue
        if fu == fd:
            # Within one 1-minute bar OHLC cannot identify which barrier was first.
            # Conservatively discard the event instead of guessing.
            continue
        long_win[i] = 1.0 if fu < fd else 0.0
        short_win[i] = 1.0 if fd < fu else 0.0
    return pd.DataFrame({"timestamp": bars.timestamp, "long_win": long_win, "short_win": short_win, "atr_abs": atr_abs})


def wilson_lower_bound(wins: int, n: int, z: float = 1.959963984540054) -> float | None:
    if n <= 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((centre - adj) / denom)


def _fit_model(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=260,
        max_leaf_nodes=21,
        learning_rate=0.05,
        l2_regularization=8.0,
        min_samples_leaf=120,
        random_state=17,
    )
    model.fit(X, y)
    return model


def _apply_calibration(model, X_cal: pd.DataFrame, y_cal: np.ndarray):
    raw = model.predict_proba(X_cal)[:, 1]
    if len(np.unique(y_cal)) < 2:
        return raw, None
    # Isotonic is flexible but can overfit small calibration sets. Use sigmoid/Platt on smaller windows.
    if len(y_cal) < 1000:
        cal = LogisticRegression(C=10.0, solver="lbfgs", random_state=17)
        cal.fit(raw.reshape(-1, 1), y_cal)
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1], cal
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y_cal)
    return iso.predict(raw), iso


def _predict_success(model, calibrator, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    if calibrator is None:
        return raw
    if isinstance(calibrator, LogisticRegression):
        return np.asarray(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(calibrator.predict(raw), dtype=float)


def trade_stats(success: np.ndarray, prob: np.ndarray, stop_pct: np.ndarray, threshold: float, margin: float, cooldown: int, cfg: ResearchConfig) -> dict:
    if len(success) == 0:
        return {"signals": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None, "max_drawdown_r": None, "coverage": 0.0, "wilson_lb": None}
    take = np.isfinite(success) & np.isfinite(prob) & (prob >= threshold)
    # Since this is a binary success model, margin is expressed as distance from 0.5.
    take &= (prob - 0.5) >= margin
    chosen: list[int] = []
    next_allowed = -1
    for idx in np.flatnonzero(take):
        if int(idx) >= next_allowed:
            chosen.append(int(idx))
            next_allowed = int(idx) + int(cooldown)
    if not chosen:
        return {"signals": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None, "max_drawdown_r": None, "coverage": 0.0, "wilson_lb": None}
    s = success[chosen].astype(int)
    # Convert percentage transaction costs to R using the actual stop distance for each event.
    roundtrip_cost_pct = 2.0 * (cfg.fee_per_side + cfg.slippage_per_side)
    costs_r = roundtrip_cost_pct / np.maximum(stop_pct[chosen], 1e-8)
    pnl = np.where(s == 1, cfg.target_r - costs_r, -1.0 - costs_r)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = float(np.min(eq - peak[1:])) if len(eq) else 0.0
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    pf = float(wins.sum() / losses.sum()) if len(losses) else None
    return {
        "signals": int(len(chosen)),
        "win_rate": float(s.mean()),
        "expectancy_r": float(pnl.mean()),
        "profit_factor": pf,
        "max_drawdown_r": dd,
        "coverage": float(len(chosen) / len(success)),
        "wilson_lb": wilson_lower_bound(int(s.sum()), int(len(s))),
        "avg_win_r": float(wins.mean()) if len(wins) else None,
        "avg_loss_r": float(losses.mean()) if len(losses) else None,
    }


def _feature_columns(df: pd.DataFrame) -> list[str]:
    ignore = {"timestamp", "side", "primary_score", "strategy_id", "stop_atr", "horizon_hours", "success", "stop_pct", "long_win", "short_win", "atr_abs"}
    return [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]


def _evaluate_candidate(train_events: pd.DataFrame, test_events: pd.DataFrame, cfg: ResearchConfig) -> tuple[dict, object, object, list[str]]:
    if len(train_events) < 1000 or len(test_events) < 100:
        return {"eligible": False, "reason": "insufficient_events"}, None, None, []
    cut = max(int(len(train_events) * 0.75), 500)
    fit = train_events.iloc[:cut].copy()
    cal = train_events.iloc[cut:].copy()
    cols = _feature_columns(train_events)
    X_fit = fit[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_cal = cal[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_events[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = _fit_model(X_fit, fit.success.to_numpy(int))
    cal_prob, calibrator = _apply_calibration(model, X_cal, cal.success.to_numpy(int))
    stop_pct_cal = cal.stop_pct.to_numpy(float)
    best = None
    for threshold, margin in product(cfg.probability_grid, cfg.margin_grid):
        if threshold < cfg.model_probability_floor:
            continue
        st = trade_stats(cal.success.to_numpy(int), cal_prob, stop_pct_cal, float(threshold), float(margin), cfg.cooldown_bars, cfg)
        if st["signals"] < cfg.min_calibration_signals or st["win_rate"] is None:
            continue
        feasible = st["win_rate"] >= cfg.signal_win_rate_floor
        # Hard pass first; then maximize lower confidence bound, win rate and expectancy.
        key = (
            1 if feasible else 0,
            st["wilson_lb"] if st["wilson_lb"] is not None else -1,
            st["win_rate"] if st["win_rate"] is not None else -1,
            st["expectancy_r"] if st["expectancy_r"] is not None else -999,
            st["signals"],
        )
        candidate = {"threshold": float(threshold), "margin": float(margin), "calibration": st, "score_key": key}
        if best is None or key > best["score_key"]:
            best = candidate
    if best is None:
        return {"eligible": False, "reason": "no_calibration_gate"}, None, None, cols
    return best, model, calibrator, cols


def _walkforward_from_events(d: pd.DataFrame, weekly: bool, strategy: str, spec: BarrierSpec, cfg: ResearchConfig) -> dict:
    n = len(d)
    holdout_n = max(int(n * cfg.final_holdout_fraction), cfg.min_oos_signals)
    if holdout_n >= n:
        return {"weekly": weekly, "strategy": strategy, "barrier": asdict(spec), "folds": [], "eligible": False, "reason": "holdout_consumes_sample"}
    dev = d.iloc[:-holdout_n].copy()
    hold = d.iloc[-holdout_n:].copy()
    fold = max(len(dev) // (cfg.folds + 1), 500)
    results = []
    for k in range(1, cfg.folds + 1):
        a = k * fold
        b = min((k + 1) * fold, len(dev))
        if b <= a:
            continue
        train_end = max(a - cfg.purge_bars, 0)
        train = dev.iloc[:train_end].copy()
        test = dev.iloc[a:b].copy()
        if len(train) < 1500 or len(test) < 100:
            continue
        gate, model, calibrator, cols = _evaluate_candidate(train, test, cfg)
        if model is None:
            results.append({"fold": k, "train_rows": len(train), "test_rows": len(test), "gate": gate, "trade": None, "passed": False})
            continue
        X_test = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_prob = _predict_success(model, calibrator, X_test)
        st = trade_stats(test.success.to_numpy(int), test_prob, test.stop_pct.to_numpy(float), gate["threshold"], gate["margin"], cfg.cooldown_bars, cfg)
        passed = bool(st["signals"] >= cfg.min_fold_signals and st["win_rate"] is not None and st["win_rate"] >= cfg.min_fold_win_rate)
        results.append({"fold": k, "train_rows": len(train), "test_rows": len(test), "gate": gate, "trade": st, "passed": passed})

    usable = [r for r in results if r.get("trade") and r["trade"].get("signals", 0) > 0]
    total_signals = int(sum(r["trade"]["signals"] for r in usable))
    total_wins = int(sum(int(round(r["trade"]["signals"] * r["trade"]["win_rate"])) for r in usable))
    pooled_win = (total_wins / total_signals) if total_signals else None
    fold_rates = [r["trade"]["win_rate"] for r in usable if r["trade"].get("win_rate") is not None]
    mean_precision = float(np.mean(fold_rates)) if fold_rates else None
    each_fold_pass = len(results) >= cfg.folds and all(r.get("passed") for r in results if r.get("trade") is not None)
    pooled_pass = bool(total_signals >= cfg.min_oos_signals and pooled_win is not None and pooled_win >= cfg.signal_win_rate_floor)
    eligible = bool(pooled_pass and (each_fold_pass if cfg.require_every_fold else True))
    holdout = {"signals": 0, "win_rate": None, "passed": False}
    if eligible:
        gate, model, calibrator, cols = _evaluate_candidate(dev, hold, cfg)
        if model is not None:
            Xh = hold[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ph = _predict_success(model, calibrator, Xh)
            hst = trade_stats(hold.success.to_numpy(int), ph, hold.stop_pct.to_numpy(float), gate["threshold"], gate["margin"], cfg.cooldown_bars, cfg)
            holdout = {**hst, "threshold": gate["threshold"], "margin": gate["margin"], "passed": bool(hst.get("signals", 0) >= cfg.min_fold_signals and hst.get("win_rate") is not None and hst["win_rate"] >= cfg.signal_win_rate_floor)}
            eligible = bool(eligible and holdout["passed"])
    return {"weekly": bool(weekly), "strategy": strategy, "barrier": asdict(spec), "folds": results, "pooled_oos_signals": total_signals, "pooled_oos_wins": total_wins, "pooled_oos_win_rate": pooled_win, "mean_fold_win_rate": mean_precision, "min_fold_win_rate": min(fold_rates) if fold_rates else None, "eligible": eligible, "holdout": holdout, "reason": "passed" if eligible else "failed_hard_gate"}

def _walkforward_strategy(raw: pd.DataFrame, weekly: bool, strategy: str, spec: BarrierSpec, cfg: ResearchConfig) -> dict:
    features = build_feature_table(raw, weekly)
    labels = barrier_labels(raw, spec)
    d = features.merge(labels, on="timestamp", how="inner").dropna(subset=["5m_atr"]).reset_index(drop=True)
    side, pscore = primary_side(d, strategy)
    d["side"] = side
    d["primary_score"] = pscore
    d["strategy_id"] = float(list(cfg.strategies).index(strategy))
    d["stop_atr"] = float(spec.stop_atr)
    d["horizon_hours"] = float(spec.horizon_minutes) / 60.0
    d["success"] = np.where(d.side == 1, d.long_win, np.where(d.side == -1, d.short_win, np.nan))
    d["stop_pct"] = np.maximum(d["5m_atr"].astype(float) * float(spec.stop_atr), 1e-8)
    d = d[(d.side != 0) & np.isfinite(d.success)].copy().reset_index(drop=True)
    d.success = d.success.astype(int)
    return _walkforward_from_events(d, weekly, strategy, spec, cfg)


def _rank_key(r: dict) -> tuple:
    hold = r.get("holdout", {})
    return (
        1 if r.get("eligible") else 0,
        hold.get("win_rate") or -1,
        r.get("pooled_oos_win_rate") or -1,
        r.get("mean_fold_win_rate") or -1,
        sum((f.get("trade") or {}).get("expectancy_r") or -999 for f in r.get("folds", [])),
    )


def build_high_precision_models(data_dir: str, out_dir: str, folds: int = 5) -> dict:
    cfg = ResearchConfig.from_env(folds=folds)
    if cfg.target_r < cfg.rr_floor:
        raise RuntimeError("Target R must be at least the configured gross R:R floor.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "selected_model.joblib"
    selection_path = out / "selection.json"
    # Never allow a stale model to survive a failed rebuild.
    if model_path.exists():
        model_path.unlink()

    raw = load_1m(data_dir)
    results: list[dict] = []
    specs = [BarrierSpec(h, s, cfg.target_r) for h, s in product(cfg.horizon_minutes, cfg.stop_atr)]
    for weekly in (False, True):
        features = build_feature_table(raw, weekly)
        label_cache: dict[tuple[int, float], pd.DataFrame] = {}
        for spec in specs:
            label_cache[(spec.horizon_minutes, spec.stop_atr)] = barrier_labels(raw, spec)
        for strategy in cfg.strategies:
            for spec in specs:
                try:
                    labels = label_cache[(spec.horizon_minutes, spec.stop_atr)]
                    # Reuse the already-built features/labels rather than recomputing them for each strategy.
                    d = features.merge(labels, on="timestamp", how="inner").dropna(subset=["5m_atr"]).reset_index(drop=True)
                    side, pscore = primary_side(d, strategy)
                    d["side"] = side
                    d["primary_score"] = pscore
                    d["strategy_id"] = float(list(cfg.strategies).index(strategy))
                    d["stop_atr"] = float(spec.stop_atr)
                    d["horizon_hours"] = float(spec.horizon_minutes) / 60.0
                    d["success"] = np.where(d.side == 1, d.long_win, np.where(d.side == -1, d.short_win, np.nan))
                    d["stop_pct"] = np.maximum(d["5m_atr"].astype(float) * float(spec.stop_atr), 1e-8)
                    d = d[(d.side != 0) & np.isfinite(d.success)].copy().reset_index(drop=True)
                    d.success = d.success.astype(int)
                    if len(d) < cfg.min_oos_signals * 2:
                        r = {"weekly": weekly, "strategy": strategy, "barrier": asdict(spec), "eligible": False, "reason": "too_few_candidate_events", "folds": []}
                    else:
                        r = _walkforward_from_events(d, weekly, strategy, spec, cfg)
                except Exception as exc:
                    r = {"weekly": weekly, "strategy": strategy, "barrier": asdict(spec), "eligible": False, "reason": f"error: {exc}", "folds": []}
                results.append(r)
                print(json.dumps({"weekly": weekly, "strategy": strategy, "barrier": asdict(spec), "eligible": r.get("eligible"), "pooled_oos_win_rate": r.get("pooled_oos_win_rate"), "holdout_win_rate": r.get("holdout", {}).get("win_rate")}, separators=(",", ":")))

    eligible = [r for r in results if r.get("eligible")]
    summary = {
        "eligible": bool(eligible),
        "hard_requirements": {
            "gross_rr_floor": cfg.rr_floor,
            "target_r": cfg.target_r,
            "signal_win_rate_floor": cfg.signal_win_rate_floor,
            "model_probability_floor": cfg.model_probability_floor,
            "require_every_walkforward_fold": cfg.require_every_fold,
            "minimum_oos_signals": cfg.min_oos_signals,
            "final_holdout_fraction": cfg.final_holdout_fraction,
        },
        "tested_candidates": len(results),
        "candidate_results": sorted(results, key=_rank_key, reverse=True),
        "config": asdict(cfg),
        "data": {"rows_1m": int(len(raw)), "first_timestamp": str(raw.timestamp.iloc[0]), "last_timestamp": str(raw.timestamp.iloc[-1])},
    }

    if not eligible:
        selection_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise RuntimeError(
            "No strategy passed the hard 2:1 R:R + 70% walk-forward win-rate + final holdout gates. "
            "No live model was produced. See models/selection.json for all candidate results."
        )

    chosen = max(eligible, key=_rank_key)
    # Deployment model uses the same fitted model/calibrator pair that generated the final development gate.
    # Keeping the pair aligned avoids applying a calibrator learned from one model to a different refit model.
    features = build_feature_table(raw, bool(chosen["weekly"]))
    spec = BarrierSpec(**chosen["barrier"])
    labels = barrier_labels(raw, spec)
    d = features.merge(labels, on="timestamp", how="inner").dropna(subset=["5m_atr"]).copy()
    side, pscore = primary_side(d, chosen["strategy"])
    d["side"] = side
    d["primary_score"] = pscore
    d["strategy_id"] = float(list(cfg.strategies).index(chosen["strategy"]))
    d["stop_atr"] = float(spec.stop_atr)
    d["horizon_hours"] = float(spec.horizon_minutes) / 60.0
    d["success"] = np.where(d.side == 1, d.long_win, np.where(d.side == -1, d.short_win, np.nan))
    d["stop_pct"] = np.maximum(d["5m_atr"].astype(float) * float(spec.stop_atr), 1e-8)
    d = d[(d.side != 0) & np.isfinite(d.success)].copy().reset_index(drop=True)
    d.success = d.success.astype(int)
    hold_n = max(int(len(d) * cfg.final_holdout_fraction), cfg.min_oos_signals)
    dev = d.iloc[:-hold_n].copy()
    cols = _feature_columns(dev)
    gate, model, calibrator, cols = _evaluate_candidate(dev, d.iloc[-hold_n:].copy(), cfg)
    if model is None or gate.get("calibration", {}).get("win_rate") is None or gate["calibration"]["win_rate"] < cfg.signal_win_rate_floor:
        selection_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise RuntimeError("Selected candidate lost its final calibration gate during deployment fit; no live model produced.")
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "weekly": bool(chosen["weekly"]),
        "strategy": chosen["strategy"],
        "barrier": chosen["barrier"],
        "features": cols,
        "trained_rows": int(len(dev) * 0.75),
        "signal_gate": {k: v for k, v in gate.items() if k != "score_key"},
        "config": asdict(cfg),
        "hard_requirements_passed": True,
        "selection": chosen,
    }
    joblib.dump(artifact, model_path)
    summary["selected"] = chosen
    summary["deployment_gate"] = artifact["signal_gate"]
    selection_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def live_prediction(snapshot: dict, model_path: str) -> dict:
    b = joblib.load(model_path)
    cfg = b.get("config", {})
    if not b.get("hard_requirements_passed"):
        return {"eligible": False, "reason": "artifact_hard_gate_failed"}
    tf = snapshot["timeframes"]
    weekly = bool(b.get("weekly"))
    order = ["1w", "1d", "4h", "1h", "15m", "5m"] if weekly else ["1d", "4h", "1h", "15m", "5m"]
    row: dict[str, float] = {}
    for name in order:
        z = tf[name]
        p = name
        fields = {
            "ret1": z.get("ret1"), "ret4": z.get("ret4"), "ret12": z.get("ret12"),
            "ema20": z.get("ema20_gap"), "ema50": z.get("ema50_gap"), "ema_slope": z.get("ema_slope"),
            "atr": z.get("atr"), "rsi": z.get("rsi"), "volz": z.get("volume_z"),
            "range20": z.get("range20"), "taker_delta": z.get("taker_delta_ratio"),
            "quote_delta": z.get("taker_quote_delta", z.get("taker_delta_ratio")),
            "body": z.get("body"), "upper_wick": z.get("upper_wick"), "lower_wick": z.get("lower_wick"),
            "breakout_up": z.get("breakout_up"), "breakout_down": z.get("breakout_down"),
        }
        for suffix, value in fields.items():
            row[f"{p}_{suffix}"] = 0.0 if value is None or not np.isfinite(value) else float(value)
    trend_cols = [f"{tf.lower()}_ema20" for tf in DAILY_TFS]
    mom_cols = [f"{tf.lower()}_ret4" for tf in DAILY_TFS]
    row["trend_votes"] = sum(np.sign(row.get(c, 0.0)) for c in trend_cols)
    row["momentum_votes"] = sum(np.sign(row.get(c, 0.0)) for c in mom_cols)
    row["flow_score"] = row.get("15m_taker_delta", 0.0) + row.get("5m_taker_delta", 0.0)
    ts = pd.Timestamp(snapshot.get("timestamp_utc") or pd.Timestamp.now(tz="UTC"))
    row["hour"] = float(ts.hour)
    row["dow"] = float(ts.dayofweek)
    row["hour_sin"] = math.sin(2 * math.pi * ts.hour / 24.0)
    row["hour_cos"] = math.cos(2 * math.pi * ts.hour / 24.0)
    row["dow_sin"] = math.sin(2 * math.pi * ts.dayofweek / 7.0)
    row["dow_cos"] = math.cos(2 * math.pi * ts.dayofweek / 7.0)

    base_df = pd.DataFrame([row])
    primary, pscore = primary_side(pd.DataFrame([row]), b["strategy"])
    side = int(primary[0])
    row["side"] = float(side)
    row["primary_score"] = float(pscore[0])
    row["strategy_id"] = float(list(ResearchConfig.from_env().strategies).index(b["strategy"]))
    row["stop_atr"] = float(b["barrier"]["stop_atr"])
    row["horizon_hours"] = float(b["barrier"]["horizon_minutes"]) / 60.0
    x = pd.DataFrame([{c: row.get(c, 0.0) for c in b["features"]}]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_prob = float(b["model"].predict_proba(x)[0, 1])
    calibrator = b.get("calibrator")
    prob = float(calibrator.predict([raw_prob])[0]) if calibrator is not None else raw_prob
    gate = b["signal_gate"]
    margin_ok = (prob - 0.5) >= float(gate.get("margin", 0.05))
    prob_ok = prob >= max(float(gate.get("threshold", 0.70)), float(cfg.get("model_probability_floor", 0.70)))
    hist = b["selection"]
    eligible = bool(side != 0 and prob_ok and margin_ok and b.get("hard_requirements_passed") and hist.get("holdout", {}).get("passed", False))
    return {
        "eligible": eligible,
        "reason": "passed" if eligible else "primary_side_or_probability_gate_failed",
        "side": side,
        "direction": "BUY" if side == 1 else "SELL" if side == -1 else "WAIT",
        "primary_score": float(pscore[0]),
        "model_probability": prob,
        "signal_threshold": float(gate.get("threshold", 0.70)),
        "probability_margin": float(prob - 0.5),
        "margin_threshold": float(gate.get("margin", 0.05)),
        "gross_rr": float(b["barrier"]["target_r"]),
        "rr_floor": float(cfg.get("rr_floor", 2.0)),
        "historical_oos_win_rate": float(hist.get("pooled_oos_win_rate") or 0.0),
        "historical_holdout_win_rate": float(hist.get("holdout", {}).get("win_rate") or 0.0),
        "strategy": b["strategy"],
        "barrier": b["barrier"],
        "selected_walkforward": hist,
    }


def build_models(data_dir: str, out_dir: str):
    return build_high_precision_models(data_dir, out_dir)
