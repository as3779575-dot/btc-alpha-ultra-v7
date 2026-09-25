# BTC Alpha Ultra V7 — Deep Precision Cloud Monitor (BTCUSDT + ETHUSDT)

This version is designed around **high-selectivity BTCUSDT and ETHUSDT alerts**. Each asset uses its own historical model and its own strict deployment gate. It does not place trades; it only sends Telegram research alerts for manual execution.

## Signal hierarchy

4H = regime and major market structure

1H = structural direction and available target room

30M = setup formation and breakout/reclaim context

15M = execution structure, ATR, momentum, volume and taker flow

5M = statistical model confirmation and short-term confirmation

## Historical gate

The bootstrap workflow downloads BTCUSDT and ETHUSDT 1-minute Binance Futures history and runs the strict historical model-selection process separately for each asset. A live model is only accepted when the selected artifact passes the configured hard gate, including:

- gross target >= 2R
- pooled walk-forward OOS win rate >= 70%
- final holdout win rate >= 70%
- every required walk-forward fold passing
- minimum OOS sample
- holdout pass

The live monitor exits instead of trading when this artifact gate is not satisfied.

## Asset-specific models

BTCUSDT and ETHUSDT are not scored with the same performance claim. Each asset must pass the configured historical OOS and final-holdout gates independently before its live monitor can run.

## Live deep research

The live gate adds independent current evidence:

- confirmed 4H/1H/30M/15M structure
- HH/HL/LH/LL swing context and BOS
- support/resistance and 2R target room
- 15M ATR/volatility sanity
- volume and taker-flow confirmation
- live bid/ask spread
- live order-book imbalance
- futures price/OI relationship
- funding
- futures basis
- global long/short ratio
- ADL risk rating
- BTC options open interest around upcoming expiries and near-ATM contracts when available
- recent Bitcoin/crypto headline risk via public RSS
- adversarial vetoes for contradictory positioning/crowding
- late-entry protection
- no daily alert ceiling and no live alert cooldown

Live options, derivatives, and news inputs are fail-closed for a fresh entry: an unavailable required layer blocks the signal rather than silently substituting a neutral value. A known high-impact recent headline also blocks a fresh entry.

## Cloud setup

1. Create a **public GitHub repository** and upload this folder's contents.
2. Go to **Actions → BTC Alpha Ultra V7 Bootstrap → Run workflow**.
3. Let bootstrap finish. It commits the BTCUSDT artifacts at `models/selected_model.joblib` and `models/selection.json`, plus the ETHUSDT artifacts at `models/ETHUSDT/selected_model.joblib` and `models/ETHUSDT/selection.json`, only when their strict historical gates pass.
4. Add repository secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Go to **Actions → BTC Alpha Ultra V7 Deep Live Monitor → Run workflow**.
6. The live workflow subsequently runs BTCUSDT and ETHUSDT monitors concurrently on the scheduled blocks.

## Important

This is a signal research system, not guaranteed profitability. A historical >70% win rate is an out-of-sample measurement of the selected historical rules; it is not a promise that future live trades will exceed 70%.

Historical 1-minute OHLCV cannot reconstruct the historical order book, live spread, options chain, or news tape exactly. Those are therefore used as **live veto/confirmation layers**, not falsely claimed as part of the historical backtest. The live feature construction is kept aligned with the historical feature definitions, including the selected strategy, barrier parameters, weekly configuration, volatility regime, and completed-candle timing.
