"""
Train a Voting Ensemble (LightGBM + RandomForest + LogisticRegression) on historical OANDA data.

Trains on TRAIN_TIMEFRAME (H1 by default — same as live inference to avoid distribution mismatch).
Reserves the last 90 days as a held-out test set that is never touched during training.

Lookahead fix:
  - Signal detected at bar i   (dc_high/dc_low built from bars i-DONCHIAN_PERIOD..i-1)
  - Entry at bar i+1 open
  - Features taken from bar i  (no bar i+1 data used — zero lookahead)

Usage:
    python train_ml.py                        # all config pairs, 365 days, H1
    python train_ml.py --days 730             # more history
    python train_ml.py --pair EUR_USD         # single pair
    python train_ml.py --train-timeframe H4   # override timeframe

Saves model to config.ML_MODEL_PATH (model.pkl).
"""

import argparse
import pickle
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

import config
from strategy import add_indicators
from ml_model import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_HOLDOUT_DAYS = 90   # last 90 days reserved as held-out test set


# ─── FETCH HISTORY ────────────────────────────────────────────────────────────

_BATCH_SIZE = 500   # OANDA max candles per request

# minutes per candle for each granularity (used to advance the batch window)
_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}


def fetch_history(pair: str, days: int, timeframe: str = None) -> pd.DataFrame:
    """Fetch historical candles in batches to stay within OANDA's per-request limit."""
    tf = timeframe or config.TRAIN_TIMEFRAME
    step = timedelta(minutes=_TF_MINUTES.get(tf, 60))

    client = oandapyV20.API(
        access_token=config.OANDA_API_KEY,
        environment=config.OANDA_ENVIRONMENT,
    )
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_rows = []
    batch_start = start

    while batch_start < end:
        params = {
            "from":        batch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count":       _BATCH_SIZE,
            "granularity": tf,
            "price":       "M",
        }
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)

        candles = r.response["candles"]
        if not candles:
            break

        for candle in candles:
            if not candle["complete"]:
                continue
            mid = candle["mid"]
            t   = pd.to_datetime(candle["time"])
            if t >= end:
                break
            all_rows.append({
                "time":   t,
                "open":   float(mid["o"]),
                "high":   float(mid["h"]),
                "low":    float(mid["l"]),
                "close":  float(mid["c"]),
                "volume": int(candle["volume"]),
            })

        last_time = pd.to_datetime(candles[-1]["time"])
        if last_time <= batch_start:
            break
        batch_start = last_time + step

    df = pd.DataFrame(all_rows).set_index("time")
    df = df[~df.index.duplicated(keep="first")]
    log.info(f"Fetched {len(df)} {tf} candles for {pair}.")
    return df


# ─── LABEL TRADES ────────────────────────────────────────────────────────────

def _pip(pair: str) -> float:
    return 0.0001 if "JPY" not in pair else 0.01


def label_signals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Detect every Donchian breakout signal and label it 1 (win) or 0 (loss).

    Lookahead-free methodology:
      - Bar i:   signal fires when close[i] breaks dc_high[i] or dc_low[i].
                 dc_high/dc_low are computed with .shift(1) in add_indicators,
                 so they only use data up to bar i-1.
      - Bar i+1: entry at open[i+1]  (this is when the bot would actually fill)
      - Features come from bar i     (no future data)
      - SL/TP based on ATR from bar i

    The label row is indexed at bar i (the signal bar) so it aligns with
    the feature row built by build_features.
    """
    df      = add_indicators(df)
    records = []

    lookback = max(config.DONCHIAN_PERIOD, 20) + 1   # enough warm-up bars

    for i in range(lookback, len(df) - 1):
        bar   = df.iloc[i]
        entry_bar = df.iloc[i + 1]   # next bar — where we actually enter

        if pd.isna(bar["dc_high"]) or pd.isna(bar["vol_ma"]) or pd.isna(bar["adx"]):
            continue

        vol_surge = bar["volume"] > config.VOLUME_SURGE_FACTOR * bar["vol_ma"]
        adx_ok    = bar["adx"] > config.ADX_MIN_TREND
        buf       = config.BREAKOUT_ATR_BUFFER * bar["atr"]

        candle_range = bar["high"] - bar["low"]
        close_pct    = (bar["close"] - bar["low"]) / candle_range if candle_range > 0 else 0.5

        if (bar["close"] > bar["dc_high"] + buf
                and bar["rsi"] > config.RSI_BUY_MIN
                and vol_surge
                and adx_ok
                and bar["adx_pos"] > bar["adx_neg"]
                and close_pct >= config.BREAKOUT_CANDLE_CLOSE_PCT):
            signal = "buy"
        elif (bar["close"] < bar["dc_low"] - buf
                and bar["rsi"] < config.RSI_SELL_MAX
                and vol_surge
                and adx_ok
                and bar["adx_neg"] > bar["adx_pos"]
                and close_pct <= (1.0 - config.BREAKOUT_CANDLE_CLOSE_PCT)):
            signal = "sell"
        else:
            continue

        atr   = bar["atr"]
        entry = entry_bar["open"]

        if signal == "buy":
            sl = entry - atr * config.ATR_SL_MULTIPLIER
            tp = entry + atr * config.ATR_TP_MULTIPLIER
        else:
            sl = entry + atr * config.ATR_SL_MULTIPLIER
            tp = entry - atr * config.ATR_TP_MULTIPLIER

        # simulate forward from bar i+1 onwards
        outcome = None
        for j in range(i + 1, min(i + 200, len(df))):
            hi = df.iloc[j]["high"]
            lo = df.iloc[j]["low"]
            if signal == "buy":
                if lo <= sl:  outcome = 0; break
                if hi >= tp:  outcome = 1; break
            else:
                if hi >= sl:  outcome = 0; break
                if lo <= tp:  outcome = 1; break

        if outcome is None:
            continue   # trade never resolved — skip

        records.append({
            "index":  df.index[i],   # signal bar index → aligns with feature row
            "signal": 1 if signal == "buy" else 0,
            "label":  outcome,
        })

    wins   = sum(r["label"] for r in records)
    losses = len(records) - wins
    log.info(f"Labelled {len(records)} signals — {wins} wins, {losses} losses.")
    return pd.DataFrame(records).set_index("index")


# ─── BUILD MODEL ─────────────────────────────────────────────────────────────

def build_voting_ensemble() -> VotingClassifier:
    """
    Soft-voting ensemble of three diverse learners with strong regularisation
    to prevent overfitting on ~1000–2000 samples.

    Changes from original:
      - LightGBM: fewer trees (100), larger min_child_samples (30), explicit L1/L2 reg
      - RandomForest replaced with ExtraTrees — extra randomness reduces variance
      - LogisticRegression: C=0.05 (stronger penalty)
    """
    lgbm = LGBMClassifier(
        n_estimators=100,           # fewer trees → less overfitting
        max_depth=3,                # shallower
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,       # at least 30 samples to create a leaf
        reg_alpha=0.1,              # L1 regularisation
        reg_lambda=1.0,             # L2 regularisation
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    # ExtraTrees uses extra randomness in split selection — lower variance than RF
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=10,        # conservative leaf size
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, max_iter=1000,
                                   class_weight="balanced", random_state=42)),
    ])
    return VotingClassifier(
        estimators=[("lgbm", lgbm), ("et", et), ("lr", lr)],
        voting="soft",
    )


def _cv_and_fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: VotingClassifier,
    label: str,
) -> VotingClassifier:
    """
    Walk-forward CV on training set → report metrics → final fit on full training set
    → evaluate on held-out test set.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []

    log.info(f"Walk-forward CV on {len(X_train)} train samples ({label})...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, probs)
            aucs.append(auc)
            log.info(f"  Fold {fold} AUC: {auc:.3f}")

    log.info(f"Mean CV AUC: {np.mean(aucs):.3f}" if aucs else "CV: no folds with both classes")

    # Final fit: wrap with isotonic calibration so predicted probabilities are
    # meaningful (i.e. model.predict_proba(...)[1] == 0.60 → ~60% actual win rate).
    # CalibratedClassifierCV with cv=5 handles this entirely on X_train via
    # cross-validation — the held-out test set is never touched.
    log.info("Fitting calibrated model on full training set...")
    calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)

    # ── Held-out test set evaluation ─────────────────────────────────────────
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        test_probs = calibrated.predict_proba(X_test)[:, 1]
        test_preds = calibrated.predict(X_test)
        test_auc   = roc_auc_score(y_test, test_probs)
        log.info(f"\n{'='*55}")
        log.info(f"HELD-OUT TEST SET ({len(X_test)} samples, last {_HOLDOUT_DAYS} days)")
        log.info(f"Test AUC: {test_auc:.3f}  (>0.55 = useful, >0.60 = good)")
        log.info(f"{'='*55}")
        print(f"\nHeld-out test set classification report:")
        print(classification_report(y_test, test_preds, target_names=["loss", "win"]))

        # ── threshold recommendation ─────────────────────────────────────
        # With a calibrated model the predicted probabilities reflect the true
        # base win rate (~27%).  A threshold of 0.50 will reject everything.
        # Instead we look for the threshold that achieves precision >= base_rate+0.10
        # (i.e. at least 10 percentage points above the signal's natural hit rate)
        # while keeping recall >= 5% so the filter doesn't block all trades.
        base_rate = float(y_test.mean())
        min_prec  = base_rate + 0.10   # meaningful improvement over random signal
        log.info(
            f"Base win rate on test set: {base_rate:.1%} | "
            f"Probability dist: mean={test_probs.mean():.3f}, "
            f"p50={np.percentile(test_probs, 50):.3f}, "
            f"p75={np.percentile(test_probs, 75):.3f}, "
            f"p90={np.percentile(test_probs, 90):.3f}"
        )
        precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
        recommended = None
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            if p >= min_prec and r >= 0.05:
                recommended = t
                break
        if recommended is not None:
            log.info(
                f"Recommended ML_MIN_CONFIDENCE: {recommended:.2f}  "
                f"(achieves ≥{min_prec:.0%} win precision, >{base_rate:.0%} base rate)"
            )
        else:
            log.warning(
                "No threshold achieves base_rate+10pp precision on the test set. "
                "Consider setting ML_MIN_CONFIDENCE = 0.0 to bypass the ML filter."
            )
    else:
        log.warning("Held-out test set too small or single-class — skipping test evaluation.")

    return calibrated


# ─── PREPARE DATASET ─────────────────────────────────────────────────────────

def _prepare_dataset(pair: str, days: int, tf: str) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch, label, and build features for one pair. Returns (X, y) with aligned index."""
    df       = fetch_history(pair, days, timeframe=tf)
    labels   = label_signals(df, pair)
    features = build_features(df)

    common = features.index.intersection(labels.index)
    X      = features.loc[common].copy()
    X["signal"] = labels.loc[common, "signal"]
    y      = labels.loc[common, "label"]
    return X, y


def _train_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Time-based split using DatetimeIndex: train = everything before cutoff,
    test = last HOLDOUT_DAYS.  Must be called while X still has a DatetimeIndex
    (i.e. before pd.concat with ignore_index=True).
    """
    cutoff = X.index.max() - pd.Timedelta(days=_HOLDOUT_DAYS)
    mask   = X.index <= cutoff
    return X[mask], y[mask], X[~mask], y[~mask]


# ─── TRAIN (SINGLE PAIR) ─────────────────────────────────────────────────────

def train(pair: str, days: int, train_timeframe: str = None) -> None:
    tf = train_timeframe or config.TRAIN_TIMEFRAME
    log.info(f"Training on {pair}, {days} days of {tf} data...")

    X, y = _prepare_dataset(pair, days, tf)

    if len(X) < 30:
        log.error("Not enough labelled samples to train. Try more days.")
        return

    X_train, y_train, X_test, y_test = _train_holdout_split(X, y)
    log.info(f"Train: {len(X_train)} samples | Test (holdout): {len(X_test)} samples")

    model = _cv_and_fit(X_train, y_train, X_test, y_test, build_voting_ensemble(), pair)

    with open(config.ML_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved to {config.ML_MODEL_PATH}")

    preds = model.predict(X_train)  # model is already the calibrated wrapper
    print("\nTraining set classification report:")
    print(classification_report(y_train, preds, target_names=["loss", "win"]))


# ─── MULTI-PAIR TRAINING ─────────────────────────────────────────────────────

def train_combined(pairs: list, days: int, train_timeframe: str = None) -> None:
    """
    Train one shared model across all pairs.

    Holdout split happens PER PAIR (while DatetimeIndex is intact) before concat,
    guaranteeing that each pair's last HOLDOUT_DAYS are in the test set — not an
    arbitrary positional slice of the concatenated array.
    """
    tf = train_timeframe or config.TRAIN_TIMEFRAME
    log.info(f"Training combined model on {pairs}, {days} days of {tf} data...")

    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    for pair in pairs:
        X, y = _prepare_dataset(pair, days, tf)
        if len(X) < 10:
            log.warning(f"{pair}: too few samples — skipping.")
            continue
        # split while DatetimeIndex is available
        X_tr, y_tr, X_te, y_te = _train_holdout_split(X, y)
        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_test_all.append(X_te)
        y_test_all.append(y_te)

    if not X_train_all:
        log.error("No pairs produced enough samples.")
        return

    # now reset_index to avoid DatetimeIndex collisions across pairs
    X_train = pd.concat(X_train_all, ignore_index=True)
    y_train = pd.concat(y_train_all, ignore_index=True)
    X_test  = pd.concat(X_test_all,  ignore_index=True)
    y_test  = pd.concat(y_test_all,  ignore_index=True)

    total = len(X_train) + len(X_test)
    log.info(
        f"Combined dataset: {total} samples — "
        f"{int((pd.concat(y_train_all + y_test_all)).sum())} wins, "
        f"{int((1 - pd.concat(y_train_all + y_test_all)).sum())} losses"
    )
    log.info(f"Train: {len(X_train)} samples | Test (holdout): {len(X_test)} samples")

    model = _cv_and_fit(X_train, y_train, X_test, y_test, build_voting_ensemble(), f"{pairs}")

    with open(config.ML_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Combined model saved to {config.ML_MODEL_PATH}")

    preds = model.predict(X_train)  # model is already the calibrated wrapper
    print("\nTraining set classification report:")
    print(classification_report(y_train, preds, target_names=["loss", "win"]))


# ─── ENTRY ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FX Bot voting ensemble")
    parser.add_argument("--pair", default=None,
                        help="single pair (default: all config pairs combined)")
    parser.add_argument("--days", type=int, default=365,
                        help="days of history to fetch (default: 365)")
    parser.add_argument("--train-timeframe", default=None,
                        help=f"candle granularity (default: config.TRAIN_TIMEFRAME={config.TRAIN_TIMEFRAME})")
    args = parser.parse_args()

    tf = args.train_timeframe

    if args.pair:
        train(args.pair, args.days, train_timeframe=tf)
    else:
        train_combined(config.PAIRS, args.days, train_timeframe=tf)
