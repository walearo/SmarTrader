"""
Backtest the Donchian channel breakout strategy on historical OANDA data.

Improvements over the old EMA backtest:
  - Batched OANDA fetch (no 5000-candle single-request limit)
  - Donchian breakout signal (matches live strategy.py)
  - 1-pip round-trip spread cost deducted from every trade
  - Max trade duration: force-close after BACKTEST_MAX_TRADE_DURATION hours
  - Session filter mirrors the live bot

Usage:
    python backtest.py                        # uses config.py defaults
    python backtest.py --pair EUR_USD --days 365
    python backtest.py --pair GBP_USD --days 180 --units 2000
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

import config
from strategy import add_indicators


# ─── DATA ─────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 500
_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}


def fetch_history(pair: str, days: int) -> pd.DataFrame:
    """Fetch H1 candles in batches — no single-request candle limit."""
    tf   = config.TIMEFRAME
    step = timedelta(minutes=_TF_MINUTES.get(tf, 60))

    client = oandapyV20.API(
        access_token=config.OANDA_API_KEY,
        environment=config.OANDA_ENVIRONMENT,
    )
    end         = datetime.now(timezone.utc)
    start       = end - timedelta(days=days)
    batch_start = start
    all_rows    = []

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
    print(f"  Fetched {len(df)} {tf} candles for {pair}.")
    return df


# ─── SIMULATION ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    pair:       str
    direction:  str        # 'buy' | 'sell'
    entry_time: datetime
    entry:      float
    sl:         float
    tp:         float
    exit_time:  datetime  = None
    exit_price: float     = None
    result:     str       = None   # 'win' | 'loss' | 'timeout'
    pnl_pips:   float     = 0.0


@dataclass
class BacktestResult:
    pair:            str
    trades:          list  = field(default_factory=list)
    starting_equity: float = 10_000.0

    def pip(self):
        return 0.0001 if "JPY" not in self.pair else 0.01

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.result]
        if not closed:
            return {"error": "No closed trades."}

        wins    = [t for t in closed if t.result == "win"]
        losses  = [t for t in closed if t.result in ("loss", "timeout")]

        gross_profit = sum(t.pnl_pips for t in wins)
        gross_loss   = abs(sum(t.pnl_pips for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

        # equity curve (1 pip ≈ $0.10 per micro-lot / 1000 units)
        pip_value = 0.10 * (config.UNITS / 1000)
        equity = self.starting_equity
        peak   = equity
        max_dd = 0.0
        for t in closed:
            equity += t.pnl_pips * pip_value
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        timeouts = len([t for t in closed if t.result == "timeout"])
        return {
            "pair":           self.pair,
            "total_trades":   len(closed),
            "wins":           len(wins),
            "losses":         len(losses),
            "timeouts":       timeouts,
            "win_rate":       f"{len(wins)/len(closed)*100:.1f}%",
            "gross_profit_p": f"{gross_profit:.1f} pips",
            "gross_loss_p":   f"{gross_loss:.1f} pips",
            "net_pips":       f"{gross_profit - gross_loss:.1f} pips",
            "profit_factor":  f"{profit_factor:.2f}",
            "max_drawdown":   f"{max_dd:.1f}%",
            "final_equity":   f"${equity:,.2f}",
            "spread_cost_p":  f"{len(closed) * config.BACKTEST_SPREAD_PIPS:.1f} pips total",
        }


def run_backtest(pair: str, df: pd.DataFrame) -> BacktestResult:
    df     = add_indicators(df)
    result = BacktestResult(pair=pair)
    pip    = result.pip()

    # spread cost per trade in price units (round-trip)
    spread_cost = config.BACKTEST_SPREAD_PIPS * pip

    in_trade = False
    trade: Trade = None

    lookback = max(config.DONCHIAN_PERIOD, 20) + 1

    for i in range(lookback, len(df)):
        candle = df.iloc[i]

        # ── manage open trade ──────────────────────────────────────────────
        if in_trade:
            hi, lo = candle["high"], candle["low"]

            # max duration check — force-close at current candle's close
            bars_open = i - list(df.index).index(trade.entry_time)
            if bars_open >= config.BACKTEST_MAX_TRADE_DURATION:
                close_px = candle["close"]
                trade.exit_time  = candle.name
                trade.exit_price = close_px
                trade.result     = "timeout"
                if trade.direction == "buy":
                    trade.pnl_pips = (close_px - trade.entry) / pip - config.BACKTEST_SPREAD_PIPS
                else:
                    trade.pnl_pips = (trade.entry - close_px) / pip - config.BACKTEST_SPREAD_PIPS
                result.trades.append(trade)
                in_trade = False
                continue

            if trade.direction == "buy":
                if lo <= trade.sl:
                    trade.exit_time  = candle.name
                    trade.exit_price = trade.sl
                    trade.result     = "loss"
                    trade.pnl_pips   = (trade.sl - trade.entry) / pip - config.BACKTEST_SPREAD_PIPS
                    in_trade = False
                elif hi >= trade.tp:
                    trade.exit_time  = candle.name
                    trade.exit_price = trade.tp
                    trade.result     = "win"
                    trade.pnl_pips   = (trade.tp - trade.entry) / pip - config.BACKTEST_SPREAD_PIPS
                    in_trade = False
            else:  # sell
                if hi >= trade.sl:
                    trade.exit_time  = candle.name
                    trade.exit_price = trade.sl
                    trade.result     = "loss"
                    trade.pnl_pips   = (trade.entry - trade.sl) / pip - config.BACKTEST_SPREAD_PIPS
                    in_trade = False
                elif lo <= trade.tp:
                    trade.exit_time  = candle.name
                    trade.exit_price = trade.tp
                    trade.result     = "win"
                    trade.pnl_pips   = (trade.entry - trade.tp) / pip - config.BACKTEST_SPREAD_PIPS
                    in_trade = False

            if not in_trade:
                result.trades.append(trade)
            continue   # one trade at a time

        # ── session filter ────────────────────────────────────────────────
        hour = candle.name.hour
        if not any(start <= hour < end for start, end in config.SESSIONS):
            continue

        # ── Donchian breakout signal (mirrors strategy.py get_signal exactly) ──
        if pd.isna(candle["dc_high"]) or pd.isna(candle["vol_ma"]) or pd.isna(candle["adx"]):
            continue

        vol_surge    = candle["volume"] > config.VOLUME_SURGE_FACTOR * candle["vol_ma"]
        adx_ok       = candle["adx"] > config.ADX_MIN_TREND
        buf          = config.BREAKOUT_ATR_BUFFER * candle["atr"]
        candle_range = candle["high"] - candle["low"]
        close_pct    = (candle["close"] - candle["low"]) / candle_range if candle_range > 0 else 0.5

        if (candle["close"] > candle["dc_high"] + buf
                and candle["rsi"] > config.RSI_BUY_MIN
                and vol_surge and adx_ok
                and candle["adx_pos"] > candle["adx_neg"]
                and close_pct >= config.BREAKOUT_CANDLE_CLOSE_PCT):
            signal = "buy"
        elif (candle["close"] < candle["dc_low"] - buf
                and candle["rsi"] < config.RSI_SELL_MAX
                and vol_surge and adx_ok
                and candle["adx_neg"] > candle["adx_pos"]
                and close_pct <= (1.0 - config.BREAKOUT_CANDLE_CLOSE_PCT)):
            signal = "sell"
        else:
            continue

        # entry at next candle open (mirrors live bot and label_signals)
        if i + 1 >= len(df):
            continue
        next_candle = df.iloc[i + 1]
        atr         = candle["atr"]
        entry       = next_candle["open"]

        if signal == "buy":
            sl = round(entry - atr * config.ATR_SL_MULTIPLIER, 5)
            tp = round(entry + atr * config.ATR_TP_MULTIPLIER, 5)
        else:
            sl = round(entry + atr * config.ATR_SL_MULTIPLIER, 5)
            tp = round(entry - atr * config.ATR_TP_MULTIPLIER, 5)

        trade    = Trade(pair, signal, candle.name, entry, sl, tp)
        in_trade = True

    return result


# ─── REPORT ───────────────────────────────────────────────────────────────────

def print_report(result: BacktestResult) -> None:
    import sys
    # ensure Unicode box characters print on Windows without crashing
    out = sys.stdout if sys.stdout.encoding and sys.stdout.encoding.lower() in ("utf-8", "utf-16") \
          else open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False, buffering=1)

    def p(text=""):
        print(text, file=out)

    s = result.summary()
    if "error" in s:
        p(f"\n  {s['error']}")
        return

    p(f"\n{'─'*50}")
    p(f"  BACKTEST REPORT — {s['pair']}")
    p(f"{'─'*50}")
    for k, v in s.items():
        if k == "pair":
            continue
        label = k.replace("_", " ").title()
        p(f"  {label:<26} {v}")
    p(f"{'─'*50}\n")

    # trade log
    p("  Recent trades (last 10):")
    p(f"  {'#':<4} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'Pips':>8} {'Result'}")
    p(f"  {'-'*55}")
    for i, t in enumerate(result.trades[-10:], 1):
        pips = f"{t.pnl_pips:+.1f}"
        p(f"  {i:<4} {t.direction:<5} {t.entry:>10.5f} {t.exit_price:>10.5f} {pips:>8} {t.result}")
    p()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FX Bot Backtester — Donchian breakout")
    parser.add_argument("--pair",  default=None,  help="e.g. EUR_USD (default: all pairs in config)")
    parser.add_argument("--days",  type=int, default=180, help="history window in days (default: 180)")
    parser.add_argument("--units", type=int, default=None, help="override position size")
    args = parser.parse_args()

    if args.units:
        config.UNITS = args.units

    pairs = [args.pair] if args.pair else config.PAIRS

    print(
        f"\nBacktest: {config.TIMEFRAME} | Donchian {config.DONCHIAN_PERIOD} | "
        f"VolSurge {config.VOLUME_SURGE_FACTOR}x | RSI {config.RSI_PERIOD} | "
        f"Spread {config.BACKTEST_SPREAD_PIPS}pip | MaxDur {config.BACKTEST_MAX_TRADE_DURATION}h | "
        f"{args.days}d history"
    )

    for pair in pairs:
        print(f"\nFetching data for {pair}...")
        try:
            df     = fetch_history(pair, args.days)
            result = run_backtest(pair, df)
            print_report(result)
        except Exception as e:
            print(f"  ERROR for {pair}: {e}")


if __name__ == "__main__":
    main()
