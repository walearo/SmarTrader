"""
FX Bot — Live Web Dashboard
Features: Auth, Manual Controls, Charts, Mobile-Optimized

Run:  python dashboard.py
Open: http://127.0.0.1:8000   (localhost only — use SSH tunnel for remote access)
"""

import json
import os
import asyncio
from collections import deque
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

import oandapyV20
import oandapyV20.endpoints.accounts as accounts_ep
import oandapyV20.endpoints.trades as trades_ep
import oandapyV20.endpoints.positions as positions_ep

import config
import db
import bot_control
from risk_manager import get_drawdown_status

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=config.DASHBOARD_SECRET_KEY)

_equity_history: deque = deque(maxlen=300)


# ─── AUTH ─────────────────────────────────────────────────────────────────────

def authenticated(request: Request) -> bool:
    return request.session.get("authenticated") is True


# ─── OANDA ────────────────────────────────────────────────────────────────────

def _client():
    return oandapyV20.API(
        access_token=config.OANDA_API_KEY,
        environment=config.OANDA_ENVIRONMENT,
    )


def _account() -> dict:
    r = accounts_ep.AccountSummary(accountID=config.OANDA_ACCOUNT_ID)
    _client().request(r)
    a = r.response["account"]
    nav = float(a["NAV"])
    margin = float(a["marginUsed"])
    return {
        "nav":           round(nav, 2),
        "balance":       round(float(a["balance"]), 2),
        "unrealized_pl": round(float(a["unrealizedPL"]), 2),
        "realized_pl":   round(float(a["pl"]), 2),
        "margin_used":   round(margin, 2),
        "margin_pct":    round(margin / nav * 100, 1) if nav > 0 else 0,
        "open_trades":   int(a["openTradeCount"]),
        "currency":      a["currency"],
    }


_PRICE_DP = {
    "XAU_USD":  2,
    "WTICO_USD": 3,
    "USD_JPY":  3,
    "USD_CHF":  5,
}
_PRICE_DP_DEFAULT = 5

_UNITS_SUFFIX = {
    "XAU_USD":   " oz",
    "WTICO_USD": " bbl",
}


def _open_trades() -> list:
    r = trades_ep.OpenTrades(accountID=config.OANDA_ACCOUNT_ID)
    _client().request(r)
    result = []
    for t in r.response.get("trades", []):
        try:
            inst  = t["instrument"]
            units = float(t["currentUnits"])
            sl    = t.get("stopLossOrder", {}).get("price", "—")
            tp    = t.get("takeProfitOrder", {}).get("price", "—")
            result.append({
                "id":           t["id"],
                "instrument":   inst,
                "display_name": _fmt_instrument(inst),
                "price_dp":     _PRICE_DP.get(inst, _PRICE_DP_DEFAULT),
                "units_suffix": _UNITS_SUFFIX.get(inst, ""),
                "direction":    "BUY" if units > 0 else "SELL",
                "units":        abs(int(units)),
                "entry":        float(t["price"]),
                "sl":           sl,
                "tp":           tp,
                "unrealized":   round(float(t["unrealizedPL"]), 2),
                "opened":       t["openTime"][:16].replace("T", " "),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return result


def _safe_open_trades() -> list:
    try:
        return _open_trades()
    except Exception:
        return []


def _closed_trades(count: int = 30) -> list:
    params = {"state": "CLOSED", "count": count}
    r = trades_ep.TradesList(accountID=config.OANDA_ACCOUNT_ID, params=params)
    _client().request(r)
    result = []
    for t in r.response.get("trades", []):
        try:
            units = float(t.get("initialUnits", 0))
            pl    = round(float(t.get("realizedPL", 0)), 2)

            duration = "—"
            try:
                from datetime import datetime as _dt
                open_t  = _dt.fromisoformat(t["openTime"].replace("Z", "+00:00"))
                close_t = _dt.fromisoformat(t["closeTime"].replace("Z", "+00:00"))
                secs    = int((close_t - open_t).total_seconds())
                h, m    = divmod(secs // 60, 60)
                duration = f"{h}h {m}m" if h > 0 else f"{m}m"
            except Exception:
                pass

            inst = t["instrument"]
            result.append({
                "instrument":   inst,
                "display_name": _fmt_instrument(inst),
                "price_dp":     _PRICE_DP.get(inst, _PRICE_DP_DEFAULT),
                "units_suffix": _UNITS_SUFFIX.get(inst, ""),
                "direction":    "BUY" if units > 0 else "SELL",
                "units":        abs(int(units)),
                "entry":        float(t["price"]),
                "close":        float(t.get("averageClosePrice", 0)),
                "pl":           pl,
                "result":       "WIN" if pl > 0 else "LOSS",
                "closed":       t.get("closeTime", "")[:16].replace("T", " "),
                "duration":     duration,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return result


def _close_trade(trade_id: str) -> dict:
    r = trades_ep.TradeClose(accountID=config.OANDA_ACCOUNT_ID, tradeID=trade_id)
    _client().request(r)
    return r.response


def _close_all_trades() -> list:
    results = []
    for pair in config.PAIRS:
        try:
            body = {"longUnits": "ALL", "shortUnits": "ALL"}
            r = positions_ep.PositionClose(
                accountID=config.OANDA_ACCOUNT_ID,
                instrument=pair,
                data=body,
            )
            _client().request(r)
            results.append({"pair": pair, "status": "closed"})
        except Exception as e:
            results.append({"pair": pair, "status": "error", "detail": str(e)})
    return results


def _bot_log(count: int = 40) -> list:
    try:
        return db.log_recent(count)
    except Exception:
        return []


def _news() -> list:
    try:
        from news import upcoming_events
        return upcoming_events(hours=8)
    except Exception:
        return []


def _journal_analytics() -> dict:
    try:
        entries = db.journal_all()
    except Exception:
        return {"entries": [], "score_dist": {}, "regime_stats": {}, "themes": [], "total": 0}

    score_dist: dict[int, int] = {i: 0 for i in range(1, 11)}
    regime_stats: dict[str, dict] = {}
    themes: list[str] = []

    for e in entries:
        a = e.get("analysis", {})
        t = e.get("trade", {})

        score = a.get("score")
        if isinstance(score, (int, float)):
            k = max(1, min(10, int(score)))
            score_dist[k] = score_dist.get(k, 0) + 1

        regime = t.get("regime", "unknown")
        if regime not in regime_stats:
            regime_stats[regime] = {"wins": 0, "total": 0}
        regime_stats[regime]["total"] += 1
        if t.get("pl_pips", 0) > 0:
            regime_stats[regime]["wins"] += 1

        improve = a.get("improve", "")
        if improve and improve not in themes:
            themes.append(improve)

    recent = []
    for e in reversed(entries[-20:]):
        a = e.get("analysis", {})
        t = e.get("trade", {})
        recent.append({
            "time":         e.get("time", "")[:16].replace("T", " "),
            "instrument":   t.get("instrument", ""),
            "display_name": _fmt_instrument(t.get("instrument", "")),
            "direction":    t.get("direction", ""),
            "pl_pips":    round(float(t.get("pl_pips", 0)), 1),
            "score":      a.get("score", "—"),
            "why":        a.get("why", ""),
            "done_well":  a.get("done_well", ""),
            "improve":    a.get("improve", ""),
            "regime":     t.get("regime", ""),
        })

    return {
        "entries":      recent,
        "score_dist":   score_dist,
        "regime_stats": regime_stats,
        "themes":       themes[-10:],
        "total":        len(entries),
    }


_DISPLAY_NAMES = {
    "XAU_USD":   "XAU/USD",
    "WTICO_USD": "WTI/USD",
}

def _fmt_instrument(instrument: str) -> str:
    return _DISPLAY_NAMES.get(instrument, instrument.replace("_", "/"))


def _chart_data(closed: list) -> dict:
    wins   = [t for t in closed if t["result"] == "WIN"]
    losses = [t for t in closed if t["result"] == "LOSS"]

    pair_pips: dict[str, float] = {}
    for t in closed:
        p   = t["instrument"]
        pip = config.INSTRUMENT_PIP.get(p, config.INSTRUMENT_PIP_DEFAULT)
        if t["direction"] == "BUY":
            pips = (t["close"] - t["entry"]) / pip
        else:
            pips = (t["entry"] - t["close"]) / pip
        pair_pips[p] = round(pair_pips.get(p, 0) + pips, 1)

    return {
        "win_loss":    [len(wins), len(losses)],
        "pair_labels": [_fmt_instrument(p) for p in pair_pips],
        "pair_pips":   list(pair_pips.values()),
    }


# ─── ROUTES — AUTH ────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="color-scheme" content="dark"/>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<title>FX Bot — Login</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  html{background:#0d1117}
  body{background:#0d1117;display:flex;align-items:center;justify-content:center;min-height:100vh;font-family:'Segoe UI',system-ui,sans-serif}
  .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:40px 36px;width:100%;max-width:380px}
  h1{color:#e6edf3;font-size:20px;margin-bottom:6px;text-align:center}
  .sub{color:#8b949e;font-size:13px;text-align:center;margin-bottom:28px}
  label{display:block;color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:14px;padding:10px 14px;margin-bottom:18px;outline:none}
  input:focus{border-color:#58a6ff}
  button{width:100%;background:#1f6feb;border:none;border-radius:6px;color:#fff;cursor:pointer;font-size:14px;font-weight:600;padding:12px;transition:.2s}
  button:hover{background:#388bfd}
  .error{background:#3a1a1a;border:1px solid #f85149;border-radius:6px;color:#f85149;font-size:13px;padding:10px 14px;margin-bottom:16px;text-align:center}
</style>
</head>
<body>
<div class="card">
  <h1>⚡ FX Bot</h1>
  <div class="sub">Sign in to your dashboard</div>
  {error}
  <form method="post" action="/login">
    <label>Username</label>
    <input type="text" name="username" autocomplete="username" required autofocus/>
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password" required/>
    <button type="submit">Sign In</button>
  </form>
</div>
</body>
</html>"""


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return LOGIN_HTML.replace("{error}", "")


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == config.DASHBOARD_USERNAME and password == config.DASHBOARD_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)
    error = '<div class="error">Invalid username or password</div>'
    return HTMLResponse(LOGIN_HTML.replace("{error}", error), status_code=401)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ─── ROUTES — CONTROLS ────────────────────────────────────────────────────────

@app.post("/api/control/pause")
def api_pause(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    bot_control.pause("dashboard")
    return {"status": "paused"}


@app.post("/api/control/resume")
def api_resume(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    bot_control.resume()
    return {"status": "active"}


@app.post("/api/control/close/{trade_id}")
def api_close_trade(trade_id: str, request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    try:
        result = _close_trade(trade_id)
        return {"status": "closed", "detail": result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/control/close-all")
def api_close_all(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    return {"results": _close_all_trades()}


@app.get("/api/alerts")
def api_alerts(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    return bot_control.get_alert_settings()


@app.post("/api/alerts/toggle/{alert_type}")
def api_toggle_alert(alert_type: str, request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    if alert_type not in bot_control.ALERT_TYPES:
        return {"error": "unknown alert type"}
    current = bot_control.is_alert_enabled(alert_type)
    bot_control.set_alert(alert_type, not current)
    return {"alert_type": alert_type, "enabled": not current}


@app.get("/api/settings")
def api_get_settings(request: Request):
    if not authenticated(request):
        return {"unauthorized": True}
    return bot_control.get_bot_settings()


@app.post("/api/settings")
async def api_save_settings(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    try:
        body = await request.json()
    except Exception:
        return {"error": "invalid JSON"}

    errors = []
    _int_ranges   = [("MAX_CONCURRENT_TRADES", 1, 10), ("UNITS", 100, 100_000),
                     ("ALERT_PRICE_MOVE_PIPS", 1, 500), ("NEWS_BLACKOUT_MINUTES", 0, 180),
                     ("NEWS_POST_EVENT_HOURS", 0, 24),
                     ("MAX_TRADE_HOURS", 0, 168), ("MAX_DAILY_TRADES", 0, 50),
                     ("MAX_CONSECUTIVE_LOSSES", 0, 20)]
    _float_ranges = [("RISK_PCT_PER_TRADE", 0.1, 5.0),
                     ("MAX_DAILY_LOSS_PCT", 0.5, 20.0), ("MAX_WEEKLY_LOSS_PCT", 1.0, 50.0),
                     ("SPREAD_MAX_PIPS", 0.5, 200.0)]

    for key, lo, hi in _int_ranges:
        if key in body:
            v = body[key]
            if not isinstance(v, int) or not (lo <= v <= hi):
                errors.append(f"{key} must be an integer {lo}–{hi}")

    for key, lo, hi in _float_ranges:
        if key in body:
            v = body[key]
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                errors.append(f"{key} must be {lo}–{hi}")

    if "PAIRS" in body:
        v = body["PAIRS"]
        valid = set(bot_control.ALL_PAIRS)
        if not isinstance(v, list) or not v or not all(p in valid for p in v):
            errors.append("PAIRS must be a non-empty list of valid pair names")

    if "SESSIONS" in body:
        v = body["SESSIONS"]
        if not isinstance(v, list) or not v:
            errors.append("SESSIONS must be a non-empty list")
        else:
            for s in v:
                if not isinstance(s, list) or len(s) != 2:
                    errors.append("Each session must be [start_hour, end_hour]"); break
                start, end = s
                if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= 24):
                    errors.append("Session hours: integers with 0 ≤ start < end ≤ 24"); break

    if errors:
        return {"error": "; ".join(errors)}

    bot_control.save_bot_settings(body)
    return {"status": "saved"}


@app.post("/api/settings/reset")
def api_reset_settings(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    bot_control.reset_bot_settings()
    return {"status": "reset"}


@app.get("/api/pair-stats")
def api_pair_stats(request: Request):
    if not authenticated(request):
        return {"error": "unauthorized"}
    import statistics
    pair_stats = db.pair_stats()
    pnl_series = db.trade_pnl_series()
    sharpe = None
    if len(pnl_series) >= 2:
        mean  = statistics.mean(pnl_series)
        stdev = statistics.stdev(pnl_series)
        sharpe = round(mean / stdev, 3) if stdev > 0 else 0.0
    avg_win  = round(statistics.mean([p for p in pnl_series if p > 0]), 1) if any(p > 0 for p in pnl_series) else 0
    avg_loss = round(statistics.mean([p for p in pnl_series if p < 0]), 1) if any(p < 0 for p in pnl_series) else 0
    total_trades = len(pnl_series)
    total_pnl    = round(sum(pnl_series), 1)
    wins         = sum(1 for p in pnl_series if p > 0)
    overall_wr   = round(wins / total_trades * 100, 1) if total_trades else 0
    return {
        "pair_stats":    pair_stats,
        "sharpe":        sharpe,
        "avg_win_pips":  avg_win,
        "avg_loss_pips": avg_loss,
        "total_trades":  total_trades,
        "total_pnl":     total_pnl,
        "overall_wr":    overall_wr,
    }


# ─── ROUTES — DATA ────────────────────────────────────────────────────────────

@app.get("/api/data")
def api_data(request: Request):
    if not authenticated(request):
        return {"unauthorized": True}

    bot_control.apply_bot_settings()

    try:
        acc = _account()
    except Exception as e:
        return {"error": f"Account fetch failed: {e}"}

    try:
        closed = _closed_trades(50)
    except Exception:
        closed = []

    wins     = sum(1 for t in closed if t["result"] == "WIN")
    win_rate = f"{wins/len(closed)*100:.0f}%" if closed else "—"

    _equity_history.append({
        "time": datetime.now(timezone.utc).strftime("%H:%M"),
        "nav":  acc["nav"],
    })

    ctrl = bot_control.status()

    return {
        "updated":        datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "account":        acc,
        "win_rate":       win_rate,
        "trade_count":    len(closed),
        "bot_paused":     ctrl.get("paused", False),
        "pause_reason":   ctrl.get("reason"),
        "open_trades":    _safe_open_trades(),
        "closed_trades":  closed[:20],
        "equity_history": list(_equity_history),
        "chart_data":     _chart_data(closed),
        "bot_log":        _bot_log(40),
        "news":           _news(),
        "drawdown":       get_drawdown_status(),
        "alert_settings": bot_control.get_alert_settings(),
        "bot_settings":   {"MAX_CONCURRENT_TRADES": config.MAX_CONCURRENT_TRADES},
    }


@app.get("/api/journal")
def api_journal(request: Request):
    if not authenticated(request):
        return {"unauthorized": True}
    return _journal_analytics()


@app.get("/stream")
async def stream(request: Request):
    if not authenticated(request):
        async def unauth():
            yield 'event: unauthorized\ndata: {}\n\n'
        return StreamingResponse(unauth(), media_type="text/event-stream")

    async def generator():
        while True:
            try:
                data = api_data(request)
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── DASHBOARD HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="color-scheme" content="dark"/>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<title>FX Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── Reset & Variables ─────────────────────────────────────────────────────── */
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;
  --green:#3fb950;--red:#f85149;--blue:#58a6ff;
  --yellow:#d29922;--purple:#bc8cff;--orange:#f0883e;
}
*{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:dark;background:#0d1117!important;height:100%}
body{background:#0d1117!important;color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;min-height:100vh;overflow-x:hidden}

/* ── Header ────────────────────────────────────────────────────────────────── */
header{background:var(--card);border-bottom:1px solid var(--border);padding:0 20px;height:52px;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:12px;position:sticky;top:0;z-index:100;flex-shrink:0}
.hdr-left{display:flex;align-items:center;gap:12px}
.hdr-right{display:flex;align-items:center;gap:10px}
header h1{font-size:17px;font-weight:700;white-space:nowrap;line-height:1}
.status-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;line-height:1;white-space:nowrap}
.status-badge::before{content:'';display:block;width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
.badge-active{background:#1a3a1f;color:var(--green);border:1px solid var(--green)}
.badge-paused{background:#3a1a1a;color:var(--red);border:1px solid var(--red)}
.badge-connecting{background:#1e2227;color:var(--muted);border:1px solid var(--border)}
.updated{color:var(--muted);font-size:11px;white-space:nowrap}
.hdr-btn{padding:6px 14px;border-radius:6px;font-size:12px;font-weight:600;line-height:1;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--text);transition:.15s}
@keyframes spin{to{transform:rotate(360deg)}}
.spinning{display:inline-block;animation:spin .7s linear infinite}
.hdr-btn:hover{background:#21262d}
.hdr-btn.danger{border-color:var(--red);color:var(--red)}
.hdr-btn.danger:hover{background:#3a1a1a}
.hdr-btn.primary{border-color:var(--blue);color:var(--blue)}
.hdr-btn.primary:hover{background:#1c2a3a}
.hdr-btn.success{border-color:var(--green);color:var(--green)}
.hdr-btn.success:hover{background:#1a3a1f}

/* ── Layout ────────────────────────────────────────────────────────────────── */
.container{max-width:1500px;margin:0 auto;padding:18px 20px;background-color:var(--bg);min-height:calc(100vh - 52px)}

/* ── Stat Cards ────────────────────────────────────────────────────────────── */
.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
.card .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.card .val{font-size:24px;font-weight:700;line-height:1}
.card .sub{color:var(--muted);font-size:12px;margin-top:5px}
.c-green{color:var(--green)}.c-red{color:var(--red)}.c-blue{color:var(--blue)}.c-yellow{color:var(--yellow)}

/* ── Drawdown Bars ─────────────────────────────────────────────────────────── */
.drawdown-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.dd-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 18px}
.dd-labels{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
.dd-name{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.dd-val{font-size:12px;font-weight:700;color:var(--text)}
.dd-bar-bg{height:6px;background:#21262d;border-radius:4px;overflow:hidden}
.dd-bar-fill{height:100%;border-radius:4px;transition:width .4s ease,background-color .4s}

/* ── Charts Row ────────────────────────────────────────────────────────────── */
.charts-row{display:grid;grid-template-columns:1fr minmax(200px,240px) minmax(240px,280px);gap:12px;margin-bottom:16px}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 18px;overflow:hidden}
.chart-card h2{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px}
.chart-wrap{position:relative;overflow:hidden}
.equity-wrap{height:240px}
.donut-wrap{height:180px;display:flex;align-items:center;justify-content:center}
.bar-wrap{height:180px}
canvas{display:block;background-color:#161b22}

/* ── Content Rows ──────────────────────────────────────────────────────────── */
.row2{display:grid;grid-template-columns:1fr 380px;gap:12px;margin-bottom:16px}
.row-full{margin-bottom:16px}
.tcard{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
.tcard-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.tcard-hdr h2{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}

/* ── Tables ────────────────────────────────────────────────────────────────── */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:560px}
th{color:var(--muted);font-weight:500;text-align:left;padding:6px 10px;border-bottom:1px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:.4px;white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid #21262d;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(255,255,255,.02)}
.t-win{color:var(--green);font-weight:700}
.t-loss{color:var(--red);font-weight:700}
.t-buy{color:var(--green)}.t-sell{color:var(--red)}

/* ── Buttons in table ──────────────────────────────────────────────────────── */
.btn-close{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--red);background:transparent;color:var(--red);transition:.15s}
.btn-close:hover{background:#3a1a1a}
.btn-close:disabled{opacity:.4;cursor:not-allowed}
.btn-sm{padding:4px 12px;border-radius:5px;font-size:11px;font-weight:600;cursor:pointer;background:transparent;transition:.15s}
.btn-danger{border:1px solid var(--red);color:var(--red)}
.btn-danger:hover{background:#3a1a1a}

/* ── Empty states ──────────────────────────────────────────────────────────── */
.empty{color:var(--muted);text-align:center;padding:20px;font-style:italic;font-size:13px}

/* ── News ──────────────────────────────────────────────────────────────────── */
.news-item{display:flex;flex-direction:column;gap:3px;padding:9px 0;border-bottom:1px solid #21262d}
.news-item:last-child{border-bottom:none}
.news-row1{display:flex;align-items:center;gap:8px}
.news-cur{background:#1c2a3a;color:var(--blue);padding:1px 7px;border-radius:4px;font-size:10px;font-weight:700}
.news-impact{background:#3a2a1a;color:var(--orange);padding:1px 7px;border-radius:4px;font-size:10px;font-weight:700}
.news-title{color:var(--text);font-size:12px}
.news-time{color:var(--muted);font-size:11px}
.no-events{color:var(--muted);font-style:italic;font-size:13px;padding:10px 0}
#news-list{max-height:280px;overflow-y:auto}

/* ── Chart range toggle ─────────────────────────────────────────────────────── */
.chart-range{display:flex;gap:4px;margin-bottom:8px;justify-content:flex-end}
.range-btn{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--muted);transition:.15s}
.range-btn:hover{color:var(--text)}
.range-btn.active{background:#1c2a3a;border-color:var(--blue);color:var(--blue)}

/* ── Log Filters ────────────────────────────────────────────────────────────── */
.log-filters{display:flex;gap:6px;align-items:center}
.log-filter-btn{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--muted);transition:.15s}
.log-filter-btn:hover{background:#21262d;color:var(--text)}
.log-filter-btn.active{background:#1c2a3a;border-color:var(--blue);color:var(--blue)}
.err-badge{display:none;background:var(--red);color:#fff;border-radius:10px;font-size:9px;font-weight:700;padding:1px 5px;margin-left:3px;vertical-align:middle}

/* ── Bot Log ────────────────────────────────────────────────────────────────── */
.log-wrap{max-height:420px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#30363d #161b22}
.log-wrap::-webkit-scrollbar{width:6px}
.log-wrap::-webkit-scrollbar-track{background:#161b22;border-radius:3px}
.log-wrap::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.log-wrap::-webkit-scrollbar-thumb:hover{background:#484f58}
.log-entry{display:flex;gap:10px;align-items:flex-start;padding:6px 0;border-bottom:1px solid #21262d;font-size:12px}
.log-entry:last-child{border-bottom:none}
.log-time{color:var(--muted);white-space:nowrap;font-size:10px;min-width:138px;padding-top:1px}
.log-badge{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap}
.lb-signal{background:#1a2a3a;color:var(--blue)}
.lb-trade_open{background:#1a3a1f;color:var(--green)}
.lb-trade_close{background:#1f1a2a;color:var(--purple)}
.lb-filter{background:#2a2a1a;color:var(--yellow)}
.lb-kill_switch{background:#3a1a1a;color:var(--red)}
.lb-error{background:#3a1a1a;color:#ff6b6b}
.lb-info{background:#1e2227;color:var(--muted)}
.log-text{color:var(--text);flex:1;line-height:1.4}

/* ── Toast ──────────────────────────────────────────────────────────────────── */
#toast{position:fixed;bottom:24px;right:24px;background:#161b22;border:1px solid var(--border);border-radius:8px;padding:12px 18px;font-size:13px;display:none;z-index:999;max-width:300px;box-shadow:0 8px 24px rgba(0,0,0,.4)}

/* ── Confirm Modal ───────────────────────────────────────────────────────────── */
#modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:1000;display:none;align-items:center;justify-content:center}
#modal-overlay.open{display:flex}
#modal-box{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:28px 28px 22px;width:100%;max-width:380px;box-shadow:0 16px 48px rgba(0,0,0,.6);animation:modal-in .15s ease}
@keyframes modal-in{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
#modal-icon{font-size:26px;margin-bottom:12px}
#modal-title{font-size:16px;font-weight:700;color:var(--text);margin-bottom:8px}
#modal-msg{font-size:13px;color:var(--muted);line-height:1.5;margin-bottom:22px}
.modal-actions{display:flex;gap:10px;justify-content:flex-end}
.modal-btn{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--text);transition:.15s}
.modal-btn:hover{background:#21262d}
.modal-btn.confirm-danger{background:#3a1a1a;border-color:var(--red);color:var(--red)}
.modal-btn.confirm-danger:hover{background:#4a2020}

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:18px}
.tab-btn{padding:9px 20px;font-size:13px;font-weight:600;cursor:pointer;border:none;background:transparent;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px;transition:.15s}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue)}
.tab-pane{display:none}.tab-pane.active{display:block}

/* ── Per-pair stats ──────────────────────────────────────────────────────────── */
.ps-summary{display:flex;gap:24px;flex-wrap:wrap;padding:10px 0 14px;border-bottom:1px solid #21262d;margin-bottom:12px}
.ps-summary-item{display:flex;flex-direction:column;align-items:center}
.ps-summary-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.ps-summary-value{font-size:18px;font-weight:700;color:var(--text)}
.ps-table{width:100%;border-collapse:collapse;font-size:13px}
.ps-table th{text-align:left;padding:6px 10px;color:var(--muted);font-weight:600;font-size:11px;border-bottom:1px solid #21262d}
.ps-table td{padding:6px 10px;border-bottom:1px solid #161b22}
.ps-table tr:last-child td{border-bottom:none}
.ps-pos{color:var(--green)}.ps-neg{color:var(--red)}

/* ── Journal ────────────────────────────────────────────────────────────────── */
.journal-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.score-bars{display:flex;flex-direction:column;gap:5px}
.score-row{display:flex;align-items:center;gap:8px;font-size:12px}
.score-lbl{width:16px;color:var(--muted);text-align:right;flex-shrink:0}
.score-bg{flex:1;height:12px;background:#21262d;border-radius:3px;overflow:hidden}
.score-fill{height:100%;border-radius:3px;background:var(--blue);transition:width .4s}
.score-cnt{width:22px;color:var(--muted);font-size:11px;flex-shrink:0}
.regime-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid #21262d;font-size:12px}
.regime-row:last-child{border-bottom:none}
.regime-name{color:var(--text);font-weight:600;min-width:120px}
.regime-wr{font-weight:700}
.journal-entry{padding:12px 0;border-bottom:1px solid #21262d}
.journal-entry:last-child{border-bottom:none}
.je-header{display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.je-pair{font-weight:700;font-size:13px}
.je-dir{font-size:11px;font-weight:700;padding:1px 7px;border-radius:3px}
.je-dir-buy{background:#1a3a1f;color:var(--green)}
.je-dir-sell{background:#3a1a1a;color:var(--red)}
.je-pl{font-size:12px;font-weight:700}
.je-score{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;background:#1c2a3a;color:var(--blue)}
.je-regime{font-size:11px;color:var(--muted)}
.je-time{font-size:11px;color:var(--muted);margin-left:auto}
.je-field{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.4}
.je-field strong{color:var(--text)}
.themes-list{display:flex;flex-direction:column;gap:6px}
.theme-item{background:#1e2227;border-left:3px solid var(--yellow);padding:8px 12px;border-radius:0 5px 5px 0;font-size:12px;color:var(--text);line-height:1.4}

/* ── Settings Form ──────────────────────────────────────────────────────────── */
.settings-sections{display:flex;flex-direction:column;gap:20px}
.settings-section-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.settings-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px 24px}
.settings-row{display:flex;flex-direction:column;gap:5px}
.settings-label{font-size:12px;color:var(--muted);font-weight:500}
.settings-input{background:#0d1117;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px;padding:8px 12px;outline:none;width:100%;transition:.15s}
.settings-input:focus{border-color:var(--blue)}
.pairs-grid{display:flex;flex-wrap:wrap;gap:8px}
.pair-checkbox{display:flex;align-items:center;gap:6px;padding:6px 11px;background:#1e2227;border:1px solid var(--border);border-radius:6px;cursor:pointer;user-select:none;font-size:12px;font-weight:600;color:var(--text);transition:.15s}
.pair-checkbox:hover{border-color:#484f58}
.pair-checkbox.pair-active{border-color:var(--green)}
.pair-checkbox input{accent-color:var(--green)}
.sessions-tbl{border-collapse:collapse;font-size:13px}
.sessions-tbl th{font-size:11px;color:var(--muted);font-weight:500;text-align:left;padding:4px 10px 6px}
.sessions-tbl td{padding:4px 10px}
.session-input{background:#0d1117;border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:13px;padding:5px 8px;width:72px;outline:none;text-align:center}
.session-input:focus{border-color:var(--blue)}
.settings-note{font-size:11px;color:var(--muted);font-style:italic}

/* ── Alert Toggles ──────────────────────────────────────────────────────────── */
.alert-toggles{display:flex;flex-wrap:wrap;gap:10px;padding:4px 0}
.alert-toggle-item{display:flex;align-items:center;gap:10px;padding:9px 14px;background:#1e2227;border:1px solid var(--border);border-radius:8px;cursor:pointer;user-select:none;transition:.15s;min-width:160px}
.alert-toggle-item:hover{border-color:#484f58}
.alert-toggle-item.a-enabled{border-color:var(--green)}
.alert-toggle-item.a-disabled{border-color:#444;opacity:.75}
.toggle-label{font-size:13px;font-weight:600;color:var(--text);flex:1}
.toggle-state{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:700;white-space:nowrap}
.toggle-state.on{background:#1a3a1f;color:var(--green)}
.toggle-state.off{background:#2a1a1a;color:var(--red)}
.tgl-sw{position:relative;width:36px;height:20px;flex-shrink:0}
.tgl-sw input{opacity:0;width:0;height:0;position:absolute}
.tgl-slider{position:absolute;inset:0;background:#30363d;border-radius:10px;transition:.2s;pointer-events:none}
.tgl-slider:before{content:'';position:absolute;width:14px;height:14px;left:3px;top:3px;background:var(--muted);border-radius:50%;transition:.2s}
input:checked~.tgl-slider{background:var(--green)}
input:checked~.tgl-slider:before{transform:translateX(16px);background:#fff}

/* ── Scrollbar ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

/* ── Mobile ─────────────────────────────────────────────────────────────────── */
@media(max-width:1100px){
  .stats{grid-template-columns:repeat(3,1fr)}
}
@media(max-width:900px){
  .stats{grid-template-columns:repeat(3,1fr)}
  .drawdown-row{grid-template-columns:1fr}
  .charts-row{grid-template-columns:1fr 1fr;grid-template-rows:auto auto}
  .charts-row .chart-card:first-child{grid-column:1/-1}
  .row2{grid-template-columns:1fr}
  .journal-grid{grid-template-columns:1fr}
  header h1{font-size:15px;line-height:1}
  .updated{display:none}
  .col-hide-mobile{display:none}
}
@media(max-width:560px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .charts-row{grid-template-columns:1fr}
  .journal-grid{grid-template-columns:1fr}
  .hdr-right .updated{display:none}
  .card .val{font-size:20px}
  .container{padding:12px}
  .hdr-btn{padding:5px 10px;font-size:11px}
  .col-hide-mobile{display:none}
}
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="hdr-left">
    <h1>⚡ FX Bot</h1>
    <span class="status-badge badge-connecting" id="status-badge">CONNECTING</span>
  </div>
  <div class="hdr-right">
    <span class="updated" id="updated-time">Connecting...</span>
    <button class="hdr-btn" id="refresh-btn" onclick="refreshNow()" title="Refresh now">↻</button>
    <button class="hdr-btn primary" id="pause-btn" onclick="togglePause()">⏸ Pause Bot</button>
    <a class="hdr-btn" href="/logout">Sign Out</a>
  </div>
</header>

<div class="container">

  <!-- TABS -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('main')">Dashboard</button>
    <button class="tab-btn" onclick="switchTab('journal')">Journal Analytics</button>
    <button class="tab-btn" onclick="switchTab('settings')">Settings</button>
  </div>

  <div id="tab-main" class="tab-pane active">

  <!-- STAT CARDS -->
  <div class="stats">
    <div class="card">
      <div class="lbl">Account NAV</div>
      <div class="val c-blue" id="nav">—</div>
      <div class="sub" id="balance-sub">Balance: —</div>
    </div>
    <div class="card">
      <div class="lbl">Unrealized P&L</div>
      <div class="val" id="unrealized">—</div>
      <div class="sub" id="realized-sub">Realized: —</div>
    </div>
    <div class="card">
      <div class="lbl">Open Trades</div>
      <div class="val c-blue" id="open-count">—</div>
      <div class="sub" id="open-count-sub">— / — max</div>
    </div>
    <div class="card">
      <div class="lbl">Win Rate</div>
      <div class="val c-green" id="win-rate">—</div>
      <div class="sub" id="wr-sub">— trades sampled</div>
    </div>
    <div class="card">
      <div class="lbl">Margin Used</div>
      <div class="val c-yellow" id="margin">—</div>
      <div class="sub" id="margin-sub">— of NAV · <span id="currency-sub">—</span></div>
    </div>
  </div>

  <!-- DRAWDOWN BARS -->
  <div class="drawdown-row">
    <div class="dd-card">
      <div class="dd-labels">
        <span class="dd-name">Daily Drawdown</span>
        <span class="dd-val" id="dd-daily-val">0.00% / 3.0%</span>
      </div>
      <div class="dd-bar-bg"><div class="dd-bar-fill" id="dd-daily-bar" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="dd-card">
      <div class="dd-labels">
        <span class="dd-name">Weekly Drawdown</span>
        <span class="dd-val" id="dd-weekly-val">0.00% / 8.0%</span>
      </div>
      <div class="dd-bar-bg"><div class="dd-bar-fill" id="dd-weekly-bar" style="width:0%;background:var(--green)"></div></div>
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div class="charts-row">
    <div class="chart-card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <h2 style="margin-bottom:0">Equity Curve</h2>
        <div class="chart-range">
          <button class="range-btn" onclick="setEquityRange(60,event)">10m</button>
          <button class="range-btn" onclick="setEquityRange(180,event)">30m</button>
          <button class="range-btn active" onclick="setEquityRange(0,event)">All</button>
        </div>
      </div>
      <div class="chart-wrap equity-wrap"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h2>Win / Loss</h2>
      <div class="chart-wrap donut-wrap"><canvas id="winlossChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h2>P&L by Instrument (pips)</h2>
      <div class="chart-wrap bar-wrap"><canvas id="pairChart"></canvas></div>
    </div>
  </div>

  <!-- OPEN TRADES + NEWS -->
  <div class="row2">
    <div class="tcard">
      <div class="tcard-hdr">
        <h2>Open Trades</h2>
        <button class="btn-sm btn-danger" id="close-all-btn" onclick="closeAll()" disabled>Close All</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Pair</th><th>Dir</th><th class="col-hide-mobile">Units</th><th>Entry</th><th>SL</th><th>TP</th><th>P&L</th><th class="col-hide-mobile">Opened</th><th></th></tr></thead>
          <tbody id="open-tbody"><tr><td colspan="9" class="empty">No open trades</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="tcard">
      <div class="tcard-hdr"><h2>Upcoming News (8h)</h2></div>
      <div id="news-list"><div class="no-events">Loading...</div></div>
    </div>
  </div>

  <!-- CLOSED TRADES -->
  <div class="row-full tcard">
    <div class="tcard-hdr"><h2>Recent Closed Trades</h2></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Dir</th><th class="col-hide-mobile">Units</th><th>Entry</th><th>Close</th><th>P&L</th><th>Result</th><th class="col-hide-mobile">Duration</th><th class="col-hide-mobile">Closed</th></tr></thead>
        <tbody id="closed-tbody"><tr><td colspan="9" class="empty">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- BOT LOG -->
  <div class="row-full tcard">
    <div class="tcard-hdr">
      <h2>Bot Log</h2>
      <div class="log-filters">
        <button class="log-filter-btn active" data-filter="all"     onclick="setLogFilter('all')">All</button>
        <button class="log-filter-btn"        data-filter="trades"  onclick="setLogFilter('trades')">Trades</button>
        <button class="log-filter-btn"        data-filter="blocked" onclick="setLogFilter('blocked')">Blocked</button>
        <button class="log-filter-btn"        data-filter="errors"  id="err-filter-btn" onclick="setLogFilter('errors')">Errors<span class="err-badge" id="err-badge"></span></button>
      </div>
    </div>
    <div class="log-wrap" id="bot-log"><div class="no-events">Waiting for bot events...</div></div>
  </div>

  </div><!-- end tab-main -->

  <!-- JOURNAL TAB -->
  <div id="tab-journal" class="tab-pane">
    <div class="journal-grid">
      <div class="tcard">
        <div class="tcard-hdr"><h2>Score Distribution <span id="j-total" style="color:var(--muted);font-size:10px"></span></h2></div>
        <div class="score-bars" id="j-score-bars"><div class="empty">No journal data yet</div></div>
      </div>
      <div class="tcard">
        <div class="tcard-hdr"><h2>Win Rate by Regime</h2></div>
        <div id="j-regime-stats"><div class="empty">No journal data yet</div></div>
      </div>
    </div>
    <div class="row-full tcard" style="margin-bottom:16px">
      <div class="tcard-hdr"><h2>Improvement Themes (recent)</h2></div>
      <div class="themes-list" id="j-themes"><div class="empty">No themes yet</div></div>
    </div>
    <div class="row-full tcard">
      <div class="tcard-hdr">
        <h2>Recent Journal Entries</h2>
        <button class="hdr-btn" onclick="loadJournal()">↻ Refresh</button>
      </div>
      <div id="j-entries"><div class="empty">No entries yet</div></div>
    </div>
    <!-- Per-Pair Stats + Sharpe -->
    <div class="row-full tcard" style="margin-bottom:16px">
      <div class="tcard-hdr">
        <h2>Per-Pair Performance</h2>
        <button class="hdr-btn" onclick="loadPairStats(true)">↻ Refresh</button>
      </div>
      <div id="pair-stats-body"><div class="empty">Loading...</div></div>
    </div>
  </div><!-- end tab-journal -->

  <!-- SETTINGS TAB -->
  <div id="tab-settings" class="tab-pane">

    <!-- Trading Parameters -->
    <div class="row-full tcard" style="margin-bottom:16px">
      <div class="tcard-hdr">
        <div>
          <h2>Trading Parameters</h2>
          <div class="settings-note" style="margin-top:3px">Changes apply on the next bot cycle (~60s)</div>
        </div>
        <div style="display:flex;gap:8px">
          <button class="hdr-btn" onclick="resetSettings()">Reset Defaults</button>
          <button class="hdr-btn primary" onclick="saveSettings()">Save Changes</button>
        </div>
      </div>
      <div class="settings-sections" id="settings-form">
        <div class="empty">Loading...</div>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:18px;padding-top:14px;border-top:1px solid var(--border)">
        <button class="hdr-btn" onclick="resetSettings()">Reset Defaults</button>
        <button class="hdr-btn primary" onclick="saveSettings()">Save Changes</button>
      </div>
    </div>

    <!-- Telegram Alerts -->
    <div class="row-full tcard">
      <div class="tcard-hdr"><h2>Telegram Alerts</h2></div>
      <div class="alert-toggles" id="alert-toggles"><div class="empty">Loading...</div></div>
    </div>

  </div><!-- end tab-settings -->

</div>

<!-- CONFIRM MODAL -->
<div id="modal-overlay">
  <div id="modal-box">
    <div id="modal-icon">⚠️</div>
    <div id="modal-title">Confirm</div>
    <div id="modal-msg"></div>
    <div class="modal-actions">
      <button class="modal-btn" onclick="modalCancel()">Cancel</button>
      <button class="modal-btn confirm-danger" id="modal-confirm-btn" onclick="modalConfirm()">Confirm</button>
    </div>
  </div>
</div>

<!-- TOAST -->
<div id="toast"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let botPaused      = false;
let _openTradeCount = 0;
let _logFilter     = 'all';
let _lastLog       = [];
let _staleTimer    = null;

// ── Dark canvas background plugin (prevents GPU compositing white-flash) ──────
const darkCanvasPlugin = {
  id: 'darkCanvas',
  beforeDraw(chart) {
    const {ctx, width, height} = chart;
    ctx.save();
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  }
};
Chart.register(darkCanvasPlugin);

// ── Donut center-label plugin ─────────────────────────────────────────────────
const centerTextPlugin = {
  id: 'centerText',
  afterDraw(chart) {
    if (chart.config.type !== 'doughnut') return;
    const {ctx, chartArea: {left, right, top, bottom}} = chart;
    const cx = (left + right) / 2, cy = (top + bottom) / 2;
    const data  = chart.data.datasets[0].data;
    const total = data.reduce((a, b) => a + b, 0);
    const wins  = data[0] || 0;
    const pct   = total > 0 ? Math.round(wins / total * 100) : null;
    ctx.save();
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e6edf3';
    ctx.font = 'bold 20px Segoe UI, system-ui, sans-serif';
    ctx.fillText(pct !== null ? pct + '%' : '—', cx, cy - 8);
    ctx.font = '11px Segoe UI, system-ui, sans-serif';
    ctx.fillStyle = '#8b949e';
    ctx.fillText(total + ' trades', cx, cy + 12);
    ctx.restore();
  }
};
Chart.register(centerTextPlugin);

// ── Chart.js setup ────────────────────────────────────────────────────────────
const _tooltipDefaults = {
  backgroundColor: '#1e2430',
  titleColor: '#e6edf3',
  bodyColor: '#8b949e',
  borderColor: '#30363d',
  borderWidth: 1,
  padding: 8,
  cornerRadius: 6,
};

let _equityRangePoints = 0; // 0 = show all
let _equityHistoryFull = [];

const equityChart = new Chart(document.getElementById('equityChart'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,.08)',
      borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { ..._tooltipDefaults, callbacks: { label: ctx => ' ' + ctx.parsed.y.toFixed(2) } },
    },
    scales: {
      x: { ticks: { color: '#8b949e', maxTicksLimit: 6 }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    }
  }
});

const winlossChart = new Chart(document.getElementById('winlossChart'), {
  type: 'doughnut',
  data: {
    labels: ['Wins', 'Losses'],
    datasets: [{
      data: [0, 0],
      backgroundColor: ['#3fb950', '#f85149'],
      borderColor: '#161b22', borderWidth: 3,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    cutout: '65%',
    plugins: {
      legend: { position: 'bottom', labels: { color: '#8b949e', font: { size: 11 }, padding: 10 } },
      tooltip: _tooltipDefaults,
    }
  }
});

const pairChart = new Chart(document.getElementById('pairChart'), {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{
      data: [], backgroundColor: [], borderRadius: 4,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
      tooltip: { ..._tooltipDefaults, callbacks: { label: ctx => ' ' + ctx.parsed.x.toFixed(1) + ' pips' } },
    },
    scales: {
      x: { ticks: { color: '#8b949e', font: { size: 11 } }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e', font: { size: 11 } }, grid: { display: false } },
    }
  }
});

function setEquityRange(n, e) {
  _equityRangePoints = n;
  document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
  e.currentTarget.classList.add('active');
  _applyEquityRange();
}

function _applyEquityRange() {
  const src = _equityHistoryFull;
  const slice = _equityRangePoints > 0 ? src.slice(-_equityRangePoints) : src;
  equityChart.data.labels = slice.map(h => h.time);
  equityChart.data.datasets[0].data = slice.map(h => h.nav);
  equityChart.update('none');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmtPL = n => n >= 0
  ? `<span class="c-green">+${n.toFixed(2)}</span>`
  : `<span class="c-red">${n.toFixed(2)}</span>`;
const fmtCcy = (ccy, n) => `${ccy} ${n.toLocaleString('en', {minimumFractionDigits: 2})}`;

function toLocal(utcStr) {
  // Accepts "2026-04-25 18:19:09 UTC" or ISO strings, returns local HH:MM or date+time
  if (!utcStr || utcStr === '—') return utcStr;
  const s = utcStr.replace(' UTC', '').replace(' ', 'T') + (utcStr.includes('T') ? '' : 'Z');
  const d = new Date(s.endsWith('Z') ? s : s + 'Z');
  if (isNaN(d)) return utcStr;
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  }
  return d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
}

function toast(msg, color='#3fb950') {
  const t = $('toast');
  t.textContent = msg;
  t.style.borderColor = color;
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 3500);
}

async function post(url) {
  try {
    const r = await fetch(url, { method: 'POST' });
    if (!r.ok) return { error: `HTTP ${r.status}` };
    return r.json();
  } catch (e) {
    return { error: 'Network error' };
  }
}

// ── Confirm Modal ─────────────────────────────────────────────────────────────
let _modalCallback = null;

function showConfirm({ title, message, confirmText = 'Confirm', onConfirm }) {
  $('modal-title').textContent   = title;
  $('modal-msg').textContent     = message;
  $('modal-confirm-btn').textContent = confirmText;
  _modalCallback = onConfirm;
  $('modal-overlay').classList.add('open');
}

function modalConfirm() {
  $('modal-overlay').classList.remove('open');
  if (_modalCallback) { _modalCallback(); _modalCallback = null; }
}

function modalCancel() {
  $('modal-overlay').classList.remove('open');
  _modalCallback = null;
}

// Close on backdrop click
$('modal-overlay').addEventListener('click', e => {
  if (e.target === $('modal-overlay')) modalCancel();
});

// Close on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') modalCancel();
});

// ── Controls ──────────────────────────────────────────────────────────────────
async function _doTogglePause() {
  const btn = $('pause-btn');
  btn.disabled = true;
  const url = botPaused ? '/api/control/resume' : '/api/control/pause';
  const d = await post(url);
  btn.disabled = false;
  if (d.error) { toast('Error: ' + d.error, '#f85149'); return; }
  toast(botPaused ? 'Bot resumed' : 'Bot paused', botPaused ? '#3fb950' : '#f85149');
}

function togglePause() {
  if (botPaused && _openTradeCount > 0) {
    showConfirm({
      title:       'Resume Bot',
      message:     `There are ${_openTradeCount} open trade${_openTradeCount !== 1 ? 's' : ''}. Resume the bot and allow new signals?`,
      confirmText: 'Resume',
      onConfirm:   _doTogglePause,
    });
  } else {
    _doTogglePause();
  }
}

function closeTrade(id, btn) {
  showConfirm({
    title:       'Close Trade',
    message:     `Close trade #${id}? This will execute a market order immediately.`,
    confirmText: 'Close Trade',
    onConfirm:   async () => {
      btn.disabled = true;
      btn.textContent = '...';
      const d = await post(`/api/control/close/${id}`);
      if (d.status === 'closed') toast('Trade closed', '#3fb950');
      else { toast('Error: ' + (d.error || d.detail || 'unknown'), '#f85149'); btn.disabled = false; btn.textContent = 'Close'; }
    }
  });
}

function closeAll() {
  if (_openTradeCount === 0) return;
  showConfirm({
    title:       'Close All Positions',
    message:     'This will immediately close ALL open trades at market price. This action cannot be undone.',
    confirmText: 'Close All',
    onConfirm:   async () => {
      const d = await post('/api/control/close-all');
      const results = d.results || [];
      const errors  = results.filter(r => r.status !== 'closed');
      if (errors.length === 0) {
        toast(`All ${results.length} position${results.length !== 1 ? 's' : ''} closed`, '#3fb950');
      } else {
        toast(`${results.length - errors.length} closed, ${errors.length} error(s)`, '#f85149');
      }
    }
  });
}

// ── Alert Toggles ─────────────────────────────────────────────────────────────
const ALERT_LABELS = {
  price_alert:  'Price Alerts',
  trade_open:   'Trade Opened',
  trade_close:  'Trade Closed',
  kill_switch:  'Kill Switch',
  error_alert:  'Error Alerts',
};

function renderAlertToggles(settings) {
  if (!settings) return;
  const el = $('alert-toggles');
  el.innerHTML = Object.entries(ALERT_LABELS).map(([key, label]) => {
    const on = settings[key] !== false;
    return `<label class="alert-toggle-item ${on ? 'a-enabled' : 'a-disabled'}" for="atgl-${key}">
      <div class="tgl-sw">
        <input type="checkbox" id="atgl-${key}" ${on ? 'checked' : ''} onchange="toggleAlert('${key}', this.checked)">
        <span class="tgl-slider"></span>
      </div>
      <span class="toggle-label">${label}</span>
      <span class="toggle-state ${on ? 'on' : 'off'}" id="astate-${key}">${on ? 'ON' : 'OFF'}</span>
    </label>`;
  }).join('');
}

async function toggleAlert(alertType, checked) {
  const d = await post(`/api/alerts/toggle/${alertType}`);
  if (d.error) { toast('Error: ' + d.error, '#f85149'); return; }
  const on = d.enabled;
  const item = document.querySelector(`label[for="atgl-${alertType}"]`);
  if (item) { item.classList.toggle('a-enabled', on); item.classList.toggle('a-disabled', !on); }
  const stateEl = $('astate-' + alertType);
  if (stateEl) { stateEl.textContent = on ? 'ON' : 'OFF'; stateEl.className = 'toggle-state ' + (on ? 'on' : 'off'); }
  const chk = $('atgl-' + alertType);
  if (chk) chk.checked = on;
  toast((ALERT_LABELS[alertType] || alertType) + ' ' + (on ? 'enabled' : 'disabled'), on ? '#3fb950' : '#f85149');
}

// ── Per-Pair Stats ─────────────────────────────────────────────────────────────
let _pairStatsLoaded = false;
async function loadPairStats(force = false) {
  if (_pairStatsLoaded && !force) return;
  const el = $('pair-stats-body');
  if (!el) return;
  const d = await fetch('/api/pair-stats').then(r => r.json());
  _pairStatsLoaded = true;
  if (d.error) { el.innerHTML = `<div class="empty">${d.error}</div>`; return; }
  const wr = (d.overall_wr ?? 0).toFixed(1);
  const sharpe = d.sharpe !== null && d.sharpe !== undefined ? d.sharpe.toFixed(3) : '—';
  const pnlClass = d.total_pnl >= 0 ? 'ps-pos' : 'ps-neg';
  const pnlSign  = d.total_pnl >= 0 ? '+' : '';
  el.innerHTML = `
    <div class="ps-summary">
      <div class="ps-summary-item"><span class="ps-summary-label">Total Trades</span><span class="ps-summary-value">${d.total_trades}</span></div>
      <div class="ps-summary-item"><span class="ps-summary-label">Win Rate</span><span class="ps-summary-value">${wr}%</span></div>
      <div class="ps-summary-item"><span class="ps-summary-label">Total P&amp;L</span><span class="ps-summary-value ${pnlClass}">${pnlSign}${d.total_pnl} pips</span></div>
      <div class="ps-summary-item"><span class="ps-summary-label">Avg Win</span><span class="ps-summary-value ps-pos">+${d.avg_win_pips}</span></div>
      <div class="ps-summary-item"><span class="ps-summary-label">Avg Loss</span><span class="ps-summary-value ps-neg">${d.avg_loss_pips}</span></div>
      <div class="ps-summary-item"><span class="ps-summary-label">Sharpe (per trade)</span><span class="ps-summary-value">${sharpe}</span></div>
    </div>
    ${d.pair_stats.length === 0 ? '<div class="empty">No closed trades yet</div>' : `
    <table class="ps-table">
      <thead><tr><th>Pair</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Total P&amp;L (pips)</th></tr></thead>
      <tbody>${d.pair_stats.map(p => {
        const cls = p.pnl_pips >= 0 ? 'ps-pos' : 'ps-neg';
        const sign = p.pnl_pips >= 0 ? '+' : '';
        return `<tr>
          <td style="font-weight:600">${p.pair}</td>
          <td>${p.trades}</td>
          <td class="ps-pos">${p.wins}</td>
          <td class="ps-neg">${p.losses}</td>
          <td>${p.win_rate}%</td>
          <td class="${cls}">${sign}${p.pnl_pips}</td>
        </tr>`;
      }).join('')}</tbody>
    </table>`}`;
}

// ── Settings ──────────────────────────────────────────────────────────────────
const ALL_PAIRS = ['EUR_USD','GBP_USD','USD_JPY','AUD_USD','USD_CHF','USD_CAD','NZD_USD','XAU_USD','WTICO_USD'];
const SESSION_NAMES = ['London','New York','Asian','Custom'];
let _settingsLoaded = false;

async function loadSettings(force=false) {
  if (_settingsLoaded && !force) return;
  const d = await fetch('/api/settings').then(r => r.json());
  if (d.unauthorized) { window.location.href = '/login'; return; }
  _settingsLoaded = true;
  renderSettingsForm(d);
}

function renderSettingsForm(d) {
  const el = $('settings-form');
  if (!el) return;

  const activePairs = new Set(d.PAIRS || []);
  const sessions    = d.SESSIONS || [[7,16],[13,22]];

  const pairsHTML = ALL_PAIRS.map(p => `
    <label class="pair-checkbox ${activePairs.has(p) ? 'pair-active' : ''}" id="pair-lbl-${p}" for="pair-${p}">
      <input type="checkbox" id="pair-${p}" ${activePairs.has(p) ? 'checked' : ''}
             onchange="$('pair-lbl-${p}').classList.toggle('pair-active',this.checked)">
      ${p.replace('_','/')}
    </label>`).join('');

  const sessHTML = sessions.map((s, i) => `
    <tr>
      <td style="color:var(--muted);font-size:12px;padding-right:8px">${SESSION_NAMES[i] || 'Session '+(i+1)}</td>
      <td><input type="number" class="session-input" id="sess-start-${i}" value="${s[0]}" min="0" max="23"></td>
      <td style="color:var(--muted);padding:0 6px;font-size:12px">to</td>
      <td><input type="number" class="session-input" id="sess-end-${i}" value="${s[1]}" min="1" max="24"></td>
      <td style="color:var(--muted);font-size:11px;padding-left:8px">UTC</td>
    </tr>`).join('');

  el.innerHTML = `
    <div class="settings-section">
      <div class="settings-section-title">Position Management</div>
      <div class="settings-grid">
        <div class="settings-row"><label class="settings-label">Max Concurrent Trades</label>
          <input type="number" id="s-MAX_CONCURRENT_TRADES" class="settings-input" value="${d.MAX_CONCURRENT_TRADES}" min="1" max="10" step="1"></div>
        <div class="settings-row"><label class="settings-label">Base Trade Size (units)</label>
          <input type="number" id="s-UNITS" class="settings-input" value="${d.UNITS}" min="100" max="100000" step="100"></div>
        <div class="settings-row"><label class="settings-label">Risk % per Trade</label>
          <input type="number" id="s-RISK_PCT_PER_TRADE" class="settings-input" value="${d.RISK_PCT_PER_TRADE}" min="0.1" max="5.0" step="0.1"></div>
      </div>
    </div>
    <div class="settings-section">
      <div class="settings-section-title">Risk Limits</div>
      <div class="settings-grid">
        <div class="settings-row"><label class="settings-label">Max Daily Loss %</label>
          <input type="number" id="s-MAX_DAILY_LOSS_PCT" class="settings-input" value="${d.MAX_DAILY_LOSS_PCT}" min="0.5" max="20" step="0.5"></div>
        <div class="settings-row"><label class="settings-label">Max Weekly Loss %</label>
          <input type="number" id="s-MAX_WEEKLY_LOSS_PCT" class="settings-input" value="${d.MAX_WEEKLY_LOSS_PCT}" min="1" max="50" step="0.5"></div>
      </div>
    </div>
    <div class="settings-section">
      <div class="settings-section-title">Filters &amp; Limits</div>
      <div class="settings-grid">
        <div class="settings-row"><label class="settings-label">News Blackout (minutes)</label>
          <input type="number" id="s-NEWS_BLACKOUT_MINUTES" class="settings-input" value="${d.NEWS_BLACKOUT_MINUTES}" min="0" max="180" step="5"></div>
        <div class="settings-row"><label class="settings-label">Post-Event Cool-down (hours)</label>
          <input type="number" id="s-NEWS_POST_EVENT_HOURS" class="settings-input" value="${d.NEWS_POST_EVENT_HOURS}" min="0" max="24" step="1"></div>
        <div class="settings-row"><label class="settings-label">Price Alert Threshold (pips)</label>
          <input type="number" id="s-ALERT_PRICE_MOVE_PIPS" class="settings-input" value="${d.ALERT_PRICE_MOVE_PIPS}" min="1" max="500" step="1"></div>
        <div class="settings-row"><label class="settings-label">Max Spread (pips, 0=off)</label>
          <input type="number" id="s-SPREAD_MAX_PIPS" class="settings-input" value="${d.SPREAD_MAX_PIPS}" min="0.5" max="200" step="0.5"></div>
        <div class="settings-row"><label class="settings-label">Trade Timeout (hours, 0=off)</label>
          <input type="number" id="s-MAX_TRADE_HOURS" class="settings-input" value="${d.MAX_TRADE_HOURS}" min="0" max="168" step="1"></div>
        <div class="settings-row"><label class="settings-label">Max Trades/Day (0=unlimited)</label>
          <input type="number" id="s-MAX_DAILY_TRADES" class="settings-input" value="${d.MAX_DAILY_TRADES}" min="0" max="50" step="1"></div>
        <div class="settings-row"><label class="settings-label">Consecutive Loss Limit (0=off)</label>
          <input type="number" id="s-MAX_CONSECUTIVE_LOSSES" class="settings-input" value="${d.MAX_CONSECUTIVE_LOSSES}" min="0" max="20" step="1"></div>
      </div>
    </div>
    <div class="settings-section">
      <div class="settings-section-title">Trading Sessions (UTC hours)</div>
      <table class="sessions-tbl">
        <thead><tr><th>Session</th><th>Start</th><th></th><th>End</th><th></th></tr></thead>
        <tbody id="s-sessions-body">${sessHTML}</tbody>
      </table>
    </div>
    <div class="settings-section">
      <div class="settings-section-title">Active Trading Pairs</div>
      <div class="pairs-grid">${pairsHTML}</div>
    </div>`;
}

async function saveSettings() {
  const sessions = [];
  let i = 0;
  while ($('sess-start-' + i)) {
    sessions.push([parseInt($('sess-start-'+i).value), parseInt($('sess-end-'+i).value)]);
    i++;
  }
  const pairs = ALL_PAIRS.filter(p => { const el = $('pair-'+p); return el && el.checked; });
  if (!pairs.length) { toast('Select at least one trading pair', '#f85149'); return; }

  const body = {
    MAX_CONCURRENT_TRADES: parseInt($('s-MAX_CONCURRENT_TRADES').value),
    UNITS:                 parseInt($('s-UNITS').value),
    RISK_PCT_PER_TRADE:    parseFloat($('s-RISK_PCT_PER_TRADE').value),
    MAX_DAILY_LOSS_PCT:    parseFloat($('s-MAX_DAILY_LOSS_PCT').value),
    MAX_WEEKLY_LOSS_PCT:   parseFloat($('s-MAX_WEEKLY_LOSS_PCT').value),
    ALERT_PRICE_MOVE_PIPS:  parseInt($('s-ALERT_PRICE_MOVE_PIPS').value),
    NEWS_BLACKOUT_MINUTES:  parseInt($('s-NEWS_BLACKOUT_MINUTES').value),
    NEWS_POST_EVENT_HOURS:  parseInt($('s-NEWS_POST_EVENT_HOURS').value),
    SPREAD_MAX_PIPS:        parseFloat($('s-SPREAD_MAX_PIPS').value),
    MAX_TRADE_HOURS:        parseInt($('s-MAX_TRADE_HOURS').value),
    MAX_DAILY_TRADES:       parseInt($('s-MAX_DAILY_TRADES').value),
    MAX_CONSECUTIVE_LOSSES: parseInt($('s-MAX_CONSECUTIVE_LOSSES').value),
    PAIRS: pairs, SESSIONS: sessions,
  };

  const d = await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(r => r.json());

  if (d.error) { toast('Error: ' + d.error, '#f85149'); return; }
  toast('Settings saved — takes effect next bot cycle', '#3fb950');
}

async function resetSettings() {
  showConfirm({
    title: 'Reset to Defaults',
    message: 'Reset all trading parameters to the values defined in config.py / .env?',
    confirmText: 'Reset',
    onConfirm: async () => {
      const d = await fetch('/api/settings/reset', { method: 'POST' }).then(r => r.json());
      if (d.error) { toast('Error: ' + d.error, '#f85149'); return; }
      _settingsLoaded = false;
      await loadSettings(true);
      toast('Reset to defaults', '#3fb950');
    }
  });
}

// ── Update functions ──────────────────────────────────────────────────────────
function updateStats(d) {
  const a = d.account;
  $('nav').textContent         = fmtCcy(a.currency, a.nav);
  $('balance-sub').textContent = `Balance: ${fmtCcy(a.currency, a.balance)}`;
  $('unrealized').innerHTML    = fmtPL(a.unrealized_pl);
  $('realized-sub').innerHTML  = `Realized: ${fmtPL(a.realized_pl)}`;

  // open trades card
  const maxTrades = (d.bot_settings && d.bot_settings.MAX_CONCURRENT_TRADES) || '—';
  $('open-count').textContent     = a.open_trades;
  $('open-count-sub').textContent = `${a.open_trades} / ${maxTrades} max`;

  // win rate with dynamic colour
  const wrEl  = $('win-rate');
  const wrPct = parseInt(d.win_rate);
  wrEl.textContent = d.win_rate;
  wrEl.className   = 'val ' + (isNaN(wrPct) ? 'c-blue' : wrPct >= 35 ? 'c-green' : wrPct >= 25 ? 'c-yellow' : 'c-red');
  if (d.trade_count !== undefined) $('wr-sub').textContent = `${d.trade_count} trades sampled`;

  // margin with percentage
  $('margin').textContent      = fmtCcy(a.currency, a.margin_used);
  $('margin-sub').innerHTML    = `${a.margin_pct ?? '—'}% of NAV · <span id="currency-sub">${a.currency}</span>`;

  // updated time + stale detection (fix 10)
  $('updated-time').textContent  = 'Updated ' + toLocal(d.updated);
  $('updated-time').style.color  = '';
  if (_staleTimer) clearTimeout(_staleTimer);
  _staleTimer = setTimeout(() => {
    $('updated-time').textContent = '⚠ Data may be stale — reconnecting...';
    $('updated-time').style.color = 'var(--red)';
  }, 30000);

  // close-all button state
  _openTradeCount = a.open_trades;
  const cab = $('close-all-btn');
  if (cab) cab.disabled = (_openTradeCount === 0);

  // bot status
  botPaused = d.bot_paused;
  const badge = $('status-badge');
  const btn   = $('pause-btn');
  if (botPaused) {
    badge.textContent = 'PAUSED';
    badge.className   = 'status-badge badge-paused';
    btn.textContent   = '▶ Resume Bot';
    btn.className     = 'hdr-btn success';
  } else {
    badge.textContent = 'ACTIVE';
    badge.className   = 'status-badge badge-active';
    btn.textContent   = '⏸ Pause Bot';
    btn.className     = 'hdr-btn primary';
  }

  // alert toggles
  if (d.alert_settings) renderAlertToggles(d.alert_settings);

  // drawdown bars (fix 2)
  if (d.drawdown) {
    const dd = d.drawdown;
    const dPct  = Math.min(dd.daily_pct  / dd.daily_limit  * 100, 100);
    const wPct  = Math.min(dd.weekly_pct / dd.weekly_limit * 100, 100);
    const dColor  = dPct >= 80 ? 'var(--red)' : dPct >= 50 ? 'var(--yellow)' : 'var(--green)';
    const wColor  = wPct >= 80 ? 'var(--red)' : wPct >= 50 ? 'var(--yellow)' : 'var(--green)';
    $('dd-daily-val').textContent  = `${dd.daily_pct.toFixed(2)}% / ${dd.daily_limit}%`;
    $('dd-weekly-val').textContent = `${dd.weekly_pct.toFixed(2)}% / ${dd.weekly_limit}%`;
    Object.assign($('dd-daily-bar').style,  {width: dPct + '%', background: dColor});
    Object.assign($('dd-weekly-bar').style, {width: wPct + '%', background: wColor});
  }
}

function updateCharts(history, chartData) {
  // equity curve (range-aware)
  _equityHistoryFull = history;
  _applyEquityRange();

  // win/loss donut
  winlossChart.data.datasets[0].data = chartData.win_loss;
  winlossChart.update('none');

  // pip P&L per pair
  const colors = chartData.pair_pips.map(v => v >= 0 ? '#3fb950' : '#f85149');
  pairChart.data.labels = chartData.pair_labels;
  pairChart.data.datasets[0].data = chartData.pair_pips;
  pairChart.data.datasets[0].backgroundColor = colors;
  pairChart.update('none');
}

function fmtUnits(t) {
  return t.units.toLocaleString() + (t.units_suffix || '');
}

function updateOpenTrades(trades) {
  const tbody = $('open-tbody');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty">No open trades</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => {
    const dp = t.price_dp || 5;
    return `<tr>
      <td><strong>${t.display_name || t.instrument.replace('_','/')}</strong></td>
      <td class="t-${t.direction.toLowerCase()}">${t.direction}</td>
      <td class="col-hide-mobile">${fmtUnits(t)}</td>
      <td>${t.entry.toFixed(dp)}</td>
      <td style="color:var(--red)">${t.sl}</td>
      <td style="color:var(--green)">${t.tp}</td>
      <td>${fmtPL(t.unrealized)}</td>
      <td class="col-hide-mobile" style="color:var(--muted);font-size:12px">${toLocal(t.opened)}</td>
      <td><button class="btn-close" onclick="closeTrade('${t.id}',this)">Close</button></td>
    </tr>`;
  }).join('');
}

function updateClosedTrades(trades) {
  const tbody = $('closed-tbody');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty">No closed trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => {
    const dp = t.price_dp || 5;
    return `<tr>
      <td><strong>${t.display_name || t.instrument.replace('_','/')}</strong></td>
      <td class="t-${t.direction.toLowerCase()}">${t.direction}</td>
      <td class="col-hide-mobile">${fmtUnits(t)}</td>
      <td>${t.entry.toFixed(dp)}</td>
      <td>${t.close.toFixed(dp)}</td>
      <td>${fmtPL(t.pl)}</td>
      <td class="t-${t.result.toLowerCase()}">${t.result}</td>
      <td class="col-hide-mobile" style="color:var(--muted);font-size:12px">${t.duration || '—'}</td>
      <td class="col-hide-mobile" style="color:var(--muted);font-size:12px">${toLocal(t.closed)}</td>
    </tr>`;
  }).join('');
}

function updateNews(news) {
  const el = $('news-list');
  if (!news.length) {
    el.innerHTML = '<div class="no-events">No high-impact events in next 8 hours</div>';
    return;
  }
  el.innerHTML = news.map(n => `
    <div class="news-item">
      <div class="news-row1">
        <span class="news-cur">${n.currency}</span>
        <span class="news-impact">HIGH</span>
        <span class="news-time">${n.time}</span>
      </div>
      <div class="news-title">${n.title}</div>
    </div>`).join('');
}

// ── Log filter (fix 7) ────────────────────────────────────────────────────────
function setLogFilter(filter) {
  _logFilter = filter;
  document.querySelectorAll('.log-filter-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.filter === filter)
  );
  renderLog(_lastLog);
}

function renderLog(log) {
  _lastLog = log;
  const el = $('bot-log');
  const TRADE_TYPES  = new Set(['signal', 'trade_open', 'trade_close']);
  const ERROR_TYPES  = new Set(['kill_switch', 'error']);

  const BLOCKED_TYPES = new Set(['filter']);
  let filtered = log;
  if (_logFilter === 'trades') {
    filtered = log.filter(e => TRADE_TYPES.has(e.type));
  } else if (_logFilter === 'blocked') {
    filtered = log.filter(e => BLOCKED_TYPES.has(e.type));
  } else if (_logFilter === 'errors') {
    filtered = log.filter(e => ERROR_TYPES.has(e.type));
  }

  if (!filtered.length) {
    el.innerHTML = '<div class="no-events">No matching events.</div>';
    return;
  }
  const typeMap = {
    signal: 'SIGNAL', trade_open: 'TRADE OPEN', trade_close: 'TRADE CLOSE',
    filter: 'FILTER', kill_switch: 'KILL SWITCH', info: 'INFO', error: 'ERROR',
  };
  const atTop = el.scrollTop < 40;
  el.innerHTML = filtered.map(e => {
    let text = '';
    if (e.type === 'signal')           text = `${e.pair} &rarr; ${e.signal?.toUpperCase()} &nbsp;(ML: ${e.confidence})`;
    else if (e.type === 'trade_open')  text = `${e.pair} ${e.direction} ${e.units} units @ ${e.entry} | SL:${e.sl} TP:${e.tp}`;
    else if (e.type === 'trade_close') text = `${e.pair} closed &mdash; ${e.result} ${e.pnl_pips > 0 ? '+':''}${e.pnl_pips} pips`;
    else if (e.type === 'filter')      text = `${e.pair} &mdash; ${e.reason}`;
    else if (e.type === 'kill_switch') text = e.reason;
    else if (e.type === 'error')      text = `${e.context}: ${e.message}`;
    else text = e.message || '';
    return `<div class="log-entry">
      <span class="log-time">${toLocal(e.time)}</span>
      <span class="log-badge lb-${e.type}">${typeMap[e.type] || e.type}</span>
      <span class="log-text">${text}</span>
    </div>`;
  }).join('');
  if (atTop) el.scrollTop = 0;
}

function updateLog(log) {
  renderLog(log);
  // error badge
  const errCount = (log || []).filter(e => e.type === 'error').length;
  const badge = $('err-badge');
  if (badge) {
    badge.textContent = errCount;
    badge.style.display = errCount > 0 ? 'inline' : 'none';
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  $('tab-' + name).classList.add('active');
  const labelMap = {main: 'dashboard', journal: 'journal', settings: 'settings'};
  document.querySelectorAll('.tab-btn').forEach(b => {
    if (b.textContent.toLowerCase().includes(labelMap[name] || name))
      b.classList.add('active');
  });
  if (name === 'journal')  { loadJournal(true); loadPairStats(true); }
  if (name === 'settings') loadSettings();
  try { localStorage.setItem('fxbot_tab', name); } catch(e) {}
}

// ── Restore last tab on load ───────────────────────────────────────────────────
(function() {
  try {
    const saved = localStorage.getItem('fxbot_tab');
    if (saved && saved !== 'main') switchTab(saved);
  } catch(e) {}
})();

// ── Journal ───────────────────────────────────────────────────────────────────
let _journalLoaded = false;
async function loadJournal(force = false) {
  if (_journalLoaded && !force) return;
  const data = await fetch('/api/journal').then(r => r.json());
  _journalLoaded = true;
  if (data.unauthorized) { window.location.href = '/login'; return; }

  // total count
  $('j-total').textContent = data.total ? `(${data.total} total)` : '';

  // score distribution
  const scoreBars = $('j-score-bars');
  const dist = data.score_dist || {};
  const maxCnt = Math.max(1, ...Object.values(dist));
  if (Object.values(dist).every(v => v === 0)) {
    scoreBars.innerHTML = '<div class="empty">No entries yet</div>';
  } else {
    scoreBars.innerHTML = Object.entries(dist).map(([score, cnt]) => `
      <div class="score-row">
        <span class="score-lbl">${score}</span>
        <div class="score-bg"><div class="score-fill" style="width:${cnt/maxCnt*100}%"></div></div>
        <span class="score-cnt">${cnt}</span>
      </div>`).join('');
  }

  // regime win rates
  const regimeEl = $('j-regime-stats');
  const rs = data.regime_stats || {};
  if (!Object.keys(rs).length) {
    regimeEl.innerHTML = '<div class="empty">No entries yet</div>';
  } else {
    const regimeColor = r => r === 'trending_up' ? 'var(--green)' : r === 'trending_down' ? 'var(--red)' : r === 'volatile' ? 'var(--orange)' : 'var(--muted)';
    regimeEl.innerHTML = Object.entries(rs).map(([regime, s]) => {
      const wr = s.total > 0 ? Math.round(s.wins / s.total * 100) : 0;
      return `<div class="regime-row">
        <span class="regime-name">${regime.replace('_', ' ')}</span>
        <span style="color:var(--muted);font-size:12px">${s.wins}/${s.total} trades</span>
        <span class="regime-wr" style="color:${regimeColor(regime)}">${wr}%</span>
      </div>`;
    }).join('');
  }

  // improvement themes
  const themesEl = $('j-themes');
  const themes = data.themes || [];
  if (!themes.length) {
    themesEl.innerHTML = '<div class="empty">No themes yet</div>';
  } else {
    themesEl.innerHTML = themes.slice().reverse().map(t =>
      `<div class="theme-item">${t}</div>`
    ).join('');
  }

  // recent entries
  const entriesEl = $('j-entries');
  const entries = data.entries || [];
  if (!entries.length) {
    entriesEl.innerHTML = '<div class="empty">No journal entries yet</div>';
  } else {
    entriesEl.innerHTML = entries.map(e => {
      const plColor  = e.pl_pips >= 0 ? 'var(--green)' : 'var(--red)';
      const plSign   = e.pl_pips >= 0 ? '+' : '';
      const dirClass = e.direction === 'BUY' ? 'je-dir-buy' : 'je-dir-sell';
      return `<div class="journal-entry">
        <div class="je-header">
          <span class="je-pair">${e.display_name || (e.instrument||'').replace('_','/')}</span>
          <span class="je-dir ${dirClass}">${e.direction}</span>
          <span class="je-pl" style="color:${plColor}">${plSign}${e.pl_pips} pips</span>
          <span class="je-score">Score ${e.score}/10</span>
          ${e.regime ? `<span class="je-regime">${e.regime.replace('_',' ')}</span>` : ''}
          <span class="je-time">${toLocal(e.time)}</span>
        </div>
        <div class="je-field"><strong>Why: </strong>${e.why}</div>
        <div class="je-field"><strong>Done well: </strong>${e.done_well}</div>
        <div class="je-field"><strong>Improve: </strong>${e.improve}</div>
      </div>`;
    }).join('');
  }
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect() {
  const es = new EventSource('/stream');

  es.addEventListener('unauthorized', () => {
    window.location.href = '/login';
  });

  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.unauthorized) { window.location.href = '/login'; return; }
    if (d.error) {
      console.error(d.error);
      $('status-badge').textContent = 'ERROR';
      $('status-badge').className   = 'status-badge badge-paused';
      return;
    }
    updateStats(d);
    updateCharts(d.equity_history, d.chart_data);
    updateOpenTrades(d.open_trades);
    updateClosedTrades(d.closed_trades);
    updateNews(d.news);
    updateLog(d.bot_log);
  };

  es.onerror = () => {
    $('status-badge').textContent = 'DISCONNECTED';
    $('status-badge').className   = 'status-badge badge-paused';
    es.close();
    setTimeout(connect, 5000);
  };
}

connect();

function refreshNow() {
  const btn = $('refresh-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinning">⟳</span>'; }
  fetch('/api/data').then(r => r.json()).then(d => {
    if (d.unauthorized) { window.location.href = '/login'; return; }
    if (d.error) { toast('Refresh error: ' + d.error, '#f85149'); return; }
    updateStats(d);
    updateCharts(d.equity_history, d.chart_data);
    updateOpenTrades(d.open_trades);
    updateClosedTrades(d.closed_trades);
    updateNews(d.news);
    updateLog(d.bot_log);
    _journalLoaded = false;
  }).catch(err => toast('Refresh failed', '#f85149')).finally(() => {
    if (btn) { btn.disabled = false; btn.textContent = '↻'; }   // restore plain icon
  });
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not authenticated(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(DASHBOARD_HTML, headers={"Cache-Control": "no-store"})


# ─── ENTRY ────────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    from starlette.responses import Response
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#0f0f1a"/>'
        '<text x="16" y="24" font-size="22" text-anchor="middle" '
        'font-family="sans-serif">&#x26A1;</text>'
        "</svg>"
    )
    return Response(content=svg.encode(), media_type="image/svg+xml")


@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    db.init()
    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000, reload=False)
