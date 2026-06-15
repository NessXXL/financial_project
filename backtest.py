"""
BACKTESTING ENGINE — turns the demo into science
=================================================
Proves (or disproves) the signal on historical data. Three outputs:

  1. PORTFOLIO BACKTEST  — follow BUY signals monthly, compare vs SPY
  2. HIT RATE            — % of BUY signals that beat SPY over the horizon
  3. EVENT STUDY         — avg forward return after positive vs negative
                           sentiment months (5 / 21 / 63 trading days)

Honesty rules baked in:
  • Point-in-time fundamentals: a quarter is only used 45+ days after it
    ends (real reporting lag) — no look-ahead bias.
  • Historical headlines come from GDELT (free archive). If none exist
    for a month, sentiment defaults to NEUTRAL 50 and is logged — never
    invented.
  • Every number is saved to backtest_results.json so it can be audited.

Usage:
  python backtest.py --tickers NVDA META AAPL MSFT GOOGL AMZN TSLA FUBO \
                     --start 2023-01-01 --end 2025-06-01
  python backtest.py --tickers NVDA META --no-news     # ratios-only, fast
"""

import os, json, time, math, argparse, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance -q"); import yfinance as yf

CACHE_DIR = "data/news_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

REPORTING_LAG_DAYS = 45      # quarter usable only after earnings are out
SENT_W, RATIO_W    = 0.35, 0.65
BUY_T, HOLD_T      = 70, 45

# ─────────────────────────────────────────────────────────────────────────────
# 1. POINT-IN-TIME RATIO SCORE  (from quarterly statements)
# ─────────────────────────────────────────────────────────────────────────────
def _row(df, names):
    if df is None or df.empty: return None
    for n in names:
        if n in df.index: return df.loc[n]
    return None

def _val(series, col):
    if series is None: return None
    try:
        v = float(series[col])
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None

ANNUAL_LAG_DAYS = 90        # 10-K filings take longer than 10-Qs

def load_fundamentals(ticker):
    """Pull quarterly AND annual statements once per ticker.
    Free yfinance: quarterly ≈ last 4-5 quarters, annual ≈ last 4 years —
    annual is what lets the backtest reach back to ~2022 honestly."""
    tk = yf.Ticker(ticker)
    q = {"inc": tk.quarterly_financials,
         "bal": tk.quarterly_balance_sheet,
         "cf":  tk.quarterly_cashflow}
    a = {"inc": tk.financials,
         "bal": tk.balance_sheet,
         "cf":  tk.cashflow}
    try:    shares = float(tk.info.get("sharesOutstanding") or 0) or None
    except Exception: shares = None
    return {"q": q, "a": a, "shares": shares}

def latest_usable_statement(funds, asof):
    """Most recent statement legally knowable at `asof`.
    Checks quarterly (45-day lag) and annual (90-day lag); returns whichever
    period END is freshest → (period, kind) or (None, None)."""
    best = (None, None)
    for kind, lag in (("q", REPORTING_LAG_DAYS), ("a", ANNUAL_LAG_DAYS)):
        inc = funds[kind]["inc"]
        if inc is None or inc.empty: continue
        cutoff = asof - timedelta(days=lag)
        cols = [pd.Timestamp(c) for c in inc.columns if pd.Timestamp(c) <= cutoff]
        if not cols: continue
        cand = max(cols)
        if best[0] is None or cand > best[0]:
            best = (cand, kind)
    return best

# benchmark: key -> (green_lo, green_hi)  + inverse handled by ranges
PIT_BENCH = {
    "current_ratio":(2,None,1,2),      "gross_margin":(40,None,20,40),
    "operating_margin":(15,None,5,15), "net_margin":(10,None,3,10),
    "roa":(5,None,2,5),                "roe":(15,None,8,15),
    "de_ratio":(None,1,1,2),           "da_ratio":(None,0.5,0.5,0.7),
    "interest_coverage":(3,None,1.5,3),"asset_turnover":(1,None,0.5,1),
    "pe_ratio":(10,25,25,40),          "ps_ratio":(None,3,3,8),
    "fcf_margin":(10,None,3,10),       "cf_to_ni":(1,None,0.8,1),
    "capex_to_rev":(None,5,5,15),      "ebitda_margin":(20,None,10,20),
}

def _grade(key, v):
    if v is None: return None
    g_lo, g_hi, y_lo, y_hi = PIT_BENCH[key]
    def inr(lo, hi):
        if lo is not None and v < lo: return False
        if hi is not None and v > hi: return False
        return True
    if inr(g_lo, g_hi): return 2
    if inr(y_lo, y_hi): return 1
    return 0

def ratio_score_asof(funds, q, price, kind="q"):
    """Compute a 0-100 health score using ONLY data known at that date.
    kind: "q" quarterly (flows ×4 to annualise) or "a" annual (×1)."""
    stmts = funds[kind]
    inc, bal, cf, sh = stmts["inc"], stmts["bal"], stmts["cf"], funds["shares"]
    rev   = _val(_row(inc, ["Total Revenue"]), q)
    cogs  = _val(_row(inc, ["Cost Of Revenue"]), q)
    gross = _val(_row(inc, ["Gross Profit"]), q) or (
            (rev - cogs) if (rev and cogs) else None)
    ebit  = _val(_row(inc, ["Operating Income","EBIT"]), q)
    ni    = _val(_row(inc, ["Net Income"]), q)
    ebitda= _val(_row(inc, ["EBITDA","Normalized EBITDA"]), q)
    intex = _val(_row(inc, ["Interest Expense"]), q)
    ca    = _val(_row(bal, ["Current Assets"]), q)
    cl    = _val(_row(bal, ["Current Liabilities"]), q)
    ta    = _val(_row(bal, ["Total Assets"]), q)
    debt  = _val(_row(bal, ["Total Debt","Long Term Debt"]), q)
    eq    = _val(_row(bal, ["Stockholders Equity","Common Stock Equity"]), q)
    ocf   = _val(_row(cf,  ["Operating Cash Flow","Cash Flow From Continuing Operating Activities"]), q)
    capex = _val(_row(cf,  ["Capital Expenditure"]), q)
    capex = abs(capex) if capex is not None else None
    fcf   = (ocf - capex) if (ocf is not None and capex is not None) else None

    def d(a,b): return (a/b) if (a is not None and b not in (None,0)) else None
    # annualise flows: quarterly ×4, annual already full-year
    mult = 4 if kind == "q" else 1
    A = lambda x: x*mult if x is not None else None

    vals = {
        "current_ratio":     d(ca, cl),
        "gross_margin":      d(gross, rev)*100 if d(gross,rev) is not None else None,
        "operating_margin":  d(ebit, rev)*100  if d(ebit,rev)  is not None else None,
        "net_margin":        d(ni, rev)*100    if d(ni,rev)    is not None else None,
        "ebitda_margin":     d(ebitda, rev)*100 if d(ebitda,rev) is not None else None,
        "roa":               d(A(ni), ta)*100  if d(A(ni),ta)  is not None else None,
        "roe":               d(A(ni), eq)*100  if d(A(ni),eq)  is not None else None,
        "de_ratio":          d(debt, eq),
        "da_ratio":          d(debt, ta),
        "interest_coverage": d(ebit, abs(intex) if intex else None),
        "asset_turnover":    d(A(rev), ta),
        "pe_ratio":          d(price, d(A(ni), sh)) if sh else None,
        "ps_ratio":          d(price*sh if sh else None, A(rev)) if sh else None,
        "fcf_margin":        d(fcf, rev)*100 if d(fcf,rev) is not None else None,
        "cf_to_ni":          d(ocf, ni),
        "capex_to_rev":      d(capex, rev)*100 if d(capex,rev) is not None else None,
    }
    pts, mx = 0, 0
    for k, v in vals.items():
        g = _grade(k, v)
        if g is None: continue
        pts += g; mx += 2
    return round(pts/mx*100, 1) if mx else None, vals

# ─────────────────────────────────────────────────────────────────────────────
# 2. HISTORICAL SENTIMENT  (GDELT free archive → OUR model)
# ─────────────────────────────────────────────────────────────────────────────
COMPANY_QUERY = {
    "NVDA":"NVIDIA","META":"Meta Platforms","AAPL":"Apple iPhone","MSFT":"Microsoft",
    "GOOGL":"Alphabet Google","AMZN":"Amazon","TSLA":"Tesla","FUBO":"FuboTV",
    "NFLX":"Netflix","AMD":"AMD semiconductor","INTC":"Intel",
    # ── Kazakhstan 🇰🇿 (GDR/ADR listings reachable via yfinance) ──
    "KSPI":"Kaspi.kz",            # NASDAQ ADR
    "AIRA.IL":"Air Astana",       # London GDR
    "HSBK.IL":"Halyk Bank",       # London GDR
    "KAP.IL":"Kazatomprom",       # London GDR
    "FRHC":"Freedom Holding",     # NASDAQ
}

def gdelt_headlines(ticker, y, m, maxrec=25):
    """Free historical headlines for a month. Cached on disk."""
    key = f"{ticker}_{y}{m:02d}"
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        return json.load(open(path,encoding="utf-8"))
    q = COMPANY_QUERY.get(ticker, ticker)
    start = f"{y}{m:02d}01000000"
    endd  = (datetime(y, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    end   = endd.strftime("%Y%m%d") + "235959"
    url = ("https://api.gdeltproject.org/api/v2/doc/doc"
           f"?query={requests.utils.quote(q)}%20(stock%20OR%20earnings%20OR%20revenue)"
           f"%20sourcelang:eng&mode=artlist&maxrecords={maxrec}"
           f"&startdatetime={start}&enddatetime={end}&format=json")
    heads = []
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200:
            arts = r.json().get("articles", [])
            seen = set()
            for a in arts:
                t = (a.get("title") or "").strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower()); heads.append(t)
        time.sleep(1.2)          # be polite to the free API
    except Exception:
        pass
    json.dump(heads, open(path,"w",encoding="utf-8"), ensure_ascii=False)
    return heads

def sent_score_from_engine(engine, heads):
    """0-100 sentiment score using the trained model; None if no news."""
    if not heads or engine is None: return None, 0
    res = engine.analyze_headlines(heads)
    raw = res["score"]                       # −1..+1 region
    if   raw >  0.15: return 100, len(heads)
    elif raw < -0.15: return 0,   len(heads)
    return 50, len(heads)

# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    tickers = [t.upper() for t in args.tickers]
    print(f"\n{'─'*64}\n  BACKTEST  {args.start} → {args.end}   tickers: {', '.join(tickers)}\n{'─'*64}")

    # prices (daily, adjusted)
    print("[1/4] Downloading prices ...")
    all_tickers = tickers + ["SPY"]
    # threads=False avoids yfinance's sqlite cache 'database is locked'
    # error that shows up on Windows when downloading many tickers at once
    px = yf.download(all_tickers, start=args.start, end=args.end,
                     progress=False, auto_adjust=True, threads=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    px = px.dropna(how="all")

    # retry any ticker that came back empty (transient lock / rate limit)
    missing = [t for t in all_tickers if t not in px.columns or px[t].dropna().empty]
    for t in missing:
        print(f"      retrying {t} ...")
        for attempt in range(3):
            try:
                time.sleep(1.5)
                s = yf.download(t, start=args.start, end=args.end,
                                progress=False, auto_adjust=True,
                                threads=False)["Close"]
                if not s.dropna().empty:
                    px[t] = s
                    print(f"      ✓ {t} recovered")
                    break
            except Exception:
                continue
        else:
            print(f"      ✗ {t}: still unavailable — excluded from backtest")
    px = px.dropna(how="all")
    tickers = [t for t in tickers if t in px.columns and not px[t].dropna().empty]

    # fundamentals
    print("[2/4] Downloading quarterly fundamentals ...")
    funds = {}
    for t in tickers:
        try:
            funds[t] = load_fundamentals(t)
            print(f"      ✓ {t}")
        except Exception as e:
            print(f"      ✗ {t}: {e}")

    # sentiment model (optional)
    engine = None
    if not args.no_news:
        print("[3/4] Loading sentiment model ...")
        try:
            # train.Vocabulary must be reachable as __main__.Vocabulary for
            # pickle.load to unpickle outputs/vocab.pkl from this script
            import sys
            from train import Vocabulary
            sys.modules["__main__"].Vocabulary = Vocabulary

            from financial_analysis import SentimentEngine
            engine = SentimentEngine(model_dir=args.model_dir)
            print(f"      ✓ model loaded (val acc {engine.val_acc:.1%})")
        except Exception as e:
            print(f"      ✗ model unavailable ({e}) → ratios-only mode")
    else:
        print("[3/4] --no-news → ratios-only mode")

    # ── diagnose quarterly data coverage ─────────────────────────────────
    # yfinance's free quarterly_financials/balance_sheet/cashflow only return
    # the last ~4-5 reported quarters *as of today*, not a full history.
    # If args.start is older than that window, latest_usable_quarter() will
    # return None for every date and the backtest silently produces nothing.
    earliest_usable = {}
    for t in tickers:
        if t not in funds: continue
        cands = []
        for kind, lag in (("q", REPORTING_LAG_DAYS), ("a", ANNUAL_LAG_DAYS)):
            inc = funds[t][kind]["inc"]
            if inc is None or inc.empty: continue
            cols = sorted(pd.Timestamp(c) for c in inc.columns)
            if cols:
                cands.append(cols[0] + timedelta(days=lag))
        if cands:
            earliest_usable[t] = min(cands)

    if earliest_usable:
        feasible_start = max(earliest_usable.values())
        print(f"\n  ℹ yfinance free quarterly data covers:")
        for t, d in earliest_usable.items():
            print(f"      {t}: usable from {d.date()}")
        requested_start = pd.Timestamp(args.start)
        global_min = min(earliest_usable.values())
        if global_min > requested_start:
            print(f"\n  ⚠ Requested start {args.start} predates all available "
                  f"statements → moving start to {global_min.date()}.")
            args.start = global_min.strftime("%Y-%m-%d")
        # tickers join the portfolio as their data becomes available —
        # no global clipping to the worst ticker
        late = {t: d for t, d in earliest_usable.items()
                if d > pd.Timestamp(args.start)}
        if late:
            print("  ℹ joining later (no earlier statements): " +
                  ", ".join(f"{t} {d.date()}" for t, d in late.items()))
        if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
            print("  ✗ No statements available inside the requested window.")
            return

    # monthly rebalance dates
    months = pd.date_range(args.start, args.end, freq="MS")
    rows, no_news_count = [], 0

    print("[4/4] Generating point-in-time signals ...")
    for dt in months:
        trade_days = px.index[px.index >= dt]
        if len(trade_days) == 0: continue
        d0 = trade_days[0]
        for t in tickers:
            if t not in funds or t not in px.columns: continue
            price0 = px.loc[d0, t]
            if pd.isna(price0): continue
            q, kind = latest_usable_statement(funds[t], dt)
            if q is None: continue
            rscore, _ = ratio_score_asof(funds[t], q, float(price0), kind)
            if rscore is None: continue

            sscore, n_heads = (None, 0)
            if engine is not None:
                heads = gdelt_headlines(t, dt.year, dt.month)
                sscore, n_heads = sent_score_from_engine(engine, heads)
            if sscore is None:
                sscore = 50; no_news_count += 1     # honest neutral default

            combined = SENT_W*sscore + RATIO_W*rscore
            signal = "BUY" if combined >= BUY_T else ("HOLD" if combined >= HOLD_T else "SELL")
            rows.append(dict(date=d0, ticker=t, price=float(price0),
                             ratio=rscore, sent=sscore, n_heads=n_heads,
                             stmt=kind, stmt_period=str(pd.Timestamp(q).date()),
                             score=round(combined,1), signal=signal))
        print(f"      {dt.date()}  done")

    sig = pd.DataFrame(rows)
    if sig.empty:
        print("No signals generated — check tickers/dates."); return

    # ── portfolio: equal-weight BUYs, monthly rebalance ──────────────────
    eq, spy_eq, dates = [1.0], [1.0], [sig.date.min()]
    month_groups = sig.groupby(pd.Grouper(key="date", freq="MS"))
    month_starts = sorted(sig.date.unique())
    monthly = {d: g for d, g in sig.groupby("date")}
    ms = month_starts
    for i in range(len(ms)-1):
        d0, d1 = ms[i], ms[i+1]
        buys = monthly[d0].query("signal=='BUY'").ticker.tolist()
        if buys:
            rets = []
            for t in buys:
                try: rets.append(px.loc[d1, t]/px.loc[d0, t]-1)
                except Exception: pass
            r = float(np.nanmean(rets)) if rets else 0.0
        else:
            r = 0.0                                     # cash when no BUYs
        eq.append(eq[-1]*(1+r))
        spy_eq.append(spy_eq[-1]*(px.loc[d1,"SPY"]/px.loc[d0,"SPY"]))
        dates.append(d1)

    strat_ret = (eq[-1]-1)*100
    spy_ret   = (spy_eq[-1]-1)*100

    # ── hit rate: BUY beats SPY over horizon ─────────────────────────────
    H = args.horizon
    hits = tot = 0
    for _, r in sig[sig.signal=="BUY"].iterrows():
        try:
            i0 = px.index.get_loc(r.date)
            if i0 + H >= len(px): continue
            rt  = px.iloc[i0+H][r.ticker]/r.price - 1
            rs  = px.iloc[i0+H]["SPY"]/px.iloc[i0]["SPY"] - 1
            tot += 1; hits += int(rt > rs)
        except Exception: pass
    hit_rate = hits/tot*100 if tot else float("nan")

    # ── event study on sentiment ─────────────────────────────────────────
    ev = {}
    if engine is not None:
        for hor in (5, 21, 63):
            grp = {}
            for lbl, sub in sig.groupby(sig.sent.map({100:"POS",0:"NEG",50:"NEU"})):
                fwd = []
                for _, r in sub.iterrows():
                    try:
                        i0 = px.index.get_loc(r.date)
                        if i0+hor >= len(px): continue
                        ex = (px.iloc[i0+hor][r.ticker]/r.price-1) - \
                             (px.iloc[i0+hor]["SPY"]/px.iloc[i0]["SPY"]-1)
                        fwd.append(ex)
                    except Exception: pass
                if fwd: grp[lbl] = round(float(np.mean(fwd))*100, 2)
            ev[f"{hor}d"] = grp

    # ── report ───────────────────────────────────────────────────────────
    print(f"\n{'═'*64}\n  RESULTS\n{'─'*64}")
    print(f"  Strategy (BUY signals) : {strat_ret:+8.1f}%")
    print(f"  SPY benchmark          : {spy_ret:+8.1f}%")
    print(f"  Excess return          : {strat_ret-spy_ret:+8.1f}%")
    print(f"  Hit rate (BUY beats SPY over {H}d) : {hit_rate:.0f}%  ({hits}/{tot})")
    print(f"  Signals: {len(sig)}  BUY {len(sig[sig.signal=='BUY'])} · "
          f"HOLD {len(sig[sig.signal=='HOLD'])} · SELL {len(sig[sig.signal=='SELL'])}")
    if engine is not None:
        print(f"  Months with no archived news (defaulted neutral): {no_news_count}")
        print(f"\n  EVENT STUDY — mean excess return after sentiment label")
        for hor, grp in ev.items():
            line = "   ".join(f"{k}: {v:+.2f}%" for k, v in grp.items())
            print(f"    +{hor:>3}: {line}")
    print("═"*64)

    # ── save ─────────────────────────────────────────────────────────────
    out = dict(period=[args.start, args.end], tickers=tickers,
               strategy_return_pct=round(strat_ret,2),
               spy_return_pct=round(spy_ret,2),
               excess_pct=round(strat_ret-spy_ret,2),
               hit_rate_pct=round(hit_rate,1) if tot else None,
               hit_n=[hits, tot], horizon_days=H,
               event_study=ev, no_news_months=no_news_count,
               weights={"sentiment":SENT_W,"ratios":RATIO_W},
               signals=sig.assign(date=sig.date.astype(str)).to_dict("records"))
    json.dump(out, open("backtest_results.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print("  💾 backtest_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5.5))
        plt.plot(dates, eq, lw=2.4, color="#800054", label=f"FinSent BUY signals  {strat_ret:+.1f}%")
        plt.plot(dates, spy_eq, lw=2, color="#9aa0a6", ls="--", label=f"SPY  {spy_ret:+.1f}%")
        plt.title("Backtest — portfolio of BUY signals vs S&P 500", fontweight="bold")
        plt.ylabel("Growth of $1"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("backtest_chart.png", dpi=160)
        print("  💾 backtest_chart.png")
    except Exception as e:
        print(f"  chart skipped: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start",   default="2023-01-01")
    p.add_argument("--end",     default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--horizon", type=int, default=126, help="trading days for hit-rate (≈6 mo)")
    p.add_argument("--model_dir", default="outputs")
    p.add_argument("--no-news", action="store_true", help="skip GDELT, ratios-only (fast)")
    run(p.parse_args())
