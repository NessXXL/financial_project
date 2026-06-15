"""
BUILD SITE DATA — turn engine runs into website cards, automatically
====================================================================
No more hand-copying numbers into the site. Workflow:

  1. Run the engine on as many tickers as you want (each saves a JSON):
         python financial_analysis.py --ticker AAPL
         python financial_analysis.py --ticker JPM
         python financial_analysis.py --ticker WMT
         ...
     Or batch them in PowerShell:
         "AAPL","JPM","WMT","KO","DIS" | ForEach-Object { python financial_analysis.py --ticker $_ }

  2. Run this script. It reads EVERY *_analysis.json in the folder and
     rewrites the DATA={...} block inside index.html with real results,
     each tagged as a live engine result with its date.

         python build_site_data.py

  3. Refresh index.html (or push to GitHub → Vercel redeploys).

Flags:
  --html  path to the site file       (default: index.html)
  --only  comma-separated tickers to include (default: all JSON found)
"""

import argparse
import glob
import json
import os
import re

# ── ratio display order on the site, grouped exactly like the dashboard ──
GROUPS = [
    ("LIQUIDITY", [
        ("current_ratio", "Current Ratio", "x"),
        ("quick_ratio", "Quick Ratio", "x"),
        ("cash_ratio", "Cash Ratio", "x"),
        ("ocf_ratio", "OCF Ratio", "x")]),
    ("PROFITABILITY", [
        ("gross_margin", "Gross Margin", "%"),
        ("operating_margin", "Operating Margin", "%"),
        ("net_margin", "Net Margin", "%"),
        ("roa", "ROA", "%"), ("roe", "ROE", "%"), ("roce", "ROCE", "%"),
        ("ebitda_margin", "EBITDA Margin", "%")]),
    ("LEVERAGE", [
        ("de_ratio", "Debt / Equity", "x"),
        ("da_ratio", "Debt / Assets", "x"),
        ("interest_coverage", "Interest Coverage", "x"),
        ("dscr", "DSCR", "x"),
        ("equity_multiplier", "Equity Multiplier", "x"),
        ("net_debt_ebitda", "Net Debt/EBITDA", "x")]),
    ("EFFICIENCY", [
        ("asset_turnover", "Asset Turnover", "x"),
        ("receivables_turnover", "Receivables Turn.", "x"),
        ("dso", "DSO", "d"),
        ("fixed_asset_turnover", "Fixed Asset Turn.", "x"),
        ("inventory_turnover", "Inventory Turn.", "x"),
        ("ccc", "CCC", "d")]),
    ("VALUATION", [
        ("pe_ratio", "P/E Ratio", "x"),
        ("pb_ratio", "P/B Ratio", "x"),
        ("ev_ebitda", "EV / EBITDA", "x"),
        ("ps_ratio", "P/S Ratio", "x"),
        ("eps", "EPS", "$"),
        ("dividend_yield", "Div. Yield", "%")]),
    ("CASH FLOW", [
        ("fcf", "Free Cash Flow", "$"),
        ("fcf_margin", "FCF Margin", "%"),
        ("cf_to_net_income", "CF / Net Income", "x"),
        ("capex_to_revenue", "Capex / Revenue", "%")]),
]

# benchmark table — must match financial_analysis.py grading
BENCH = {
    "current_ratio": {"g": (2, None), "y": (1, 2), "r": (None, 1)},
    "quick_ratio": {"g": (1, None), "y": (0.5, 1), "r": (None, 0.5)},
    "cash_ratio": {"g": (0.5, 1.5), "y": (0.2, 0.5), "r": (None, 0.2)},
    "ocf_ratio": {"g": (1, None), "y": (0.5, 1), "r": (None, 0.5)},
    "gross_margin": {"g": (40, None), "y": (20, 40), "r": (None, 20)},
    "operating_margin": {"g": (15, None), "y": (5, 15), "r": (None, 5)},
    "net_margin": {"g": (10, None), "y": (3, 10), "r": (None, 3)},
    "roa": {"g": (5, None), "y": (2, 5), "r": (None, 2)},
    "roe": {"g": (15, None), "y": (8, 15), "r": (None, 8)},
    "roce": {"g": (15, None), "y": (8, 15), "r": (None, 8)},
    "ebitda_margin": {"g": (20, None), "y": (10, 20), "r": (None, 10)},
    "de_ratio": {"g": (None, 1), "y": (1, 2), "r": (2, None)},
    "da_ratio": {"g": (None, 0.5), "y": (0.5, 0.7), "r": (0.7, None)},
    "interest_coverage": {"g": (3, None), "y": (1.5, 3), "r": (None, 1.5)},
    "dscr": {"g": (1.25, None), "y": (1, 1.25), "r": (None, 1)},
    "equity_multiplier": {"g": (None, 2), "y": (2, 4), "r": (4, None)},
    "net_debt_ebitda": {"g": (None, 2), "y": (2, 4), "r": (4, None)},
    "asset_turnover": {"g": (1, None), "y": (0.5, 1), "r": (None, 0.5)},
    "inventory_turnover": {"g": (6, None), "y": (3, 6), "r": (None, 3)},
    "receivables_turnover": {"g": (8, None), "y": (4, 8), "r": (None, 4)},
    "dso": {"g": (None, 45), "y": (45, 75), "r": (75, None)},
    "ccc": {"g": (None, 30), "y": (30, 60), "r": (60, None)},
    "fixed_asset_turnover": {"g": (2, None), "y": (1, 2), "r": (None, 1)},
    "pe_ratio": {"g": (10, 25), "y": (25, 40), "r": (40, None)},
    "pb_ratio": {"g": (1, 5), "y": (5, 10), "r": (10, None)},
    "ev_ebitda": {"g": (6, 12), "y": (12, 20), "r": (20, None)},
    "ps_ratio": {"g": (None, 3), "y": (3, 8), "r": (8, None)},
    "dividend_yield": {"g": (2, 5), "y": (0.1, 2), "r": (5, None)},
    "fcf_margin": {"g": (10, None), "y": (3, 10), "r": (None, 3)},
    "cf_to_net_income": {"g": (1, None), "y": (0.8, 1), "r": (None, 0.8)},
    "capex_to_revenue": {"g": (None, 5), "y": (5, 15), "r": (15, None)},
}


def status(key, v, ratios):
    """Return 1 ok / 0 watch / -1 warn / 2 n/a — mirrors the engine."""
    if v is None:
        return 2
    if key == "fcf":
        return -1 if v < 0 else 1
    if key == "eps":
        nm = ratios.get("net_margin")
        if v < 0 or (nm is not None and nm < 0):
            return -1
        return 1
    b = BENCH.get(key)
    if not b:
        return 2

    def ok(val, lo, hi):
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True
    if ok(v, *b["g"]):
        return 1
    if ok(v, *b["r"]):
        return -1
    if ok(v, *b["y"]):
        return 0
    return 2


def fmt(v, unit):
    if v is None:
        return "N/A"
    if unit == "%":
        return f"{v:.2f}%"
    if unit == "d":
        return f"{v:.0f}d"
    if unit == "$":
        a = abs(v)
        if a >= 1e9:
            return f"${v/1e9:.2f}B"
        if a >= 1e6:
            return f"${v/1e6:.2f}M"
        return f"${v:.2f}"
    return f"{v:.2f}x"


def cap_str(mc):
    if not mc:
        return "—"
    if mc >= 1e12:
        return f"{mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"{mc/1e9:.2f}B"
    return f"{mc/1e6:.0f}M"


def js(s):
    """Escape a python string for embedding in a JS double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_entry(d):
    t = d["ticker"]
    prof = d.get("profile", {}) or {}
    quote = d.get("quote", {}) or {}
    sent = d.get("sentiment", {}) or {}
    ratios = d.get("ratios", {}) or {}
    fsig = d.get("final_signal", {}) or {}
    pred = d.get("prediction", {}) or {}

    sig = fsig.get("signal", "N/A")
    if "SELL" in sig or "AVOID" in sig:
        sig_short = "SELL"
    elif "BUY" in sig:
        sig_short = "BUY"
    else:
        sig_short = "HOLD"

    date = (d.get("timestamp", "") or "")[:10]

    bd = sent.get("breakdown", {}) or {}
    heads = []
    for h in (sent.get("details", []) or [])[:6]:
        lab = h.get("label", "neutral")
        lab3 = "pos" if lab == "positive" else "neg" if lab == "negative" else "neu"
        head = js(h.get("headline", "")[:95])
        conf = int(round((h.get("confidence", 0) or 0) * 100))
        heads.append(f'["{head}","{lab3}",{conf}]')
    heads_js = ",".join(heads)

    overall = (sent.get("overall", "neutral") or "neutral").upper()
    sc = sent.get("score", 0) or 0
    sc_str = f"{sc:+.3f}"

    # ratios grouped
    rlines = []
    for gname, items in GROUPS:
        cells = []
        for key, label, unit in items:
            v = ratios.get(key)
            st = status(key, v, ratios)
            cells.append(f'["{label}","{fmt(v, unit)}",{st}]')
        rlines.append(f'  "{gname}":[' + ",".join(cells) + "]")
    ratios_js = "{\n" + ",\n".join(rlines) + "}"

    # forecast
    if pred.get("available"):
        fc = (f'{{bull:[{pred.get("bull_case",0)},{pred.get("price_bull",0)}],'
              f'base:[{pred.get("base_case",0)},{pred.get("price_base",0)}],'
              f'bear:[{pred.get("bear_case",0)},{pred.get("price_bear",0)}],')
        an = pred.get("detail", {}).get("analyst")
        if an:
            note = (f'{an.get("n_analysts","?")} analysts · '
                    f'<b>{str(an.get("recommendation","")).replace("_"," ")}</b> · '
                    f'mean <b>${an.get("target_mean","?")} '
                    f'({an.get("upside_mean",0):+.1f}%)</b> · '
                    f'methods: {", ".join(pred.get("methods_used", []))}')
        else:
            note = ("analyst targets unavailable — "
                    f'methods: {", ".join(pred.get("methods_used", []))}')
        fc += f'an:"{js(note)}"}}'
    else:
        cur = quote.get("price", 0) or 0
        fc = (f'{{bull:[0,{cur}],base:[0,{cur}],bear:[0,{cur}],'
              f'an:"Forecast unavailable for this ticker."}}')

    sent_score = fsig.get("sentiment_score", 50)
    ratio_score = fsig.get("ratio_score", 50)
    combined = fsig.get("combined_score", 50)
    price = quote.get("price", 0) or 0
    chg = quote.get("change_pct", 0) or 0
    sector = js(prof.get("sector", "") or "")
    name = js(prof.get("name", t) or t)

    key_id = f'"{t}"' if ("." in t or "-" in t) else t
    return (
        f'{key_id}:{{name:"{name}",sector:"{sector}",'
        f'price:{price},chg:{chg},cap:"{cap_str(quote.get("marketCap"))}",'
        f'live:"{date}",\n'
        f' sig:"{sig_short}",score:{combined},sent:{sent_score},ratio:{ratio_score},\n'
        f' sentiment:{{overall:"{overall}",sc:"{sc_str}",'
        f'pos:{bd.get("positive",0)},neu:{bd.get("neutral",0)},neg:{bd.get("negative",0)},'
        f'heads:[{heads_js}]}},\n'
        f' ratios:{ratios_js},\n'
        f' fc:{fc}}}'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="index.html")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    files = sorted(glob.glob("*_analysis.json"))
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        files = [f for f in files
                 if os.path.basename(f).replace("_analysis.json", "").upper() in want]
    if not files:
        print("No *_analysis.json files found. Run financial_analysis.py first.")
        return

    entries, names = [], []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            entries.append(build_entry(d))
            names.append(d["ticker"])
        except Exception as e:
            print(f"  ! skipped {f}: {e}")
    if not entries:
        print("Nothing valid to write.")
        return

    data_block = "const DATA={\n" + ",\n".join(entries) + "\n};"

    if not os.path.exists(args.html):
        # no site? just dump the JS so it can be pasted
        open("site_data.js", "w", encoding="utf-8").write(data_block)
        print(f"  {args.html} not found — wrote site_data.js with "
              f"{len(entries)} tickers ({', '.join(names)})")
        return

    html = open(args.html, encoding="utf-8").read()
    m = re.search(r"const DATA=\{.*?\n\};", html, re.S)
    if not m:
        print(f"  ✗ could not find 'const DATA={{...}};' block in {args.html}")
        open("site_data.js", "w", encoding="utf-8").write(data_block)
        print("  wrote site_data.js instead — paste it into the <script> manually")
        return

    backup = args.html + ".bak_sitedata"
    open(backup, "w", encoding="utf-8").write(html)

    # MERGE: update/insert only the tickers we just built, keep all the rest.
    # We locate each existing "TICKER:{name:...}}," entry and replace it; new
    # tickers are appended just before the closing of the DATA object.
    block = m.group(0)
    # work on the inner body, between "const DATA={" and the final "};"
    head = "const DATA={"
    tail = "\n};"
    body = block[len(head):-len(tail)]
    # split into entries on top-level boundaries is hard; instead regex-replace
    # each ticker entry INCLUDING any trailing comma, then normalise commas.
    updated, added = [], []
    for ent, tk in zip(entries, names):
        key_re = r'"?' + re.escape(tk) + r'"?:\{name:.*?\}\},?'
        pat = re.compile(key_re, re.S)
        if pat.search(body):
            body = pat.sub(ent + ",", body, count=1)
            updated.append(tk)
        else:
            body = body.rstrip().rstrip(",") + ",\n" + ent + ","
            added.append(tk)
    # normalise: collapse duplicate commas, strip trailing comma
    body = re.sub(r",\s*,", ",", body).rstrip()
    body = body.rstrip(",")
    block = head + body + tail

    html = html[:m.start()] + block + html[m.end():]
    open(args.html, "w", encoding="utf-8").write(html)
    if updated:
        print(f"  ↻ updated: {', '.join(updated)}")
    if added:
        print(f"  + added:   {', '.join(added)}")

    # update the "demo cache" list shown for unknown tickers, if present
    cache_list = " · ".join(names)
    html2 = re.sub(r"(pre-computed results for </b>?<b>)[^<]*(</b>)",
                   r"\g<1>" + cache_list + r"\g<2>", html)
    if html2 != html:
        open(args.html, "w", encoding="utf-8").write(html2)

    print(f"  ✓ {len(entries)} ticker(s) written to {args.html}")
    print(f"  • backup: {backup}")
    print("  → refresh the page or push to GitHub (Vercel redeploys).")


if __name__ == "__main__":
    main()
