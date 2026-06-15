"""
FIX SAVE — make financial_analysis.py write marketCap + prediction to JSON
==========================================================================
Your engine PRINTS the forecast and market cap correctly, but never SAVED
them into {TICKER}_analysis.json — so the website (which reads the JSON)
showed "Mkt Cap $—" and "Forecast unavailable / +0%".

This patch applies two edits to financial_analysis.py:
  1. save block: include marketCap in quote, and the full prediction object
  2. print: fix the "+-48.2%" double-sign on a negative bull case

Run once from the project folder:   python fix_save.py
Then re-run your tickers and python build_site_data.py.
"""

import ast
import os
import shutil

F = "financial_analysis.py"


def main():
    if not os.path.exists(F):
        print(f"{F} not found — run from the project folder")
        return
    src = open(F, encoding="utf-8").read()
    changed = []

    # ── EDIT 1: save block ───────────────────────────────────────────────
    old_save = (
        '         "quote":{"price":safe((quote or {}).get("price")),'
        '"change_pct":safe((quote or {}).get("changesPercentage"))},\n'
        '         "sentiment":sentiment,\n'
        '         "ratios":{k:round(v,4) if v is not None else None for k,v in ratios.items()},\n'
        '         "ratio_score":rs,"final_signal":final}'
    )
    new_save = (
        '         "quote":{"price":safe((quote or {}).get("price")),\n'
        '                  "change_pct":safe((quote or {}).get("changesPercentage")),\n'
        '                  "marketCap":safe((quote or {}).get("marketCap"))},\n'
        '         "sentiment":sentiment,\n'
        '         "ratios":{k:round(v,4) if v is not None else None for k,v in ratios.items()},\n'
        '         "ratio_score":rs,"final_signal":final,\n'
        '         "prediction":prediction}'
    )
    if '"prediction":prediction' in src:
        print("  = save block already includes prediction — skipping edit 1")
    elif old_save in src:
        src = src.replace(old_save, new_save)
        changed.append("save block now writes marketCap + prediction")
    else:
        print("  ✗ could not find the save block to patch (edit 1).")
        print("    Look for the line starting with  out={\"ticker\":ticker  ")
        print("    and ensure quote includes marketCap and you add "
              "\"prediction\":prediction.")

    # strip helper keys so the JSON stays clean (optional, safe if absent)
    helper_old = '    out={"ticker":ticker,"timestamp":datetime.now().isoformat(),'
    helper_new = ('    for _k in ("_ratio_score","_fcf_negative"):\n'
                  '        sentiment.pop(_k, None)\n'
                  '    out={"ticker":ticker,"timestamp":datetime.now().isoformat(),')
    if "_ratio_score" in src and 'sentiment.pop(_k' not in src and helper_old in src:
        src = src.replace(helper_old, helper_new)
        changed.append("internal helper keys stripped before save")

    # ── EDIT 2: double-sign on bull case print ───────────────────────────
    old_print = '🐂 Bull case  {R}: {bull_col}+{bull:.1f}%{R}'
    new_print = '🐂 Bull case  {R}: {bull_col}{bull:+.1f}%{R}'
    if old_print in src:
        src = src.replace(old_print, new_print)
        changed.append('fixed "+-48.2%" double sign on bull case')
    elif new_print in src:
        print("  = bull-case sign already correct — skipping edit 2")

    if not changed:
        print("\nNothing to change — your file is already up to date.")
        return

    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  ✗ patch produced a syntax error (line {e.lineno}) — NOT saved")
        return

    shutil.copy(F, F + ".bak_save")
    open(F, "w", encoding="utf-8").write(src)
    print("\n  ✓ applied:")
    for c in changed:
        print(f"      • {c}")
    print(f"  • backup: {F}.bak_save")
    print("\nNext:")
    print('  Re-run your tickers, e.g.:')
    print('    "NVDA","META","AAPL","TSLA","GOOGL","MSFT","AMZN","FUBO","KSPI",'
          '"AIRA.IL","HSBK.IL","KAP.IL","FRHC","VRDN" | '
          'ForEach-Object { python financial_analysis.py --ticker $_ }')
    print('  python build_site_data.py')


if __name__ == "__main__":
    main()