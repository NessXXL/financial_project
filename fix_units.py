"""
UNITS & CURRENCY REPAIR — fixes 4 bugs found on real KZ-ticker runs
===================================================================
Run once from the project folder:   python fix_units.py

1. KAP.IL bull case +35,878%  → analyst targets arrive in the HOME market
   currency (KZT on KASE) while the price is the USD GDR quote. Guard:
   drop the analyst method when targets are inconsistent with the quote.
2. HSBK.IL "BUY but −48%"     → same mismatch, same guard.
3. KSPI P/S = 0.00x graded HEALTHY → USD market cap divided by KZT revenue.
   Cross-currency micro-values are scrubbed to N/A.
4. MSFT dividend yield 93%    → yfinance garbage; values >25% dropped.
Plus: a printed warning when statements and quote use different currencies
(so $491B "FCF" for Kaspi is labeled as tenge, not dollars).
"""

import ast, os, re, shutil

F = "financial_analysis.py"

def patch(src, old, new, label):
    if new in src:
        print(f"  = already applied: {label}")
        return src, True
    if old not in src:
        print(f"  ✗ anchor not found: {label}")
        return src, False
    print(f"  ✓ {label}")
    return src.replace(old, new, 1), True


def main():
    if not os.path.exists(F):
        print(f"{F} not found — run from the project folder"); return
    src = open(F, encoding="utf-8").read()
    shutil.copy(F, F + ".bak_units")

    # ── 1+2: analyst currency-mismatch guard ────────────────────────────
    old = '''                "n_analysts":    int(n_analysts),
                "recommendation": recommendation,
            }
    except Exception as e:
        pass'''
    new = '''                "n_analysts":    int(n_analysts),
                "recommendation": recommendation,
            }
            # Currency/listing sanity check. GDR tickers (KAP.IL, HSBK.IL ...)
            # often get analyst targets quoted in the HOME market currency
            # (e.g. tenge on KASE) while current_price is the USD GDR quote.
            # Symptoms: absurd upside (+35,000%) or a BUY consensus paired
            # with a deeply negative "upside". Honest move: drop the method.
            rec_is_buy = str(recommendation).lower() in (
                "buy", "strong_buy", "strongbuy")
            if abs(analyst_upside) > 300 or (rec_is_buy and analyst_upside < -25):
                results.pop("analyst", None)
                print("  ⚠ analyst targets inconsistent with quote currency "
                      "(home-listing units?) — analyst method dropped")
    except Exception as e:
        pass'''
    src, ok1 = patch(src, old, new, "analyst currency-mismatch guard")

    # ── 3: scrub cross-currency micro-ratios ────────────────────────────
    old = '''            "pe_ratio":             safe(pe),
            "pb_ratio":             safe(pb),
            "ev_ebitda":            safe(ev_ebitda),
            "ps_ratio":             safe(ps),
            "dividend_yield":       safe(div_yield),'''
    new = '''            "pe_ratio":             safe(pe),
            # Cross-currency scrub: for foreign GDRs, USD market cap divided
            # by local-currency statements yields ~0.00x garbage. Such micro
            # values are not real and are reported as N/A instead.
            "pb_ratio":             (lambda v: v if (v is None or v >= 0.05) else None)(safe(pb)),
            "ev_ebitda":            (lambda v: v if (v is None or v >= 0.5)  else None)(safe(ev_ebitda)),
            "ps_ratio":             (lambda v: v if (v is None or v >= 0.02) else None)(safe(ps)),
            # yfinance sometimes returns dividendYield already ×100 (93 for
            # MSFT instead of 0.93). Anything above 25% is vendor garbage.
            "dividend_yield":       (lambda v: v if (v is None or 0 < v <= 25) else None)(safe(div_yield)),'''
    src, ok2 = patch(src, old, new, "micro-ratio + dividend-yield scrub")

    # ── 4: currency mismatch warning in the report ──────────────────────
    old = '''    prediction = predict_price_target(ticker_obj, fin, quote, ratios, sentiment)'''
    new = '''    prediction = predict_price_target(ticker_obj, fin, quote, ratios, sentiment)

    # Statements vs quote currency: for GDRs (KSPI statements are in KZT,
    # the quote is USD) absolute cash-flow numbers are LOCAL currency.
    try:
        _info = ticker_obj.info or {}
        _fc, _qc = _info.get("financialCurrency"), _info.get("currency")
        if _fc and _qc and _fc != _qc:
            print(f"  ⚠ statements in {_fc}, quote in {_qc} — absolute "
                  f"figures (FCF, revenue) are {_fc}, NOT {_qc}; "
                  f"cross-currency ratios shown as N/A")
    except Exception:
        pass'''
    src, ok3 = patch(src, old, new, "currency mismatch warning")

    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  ✗ syntax error after patch (line {e.lineno}) — NOT saved, "
              f"backup at {F}.bak_units")
        return
    open(F, "w", encoding="utf-8").write(src)
    n = sum([ok1, ok2, ok3])
    print(f"\n  saved — {n}/3 fixes in place. Re-run:")
    print("    python financial_analysis.py --ticker KAP.IL")
    print("    python financial_analysis.py --ticker HSBK.IL")
    print("    python financial_analysis.py --ticker KSPI")
    print("    python financial_analysis.py --ticker MSFT")


if __name__ == "__main__":
    main()
