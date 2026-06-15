"""
ONE-SHOT REPAIR — fixes the `NameError: POSITIVE_WORDS is not defined`
======================================================================
The multilingual patch inserted the RU/KZ lexicon block too early in the
file (before POSITIVE_WORDS is defined). This script moves it to the
right place. Run once from the project folder:

    python fix_lexicon.py
"""

import ast
import os
import re
import shutil

MARKER = "MULTILINGUAL LEXICON"

LEXICON_BLOCK = '''
# ======== MULTILINGUAL LEXICON (RU/KZ) — added by fix_lexicon ========
POSITIVE_WORDS |= {
    "прибыль","рост","вырос","выросла","выросли","рекорд","рекордный","рекордная",
    "превысил","превысила","успех","успешный","успешно","укрепился","укрепилась",
    "дивиденды","покупать","повышение","повысил","улучшение","улучшил","позитивный",
    "сильный","сильные","взлет","взлетел","подорожал","подорожали","прорыв",
    "лидер","максимум","увеличил","увеличение","восстановление","опередил",
    "өсім","өсті","табыс","пайда","рекордтық","сәтті","жоғарылады","күшейді",
    "оң","серпіліс","жетістік","ұлғайды","артты","дивидендтер",
}
NEGATIVE_WORDS |= {
    "убыток","убытки","падение","упал","упала","упали","обвал","обвалился",
    "кризис","банкротство","банкрот","риск","риски","снижение","снизил","снизился",
    "сократился","сокращение","штраф","расследование","дефолт","слабый","слабые",
    "негативный","продавать","подешевел","подешевели","провал","потери","потерял",
    "минимум","долг","долги","санкции","иск","скандал","крах","рухнул",
    "шығын","құлдырау","төмендеді","дағдарыс","банкроттық","тәуекел","айыппұл",
    "әлсіз","теріс","сәтсіздік","азайды","қарыз","төмендеу",
}
NEGATION_WORDS |= {"не","нет","ни","без","емес","жоқ","ешқандай"}
try:
    _stop_words |= {
        "и","в","во","на","с","со","по","за","к","ко","из","у","о","об","же","бы",
        "это","как","для","что","но","или","от","до","при","его","ее","их","также",
        "мен","және","бұл","осы","үшін","да","де","та","те","бен","пен","туралы",
    }
except NameError:
    pass
# =====================================================================
'''


def remove_old_block(lines):
    """Line-based removal of any previously injected lexicon block.
    Finds the header line containing the marker, then deletes everything
    up to and including the next full-width comment rule (a line that is
    a '#' followed by only ═ or = characters)."""
    out, i, removed = [], 0, 0
    while i < len(lines):
        line = lines[i]
        if MARKER in line and line.lstrip().startswith("#"):
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("#") and set(s[1:].strip()) <= {"═", "="} and s[1:].strip():
                    break
                j += 1
            removed += 1
            i = j + 1            # skip the closing rule line too
            continue
        out.append(line)
        i += 1
    return out, removed


def insert_after_last_set(src):
    """Insert the lexicon block right after the LAST of the three set
    definitions, whatever order they appear in."""
    last_end = -1
    for name in ("POSITIVE_WORDS", "NEGATIVE_WORDS", "NEGATION_WORDS"):
        for m in re.finditer(
                name + r"\s*=\s*(frozenset\()?\{[^}]*\}\)?", src, re.S):
            last_end = max(last_end, m.end())
    if last_end < 0:
        return None
    return src[:last_end] + "\n" + LEXICON_BLOCK + src[last_end:]


def repair(fname):
    if not os.path.exists(fname):
        print(f"  - {fname}: not found, skipped")
        return
    src = open(fname, encoding="utf-8").read()

    if MARKER not in src:
        print(f"  = {fname}: no injected block found — nothing to fix")
        return

    bak = fname + ".bak_fix"
    shutil.copy(fname, bak)

    lines, removed = remove_old_block(src.splitlines(keepends=True))
    src = "".join(lines)
    print(f"  • {fname}: removed {removed} misplaced block(s)")

    new = insert_after_last_set(src)
    if new is None:
        print(f"  ! {fname}: POSITIVE/NEGATIVE/NEGATION sets not found — "
              f"lexicon NOT re-inserted (file saved without the block)")
        new = src

    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"  ✗ {fname}: result has a syntax error at line {e.lineno} — "
              f"NOT saved; original kept, backup at {bak}")
        return

    open(fname, "w", encoding="utf-8").write(new)
    print(f"  ✓ {fname}: fixed and saved (backup: {bak})")


if __name__ == "__main__":
    print("\nLEXICON POSITION REPAIR\n" + "-" * 46)
    repair("financial_analysis.py")
    repair("train.py")
    print("-" * 46)
    print("""Verify:
    python -c "import financial_analysis; print('import OK')"
    python financial_analysis.py --ticker AAPL

Then re-run the backtest — your current chart was generated in
RATIOS-ONLY mode (the model could not load because of this exact
error). With the model fixed you also get the event study:
    python backtest.py --tickers NVDA META AAPL MSFT GOOGL AMZN TSLA FUBO --start 2023-01-01 --end 2025-06-01
""")
