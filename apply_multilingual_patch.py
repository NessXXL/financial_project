"""
MULTILINGUAL PATCH — adds Russian + Kazakh support to your existing code
========================================================================
Run ONCE from the project folder (where train.py lives):

    python apply_multilingual_patch.py

What it changes (backups saved as *.bak_multilingual):

  train.py
    1. clean_text(): keeps Cyrillic letters (а-яё + Kazakh әғқңөұүһі)
       — the old regex [^a-z0-9\\s'] silently DELETED all Russian/Kazakh text
    2. POSITIVE/NEGATIVE/NEGATION word sets: +60 RU/KZ financial terms
    3. load_datasets(): auto-merges data/multilingual/ru.csv & kk.csv
       when they exist (created by build_multilingual_data.py)

  financial_analysis.py
    Same fixes 1–2 for its own clean_text / lexicon copies, so the
    deployed engine can score Russian and Kazakh headlines too.

After patching:
    python build_multilingual_data.py --per_class 1200     # make RU/KZ data
    python train.py --mode train --glove_path data/glove.840B.300d.txt
"""

import os, re, shutil, sys

CYR = "а-яёәғқңөұүһі"          # Russian + Kazakh-specific letters (lowercase)

LEXICON_BLOCK = '''
# ════════ MULTILINGUAL LEXICON (RU/KZ) — added by apply_multilingual_patch ════════
POSITIVE_WORDS |= {
    # Russian
    "прибыль","рост","вырос","выросла","выросли","рекорд","рекордный","рекордная",
    "превысил","превысила","успех","успешный","успешно","укрепился","укрепилась",
    "дивиденды","покупать","повышение","повысил","улучшение","улучшил","позитивный",
    "сильный","сильные","взлет","взлетел","подорожал","подорожали","прорыв",
    "лидер","максимум","увеличил","увеличение","восстановление","опередил",
    # Kazakh
    "өсім","өсті","табыс","пайда","рекордтық","сәтті","жоғарылады","күшейді",
    "оң","серпіліс","жетістік","ұлғайды","артты","дивидендтер","максимум",
}
NEGATIVE_WORDS |= {
    # Russian
    "убыток","убытки","падение","упал","упала","упали","обвал","обвалился",
    "кризис","банкротство","банкрот","риск","риски","снижение","снизил","снизился",
    "сократился","сокращение","штраф","расследование","дефолт","слабый","слабые",
    "негативный","продавать","подешевел","подешевели","провал","потери","потерял",
    "минимум","долг","долги","санкции","иск","скандал","крах","рухнул",
    # Kazakh
    "шығын","құлдырау","төмендеді","дағдарыс","банкроттық","тәуекел","айыппұл",
    "әлсіз","теріс","сәтсіздік","азайды","қарыз","минимум","төмендеу",
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
# ══════════════════════════════════════════════════════════════════════════════
'''

LOADER_WRAPPER = '''
# ════════ MULTILINGUAL DATA AUTOLOAD — added by apply_multilingual_patch ════════
# If data/multilingual/ru.csv / kk.csv exist (run build_multilingual_data.py),
# they are merged into training automatically.
_orig_load_datasets = load_datasets
def load_datasets(*args, **kwargs):
    dfs = _orig_load_datasets(*args, **kwargs)
    import os as _os
    import pandas as _pd
    for _lang in ("ru", "kk"):
        _p = _os.path.join("data", "multilingual", f"{_lang}.csv")
        if _os.path.exists(_p):
            try:
                _d = _pd.read_csv(_p)
                if {"text", "sentiment"}.issubset(_d.columns) and len(_d):
                    _d = _d.dropna(subset=["text", "sentiment"])
                    _d["sentiment"] = _d["sentiment"].astype(int)
                    dfs[f"multilingual_{_lang}"] = _d[["text", "sentiment"]]
                    print(f"  + multilingual merged: {_p} ({len(_d):,} rows)")
            except Exception as _e:
                print(f"  ! multilingual skipped {_p}: {_e}")
    return dfs
# ══════════════════════════════════════════════════════════════════════════════
'''


def patch_clean_text(src: str, fname: str) -> tuple[str, bool]:
    """Make every a-z char-class in regexes Cyrillic-aware."""
    patterns = [
        (r"\[\^a-z0-9\\s\\'\]",        f"[^a-z{CYR}0-9\\\\s\\\\']"),
        (r"\[\^a-z0-9\\s'\]",          f"[^a-z{CYR}0-9\\\\s']"),
        (r"\[\^a-zA-Z0-9\\s\\'\]",     f"[^a-zA-Z{CYR}0-9\\\\s\\\\']"),
        (r"\[\^a-zA-Z0-9\\s'\]",       f"[^a-zA-Z{CYR}0-9\\\\s']"),
        (r"\[a-z\]",                   f"[a-z{CYR}]"),
    ]
    done = False
    for pat, rep in patterns:
        new = re.sub(pat, rep, src)
        if new != src:
            src, done = new, True
    if done:
        print(f"  ✓ {fname}: clean_text regex now keeps Cyrillic")
    else:
        if CYR in src:
            print(f"  = {fname}: clean_text already Cyrillic-aware")
            done = True
        else:
            print(f"  ✗ {fname}: clean_text regex NOT found — check manually "
                  f"(look for [^a-z0-9...] and add {CYR})")
    return src, done


def patch_lexicon(src: str, fname: str) -> tuple[str, bool]:
    # idempotent: strip any previously injected block first (it may sit in
    # the WRONG place — e.g. before POSITIVE_WORDS is defined → NameError)
    src = re.sub(
        r"\n?# ════════ MULTILINGUAL LEXICON.*?═════════════════════════"
        r"═════════════════════════════════════════\n?",
        "\n", src, flags=re.S)

    # insert AFTER the LAST of the three set definitions, whatever their order
    last_end = -1
    for name in ("POSITIVE_WORDS", "NEGATIVE_WORDS", "NEGATION_WORDS"):
        for m in re.finditer(name + r"\s*=\s*(frozenset\()?\{[^}]*\}\)?", src, re.S):
            last_end = max(last_end, m.end())
    if last_end < 0:
        print(f"  ✗ {fname}: lexicon sets not found — lexicon NOT added")
        return src, False
    src = src[:last_end] + "\n" + LEXICON_BLOCK + src[last_end:]
    print(f"  ✓ {fname}: RU/KZ lexicon injected after the LAST set definition")
    return src, True


def patch_loader(src: str, fname: str) -> tuple[str, bool]:
    if "MULTILINGUAL DATA AUTOLOAD" in src:
        print(f"  = {fname}: loader wrapper already present")
        return src, True
    if "def load_datasets" not in src:
        print(f"  ✗ {fname}: load_datasets() not found — wrapper NOT added")
        return src, False
    m = re.search(r'\nif __name__\s*==\s*["\']__main__["\']', src)
    if m:
        pos = m.start()
        src = src[:pos] + "\n" + LOADER_WRAPPER + src[pos:]
    else:
        src = src + "\n" + LOADER_WRAPPER
    print(f"  ✓ {fname}: load_datasets() now auto-merges data/multilingual/*.csv")
    return src, True


def process(fname: str, do_loader: bool):
    if not os.path.exists(fname):
        print(f"  ✗ {fname} not found in this folder — skipped")
        return
    src = open(fname, encoding="utf-8").read()
    bak = fname + ".bak_multilingual"
    if not os.path.exists(bak):
        shutil.copy(fname, bak)
        print(f"  • backup: {bak}")
    src, _ = patch_clean_text(src, fname)
    src, _ = patch_lexicon(src, fname)
    if do_loader:
        src, _ = patch_loader(src, fname)
    # sanity: still valid python?
    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  ✗ {fname}: patch produced a syntax error (line {e.lineno}) — "
              f"file NOT modified, restore from {bak}")
        return
    open(fname, "w", encoding="utf-8").write(src)
    print(f"  ✓ {fname}: saved\n")


if __name__ == "__main__":
    print("\nMULTILINGUAL PATCH (RU/KZ)\n" + "─" * 46)
    print("[1/2] train.py")
    process("train.py", do_loader=True)
    print("[2/2] financial_analysis.py")
    process("financial_analysis.py", do_loader=False)
    print("─" * 46)
    print("""Next steps:
  1. pip install deep-translator
  2. python build_multilingual_data.py --per_class 1200
     (creates data/multilingual/ru.csv and kk.csv, ~30-60 min, cached)
  3. python train.py --mode train --glove_path data/glove.840B.300d.txt
     (multilingual CSVs are picked up automatically)
  4. Test it:
       python predict_demo.py   — add Russian/Kazakh sentences and see
       «Компания показала рекордную прибыль» → positive
       «Компания терпит огромные убытки»     → negative

Note: GloVe embeddings cover English only. Cyrillic words start from
random vectors and learn during training — expect ~2-4% lower accuracy
on RU/KZ than on EN. That is normal and honest to report.""")
