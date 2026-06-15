"""
MULTILINGUAL DATA BUILDER — Russian + Kazakh financial sentiment
================================================================
"The first financial sentiment engine for Central Asian markets."

What it does:
  1. Loads the same 6-8 English datasets train.py uses (via its loader)
  2. Machine-translates a balanced sample to Russian and Kazakh
     (deep-translator / Google — free, rate-limited, so we cache every line)
  3. Saves data/multilingual/ru.csv and kk.csv  (text, sentiment)
  4. train.py --multilingual then merges them into training

Run:
  pip install deep-translator
  python build_multilingual_data.py --per_class 1200          # ~7.2k lines
  python build_multilingual_data.py --per_class 300 --quick   # smoke test

Then retrain:
  python train.py --mode train --multilingual --glove_path data/glove.840B.300d.txt
"""

import os, csv, time, json, random, argparse
import pandas as pd

random.seed(42)
OUT_DIR   = "data/multilingual"
CACHE_F   = os.path.join(OUT_DIR, "_translation_cache.json")
LANGS     = {"ru": "russian", "kk": "kazakh"}

os.makedirs(OUT_DIR, exist_ok=True)

def load_cache():
    """Read cache tolerantly: utf-8 → cp1251 → give up and start fresh."""
    if not os.path.exists(CACHE_F):
        return {}
    for enc in ("utf-8", "cp1251"):
        try:
            return json.load(open(CACHE_F, encoding=enc))
        except Exception:
            continue
    print("  ! translation cache is corrupt (crashed mid-write) — starting fresh")
    try: os.remove(CACHE_F)
    except OSError: pass
    return {}

def save_cache(cache):
    """Atomic utf-8 write: temp file + replace, so a crash never corrupts it."""
    tmp = CACHE_F + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_F)

def get_translator():
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        os.system("pip install deep-translator -q")
        from deep_translator import GoogleTranslator
    return GoogleTranslator

def load_english_pool():
    """Reuse train.py's dataset loader — same data, zero duplication."""
    from train import load_datasets
    dfs = load_datasets("data")
    df = pd.concat(dfs.values(), ignore_index=True)
    df = df.dropna(subset=["text","sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    df = df[df["sentiment"].isin([0,1,2])]
    df = df.drop_duplicates(subset=["text"])
    # keep sentences short enough to translate cleanly
    df = df[df["text"].str.len().between(20, 240)].reset_index(drop=True)
    return df

def balanced_sample(df, per_class):
    parts = []
    for c in (0,1,2):
        sub = df[df.sentiment==c]
        parts.append(sub.sample(min(per_class, len(sub)), random_state=42))
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

def translate_all(sample, lang, GT, cache):
    out, todo = [], []
    for _, r in sample.iterrows():
        key = f"{lang}::{r.text}"
        if key in cache:
            out.append({"text": cache[key], "sentiment": int(r.sentiment)})
        else:
            todo.append(r)
    print(f"  {lang}: {len(out)} cached, {len(todo)} to translate")

    tr = GT(source="en", target=lang)
    BATCH = 8
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        for r in chunk:
            try:
                t = tr.translate(r.text)
                if t and t.strip():
                    cache[f"{lang}::{r.text}"] = t.strip()
                    out.append({"text": t.strip(), "sentiment": int(r.sentiment)})
            except Exception as e:
                # rate-limit / transient error → wait and continue, never crash
                time.sleep(4)
        if i % 80 == 0:
            save_cache(cache)
            print(f"    {lang}: {min(i+BATCH,len(todo))}/{len(todo)}")
        time.sleep(0.7)        # stay polite with the free endpoint
    save_cache(cache)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_class", type=int, default=1200,
                   help="EN samples per class to translate (×3 classes ×2 langs)")
    p.add_argument("--langs", nargs="+", default=["ru","kk"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick: args.per_class = min(args.per_class, 300)

    print("[1/3] Loading English pool via train.load_datasets() ...")
    df = load_english_pool()
    print(f"      pool: {len(df):,} sentences")

    sample = balanced_sample(df, args.per_class)
    print(f"[2/3] Balanced sample: {len(sample):,} "
          f"({args.per_class}/class) → translating to {args.langs}")

    GT = get_translator()
    cache = load_cache()

    for lang in args.langs:
        rows = translate_all(sample, lang, GT, cache)
        out_path = os.path.join(OUT_DIR, f"{lang}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        print(f"      ✓ {out_path}: {len(rows):,} rows")

    print("[3/3] Done. Retrain with:")
    print("      python train.py --mode train --multilingual "
          "--glove_path data/glove.840B.300d.txt")
    print("\nNote: GloVe covers English only — Cyrillic tokens get trainable")
    print("random-init embeddings and learn from the data itself.")

if __name__ == "__main__":
    main()
