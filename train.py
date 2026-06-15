"""
Financial Sentiment Analysis - Custom Deep Learning Model
Target: 85%+ accuracy
Architecture: BiLSTM + CNN + Multi-Head Attention Ensemble
Embeddings: GloVe 300d (pretrained, no pretrained transformer models)
Datasets: 6-8 financial sentiment datasets combined
"""

import os
import re
import json
import random
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")


POSITIVE_WORDS = {
    "surge","soar","beat","profit","gain","rally","outperform","upgrade",
    "bullish","boom","record","growth","positive","strong","robust",
    "improvement","exceed","breakout","advance","rise","increase","expand",
    "opportunity","success","recover","rebound","upbeat","optimistic",
    "confident","innovative","dividend","revenue","margin","efficiency",
    "momentum","acceleration","beat","raised","guidance","record","high",
}
NEGATIVE_WORDS = {
    "drop","fall","loss","decline","miss","downgrade","bearish","crash",
    "slump","weak","disappointing","concern","risk","warn","cut","layoff",
    "restructure","debt","default","recession","inflation","volatility",
    "uncertainty","pressure","crisis","bankruptcy","lawsuit","investigation",
    "fraud","recall","shortfall","deficit","slowdown","headwind","downturn",
    "below","missed","lowered","warning","probe","charges","fine","penalty",
}
NEGATION_WORDS = {
    "not","no","never","neither","nor","cannot","can't","won't","don't",
    "doesn't","didn't","isn't","aren't","wasn't","weren't","hardly",
    "barely","scarcely",
}

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





_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words("english")) - NEGATION_WORDS
except Exception:
    _stop_words = set()

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\$[\d,\.]+[bBmMkK]?", " moneyamt ", text)
    text = re.sub(r"[\d,\.]+\s*%", " pctamt ", text)
    text = re.sub(r"\b\d{4}\b", " yeartoken ", text)
    text = re.sub(r"\b\d+[\d,\.]*[bBmMkK]?\b", " numtoken ", text)
    text = re.sub(r"\$[A-Z]{1,5}", " tickertoken ", text)
    text = re.sub(r"[^a-zа-яёәғқңөұүһі0-9\s\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t) > 1]

def lex_features(text: str) -> np.ndarray:
    words = set(text.lower().split())
    n_pos = len(words & POSITIVE_WORDS)
    n_neg = len(words & NEGATIVE_WORDS)
    n_neg_w = len(words & NEGATION_WORDS)
    total = max(len(words), 1)
    return np.array([
        n_pos / total,
        n_neg / total,
        n_neg_w / total,
        (n_pos - n_neg) / total,
        min(n_pos, 5) / 5.0,
        min(n_neg, 5) / 5.0,
    ], dtype=np.float32)


PAD, UNK = "<PAD>", "<UNK>"

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build(self, token_lists):
        counter = Counter(t for tokens in token_lists for t in tokens)
        for special in [PAD, UNK]:
            idx = len(self.word2idx)
            self.word2idx[special] = idx
            self.idx2word[idx] = special
        for word, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        log.info(f"Vocabulary size: {len(self.word2idx):,}")

    def encode(self, tokens):
        unk = self.word2idx[UNK]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def load_glove(path, vocab, embed_dim=300):
    log.info(f"Loading GloVe from {path} ...")
    glove = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            # Skip vectors with NaN/Inf
            if not np.isfinite(vec).all():
                continue
            glove[word] = vec

    
    scale = 0.05
    matrix = np.random.uniform(-scale, scale, (len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab.word2idx[PAD]] = 0.0

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1

    
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms * np.sqrt(embed_dim / 300.0) * 2.0

    
    matrix[vocab.word2idx[PAD]] = 0.0

    log.info(f"GloVe coverage: {found}/{len(vocab)} = {found/len(vocab):.1%}")
    return matrix


class FinSentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.labels = labels
        self.max_len = max_len
        pad_id = vocab.word2idx[PAD]
        self.ids = []
        self.lex = []
        for text in texts:
            cleaned = clean_text(text)
            toks = tokenize(cleaned)
            ids = vocab.encode(toks)
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            self.ids.append(ids)
            self.lex.append(lex_features(cleaned))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "lex":       torch.tensor(self.lex[i],  dtype=torch.float),
            "label":     torch.tensor(self.labels[i], dtype=torch.long),
        }


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k//2),
                nn.GELU(),
                nn.BatchNorm1d(out_ch),
            ) for k in [2, 3, 4, 5]
        ])

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        return torch.cat([F.adaptive_max_pool1d(c(x), 1).squeeze(-1) for c in self.convs], dim=-1)


class FinSentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256,
                 num_classes=3, num_layers=2, num_heads=4,
                 cnn_ch=128, dropout=0.3, pretrained_emb=None):
        super().__init__()

        
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.emb.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.emb_drop = nn.Dropout(0.2)

        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        lstm_out = hidden_dim * 2

       
        self.attn_w = nn.Linear(lstm_out, 1)

        
        self.cnn = CNNBlock(embed_dim, cnn_ch)
        cnn_out = cnn_ch * 4

        
        self.lex_mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
        )

       
        total_in = lstm_out + cnn_out + 16
        self.norm = nn.LayerNorm(total_in)

       
        self.clf = nn.Sequential(
            nn.Linear(total_in, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),  nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "emb" in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, input_ids, lex):
        pad_mask = (input_ids == 0)
        emb = self.emb_drop(self.emb(input_ids))          

        
        lstm_out, _ = self.lstm(emb)                      

        
        scores = self.attn_w(lstm_out).squeeze(-1)         
        scores = scores.masked_fill(pad_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  
        lstm_rep = (lstm_out * weights).sum(dim=1)         

        
        cnn_rep = self.cnn(emb)                         

        
        lex_rep = self.lex_mlp(lex)                     

       
        fused = torch.cat([lstm_rep, cnn_rep, lex_rep], dim=-1)
        fused = self.norm(fused)
        return self.clf(fused)


def load_datasets(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    from datasets import load_dataset
    all_dfs = {}

    def try_load(name, loader_fn):
        try:
            df = loader_fn()
            if df is not None and len(df) > 0:
                all_dfs[name] = df
                log.info(f"  ✓ {name}: {len(df):,} samples")
        except Exception as e:
            log.warning(f"  ✗ {name}: {e}")

    
    def load_phrasebank():
        import urllib.request, csv, io
        urls = [
            ("phrasebank_all", "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/sentences_allagree.txt"),
            ("phrasebank_75",  "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/sentences_75agree.txt"),
        ]
        frames = []
        for tag, url in urls:
            try:
                req = urllib.request.urlopen(url, timeout=15)
                content = req.read().decode("latin-1")
                rows = []
                for line in content.strip().split("\n"):
                    if "@" in line:
                        parts = line.rsplit("@", 1)
                        if len(parts) == 2:
                            text, sent = parts[0].strip(), parts[1].strip()
                            lmap = {"negative": 0, "neutral": 1, "positive": 2}
                            if sent in lmap:
                                rows.append({"text": text, "sentiment": lmap[sent]})
                if rows:
                    frames.append(pd.DataFrame(rows))
                    log.info(f"    phrasebank {tag}: {len(rows)}")
            except Exception as e:
                log.warning(f"    phrasebank {tag} direct failed: {e}")
        if frames:
            return pd.concat(frames, ignore_index=True)
        return None

    
    try:
        df = load_phrasebank()
        if df is not None and len(df) > 0:
            all_dfs["phrasebank"] = df
            log.info(f"  ✓ phrasebank: {len(df):,} samples")
        else:
            raise ValueError("empty")
    except Exception:
        
        try:
            ds = load_dataset("Rowan/financial_phrasebank_75agreed_split")
            frames = [pd.DataFrame(ds[s]) for s in ds]
            df = pd.concat(frames, ignore_index=True)
            df = df.rename(columns={"sentence": "text", "label": "sentiment"})
            df["sentiment"] = df["sentiment"].map({0: 0, 1: 1, 2: 2})
            all_dfs["phrasebank"] = df[["text", "sentiment"]].dropna()
            log.info(f"  ✓ phrasebank (fallback): {len(df):,} samples")
        except Exception as e2:
            log.warning(f"  ✗ phrasebank all methods failed: {e2}")

    
    def load_fiqa():
        ds = load_dataset("pauri32/fiqa-2018")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        if "sentiment_score" in df.columns:
            df["sentiment"] = df["sentiment_score"].apply(
                lambda s: 0 if s < -0.05 else (2 if s > 0.05 else 1))
            text_col = next((c for c in ["sentence","question","text"] if c in df.columns), None)
            if text_col:
                return df.rename(columns={text_col: "text"})[["text","sentiment"]].dropna()
        return None
    try_load("fiqa", load_fiqa)

    
    def load_twitter():
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        df["sentiment"] = df["label"].map({0: 0, 1: 2, 2: 1})
        return df[["text","sentiment"]].dropna()
    try_load("twitter", load_twitter)

  
    def load_stocktweets():
        ds = load_dataset("mjw/stock_market_tweets")
        df = pd.DataFrame(ds["train"])
        
        if len(df) > 20000:
            df = df.sample(20000, random_state=SEED)
        if "label" not in df.columns:
            return None
        df["sentiment"] = df["label"].map({
            "bullish": 2, "bearish": 0,
            "positive": 2, "negative": 0, "neutral": 1,
            0: 0, 1: 2,
        })
        text_col = next((c for c in ["text","tweet","body"] if c in df.columns), None)
        if text_col:
            return df.rename(columns={text_col: "text"})[["text","sentiment"]].dropna()
        return None
    try_load("stocktweets", load_stocktweets)

    
    def load_kaggle():
        ds = load_dataset("chiapudding/kaggle-financial-sentiment")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        text_col = next((c for c in ["Sentence","sentence","text"] if c in df.columns), None)
        sent_col = next((c for c in ["Sentiment","sentiment","label"] if c in df.columns), None)
        if text_col and sent_col:
            df = df.rename(columns={text_col:"text", sent_col:"sentiment"})
            df["sentiment"] = df["sentiment"].astype(str).str.lower().map(
                {"negative":0,"neutral":1,"positive":2})
            return df[["text","sentiment"]].dropna()
        return None
    try_load("kaggle_fin", load_kaggle)

  
    def load_auditor():
        ds = load_dataset("FinanceInc/auditor_sentiment")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        text_col = next((c for c in ["sentence","text"] if c in df.columns), None)
        sent_col = next((c for c in ["label","sentiment"] if c in df.columns), None)
        if text_col and sent_col:
            df = df.rename(columns={text_col:"text", sent_col:"sentiment"})
            df["sentiment"] = df["sentiment"].map(
                {"negative":0,"neutral":1,"positive":2, 0:0, 1:1, 2:2})
            return df[["text","sentiment"]].dropna()
        return None
    try_load("auditor", load_auditor)

   
    def load_financial_news():
        ds = load_dataset("Sigma/financial-news-sentiment-reuters")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        text_col = next((c for c in ["headline","text","title","sentence"] if c in df.columns), None)
        sent_col = next((c for c in ["sentiment","label"] if c in df.columns), None)
        if text_col and sent_col:
            df = df.rename(columns={text_col:"text", sent_col:"sentiment"})
            if df["sentiment"].dtype == object:
                df["sentiment"] = df["sentiment"].str.lower().map(
                    {"negative":0,"neutral":1,"positive":2,"bearish":0,"bullish":2})
            else:
                df["sentiment"] = df["sentiment"].map({0:0,1:1,2:2,-1:0})
            return df[["text","sentiment"]].dropna()
        return None
    try_load("reuters_news", load_financial_news)

    def load_flue():
        ds = load_dataset("Yiyanghkust/finbert-tone")
        frames = [pd.DataFrame(ds[s]) for s in ds]
        df = pd.concat(frames, ignore_index=True)
        text_col = next((c for c in ["text","sentence"] if c in df.columns), None)
        sent_col = next((c for c in ["label","sentiment"] if c in df.columns), None)
        if text_col and sent_col:
            df = df.rename(columns={text_col:"text", sent_col:"sentiment"})
            df["sentiment"] = df["sentiment"].astype(str).str.lower().map(
                {"negative":0,"neutral":1,"positive":2,"0":0,"1":1,"2":2})
            return df[["text","sentiment"]].dropna()
        return None
    try_load("finbert_tone", load_flue)

    if not all_dfs:
        raise RuntimeError("No datasets loaded!")
    return all_dfs


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
        self.weight = weight

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        with torch.no_grad():
            smooth = torch.full_like(logits, smooth_val)
            smooth.scatter_(1, targets.unsqueeze(1), confidence)
        log_prob = F.log_softmax(logits, dim=-1)
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = -(smooth * log_prob * w).sum(dim=-1).mean()
        else:
            loss = -(smooth * log_prob).sum(dim=-1).mean()
        return loss


def augment(text, label):
    cleaned = clean_text(text)
    toks = tokenize(cleaned)
    if not toks:
        return cleaned, label
    r = random.random()
    if r < 0.33:
        
        toks = [t for t in toks if random.random() > 0.1]
    elif r < 0.66:
        
        if len(toks) >= 2:
            i = random.randint(0, len(toks)-2)
            toks[i], toks[i+1] = toks[i+1], toks[i]
    else:
    
        keep = max(1, int(len(toks) * random.uniform(0.7, 1.0)))
        toks = toks[:keep]
    return " ".join(toks), label


def compute_weights(labels, n_cls=3):
    counts = Counter(labels)
    total = len(labels)
    w = torch.tensor([total / (n_cls * max(counts.get(i,1),1)) for i in range(n_cls)], dtype=torch.float)
    return w / w.sum() * n_cls


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(DEVICE)
            lex   = batch["lex"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            out   = model(ids, lex)
            preds.extend(out.argmax(-1).cpu().numpy())
            trues.extend(label.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    rep = classification_report(trues, preds,
          target_names=["negative","neutral","positive"], zero_division=0)
    return acc, f1, rep


def train(args):
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading datasets ...")
    dataset_dict = load_datasets(args.data_dir)

    combined = pd.concat(dataset_dict.values(), ignore_index=True)
    combined = combined.dropna(subset=["text","sentiment"])
    combined["sentiment"] = combined["sentiment"].astype(int)
    combined = combined[combined["sentiment"].isin([0,1,2])]
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    log.info(f"Total samples: {len(combined):,}")
    log.info(f"Class distribution:\n{combined['sentiment'].value_counts()}")

    texts  = combined["text"].tolist()
    labels = combined["sentiment"].tolist()


    counts  = Counter(labels)
    max_cnt = max(counts.values())
    aug_t, aug_l = list(texts), list(labels)
    for cls, cnt in counts.items():
        deficit = max_cnt - cnt
        if deficit <= 0:
            continue
        pool = [t for t,l in zip(texts,labels) if l==cls]
        for _ in range(deficit):
            t, l = augment(random.choice(pool), cls)
            aug_t.append(t)
            aug_l.append(l)
    log.info(f"After augmentation: {len(aug_t):,}")

    
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(aug_t, aug_l, test_size=0.2,
                                                  stratify=aug_l, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                  stratify=y_tmp, random_state=SEED)
    log.info(f"Train {len(X_tr)} | Val {len(X_val)} | Test {len(X_te)}")

    
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build([tokenize(clean_text(t)) for t in X_tr])
    vocab.save(os.path.join(args.output_dir, "vocab.pkl"))

    
    pretrained = None
    if args.glove_path and os.path.exists(args.glove_path):
        pretrained = load_glove(args.glove_path, vocab, args.embed_dim)
    else:
        log.warning("GloVe not found — using random init")


    tr_ds  = FinSentDataset(X_tr,  y_tr,  vocab, args.max_len)
    val_ds = FinSentDataset(X_val, y_val, vocab, args.max_len)
    te_ds  = FinSentDataset(X_te,  y_te,  vocab, args.max_len)

    cw = compute_weights(y_tr)
    sw = torch.tensor([cw[l].item() for l in y_tr])
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_ds,  batch_size=args.batch_size*2, shuffle=False, num_workers=0)

    
    model = FinSentModel(
        vocab_size   = len(vocab),
        embed_dim    = args.embed_dim,
        hidden_dim   = args.hidden_dim,
        num_classes  = 3,
        num_layers   = args.num_layers,
        num_heads    = 4,
        cnn_ch       = args.cnn_channels,
        dropout      = args.dropout,
        pretrained_emb = pretrained,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,}")

    
    criterion = LabelSmoothingLoss(classes=3, smoothing=0.1, weight=cw.to(DEVICE))

    
    emb_params   = list(model.emb.parameters())
    other_params = [p for n,p in model.named_parameters() if "emb" not in n]
    lr_emb = args.lr * 0.05 if pretrained is not None else args.lr

    optimizer = AdamW([
        {"params": emb_params,   "lr": lr_emb,  "weight_decay": 0.0},
        {"params": other_params, "lr": args.lr, "weight_decay": args.wd},
    ], eps=1e-8)

    scheduler = OneCycleLR(optimizer,
        max_lr=[lr_emb, args.lr],
        steps_per_epoch=len(tr_loader),
        epochs=args.epochs,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_f1, patience_cnt = 0.0, 0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss, n_batch = 0.0, 0

        for batch in tr_loader:
            ids   = batch["input_ids"].to(DEVICE)
            lex   = batch["lex"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, lex)

            
            if torch.isnan(logits).any():
                log.warning("NaN in logits — skipping batch")
                continue

            loss = criterion(logits, label)

            if torch.isnan(loss):
                log.warning("NaN loss — skipping batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item()
            n_batch  += 1

        if n_batch == 0:
            log.error("All batches had NaN — stopping")
            break

        avg_loss = tot_loss / n_batch
        val_acc, val_f1, _ = evaluate(model, val_loader)

        log.info(f"Epoch {epoch:3d}/{args.epochs} | "
                 f"Loss {avg_loss:.4f} | "
                 f"Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

        history.append({"epoch":epoch,"loss":avg_loss,"val_acc":val_acc,"val_f1":val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                        "val_acc":val_acc,"val_f1":val_f1,"args":vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            log.info(f"  ✓ Saved best (F1={val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    
    ckpt = torch.load(os.path.join(args.output_dir,"best_model.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    te_acc, te_f1, te_rep = evaluate(model, te_loader)

    log.info(f"\n{'='*60}")
    log.info(f"TEST ACCURACY : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    log.info(f"TEST MACRO F1 : {te_f1:.4f}")
    log.info(f"\n{te_rep}")
    log.info("="*60)

    with open(os.path.join(args.output_dir,"history.json"),"w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir,"label_map.json"),"w") as f:
        json.dump({"0":"negative","1":"neutral","2":"positive"}, f)

    return te_acc, te_f1


class Predictor:
    def __init__(self, model_dir):
        self.vocab = Vocabulary.load(os.path.join(model_dir,"vocab.pkl"))
        with open(os.path.join(model_dir,"label_map.json")) as f:
            self.label_map = json.load(f)
        ckpt = torch.load(os.path.join(model_dir,"best_model.pt"), map_location="cpu")
        a = argparse.Namespace(**ckpt["args"])
        self.model = FinSentModel(len(self.vocab), a.embed_dim, a.hidden_dim,
                                   3, a.num_layers, 4, a.cnn_channels, 0.0)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.max_len = a.max_len

    @torch.no_grad()
    def predict(self, texts):
        results = []
        pad_id = self.vocab.word2idx[PAD]
        for text in texts:
            cleaned = clean_text(text)
            toks = tokenize(cleaned)
            ids  = self.vocab.encode(toks)
            if len(ids) < self.max_len:
                ids = ids + [pad_id]*(self.max_len - len(ids))
            else:
                ids = ids[:self.max_len]
            inp  = torch.tensor([ids], dtype=torch.long)
            lex  = torch.tensor([lex_features(cleaned)], dtype=torch.float)
            logits = self.model(inp, lex)
            probs  = F.softmax(logits, dim=-1).squeeze().numpy()
            pred   = int(probs.argmax())
            results.append({
                "text": text,
                "label": self.label_map[str(pred)],
                "confidence": float(probs[pred]),
                "probs": {"negative":float(probs[0]),"neutral":float(probs[1]),"positive":float(probs[2])},
            })
        return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--glove_path",  default="data/glove.840B.300d.txt")
    p.add_argument("--embed_dim",   type=int,   default=300)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--cnn_channels",type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--wd",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--max_len",     type=int,   default=100)
    p.add_argument("--min_freq",    type=int,   default=2)
    p.add_argument("--mode",        default="train", choices=["train","predict"])
    p.add_argument("--texts",       nargs="+")
    return p.parse_args()


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

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        p = Predictor(args.output_dir)
        texts = args.texts or [
            "The company reported record profits, beating expectations by 15%.",
            "Revenue declined sharply amid rising costs and supply chain disruptions.",
            "The board declared a quarterly dividend of $0.50 per share.",
        ]
        for r in p.predict(texts):
            print(f"\n{r['text']}")
            print(f"  → {r['label'].upper()} ({r['confidence']:.2%})")
