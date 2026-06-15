"""
Financial Sentiment — Prediction Demo
Run: python predict_demo.py
"""

import os
import re
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = set()

NEGATION_WORDS = {
    "not","no","never","neither","nor","cannot","can't","won't","don't",
    "doesn't","didn't","isn't","aren't","wasn't","weren't","hardly","barely",
}
_stop_words -= NEGATION_WORDS

POSITIVE_WORDS = {
    "surge","soar","beat","profit","gain","rally","outperform","upgrade",
    "bullish","boom","record","growth","positive","strong","robust",
    "improvement","exceed","breakout","advance","rise","increase","expand",
    "opportunity","success","recover","rebound","upbeat","optimistic",
    "confident","innovative","dividend","revenue","margin","efficiency",
    "momentum","acceleration","raised","guidance","high",
}
NEGATIVE_WORDS = {
    "drop","fall","loss","decline","miss","downgrade","bearish","crash",
    "slump","weak","disappointing","concern","risk","warn","cut","layoff",
    "restructure","debt","default","recession","inflation","volatility",
    "uncertainty","pressure","crisis","bankruptcy","lawsuit","investigation",
    "fraud","recall","shortfall","deficit","slowdown","headwind","downturn",
    "below","missed","lowered","warning","probe","charges","fine","penalty",
}

PAD, UNK = "<PAD>", "<UNK>"


import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Vocabulary


def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\$[\d,\.]+[bBmMkK]?", " moneyamt ", text)
    text = re.sub(r"[\d,\.]+\s*%", " pctamt ", text)
    text = re.sub(r"\b\d{4}\b", " yeartoken ", text)
    text = re.sub(r"\b\d+[\d,\.]*[bBmMkK]?\b", " numtoken ", text)
    text = re.sub(r"\$[A-Z]{1,5}", " tickertoken ", text)
    text = re.sub(r"[^a-z0-9\s\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [_lemmatizer.lemmatize(t) for t in tokens
            if t not in _stop_words and len(t) > 1]


def lex_features(text):
    words = set(text.lower().split())
    n_pos  = len(words & POSITIVE_WORDS)
    n_neg  = len(words & NEGATIVE_WORDS)
    n_negw = len(words & NEGATION_WORDS)
    total  = max(len(words), 1)
    return np.array([
        n_pos / total,
        n_neg / total,
        n_negw / total,
        (n_pos - n_neg) / total,
        min(n_pos, 5) / 5.0,
        min(n_neg, 5) / 5.0,
    ], dtype=np.float32)




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
        return torch.cat([F.adaptive_max_pool1d(c(x), 1).squeeze(-1)
                          for c in self.convs], dim=-1)


class FinSentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256,
                 num_classes=3, num_layers=2, num_heads=4,
                 cnn_ch=128, dropout=0.0, pretrained_emb=None):
        super().__init__()
        self.emb      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(0.0)
        self.lstm     = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                bidirectional=True, batch_first=True,
                                dropout=0.0)
        lstm_out = hidden_dim * 2
        self.attn_w  = nn.Linear(lstm_out, 1)
        self.cnn     = CNNBlock(embed_dim, cnn_ch)
        cnn_out      = cnn_ch * 4
        self.lex_mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(32, 16), nn.ReLU(),
        )
        total_in   = lstm_out + cnn_out + 16
        self.norm  = nn.LayerNorm(total_in)
        self.clf   = nn.Sequential(
            nn.Linear(total_in, 256), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(256, 64),       nn.GELU(), nn.Dropout(0.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, lex):
        pad_mask = (input_ids == 0)
        emb      = self.emb_drop(self.emb(input_ids))
        lstm_out, _ = self.lstm(emb)
        scores   = self.attn_w(lstm_out).squeeze(-1)
        scores   = scores.masked_fill(pad_mask, -1e9)
        weights  = torch.softmax(scores, dim=-1).unsqueeze(-1)
        lstm_rep = (lstm_out * weights).sum(dim=1)
        cnn_rep  = self.cnn(emb)
        lex_rep  = self.lex_mlp(lex)
        fused    = torch.cat([lstm_rep, cnn_rep, lex_rep], dim=-1)
        fused    = self.norm(fused)
        return self.clf(fused)



class Predictor:
    LABEL_COLORS = {
        "positive": "\033[92m",   
        "neutral":  "\033[93m",   
        "negative": "\033[91m",   
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def __init__(self, model_dir="outputs"):
        print(f"Loading model from '{model_dir}' ...")

        with open(os.path.join(model_dir, "vocab.pkl"), "rb") as f:
            self.vocab = pickle.load(f)

        with open(os.path.join(model_dir, "label_map.json")) as f:
            self.label_map = json.load(f)

        ckpt = torch.load(os.path.join(model_dir, "best_model.pt"),
                          map_location="cpu")
        a = argparse.Namespace(**ckpt["args"])
        self.max_len = a.max_len

        self.model = FinSentModel(
            vocab_size  = len(self.vocab.word2idx),
            embed_dim   = a.embed_dim,
            hidden_dim  = a.hidden_dim,
            num_classes = 3,
            num_layers  = a.num_layers,
            cnn_ch      = a.cnn_channels,
            dropout     = 0.0,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        val_acc = ckpt.get("val_acc", "?")
        val_f1  = ckpt.get("val_f1",  "?")
        print(f"Model loaded! Best checkpoint — Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")

    @torch.no_grad()
    def predict_one(self, text: str) -> Dict:
        cleaned = clean_text(text)
        toks    = tokenize(cleaned)
        pad_id  = self.vocab.word2idx[PAD]
        ids     = self.vocab.encode(toks)
        if len(ids) < self.max_len:
            ids = ids + [pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        inp    = torch.tensor([ids],           dtype=torch.long)
        lex    = torch.tensor([lex_features(cleaned)], dtype=torch.float)
        logits = self.model(inp, lex)
        probs  = F.softmax(logits, dim=-1).squeeze().numpy()
        pred   = int(probs.argmax())

        return {
            "label":      self.label_map[str(pred)],
            "confidence": float(probs[pred]),
            "negative":   float(probs[0]),
            "neutral":    float(probs[1]),
            "positive":   float(probs[2]),
        }

    def bar(self, value: float, width: int = 20) -> str:
        filled = int(round(value * width))
        return "█" * filled + "░" * (width - filled)

    def display(self, text: str, result: Dict):
        label = result["label"]
        color = self.LABEL_COLORS.get(label, "")
        conf  = result["confidence"]

        print("─" * 65)
        print(f"  {self.BOLD}Text:{self.RESET} {text}")
        print(f"  {self.BOLD}Sentiment:{self.RESET} {color}{label.upper()}{self.RESET} "
              f"({conf:.1%} confidence)")
        print(f"")
        print(f"  \033[91mnegative\033[0m  {self.bar(result['negative'])}  {result['negative']:.1%}")
        print(f"  \033[93mneutral \033[0m  {self.bar(result['neutral'])}   {result['neutral']:.1%}")
        print(f"  \033[92mpositive\033[0m  {self.bar(result['positive'])}  {result['positive']:.1%}")
        print()

    def run(self, texts: List[str]):
        print("=" * 65)
        print("       FINANCIAL SENTIMENT ANALYSIS — DEMO")
        print("=" * 65 + "\n")
        for text in texts:
            result = self.predict_one(text)
            self.display(text, result)
        print("─" * 65)



EXAMPLES = [
    # Clearly POSITIVE
    "Apple reported record quarterly earnings, beating analyst expectations by 18%.",
    "The company raised its full-year guidance after strong revenue growth.",
    "Tesla shares surged 12% after delivery numbers exceeded forecasts.",
    "Goldman Sachs upgraded the stock to Buy with a price target of $200.",
    "The acquisition deal is expected to significantly boost profit margins.",

    # Clearly NEGATIVE
    "The company missed earnings expectations and cut its annual forecast.",
    "Shares crashed 25% after the CEO resigned amid a fraud investigation.",
    "Rising inflation and supply chain disruptions weighed heavily on revenue.",
    "The bank reported a massive loss due to bad loans and credit defaults.",
    "Regulators fined the firm $500 million for compliance violations.",

    # NEUTRAL / MIXED
    "The Federal Reserve held interest rates steady at its latest meeting.",
    "The company announced it will release quarterly results next Tuesday.",
    "Management said they are monitoring the situation closely.",
    "The merger is subject to regulatory approval and is expected to close in Q3.",
    "Revenue was flat compared to last year, in line with analyst estimates.",

    # negation
    "The company did not miss its earnings target this quarter.",
    "There were no signs of financial distress in the latest report.",
    "Analysts do not expect the stock to fall further from current levels.",

    # mixed signals
    "Despite strong revenue growth, profits declined due to rising costs.",
    "The stock fell initially but recovered after the earnings call.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory with best_model.pt and vocab.pkl")
    parser.add_argument("--texts", nargs="+",
                        help="Custom sentences to analyze (optional)")
    args = parser.parse_args()

    predictor = Predictor(model_dir=args.output_dir)

    sentences = args.texts if args.texts else EXAMPLES
    predictor.run(sentences)
