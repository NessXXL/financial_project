"""
Financial Analysis System
Combines: Sentiment Analysis (your trained model) + 35 Financial Ratios
Data sources: yfinance (FREE, no API key needed) + FMP API (optional)
Usage: python financial_analysis.py --ticker NVDA
       python financial_analysis.py --ticker AAPL
       python financial_analysis.py --ticker TSLA
"""

import os, re, sys, json, pickle, argparse, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional, Tuple


import nltk
for pkg in ["punkt","punkt_tab","stopwords","wordnet"]:
    try: nltk.download(pkg, quiet=True)
    except: pass
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
try:    _stop_words = set(stopwords.words("english"))
except: _stop_words = set()
NEGATION_WORDS = {"not","no","never","neither","nor","cannot","can't","won't",
                  "don't","doesn't","didn't","isn't","aren't","wasn't","weren't","hardly","barely"}


_stop_words -= NEGATION_WORDS
POSITIVE_WORDS = {"surge","soar","beat","profit","gain","rally","outperform","upgrade","bullish",
    "boom","record","growth","positive","strong","robust","improvement","exceed","breakout",
    "advance","rise","increase","expand","opportunity","success","recover","rebound","upbeat",
    "optimistic","confident","innovative","dividend","revenue","margin","efficiency","momentum",
    "acceleration","raised","guidance","high"}
NEGATIVE_WORDS = {"drop","fall","loss","decline","miss","downgrade","bearish","crash","slump",
    "weak","disappointing","concern","risk","warn","cut","layoff","restructure","debt","default",
    "recession","inflation","volatility","uncertainty","pressure","crisis","bankruptcy","lawsuit",
    "investigation","fraud","recall","shortfall","deficit","slowdown","headwind","downturn",
    "below","missed","lowered","warning","probe","charges","fine","penalty"}

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


PAD, UNK = "<PAD>", "<UNK>"

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\$[\d,\.]+[bBmMkK]?", " moneyamt ", text)
    text = re.sub(r"[\d,\.]+\s*%", " pctamt ", text)
    text = re.sub(r"\b\d{4}\b", " yeartoken ", text)
    text = re.sub(r"\b\d+[\d,\.]*[bBmMkK]?\b", " numtoken ", text)
    text = re.sub(r"\$[A-Z]{1,5}", " tickertoken ", text)
    text = re.sub(r"[^a-zа-яёәғқңөұүһі0-9\s\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    try:    tokens = word_tokenize(text)
    except: tokens = text.split()
    return [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t)>1]

def lex_features(text):
    words = set(text.lower().split())
    n_pos=len(words&POSITIVE_WORDS); n_neg=len(words&NEGATIVE_WORDS); n_negw=len(words&NEGATION_WORDS)
    total=max(len(words),1)
    return np.array([n_pos/total, n_neg/total, n_negw/total,
                     (n_pos-n_neg)/total, min(n_pos,5)/5.0, min(n_neg,5)/5.0], dtype=np.float32)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Vocabulary

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch,out_ch,k,padding=k//2),nn.GELU(),nn.BatchNorm1d(out_ch))
            for k in [2,3,4,5]])
    def forward(self, x):
        x = x.permute(0,2,1)
        return torch.cat([F.adaptive_max_pool1d(c(x),1).squeeze(-1) for c in self.convs],dim=-1)

class FinSentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=3,
                 num_layers=2, num_heads=4, cnn_ch=128, dropout=0.0, pretrained_emb=None):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,embed_dim,padding_idx=0); self.emb_drop=nn.Dropout(0.0)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,num_layers=num_layers,bidirectional=True,batch_first=True,dropout=0.0)
        lstm_out=hidden_dim*2; self.attn_w=nn.Linear(lstm_out,1); self.cnn=CNNBlock(embed_dim,cnn_ch)
        self.lex_mlp=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Dropout(0.0),nn.Linear(32,16),nn.ReLU())
        total_in=lstm_out+cnn_ch*4+16; self.norm=nn.LayerNorm(total_in)
        self.clf=nn.Sequential(nn.Linear(total_in,256),nn.GELU(),nn.Dropout(0.0),
                               nn.Linear(256,64),nn.GELU(),nn.Dropout(0.0),nn.Linear(64,num_classes))
    def forward(self, input_ids, lex):
        pad_mask=(input_ids==0); emb=self.emb_drop(self.emb(input_ids))
        lstm_out,_=self.lstm(emb)
        scores=self.attn_w(lstm_out).squeeze(-1).masked_fill(pad_mask,-1e9)
        weights=torch.softmax(scores,dim=-1).unsqueeze(-1)
        lstm_rep=(lstm_out*weights).sum(dim=1)
        fused=torch.cat([lstm_rep,self.cnn(emb),self.lex_mlp(lex)],dim=-1)
        return self.clf(self.norm(fused))


class SentimentEngine:
    def __init__(self, model_dir="outputs"):
        with open(os.path.join(model_dir,"vocab.pkl"),"rb") as f: self.vocab=pickle.load(f)
        with open(os.path.join(model_dir,"label_map.json")) as f: self.label_map=json.load(f)
        ckpt=torch.load(os.path.join(model_dir,"best_model.pt"),map_location="cpu")
        a=argparse.Namespace(**ckpt["args"]); self.max_len=a.max_len
        self.model=FinSentModel(len(self.vocab.word2idx),a.embed_dim,a.hidden_dim,3,a.num_layers,4,a.cnn_channels,0.0)
        self.model.load_state_dict(ckpt["state_dict"]); self.model.eval()
        self.val_acc=ckpt.get("val_acc",0)

    @torch.no_grad()
    def predict(self, text):
        cleaned=clean_text(text); toks=tokenize(cleaned)
        pad_id=self.vocab.word2idx[PAD]; ids=self.vocab.encode(toks)
        ids=(ids+[pad_id]*(self.max_len-len(ids)))[:self.max_len] if len(ids)<self.max_len else ids[:self.max_len]
        probs=F.softmax(self.model(torch.tensor([ids],dtype=torch.long),
                                   torch.tensor([lex_features(cleaned)],dtype=torch.float)),dim=-1).squeeze().numpy()
        pred=int(probs.argmax())
        return {"label":self.label_map[str(pred)],"confidence":float(probs[pred]),
                "negative":float(probs[0]),"neutral":float(probs[1]),"positive":float(probs[2])}

    def analyze_headlines(self, headlines):
        empty={"overall":"neutral","score":0.0,"breakdown":{"positive":0,"neutral":0,"negative":0},"details":[]}
        if not headlines: return empty
        results=[self.predict(h) for h in headlines if h.strip()]
        if not results: return empty
        pos=sum(1 for r in results if r["label"]=="positive")
        neg=sum(1 for r in results if r["label"]=="negative")
        neu=sum(1 for r in results if r["label"]=="neutral")
        score=sum(r["positive"]-r["negative"] for r in results)/len(results)
        overall="positive" if score>0.15 else ("negative" if score<-0.15 else "neutral")
        return {"overall":overall,"score":round(score,4),
                "breakdown":{"positive":pos,"neutral":neu,"negative":neg},
                "details":[{"headline":h,**r} for h,r in zip(headlines,results)]}


def safe(val, default=None):
    if val is None: return default
    try:
        import math
        f=float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except: return default

def sdiv(a, b, default=None):
    a,b=safe(a),safe(b)
    if a is None or b is None or b==0: return default
    return a/b

def smul(a, b):
    """Safe multiply — returns None if either is None."""
    a,b=safe(a),safe(b)
    if a is None or b is None: return None
    return a*b

def pct(val):
    """Safe *100 for percentages."""
    v=safe(val)
    return None if v is None else v*100


class DataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key

    
    def fetch_yfinance(self, ticker: str):
        try:
            import yfinance as yf
        except ImportError:
            print("  Installing yfinance ..."); os.system("pip install yfinance -q")
            import yfinance as yf

        print("  Fetching via yfinance (free) ...")
        stk  = yf.Ticker(ticker)
        info = stk.info or {}

        
        try:
            news_raw  = stk.news or []
            headlines = []
            for n in news_raw[:30]:
                
                title = (n.get("content") or {}).get("title") or n.get("title","")
                if title and title.strip():
                    headlines.append(title.strip())
        except:
            headlines = []

        
        profile = {
            "companyName": info.get("longName", ticker),
            "sector":      info.get("sector",""),
            "industry":    info.get("industry",""),
            "description": info.get("longBusinessSummary","")[:120]+"..." if info.get("longBusinessSummary") else "",
        }
        quote = {
            "price":             info.get("currentPrice") or info.get("regularMarketPrice"),
            "changesPercentage": info.get("regularMarketChangePercent",0),
            "marketCap":         info.get("marketCap"),
        }

        
        fin = {
            # Income
            "revenue":               info.get("totalRevenue"),
            "grossProfit":           info.get("grossProfits"),
            "ebit":                  info.get("ebit"),
            "ebitda":                info.get("ebitda"),
            "netIncome":             info.get("netIncomeToCommon"),
            "interestExpense":       info.get("interestExpense"),
            # Balance
            "totalCurrentAssets":    info.get("totalCurrentAssets"),
            "totalCurrentLiabilities": info.get("totalCurrentLiabilities"),
            "cashAndCashEquivalents":info.get("totalCash"),
            "netReceivables":        info.get("netReceivables"),
            "inventory":             info.get("inventory"),
            "totalAssets":           info.get("totalAssets"),
            "totalDebt":             info.get("totalDebt"),
            "totalStockholdersEquity": info.get("bookValue") and
                                        smul(info.get("bookValue"), info.get("sharesOutstanding")),
            "accountPayables":       None,
            "propertyPlantEquipmentNet": None,
            # Cash flow
            "operatingCashFlow":     info.get("operatingCashflow"),
            "capitalExpenditure":    info.get("capitalExpenditures"),
            "dividendsPaid":         smul(info.get("dividendRate"), info.get("sharesOutstanding")),
            # Valuation
            "pe":                    info.get("trailingPE") or info.get("forwardPE"),
            "pb":                    info.get("priceToBook"),
            "ps":                    info.get("priceToSalesTrailing12Months"),
            "ev":                    info.get("enterpriseValue"),
            "evToEbitda":            info.get("enterpriseToEbitda"),
            "eps":                   info.get("trailingEps"),
            "dividendYield":         info.get("dividendYield"),
            "bookValuePerShare":     info.get("bookValue"),
            "investedCapital":       None,
        }

  
        try:
            inc = stk.financials
            if inc is not None and not inc.empty:
                col = inc.columns[0]
                def gi(row):
                    try: return float(inc.loc[row,col]) if row in inc.index else None
                    except: return None
                fin["revenue"]       = gi("Total Revenue") or fin["revenue"]
                fin["grossProfit"]   = gi("Gross Profit") or fin["grossProfit"]
                fin["ebit"]          = gi("EBIT") or gi("Operating Income") or fin["ebit"]
                fin["ebitda"]        = gi("EBITDA") or fin["ebitda"]
                fin["netIncome"]     = gi("Net Income") or fin["netIncome"]
                fin["interestExpense"]= gi("Interest Expense") or fin["interestExpense"]
        except: pass

        try:
            bal = stk.balance_sheet
            if bal is not None and not bal.empty:
                col = bal.columns[0]
                def gb(row):
                    try: return float(bal.loc[row,col]) if row in bal.index else None
                    except: return None
                fin["totalCurrentAssets"]      = gb("Current Assets") or fin["totalCurrentAssets"]
                fin["totalCurrentLiabilities"] = gb("Current Liabilities") or fin["totalCurrentLiabilities"]
                fin["cashAndCashEquivalents"]  = gb("Cash And Cash Equivalents") or fin["cashAndCashEquivalents"]
                fin["netReceivables"]          = gb("Accounts Receivable") or gb("Net Receivables") or fin["netReceivables"]
                fin["inventory"]               = gb("Inventory") or fin["inventory"]
                fin["totalAssets"]             = gb("Total Assets") or fin["totalAssets"]
                fin["totalDebt"]               = gb("Total Debt") or gb("Long Term Debt") or fin["totalDebt"]
                fin["totalStockholdersEquity"] = gb("Stockholders Equity") or gb("Common Stock Equity") or fin["totalStockholdersEquity"]
                fin["accountPayables"]         = gb("Accounts Payable") or fin["accountPayables"]
                fin["propertyPlantEquipmentNet"]= gb("Net PPE") or fin["propertyPlantEquipmentNet"]
        except: pass

        try:
            cf = stk.cashflow
            if cf is not None and not cf.empty:
                col = cf.columns[0]
                def gc(row):
                    try: return float(cf.loc[row,col]) if row in cf.index else None
                    except: return None
                fin["operatingCashFlow"]  = gc("Operating Cash Flow") or gc("Cash From Operations") or fin["operatingCashFlow"]
                fin["capitalExpenditure"] = gc("Capital Expenditure") or gc("Capital Expenditures") or fin["capitalExpenditure"]
        except: pass

        return profile, quote, fin, headlines, stk

   
    def fetch_rss_news(self, ticker: str, company_name: str = "") -> List[str]:
        """Scrape financial news RSS feeds — completely free."""
        import urllib.request, xml.etree.ElementTree as ET

        company = company_name.split()[0] if company_name else ticker
        feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            f"https://news.google.com/rss/search?q={ticker}+stock+earnings&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={company}+revenue+profit+earnings&hl=en-US&gl=US&ceid=US:en",
        ]
        headlines = []
        for url in feeds:
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=8)
                tree = ET.parse(resp)
                root = tree.getroot()
                for item in root.iter("item"):
                    title = item.find("title")
                    if title is not None and title.text:
                        headlines.append(title.text.strip())
                if len(headlines) >= 15:
                    break
            except:
                continue
        return headlines[:20]

    
    def fetch_newsapi(self, ticker: str, company_name: str = "", api_key: str = "") -> List[str]:
        if not api_key:
            return []
        
        company = company_name.split()[0] if company_name else ticker
        queries = [
            f"{company} stock earnings revenue profit",
            f"{ticker} financial results forecast",
        ]
        headlines = []
        seen = set()
        for q in queries:
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q":        q,
                        "language": "en",
                        "sortBy":   "publishedAt",
                        "pageSize": 15,
                        "apiKey":   api_key,
                    },
                    timeout=10,
                )
                if r.status_code == 401:
                    print("  ⚠️  NewsAPI: invalid key")
                    break
                if r.status_code == 426:
                    print("  ⚠️  NewsAPI: free plan only allows developer use (localhost)")
                    break
                if r.status_code != 200:
                    break
                articles = r.json().get("articles", [])
                for a in articles:
                    title = (a.get("title") or "").strip()
                    if title and title not in seen and "[Removed]" not in title:
                        seen.add(title)
                        headlines.append(title)
                if len(headlines) >= 20:
                    break
            except Exception as e:
                print(f"  ⚠️  NewsAPI error: {e}")
                break
        return headlines[:20]

    
    def fetch_fmp(self, ticker: str):
        headers = {"User-Agent":"Mozilla/5.0"}
        base    = "https://financialmodelingprep.com/api/v3"

        def get(ep, **params):
            params["apikey"] = self.api_key
            try:
                r = requests.get(f"{base}/{ep}", params=params, headers=headers, timeout=12)
                if r.status_code in (401,403): return None
                r.raise_for_status()
                d = r.json()
                if isinstance(d,dict) and "Error Message" in d: return None
                return d
            except: return None

        news = []
        d = get("stock_news", tickers=ticker, limit=20)
        if d and isinstance(d,list):
            news = [i.get("title","") for i in d if i.get("title")]
        return news  

   
    def fetch(self, ticker: str, newsapi_key: str = ""):
        profile, quote, fin, headlines, stk = self.fetch_yfinance(ticker)
        company_name = (profile or {}).get("companyName", ticker)

       
        if len(headlines) < 8:
            print("  Fetching RSS news feeds ...")
            rss = self.fetch_rss_news(ticker, company_name)
            headlines = list(dict.fromkeys(headlines + rss))[:25]

       
        if newsapi_key:
            print("  Fetching from NewsAPI ...")
            na_headlines = self.fetch_newsapi(ticker, company_name, newsapi_key)
            if na_headlines:
                headlines = list(dict.fromkeys(na_headlines + headlines))[:30]
                print(f"  ✓ NewsAPI added {len(na_headlines)} headlines")

       
        if self.api_key and len(headlines) < 8:
            print("  Trying FMP for additional news ...")
            fmp_news = self.fetch_fmp(ticker)
            if fmp_news:
                headlines = list(dict.fromkeys(headlines + fmp_news))[:30]

        n_fields = sum(1 for v in fin.values() if v is not None)
        print(f"  ✓ {n_fields} financial fields  |  {len(headlines)} news headlines")
        return profile, quote, fin, headlines, stk


class RatioCalculator:
    def __init__(self, fin: Dict, quote: Dict):
        self.f = fin   or {}
        self.q = quote or {}

    def v(self, *keys):
        for k in keys:
            val = self.f.get(k) or self.q.get(k)
            r   = safe(val)
            if r is not None: return r
        return None

    def compute(self) -> Dict:
        revenue   = self.v("revenue","totalRevenue")
        cogs      = self.v("costOfRevenue","costOfGoodsSold")
        gross     = self.v("grossProfit") or (
                        (safe(revenue,0)-safe(cogs,0)) if revenue and cogs else None)
        ebit      = self.v("ebit","operatingIncome")
        ebitda    = self.v("ebitda")
        net_inc   = self.v("netIncome","netIncomeToCommon")
        interest  = self.v("interestExpense")
        da        = self.v("depreciationAndAmortization")
        if ebitda is None and ebit is not None:
            ebitda = safe(ebit,0) + safe(da,0)

        curr_a    = self.v("totalCurrentAssets")
        curr_l    = self.v("totalCurrentLiabilities")
        cash      = self.v("cashAndCashEquivalents","totalCash")
        recv      = self.v("netReceivables")
        inv       = self.v("inventory")
        t_assets  = self.v("totalAssets")
        t_debt    = self.v("totalDebt")
        equity    = self.v("totalStockholdersEquity")
        payables  = self.v("accountPayables")
        ppe       = self.v("propertyPlantEquipmentNet")

        op_cf     = self.v("operatingCashFlow")
        capex     = abs(safe(self.v("capitalExpenditure"),0))
        dividends = abs(safe(self.v("dividendsPaid"),0))
        inv_cap   = self.v("investedCapital")

        pe        = self.v("pe","peRatio","trailingPE")
        pb        = self.v("pb","priceToBook","priceToBookRatio")
        ev_ebitda = self.v("evToEbitda","ev_ebitda")
        ps        = self.v("ps","priceToSalesTrailing12Months")
        div_yield = self.v("dividendYield")
        eps       = self.v("eps","trailingEps")
        mktcap    = self.v("marketCap") or self.q.get("marketCap")
        ev        = self.v("ev","enterpriseValue")
        price     = self.v("price") or self.q.get("price")
        bvps      = self.v("bookValuePerShare","bookValue")

        fcf = (safe(op_cf,0)-capex) if op_cf is not None else None
        cap_emp = (safe(t_assets,0)-safe(curr_l,0)) if (t_assets and curr_l) else None
        net_debt= (safe(t_debt,0)-safe(cash,0))     if (t_debt is not None and cash is not None) else None

        if pe is None and price and eps and eps!=0: pe=safe(price)/safe(eps)
        if pb is None and price and bvps and bvps!=0: pb=safe(price)/safe(bvps)
        if ev_ebitda is None and ev and ebitda and ebitda!=0: ev_ebitda=safe(ev)/safe(ebitda)
        if ps is None and mktcap and revenue and revenue!=0: ps=safe(mktcap)/safe(revenue)
        div_payout = sdiv(dividends, net_inc) * 100 if net_inc else None

        
        if div_yield is not None:
            dv = safe(div_yield)
            if dv is not None:
                if dv < 1.0:      
                    div_yield = dv * 100
                elif dv > 25.0:   
                    div_yield = None

        dio = sdiv(inv, cogs)*365      if (inv and cogs)         else None
        dso = sdiv(recv, revenue)*365  if (recv and revenue)     else None
        dpo = sdiv(payables, cogs)*365 if (payables and cogs)    else None
        ccc = ((dio or 0)+(dso or 0)-(dpo or 0)) if (dio and dso and dpo) else None

        def p(num, denom): return smul(sdiv(num, denom), 100)

        return {
            "current_ratio":        sdiv(curr_a, curr_l),
            "quick_ratio":          sdiv(safe(cash,0)+safe(recv,0), curr_l) if curr_l else None,
            "cash_ratio":           sdiv(cash, curr_l),
            "ocf_ratio":            sdiv(op_cf, curr_l),
            "gross_margin":         p(gross, revenue),
            "operating_margin":     p(ebit, revenue),
            "net_margin":           p(net_inc, revenue),
            "roa":                  p(net_inc, t_assets),
            "roe":                  p(net_inc, equity),
            "roce":                 p(ebit, cap_emp),
            "ebitda_margin":        p(ebitda, revenue),
            "de_ratio":             sdiv(t_debt, equity),
            "da_ratio":             sdiv(t_debt, t_assets),
            "interest_coverage":    sdiv(ebit, interest),
            "dscr":                 sdiv(op_cf, interest),
            "equity_multiplier":    sdiv(t_assets, equity),
            "net_debt_ebitda":      sdiv(net_debt, ebitda),
            "asset_turnover":       sdiv(revenue, t_assets),
            "inventory_turnover":   sdiv(cogs, inv),
            "dio":                  dio,
            "receivables_turnover": sdiv(revenue, recv),
            "dso":                  dso,
            "dpo":                  dpo,
            "ccc":                  ccc,
            "fixed_asset_turnover": sdiv(revenue, ppe),
            "pe_ratio":             safe(pe),
            # Cross-currency scrub: for foreign GDRs, USD market cap divided
            # by local-currency statements yields ~0.00x garbage. Such micro
            # values are not real and are reported as N/A instead.
            "pb_ratio":             (lambda v: v if (v is None or v >= 0.05) else None)(safe(pb)),
            "ev_ebitda":            (lambda v: v if (v is None or v >= 0.5)  else None)(safe(ev_ebitda)),
            "ps_ratio":             (lambda v: v if (v is None or v >= 0.02) else None)(safe(ps)),
            # yfinance sometimes returns dividendYield already ×100 (93 for
            # MSFT instead of 0.93). Anything above 25% is vendor garbage.
            "dividend_yield":       (lambda v: v if (v is None or 0 < v <= 25) else None)(safe(div_yield)),
            "dividend_payout":      safe(div_payout),
            "eps":                  safe(eps),
            "fcf":                  safe(fcf),
            "fcf_margin":           p(fcf, revenue),
            "cf_to_net_income":     sdiv(op_cf, net_inc),
            "capex_to_revenue":     p(capex, revenue),
            "cash_roic":            p(fcf, inv_cap),
        }


BENCHMARKS = {
    "current_ratio":        {"g":(2,None),    "y":(1,2),     "r":(None,1),   "unit":"x", "desc":"Current Ratio"},
    "quick_ratio":          {"g":(1,None),    "y":(0.5,1),   "r":(None,0.5), "unit":"x", "desc":"Quick Ratio"},
    "cash_ratio":           {"g":(0.5,1.5),   "y":(0.2,0.5), "r":(None,0.2), "unit":"x", "desc":"Cash Ratio"},
    "ocf_ratio":            {"g":(1,None),    "y":(0.5,1),   "r":(None,0.5), "unit":"x", "desc":"OCF Ratio"},
    "gross_margin":         {"g":(40,None),   "y":(20,40),   "r":(None,20),  "unit":"%", "desc":"Gross Margin"},
    "operating_margin":     {"g":(15,None),   "y":(5,15),    "r":(None,5),   "unit":"%", "desc":"Operating Margin"},
    "net_margin":           {"g":(10,None),   "y":(3,10),    "r":(None,3),   "unit":"%", "desc":"Net Margin"},
    "roa":                  {"g":(5,None),    "y":(2,5),     "r":(None,2),   "unit":"%", "desc":"ROA"},
    "roe":                  {"g":(15,None),   "y":(8,15),    "r":(None,8),   "unit":"%", "desc":"ROE"},
    "roce":                 {"g":(15,None),   "y":(8,15),    "r":(None,8),   "unit":"%", "desc":"ROCE"},
    "ebitda_margin":        {"g":(20,None),   "y":(10,20),   "r":(None,10),  "unit":"%", "desc":"EBITDA Margin"},
    "de_ratio":             {"g":(None,1),    "y":(1,2),     "r":(2,None),   "unit":"x", "desc":"Debt / Equity"},
    "da_ratio":             {"g":(None,0.5),  "y":(0.5,0.7), "r":(0.7,None), "unit":"x", "desc":"Debt / Assets"},
    "interest_coverage":    {"g":(3,None),    "y":(1.5,3),   "r":(None,1.5), "unit":"x", "desc":"Interest Coverage"},
    "dscr":                 {"g":(1.25,None), "y":(1,1.25),  "r":(None,1),   "unit":"x", "desc":"DSCR"},
    "equity_multiplier":    {"g":(None,2),    "y":(2,4),     "r":(4,None),   "unit":"x", "desc":"Equity Multiplier"},
    "net_debt_ebitda":      {"g":(None,2),    "y":(2,4),     "r":(4,None),   "unit":"x", "desc":"Net Debt / EBITDA"},
    "asset_turnover":       {"g":(1,None),    "y":(0.5,1),   "r":(None,0.5), "unit":"x", "desc":"Asset Turnover"},
    "inventory_turnover":   {"g":(6,None),    "y":(3,6),     "r":(None,3),   "unit":"x", "desc":"Inventory Turnover"},
    "dio":                  {"g":(None,45),   "y":(45,90),   "r":(90,None),  "unit":"d", "desc":"DIO (days)"},
    "receivables_turnover": {"g":(8,None),    "y":(4,8),     "r":(None,4),   "unit":"x", "desc":"Receivables Turnover"},
    "dso":                  {"g":(None,45),   "y":(45,75),   "r":(75,None),  "unit":"d", "desc":"DSO (days)"},
    "dpo":                  {"g":(30,60),     "y":(15,30),   "r":(None,15),  "unit":"d", "desc":"DPO (days)"},
    "ccc":                  {"g":(None,30),   "y":(30,60),   "r":(60,None),  "unit":"d", "desc":"Cash Conv. Cycle"},
    "fixed_asset_turnover": {"g":(2,None),    "y":(1,2),     "r":(None,1),   "unit":"x", "desc":"Fixed Asset Turnover"},
    "pe_ratio":             {"g":(10,25),     "y":(25,40),   "r":(40,None),  "unit":"x", "desc":"P/E Ratio"},
    "pb_ratio":             {"g":(1,5),       "y":(5,10),    "r":(10,None),  "unit":"x", "desc":"P/B Ratio"},
    "ev_ebitda":            {"g":(6,12),      "y":(12,20),   "r":(20,None),  "unit":"x", "desc":"EV / EBITDA"},
    "ps_ratio":             {"g":(None,3),    "y":(3,8),     "r":(8,None),   "unit":"x", "desc":"P/S Ratio"},
    "dividend_yield":       {"g":(2,5),       "y":(0.1,2),   "r":(5,None),   "unit":"%", "desc":"Dividend Yield"},
    "dividend_payout":      {"g":(30,60),     "y":(60,80),   "r":(80,None),  "unit":"%", "desc":"Dividend Payout"},
    "fcf_margin":           {"g":(10,None),   "y":(3,10),    "r":(None,3),   "unit":"%", "desc":"FCF Margin"},
    "cf_to_net_income":     {"g":(1,None),    "y":(0.8,1),   "r":(None,0.8), "unit":"x", "desc":"CF / Net Income"},
    "capex_to_revenue":     {"g":(None,5),    "y":(5,15),    "r":(15,None),  "unit":"%", "desc":"Capex / Revenue"},
    "cash_roic":            {"g":(15,None),   "y":(8,15),    "r":(None,8),   "unit":"%", "desc":"Cash ROIC"},
}

def traffic_light(key, value):
    if value is None: return "⚪","N/A"
    b=BENCHMARKS.get(key,{})
    def ok(v,lo,hi):
        if lo is not None and v<lo: return False
        if hi is not None and v>hi: return False
        return True
    if ok(value,*b.get("g",(None,None))): return "🟢","HEALTHY"
    if ok(value,*b.get("r",(None,None))): return "🔴","WARNING"
    if ok(value,*b.get("y",(None,None))): return "🟡","WATCH"
    return "⚪","N/A"

def score_ratios(ratios):
    g=y=r=0
    for k,v in ratios.items():
        if v is None or k not in BENCHMARKS: continue
        icon,_=traffic_light(k,v)
        if icon=="🟢": g+=1
        elif icon=="🟡": y+=1
        elif icon=="🔴": r+=1
    total=g+y+r
    return round((g*2+y)/(total*2)*100 if total>0 else 50,1), g, y, r

def combine_signals(sentiment, ratio_score):
    sm={"positive":100,"neutral":50,"negative":0}
    ss=sm.get(sentiment["overall"],50)
    c=0.35*ss+0.65*ratio_score
    sig,em=("BUY","📈") if c>=70 else (("HOLD","➡️") if c>=45 else ("SELL / AVOID","📉"))
    return {"signal":sig,"emoji":em,"combined_score":round(c,1),
            "sentiment_score":round(ss,1),"ratio_score":round(ratio_score,1)}


def predict_price_target(ticker_obj, fin: Dict, quote: Dict, ratios: Dict, sentiment: Dict) -> Dict:
    """
    Honest price target estimation using 3 real methods:
    1. Analyst consensus target (from yfinance)
    2. DCF fair value (simplified, from FCF data)
    3. Historical volatility range (52-week data)
    Returns a range, NOT a single number — because that would be dishonest.
    """
    import math
    current_price = safe(quote.get("price"))
    if not current_price:
        return {"available": False, "reason": "No current price data"}

    results = {}

   
    try:
        info = ticker_obj.info or {}
        target_mean  = safe(info.get("targetMeanPrice"))
        target_low   = safe(info.get("targetLowPrice"))
        target_high  = safe(info.get("targetHighPrice"))
        target_median= safe(info.get("targetMedianPrice"))
        n_analysts   = safe(info.get("numberOfAnalystOpinions"), 0)
        recommendation = info.get("recommendationKey", "")

        if target_mean and current_price:
            analyst_upside = (target_mean - current_price) / current_price * 100
            results["analyst"] = {
                "target_mean":   round(target_mean, 2),
                "target_low":    round(target_low, 2)   if target_low  else None,
                "target_high":   round(target_high, 2)  if target_high else None,
                "target_median": round(target_median, 2) if target_median else None,
                "upside_mean":   round(analyst_upside, 1),
                "upside_low":    round((target_low  - current_price)/current_price*100, 1) if target_low  else None,
                "upside_high":   round((target_high - current_price)/current_price*100, 1) if target_high else None,
                "n_analysts":    int(n_analysts),
                "recommendation": recommendation,
            }
            # FX repair (Zhanibek): targets often arrive in the HOME
            # currency (e.g. tenge) while the quote is USD. Instead of
            # dropping them immediately, convert via the live FX rate.
            # Disclosed in the output. If numbers are still inconsistent
            # after conversion (GDR share-bundle ratios etc.), the sanity
            # guard below still drops the method — honestly, no guessing.
            try:
                _fc = info.get("financialCurrency")
                _qc = info.get("currency")
                if _fc and _qc and _fc != _qc and abs(analyst_upside) > 300:
                    import yfinance as _yf
                    _fx = None
                    try:
                        _fxh = _yf.Ticker(f"{_fc}{_qc}=X").history(period="5d")["Close"].dropna()
                        if len(_fxh):
                            _fx = float(_fxh.iloc[-1])
                    except Exception:
                        pass
                    if _fx and _fx > 0:
                        _a = results["analyst"]
                        for _k in ("target_mean", "target_low", "target_high", "target_median"):
                            if _a.get(_k):
                                _a[_k] = round(_a[_k] * _fx, 2)
                        analyst_upside = (_a["target_mean"] - current_price) / current_price * 100
                        _a["upside_mean"] = round(analyst_upside, 1)
                        if _a.get("target_low"):
                            _a["upside_low"] = round((_a["target_low"] - current_price) / current_price * 100, 1)
                        if _a.get("target_high"):
                            _a["upside_high"] = round((_a["target_high"] - current_price) / current_price * 100, 1)
                        _a["fx_note"] = f"targets converted {_fc}->{_qc} @ {_fx:.6f}"
                        print(f"  i analyst targets converted {_fc}->{_qc} @ {_fx:.6f}")
            except Exception:
                pass

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
        pass

   
    try:
        info = ticker_obj.info or {}

        
        op_cf = safe(fin.get("operatingCashFlow")) or                 safe(info.get("operatingCashflow")) or                 safe(info.get("freeCashflow"))

        
        capex_raw = fin.get("capitalExpenditure") or                     info.get("capitalExpenditures") or                     info.get("capitalExpenditure") or 0
        capex_val = abs(safe(capex_raw, 0))

        
        if op_cf and capex_val > 0:
            fcf = op_cf - capex_val         
        else:
            fcf = safe(info.get("freeCashflow"))  

        
        shares = safe(info.get("sharesOutstanding")) or                  safe(info.get("impliedSharesOutstanding")) or                  safe(info.get("floatShares"))

        fcf_margin = ratios.get("fcf_margin")
        roe        = ratios.get("roe")

        
        if fcf_margin and fcf_margin > 25:
            growth_rate = 0.12
        elif fcf_margin and fcf_margin > 15:
            growth_rate = 0.09
        elif fcf_margin and fcf_margin > 5:
            growth_rate = 0.06
        else:
            growth_rate = 0.03

        discount_rate = 0.10  

        if fcf and fcf > 0 and shares and shares > 0 and discount_rate > growth_rate:
            fcf_per_share = fcf / shares
            if fcf_per_share > 0:
                
                effective_gap = discount_rate - growth_rate
                if effective_gap < 0.03:   
                    pass
                else:
                    fair_value_ps = fcf_per_share * (1 + growth_rate) / effective_gap

                    
                    ratio_to_price = fair_value_ps / current_price
                    if 0.1 <= ratio_to_price <= 3.0:
                        dcf_upside = (fair_value_ps - current_price) / current_price * 100
                        results["dcf"] = {
                            "fair_value":     round(fair_value_ps, 2),
                            "upside":         round(dcf_upside, 1),
                            "growth_assumed": f"{growth_rate*100:.0f}%",
                            "discount_rate":  f"{discount_rate*100:.0f}%",
                            "fcf_per_share":  round(fcf_per_share, 2),
                            "fcf_source":     "direct" if capex_val > 0 else "estimated",
                        }
    except Exception as e:
        pass

    
    try:
        info = ticker_obj.info or {}
        high_52w = safe(info.get("fiftyTwoWeekHigh"))
        low_52w  = safe(info.get("fiftyTwoWeekLow"))
        avg_50d  = safe(info.get("fiftyDayAverage"))
        avg_200d = safe(info.get("twoHundredDayAverage"))
        beta     = safe(info.get("beta"))

        # Fallback (Zhanibek): LSE GDRs often lack fiftyTwoWeekHigh/Low
        # in yfinance info. Compute the range from 1y price history —
        # the GDR history is already in the quote currency (USD), so
        # there is no FX issue here at all.
        if not (high_52w and low_52w):
            try:
                _h = ticker_obj.history(period="1y")["Close"].dropna()
                if len(_h) > 30:
                    high_52w = float(_h.max())
                    low_52w = float(_h.min())
                    print("  i 52-week range computed from price history "
                          "(info fields were empty)")
            except Exception:
                pass

        if high_52w and low_52w and current_price:
            annual_range_pct = (high_52w - low_52w) / low_52w * 100
            
            position = (current_price - low_52w) / (high_52w - low_52w)

            
            capped_range = min(annual_range_pct, 80.0)

            if position < 0.3:
                vol_upside   =  capped_range * 0.40
                vol_downside = -capped_range * 0.20
            elif position > 0.7:
                vol_upside   =  capped_range * 0.15
                vol_downside = -capped_range * 0.35
            else:
                vol_upside   =  capped_range * 0.25
                vol_downside = -capped_range * 0.20

            results["volatility"] = {
                "high_52w":       round(high_52w, 2),
                "low_52w":        round(low_52w, 2),
                "range_pct":      round(annual_range_pct, 1),
                "position_in_range": round(position * 100, 1),
                "upside_estimate":   round(vol_upside, 1),
                "downside_estimate": round(vol_downside, 1),
                "avg_50d":        round(avg_50d, 2)  if avg_50d  else None,
                "avg_200d":       round(avg_200d, 2) if avg_200d else None,
                "beta":           round(beta, 2)      if beta     else None,
            }
    except Exception as e:
        pass

    
    upsides   = []
    downsides = []

    if "analyst" in results:
        a = results["analyst"]
        if a.get("upside_mean") is not None:
            upsides.append(a["upside_mean"])
        
        if a.get("upside_low") is not None:
            downsides.append(a["upside_low"])

    if "dcf" in results:
        dcf_up = results["dcf"]["upside"]
        if dcf_up > 0:
            upsides.append(dcf_up)
        else:
            downsides.append(dcf_up)  

    if "volatility" in results:
        upsides.append(results["volatility"]["upside_estimate"])
        downsides.append(results["volatility"]["downside_estimate"])


    if "dcf" not in results:
        try:
            fcf_chk = sentiment.get("_fcf_negative", False)
            if fcf_chk:
                downsides.append(-20.0) 
        except:
            pass

    if not upsides:
        return {"available": False, "reason": "Insufficient data for prediction"}

    avg_upside   = sum(upsides) / len(upsides)
    
    raw_downside = sum(downsides) / len(downsides) if downsides else -avg_upside * 0.4

    
    ratio_score_val = sentiment.get("_ratio_score", 50)
    if ratio_score_val < 40:       
        floor = -25.0
    elif ratio_score_val < 55:     
        floor = -15.0
    else:                          
        floor = -5.0

    avg_downside = min(raw_downside, floor)  

    return {
        "available":      True,
        "current_price":  round(current_price, 2),
        "horizon":        "12 months",
        "bull_case":      round(max(upsides), 1),
        "base_case":      round(avg_upside, 1),
        "bear_case":      round(avg_downside, 1),
        "price_bull":     round(current_price * (1 + max(upsides)/100), 2),
        "price_base":     round(current_price * (1 + avg_upside/100), 2),
        "price_bear":     round(current_price * (1 + avg_downside/100), 2),
        "methods_used":   list(results.keys()),
        "detail":         results,
        "disclaimer":     "These are estimates based on analyst targets, DCF valuation, and historical volatility. NOT a guarantee.",
    }


def print_prediction(pred: Dict, ticker: str):
    if not pred.get("available"):
        print(f"\n  {B}📈 PRICE PREDICTION{R}  —  {Y}Not available: {pred.get('reason','')}{R}")
        return

    cp   = pred["current_price"]
    bull = pred["bull_case"]
    base = pred["base_case"]
    bear = pred["bear_case"]
    pb   = pred["price_bull"]
    pba  = pred["price_base"]
    pbe  = pred["price_bear"]
    methods = ", ".join(pred["methods_used"])

    bull_col = G if bull > 0 else RE
    base_col = G if base > 0 else RE
    bear_col = RE if bear < 0 else Y

    print(f"\n{'─'*65}")
    print(f"  {B}📈 PRICE TARGET FORECAST  (12-month horizon){R}")
    print(f"{'─'*65}")
    print(f"  Current price : {B}${cp}{R}")
    print(f"")
    print(f"  {G}{B}🐂 Bull case  {R}: {bull_col}{bull:+.1f}%{R}  →  ${pb}")
    print(f"  {Y}{B}📊 Base case  {R}: {base_col}{base:+.1f}%{R}  →  ${pba}")
    print(f"  {RE}{B}🐻 Bear case  {R}: {bear_col}{bear:+.1f}%{R}  →  ${pbe}")
    print(f"")

    
    d = pred.get("detail", {})
    if "analyst" in d:
        a = d["analyst"]
        rec = a.get("recommendation","").upper()
        rec_col = G if rec in ("BUY","STRONG_BUY","STRONGBUY") else (RE if rec in ("SELL","UNDERPERFORM") else Y)
        print(f"  {B}Wall Street analysts ({a['n_analysts']} analysts):{R}")
        print(f"    Consensus:   {rec_col}{rec}{R}")
        print(f"    Target low:  ${a['target_low']}   ({a['upside_low']:+.1f}%)" if a.get("target_low") else "")
        print(f"    Target mean: ${a['target_mean']}  ({a['upside_mean']:+.1f}%)")
        print(f"    Target high: ${a['target_high']}  ({a['upside_high']:+.1f}%)" if a.get("target_high") else "")
        print()

    if "dcf" in d:
        dc = d["dcf"]
        src = " (capex estimated)" if dc.get("fcf_source") == "estimated" else ""
        print(f"  {B}DCF fair value{R}{src}:")
        print(f"    FCF per share  : ${dc['fcf_per_share']}")
        print(f"    Growth assumed : {dc['growth_assumed']} per year")
        print(f"    Discount rate  : {dc['discount_rate']}")
        print(f"    Fair value/share: ${dc['fair_value']}  ({dc['upside']:+.1f}%)")
        print()

    if "volatility" in d:
        v = d["volatility"]
        pos = v["position_in_range"]
        pos_bar = "▓" * int(pos/5) + "░" * (20 - int(pos/5))
        print(f"  {B}52-week range:{R}")
        print(f"    Low: ${v['low_52w']}  [{pos_bar}]  High: ${v['high_52w']}")
        print(f"    Current position: {pos:.0f}% of range")
        if v.get("beta"):
            beta = v["beta"]
            beta_desc = "low risk" if beta < 0.8 else ("high volatility" if beta > 1.5 else "market-like")
            print(f"    Beta: {beta}  ({beta_desc})")
        print()

    print(f"  Methods: {methods}")
    print(f"  {Y}⚠️  Estimates only — markets are unpredictable. DYOR.{R}")


B="\033[1m";R="\033[0m";G="\033[92m";Y="\033[93m";RE="\033[91m";C="\033[96m";W="\033[97m"

def fmt(val, unit="", dec=2):
    if val is None: return "N/A"
    if unit=="%": return f"{val:.{dec}f}%"
    if unit=="d": return f"{val:.0f}d"
    if unit=="$":
        if abs(val)>=1e12: return f"${val/1e12:.2f}T"
        if abs(val)>=1e9:  return f"${val/1e9:.2f}B"
        if abs(val)>=1e6:  return f"${val/1e6:.2f}M"
        return f"${val:.2f}"
    return f"{val:.{dec}f}x" if unit=="x" else f"{val:.{dec}f}"

def bar(v,w=20):
    v=max(0.0,min(1.0,v)); f=int(round(v*w))
    return "█"*f+"░"*(w-f)

GROUPS = {
    "💧 LIQUIDITY":           ["current_ratio","quick_ratio","cash_ratio","ocf_ratio"],
    "💰 PROFITABILITY":       ["gross_margin","operating_margin","net_margin","roa","roe","roce","ebitda_margin"],
    "🏦 LEVERAGE & SOLVENCY": ["de_ratio","da_ratio","interest_coverage","dscr","equity_multiplier","net_debt_ebitda"],
    "⚙️  EFFICIENCY":         ["asset_turnover","inventory_turnover","dio","receivables_turnover","dso","dpo","ccc","fixed_asset_turnover"],
    "📊 VALUATION":           ["pe_ratio","pb_ratio","ev_ebitda","ps_ratio","dividend_yield","dividend_payout","eps"],
    "💵 CASH FLOW":           ["fcf","fcf_margin","cf_to_net_income","capex_to_revenue","cash_roic"],
}

def print_report(ticker, profile, quote, sentiment, ratios, rs, green, yellow, red, final):
    name   = (profile or {}).get("companyName", ticker)
    sector = (profile or {}).get("sector","")
    ind    = (profile or {}).get("industry","")
    price  = safe((quote or {}).get("price"))
    chg    = safe((quote or {}).get("changesPercentage"),0)
    mktcap = safe((quote or {}).get("marketCap"))
    chg_s  = f"{G}+{chg:.2f}%{R}" if chg>=0 else f"{RE}{chg:.2f}%{R}"

    print(f"\n{'═'*65}")
    print(f"  {B}{W}{name}  ({ticker}){R}")
    print(f"  {sector}  {'|  '+ind if ind else ''}")
    price_s = f"${price:.2f}" if price else "N/A"
    print(f"  Price: {B}{price_s}{R}  {chg_s}   Mkt Cap: {fmt(mktcap,'$')}")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*65}")

    
    s=sentiment["overall"]; sc=sentiment["score"]; bd=sentiment["breakdown"]
    total_h= bd["positive"]+bd["neutral"]+bd["negative"]
    sc_col = G if s=="positive" else (RE if s=="negative" else Y)
    print(f"\n  {B}📰 NEWS SENTIMENT{R}  ({total_h} headlines analyzed)")
    if total_h == 0:
        print(f"  {Y}  No headlines retrieved for this ticker.{R}")
    else:
        print(f"  Overall: {sc_col}{B}{s.upper()}{R}   Score: {sc:+.3f}")
        print(f"  {G}▲ {bd['positive']} positive{R}  {Y}= {bd['neutral']} neutral{R}  {RE}▼ {bd['negative']} negative{R}")
        print(f"\n  Headlines:")
        for d in sentiment.get("details",[])[:6]:
            icon="🟢" if d["label"]=="positive" else ("🔴" if d["label"]=="negative" else "🟡")
            hl=d["headline"][:67]+("…" if len(d["headline"])>67 else "")
            print(f"    {icon} [{d['confidence']:.0%}] {hl}")

    
    avail=sum(1 for v in ratios.values() if v is not None)
    print(f"\n  {B}📋 FINANCIAL RATIOS  ({avail}/35 available){R}")
    for group, keys in GROUPS.items():
        has = any(ratios.get(k) is not None for k in keys)
        if not has: continue
        print(f"\n  {C}{B}{group}{R}")
        print(f"  {'Metric':<30} {'Value':>13}  Status")
        print(f"  {'─'*29} {'─'*13}  {'─'*10}")
        for key in keys:
            val=ratios.get(key)
            b=BENCHMARKS.get(key,{})
            unit=b.get("unit",""); desc=b.get("desc",key)
            icon,status=traffic_light(key,val)
            if key=="fcf":
                val_s=fmt(val,"$")
                if val is not None and val < 0:
                    icon,status = "🔴","WARNING" 
            elif key=="eps":
                val_s=f"${val:.2f}" if val is not None else "N/A"
                net_m = ratios.get("net_margin")
                if val is not None and (val < 0 or (net_m is not None and net_m < 0)):
                    icon,status = "🔴","WARNING"
                elif val is not None and val > 0:
                    icon,status = "🟢","HEALTHY"
            else:            val_s=fmt(val,unit)
            col=G if icon=="🟢" else (RE if icon=="🔴" else (Y if icon=="🟡" else ""))
            print(f"  {desc:<30} {col}{val_s:>13}{R}  {icon} {status}")

    
    sig=final["signal"]; sig_col=G if "BUY" in sig else (RE if "SELL" in sig else Y)
    print(f"\n{'═'*65}")
    print(f"  {B}🎯 FINAL SIGNAL:  {sig_col}{sig}  {final['emoji']}{R}")
    print(f"{'─'*65}")
    print(f"  Combined score   {B}{final['combined_score']}/100{R}")
    print(f"  {bar(final['combined_score']/100)}")
    print(f"\n  News sentiment  (35%): {final['sentiment_score']:.0f}/100")
    print(f"  Financial health(65%): {final['ratio_score']:.1f}/100  "
          f"  {G}{green}🟢{R} healthy  {Y}{yellow}🟡{R} watch  {RE}{red}🔴{R} warning")
    print(f"{'═'*65}")
    print(f"\n  ⚠️  NOT financial advice. Always do your own due diligence.\n")


def run(ticker, api_key, model_dir="outputs", news_limit=20, newsapi_key=""):
    ticker=ticker.upper()
    print(f"\n{'─'*65}\n  🔍 Analyzing {ticker} ...\n{'─'*65}")

    print("\n[1/3] Loading sentiment model ...")
    engine=SentimentEngine(model_dir=model_dir)
    print(f"      ✓ Val accuracy: {engine.val_acc:.1%}")

    print("\n[2/3] Fetching data ...")
    fetcher=DataFetcher(api_key)
    profile, quote, fin, headlines, ticker_obj = fetcher.fetch(ticker, newsapi_key=newsapi_key)

    print("\n[3/3] Computing ...")
    calc=RatioCalculator(fin, quote)
    ratios=calc.compute()

    
    sentiment = engine.analyze_headlines(headlines)
    sentiment["news_score"] = sentiment["score"]
    sentiment["fin_score"]  = 0.0

    rs,g,y,r=score_ratios(ratios)
    sentiment["_ratio_score"] = rs   
    fcf_val = ratios.get("fcf")
    sentiment["_fcf_negative"] = (fcf_val is not None and fcf_val < 0)
    final=combine_signals(sentiment,rs)


    print("  Computing price targets ...")
    prediction = predict_price_target(ticker_obj, fin, quote, ratios, sentiment)

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
        pass

    print_report(ticker, profile, quote, sentiment, ratios, rs, g, y, r, final)
    print_prediction(prediction, ticker)

    for _k in ("_ratio_score","_fcf_negative"):
        sentiment.pop(_k, None)
    out={"ticker":ticker,"timestamp":datetime.now().isoformat(),
         "profile":{"name":(profile or {}).get("companyName"),"sector":(profile or {}).get("sector")},
         "quote":{"price":safe((quote or {}).get("price")),
                  "change_pct":safe((quote or {}).get("changesPercentage")),
                  "marketCap":safe((quote or {}).get("marketCap"))},
         "sentiment":sentiment,
         "ratios":{k:round(v,4) if v is not None else None for k,v in ratios.items()},
         "ratio_score":rs,"final_signal":final,
         "prediction":prediction}
    path=f"{ticker}_analysis.json"
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print(f"  💾 Saved to: {path}\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ticker",     required=True)
    p.add_argument("--api_key",    default="RsWo7w7mmqNvziobX7IwYpaVwoEOqJbb")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--news",        type=int, default=20)
    p.add_argument("--newsapi_key", default="",
                   help="NewsAPI.org free key — get at newsapi.org (100 req/day, ~15min delay)")
    args=p.parse_args()
    run(args.ticker, args.api_key, args.output_dir, args.news, args.newsapi_key)
