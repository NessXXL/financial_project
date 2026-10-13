# Virada AI

> AI-powered stock analysis combining financial fundamentals, financial news sentiment and an experimental price forecasting module.

🌐 **Live MVP:** https://financial-project-phi.vercel.app/

---

## Overview

**Virada AI** is a financial analysis platform designed to make stock research faster and easier to understand.

Instead of relying on a single source of information, the system combines:

- company financial statements;
- 35 financial ratios;
- financial news sentiment;
- a news coverage check (how many independent outlets reported a story);
- experimental price forecasting.

The goal is to turn large amounts of financial and textual information into a clear and explainable stock analysis.

Virada currently produces a combined analytical signal:

**BUY · HOLD · SELL**

and lets users see the factors behind that result.

---

## The Problem

Individual investors often need several different tools to analyze one company.

They may need to separately:

- read financial statements;
- calculate financial ratios;
- follow dozens of news articles;
- estimate market sentiment;
- check whether a news event is widely reported;
- analyze historical price movements.

This takes time and requires financial knowledge. Virada AI combines these tasks into one system.

---

## How Virada Works

Virada uses two main analytical engines.

### 1. Fundamental Analysis

The system processes company financial statements and calculates **35 financial ratios** across six categories:

- liquidity;
- profitability;
- leverage and solvency;
- operational efficiency;
- valuation;
- cash flow.

Each ratio is compared with a benchmark range and graded as healthy, watch or warning. The grades are combined into a financial score from **0 to 100**. Missing data is shown as N/A and is never filled in.

### 2. Financial News Analysis

Financial headlines are processed by a custom neural network built for sentiment classification. The model has three parallel branches:

**BiLSTM + Attention.** Reads word order and context in both directions and identifies the most important words in a headline.

**CNN.** Detects important local expressions and combinations of financial terms.

**Financial Lexicon.** Adds positive, negative and negation-based lexical features. This branch also supports Russian and Kazakh words.

The three branches are combined into a single representation before the final classification. Each headline is classified as:

- **Negative**
- **Neutral**
- **Positive**

---

## Model Architecture

```text
                    Financial Headline
                            │
                            ▼
                  GloVe 300d Embeddings
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   BiLSTM (2 layers)   CNN (kernels      Financial Lexicon
          │            2, 3, 4, 5)        (6 features)
      Attention             │                 │
          │             Max-pool             MLP
        512d              512d               16d
          └─────────────────┼─────────────────┘
                            ▼
                Feature Fusion (1040d)
                            │
                        LayerNorm
                            │
                            ▼
                        Classifier
                            │
                            ▼
              Negative / Neutral / Positive
```

The NLP model was implemented in PyTorch.

---

## Sentiment Model Performance

On the held-out test set (3,278 sentences), the sentiment model achieved:

| Metric | Result |
|---|---|
| Test Accuracy | 91.1% |
| Macro-F1 | 0.910 |
| Validation Accuracy | 91.6% |

Performance by sentiment class:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.92 | 0.96 | 0.94 |
| Neutral | 0.90 | 0.86 | 0.88 |
| Positive | 0.91 | 0.91 | 0.91 |

> **Note:** in the reported run, minority-class augmentation was applied before the train/test split, which may make the test score look higher than it really is. A clean re-run (split first, then augment only the training data) is planned.

---

## Evidence Module

Virada includes an Evidence Module that shows how widely each news story was reported.

The system searches for similar coverage using **GDELT** (Global Database of Events, Language and Tone) and counts how many **different outlets** published the same story:

- **3+ sources:** widely reported
- **1–2 sources:** limited coverage
- **0 sources:** unconfirmed

The module does not claim whether a headline is true or false; it measures how widely the story spread. The result is shown next to each headline so the user can judge how much weight to give it. It does **not** change the sentiment score.

---

## Combined Investment Signal

The financial and sentiment components are normalized to the same 0–100 scale. The current experimental scoring formula is:

```text
Combined Score = 0.35 × Sentiment Score + 0.65 × Financial Ratio Score
```

The financial component receives a larger weight because company fundamentals usually change more slowly than short-term news sentiment.

The prototype converts the combined score into three categories:

| Combined Score | Signal |
|---|---|
| ≥ 70 | BUY |
| 45–69 | HOLD |
| < 45 | SELL |

These weights and thresholds are part of the current experimental methodology and are still being validated.

---

## Short-Term Price Forecasting (Experimental)

Virada also includes an experimental module that predicts the next trading day's closing price.

The model uses **Ridge Regression** with **walk-forward evaluation**: at each step it is trained only on past data and predicts the next day.

Input features:

- 1-day, 5-day and 10-day returns;
- 5-, 10- and 20-day moving averages;
- historical volatility;
- RSI (computed only from past prices).

### KSPI Experiment

A separate experiment was run on two years of KSPI historical stock data.

| Metric | Result |
|---|---|
| RMSE | 1.92 |
| MAPE | 1.70% |
| Durbin–Watson | 1.79 |
| p-value (paired t-test) | 0.101 |

Daily stock prices are highly autocorrelated, so a low error is expected even from simply using yesterday's price. This experiment does not show trading skill, and its results should not be read as a guarantee of future market performance.

---

## Training Data

The sentiment model was trained on five public financial NLP datasets:

- Twitter Financial News Sentiment;
- Kaggle Financial Sentiment;
- Auditor Sentiment;
- FiQA 2018;
- Financial PhraseBank.

Before training, the datasets were standardized to three-class labels and de-duplicated. The data was split (stratified 80 / 10 / 10) into:

- Training
- Validation
- Testing

so the model is evaluated on data it did not see during training.

---

## Why Virada Is Different

Most investment platforms provide either financial data, technical indicators or market news. Virada combines several forms of analysis into one explainable workflow:

```text
Financial Statements
        +
35 Financial Ratios
        +
Financial News
        +
Custom Sentiment AI
        +
News Coverage Check
        +
Price Forecasting (experimental)
        ↓
Explainable Stock Analysis
```

The aim is not a black-box recommendation system. Users should be able to understand why the system produced a particular result.

---

## Current MVP

The current web prototype allows users to:

- search for public companies;
- analyze company fundamentals;
- process financial news;
- receive sentiment scores;
- calculate financial-ratio scores;
- generate BUY / HOLD / SELL signals;
- compare companies;
- explore the model methodology.

**Live demo:** https://financial-project-phi.vercel.app/

---

## Current Development

Virada is an experimental research project and MVP. Next development priorities:

- improving data reliability;
- expanding company coverage;
- improving website security;
- strengthening the Evidence Module;
- expanding multilingual financial NLP;
- improving model validation;
- adding portfolio-level analysis;
- testing the platform with finance professionals and investors.

A future version may explore automated portfolio execution, but the current research system should not be treated as an autonomous investment manager.

---

## Research

Virada was developed alongside a research project investigating whether combining financial fundamentals and financial-news sentiment can improve stock analysis.

The research covers:

- financial-ratio analysis;
- financial sentiment classification;
- neural-network architecture (including an ablation study);
- news coverage checking;
- model evaluation;
- price forecasting;
- statistical evaluation.

---

## Technology

- Python
- PyTorch
- GloVe Embeddings
- BiLSTM
- Convolutional Neural Networks
- Attention Mechanism
- Ridge Regression
- GDELT
- Yahoo Finance
- HTML / CSS / JavaScript (web interface, deployed on Vercel)

---

## Project Status

| Component | Status |
|---|---|
| Research model | ✅ |
| Sentiment classifier | ✅ |
| Financial analysis | ✅ |
| Evidence module | ✅ |
| Web MVP | ✅ |
| Price forecasting | 🧪 Experimental |
| Portfolio automation | 🚧 Future development |

---

## Disclaimer

Virada AI is an experimental research and educational project.

The information and signals generated by the platform are **not financial advice** and should not be considered a recommendation to buy, sell or hold any security. Financial markets involve risk, and past model performance does not guarantee future results.

---

## Author

**Zhanibek** · Almaty, Kazakhstan

🌐 https://financial-project-phi.vercel.app/
