```md
# Financial Sentiment Analysis

Custom neural network for financial sentiment analysis.

This project classifies financial text into:

- negative
- neutral
- positive

---

# Model Architecture

The final model uses:

- custom vocabulary
- embedding layer
- CNN layers
- bidirectional GRU
- attention pooling
- fully connected classifier

---

# Datasets

The model was trained using:

1. Financial PhraseBank
2. Twitter financial news sentiment dataset

---

# Technologies

- Python
- PyTorch
- Hugging Face Datasets
- scikit-learn

---

# Results

Custom neural network achieved approximately:

- Accuracy: ~75%
- Weighted F1: ~0.74

---

# Files

- `train_big_neural_sentiment.py` — training script
- `predict_big_neural_sentiment.py` — inference script
- `big_neural_sentiment.pt` — trained model
- `big_neural_vocab.joblib` — vocabulary

---

# Installation

```bash
pip install torch datasets scikit-learn tqdm joblib numpy

## Author

Zhanibek 