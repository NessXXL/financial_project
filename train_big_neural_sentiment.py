import re
import random
import joblib
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, concatenate_datasets, DatasetDict, Value
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm


# =========================
# CONFIG
# =========================

SEED = 42

MODEL_PATH = "big_neural_sentiment.pt"
VOCAB_PATH = "big_neural_vocab.joblib"

MAX_VOCAB_SIZE = 60000
MAX_LEN = 128

EMBED_DIM = 300
NUM_FILTERS = 160
HIDDEN_DIM = 320

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4

NUM_CLASSES = 3

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# =========================
# UTILS
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)

    text = re.sub(r"\$([a-z]+)", r"\1", text)
    text = re.sub(r"[^a-z0-9%$€£.,!?'-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return clean_text(text).split()


def build_vocab(sentences, max_vocab_size):
    counter = Counter()

    for sentence in sentences:
        counter.update(tokenize(sentence))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)

    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# =========================
# DATASETS
# =========================

def load_financial_phrasebank():
    print("Loading FinancialPhraseBank...")
    ds = load_dataset("lmassaron/FinancialPhraseBank")

    def normalize(example):
        return {
            "text": str(example["sentence"]),
            "label": int(example["label"]),
        }

    train = ds["train"].map(
        normalize,
        remove_columns=ds["train"].column_names,
    )

    valid = ds["validation"].map(
        normalize,
        remove_columns=ds["validation"].column_names,
    )

    test = ds["test"].map(
        normalize,
        remove_columns=ds["test"].column_names,
    )

    train = train.cast_column("label", Value("int64"))
    valid = valid.cast_column("label", Value("int64"))
    test = test.cast_column("label", Value("int64"))

    return DatasetDict({
        "train": train,
        "validation": valid,
        "test": test,
    })


def load_jppgks_twitter_financial_news():
    print("Loading jppgks/twitter-financial-news-sentiment...")
    ds = load_dataset("jppgks/twitter-financial-news-sentiment")

    def normalize(example):
        text = example["input"]
        label_text = str(example["output"]).strip().lower()

        if label_text == "negative":
            label = 0
        elif label_text == "neutral":
            label = 1
        elif label_text == "positive":
            label = 2
        else:
            raise ValueError(f"Unknown label: {label_text}")

        return {
            "text": str(text),
            "label": int(label),
        }

    train = ds["train"].map(
        normalize,
        remove_columns=ds["train"].column_names,
    )

    valid = ds["validation"].map(
        normalize,
        remove_columns=ds["validation"].column_names,
    )

    train = train.cast_column("label", Value("int64"))
    valid = valid.cast_column("label", Value("int64"))

    return DatasetDict({
        "train": train,
        "validation": valid,
    })


def load_zeroshot_twitter_financial_news():
    print("Loading zeroshot/twitter-financial-news-sentiment...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

    print(ds)

    # This dataset mapping:
    # LABEL_0 = Bearish  -> negative
    # LABEL_1 = Bullish  -> positive
    # LABEL_2 = Neutral  -> neutral

    def normalize(example):
        # Usually columns are "text" and "label".
        text = str(example["text"])
        old_label = int(example["label"])

        if old_label == 0:
            label = 0  # bearish -> negative
        elif old_label == 1:
            label = 2  # bullish -> positive
        elif old_label == 2:
            label = 1  # neutral -> neutral
        else:
            raise ValueError(f"Unknown label: {old_label}")

        return {
            "text": text,
            "label": int(label),
        }

    train_split_name = "train"

    if train_split_name not in ds:
        first_split = list(ds.keys())[0]
        all_data = ds[first_split]
        split = all_data.train_test_split(test_size=0.15, seed=SEED)
        train_raw = split["train"]
        valid_raw = split["test"]
    else:
        train_raw = ds["train"]

        if "validation" in ds:
            valid_raw = ds["validation"]
        elif "test" in ds:
            valid_raw = ds["test"]
        else:
            split = train_raw.train_test_split(test_size=0.15, seed=SEED)
            train_raw = split["train"]
            valid_raw = split["test"]

    train = train_raw.map(
        normalize,
        remove_columns=train_raw.column_names,
    )

    valid = valid_raw.map(
        normalize,
        remove_columns=valid_raw.column_names,
    )

    train = train.cast_column("label", Value("int64"))
    valid = valid.cast_column("label", Value("int64"))

    return DatasetDict({
        "train": train,
        "validation": valid,
    })


def prepare_combined_dataset():
    phrasebank = load_financial_phrasebank()
    jppgks = load_jppgks_twitter_financial_news()
    zeroshot = load_zeroshot_twitter_financial_news()

    # PhraseBank is the main target dataset.
    # We duplicate it to keep the model focused on the evaluation domain.
    combined_train = concatenate_datasets([
        phrasebank["train"],
        phrasebank["train"],
        phrasebank["train"],
        jppgks["train"],
        zeroshot["train"],
    ])

    # Validation and test stay only on PhraseBank.
    # This prevents the model from being selected for Twitter-style text only.
    combined_valid = phrasebank["validation"]
    combined_test = phrasebank["test"]

    combined_train = combined_train.shuffle(seed=SEED)

    print("\nCombined dataset:")
    print("Train:", len(combined_train))
    print("Validation:", len(combined_valid))
    print("Test:", len(combined_test))

    return DatasetDict({
        "train": combined_train,
        "validation": combined_valid,
        "test": combined_test,
    })


# =========================
# PYTORCH DATASET
# =========================

class SentimentDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.texts = hf_dataset["text"]
        self.labels = hf_dataset["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode_text(self.texts[idx], self.vocab, MAX_LEN)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# =========================
# MODEL: CNN + BiGRU + ATTENTION
# =========================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, mask):
        scores = self.attention(sequence_output).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence_output * weights.unsqueeze(-1), dim=1)

        return context


class BigNeuralSentimentModel(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=pad_idx,
        )

        self.spatial_dropout = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(
            in_channels=EMBED_DIM,
            out_channels=NUM_FILTERS,
            kernel_size=3,
            padding=1,
        )

        self.conv5 = nn.Conv1d(
            in_channels=EMBED_DIM,
            out_channels=NUM_FILTERS,
            kernel_size=5,
            padding=2,
        )

        self.gru = nn.GRU(
            input_size=NUM_FILTERS * 2,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        self.attention_pooling = AttentionPooling(HIDDEN_DIM * 2)

        self.dropout = nn.Dropout(0.45)

        self.fc1 = nn.Linear(HIDDEN_DIM * 2, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.35)

        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, input_ids):
        mask = (input_ids != 0).long()

        embedded = self.embedding(input_ids)
        embedded = self.spatial_dropout(embedded)

        x = embedded.transpose(1, 2)

        conv3_out = torch.relu(self.conv3(x))
        conv5_out = torch.relu(self.conv5(x))

        conv_out = torch.cat([conv3_out, conv5_out], dim=1)
        conv_out = conv_out.transpose(1, 2)

        gru_out, _ = self.gru(conv_out)

        context = self.attention_pooling(gru_out, mask)

        x = self.dropout(context)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        logits = self.classifier(x)

        return logits


# =========================
# TRAIN / EVALUATE
# =========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(loader), acc, weighted_f1, macro_f1, all_labels, all_preds


# =========================
# MAIN
# =========================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = prepare_combined_dataset()

    print("\nBuilding vocabulary...")
    vocab = build_vocab(dataset["train"]["text"], MAX_VOCAB_SIZE)
    print("Vocab size:", len(vocab))

    train_dataset = SentimentDataset(dataset["train"], vocab)
    valid_dataset = SentimentDataset(dataset["validation"], vocab)
    test_dataset = SentimentDataset(dataset["test"], vocab)

    train_labels = dataset["train"]["label"]

    class_counts = torch.bincount(
        torch.tensor(train_labels),
        minlength=NUM_CLASSES,
    ).float()

    print("Class counts:", class_counts)
    print("Using normal shuffled DataLoader")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BigNeuralSentimentModel(
        vocab_size=len(vocab),
        pad_idx=vocab["<PAD>"],
    ).to(device)

    class_weights = 1.0 / torch.sqrt(class_counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.03,
    )

    print("Loss class weights:", class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_valid_f1 = 0
    patience = 7
    patience_counter = 0

    print("\nStarting training...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        valid_loss, valid_acc, valid_weighted_f1, valid_macro_f1, _, _ = evaluate(
            model,
            valid_loader,
            criterion,
            device,
        )

        scheduler.step(valid_weighted_f1)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Valid loss: {valid_loss:.4f}")
        print(f"Valid accuracy: {valid_acc:.4f}")
        print(f"Valid weighted F1: {valid_weighted_f1:.4f}")
        print(f"Valid macro F1: {valid_macro_f1:.4f}")

        if valid_weighted_f1 > best_valid_f1:
            best_valid_f1 = valid_weighted_f1
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            joblib.dump(vocab, VOCAB_PATH)

            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping.")
                break

    print("\nLoading best model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_loss, test_acc, test_weighted_f1, test_macro_f1, test_labels, test_preds = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal Test Results")
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Test weighted F1:", round(test_weighted_f1, 4))
    print("Test macro F1:", round(test_macro_f1, 4))

    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=["negative", "neutral", "positive"],
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    main()