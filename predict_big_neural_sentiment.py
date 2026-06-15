import re
import joblib
import torch
import torch.nn as nn


MODEL_PATH = "big_neural_sentiment.pt"
VOCAB_PATH = "big_neural_vocab.joblib"

MAX_LEN = 128
EMBED_DIM = 300
NUM_FILTERS = 160
HIDDEN_DIM = 320
NUM_CLASSES = 3

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


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


def encode_text(text, vocab):
    tokens = tokenize(text)

    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids


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


def predict(text, model, vocab, device):
    ids = encode_text(text, vocab)

    input_tensor = torch.tensor([ids], dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return ID2LABEL[pred], probs.cpu().numpy()[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = joblib.load(VOCAB_PATH)

    model = BigNeuralSentimentModel(
        vocab_size=len(vocab),
        pad_idx=vocab["<PAD>"],
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    examples = [
        "The company reported strong profit growth.",
        "The bank faces declining revenue and liquidity problems.",
        "The company announced a new office in Finland.",
        "Revenue increased significantly compared to last year.",
        "The company suffered major losses due to weak demand.",
        "The firm published its annual report.",
        "Operating profit rose sharply in the third quarter.",
        "The company lowered its sales forecast.",
        "The acquisition is expected to improve profitability.",
    ]

    for text in examples:
        label, probs = predict(text, model, vocab, device)

        print("\nText:", text)
        print("Prediction:", label)
        print("Probabilities:")
        print("negative:", round(float(probs[0]), 4))
        print("neutral :", round(float(probs[1]), 4))
        print("positive:", round(float(probs[2]), 4))


if __name__ == "__main__":
    main()