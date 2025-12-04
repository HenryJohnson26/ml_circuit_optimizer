# train_pushback_cnn.py

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data/tokenized"
WINDOW = 32
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 32
HIDDEN_DIM = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load tokenized circuits
# ----------------------------
raw_sequences = []
opt_sequences = []

for f in Path(DATA_DIR).glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
        raw_sequences.append(data["raw_ints"])
        opt_sequences.append(data["opt_ints"])

print(f"Loaded {len(raw_sequences)} circuits.")

# ----------------------------
# Create training samples via sliding window
# ----------------------------
X = []
Y = []

for raw, opt in zip(raw_sequences, opt_sequences):
    length = min(len(raw), len(opt))

    for i in range(length - WINDOW - 1):
        X.append(raw[i:i+WINDOW])  # WINDOW tokens from raw
        Y.append(opt[i+WINDOW][0]) # Predict next token's gate type only!

# Convert to tensor
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

print("Dataset size:", X.shape, Y.shape)

# Dataloader
train_loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Define Model
# ----------------------------
# Vocabulary size = highest token integer + 1
VOCAB_SIZE = int(X.max().item() + 1)

class CNNPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.conv = nn.Conv1d(EMBED_DIM, HIDDEN_DIM, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(HIDDEN_DIM * WINDOW, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)             # (B, W, E)
        x = x.transpose(1, 2)         # (B, E, W)
        x = self.relu(self.conv(x))   # (B, H, W)
        x = x.flatten(1)              # (B, H*W)
        return self.fc(x)             # (B, vocab)

model = CNNPredictor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {avg_loss:.4f}")

print("Training complete.")
