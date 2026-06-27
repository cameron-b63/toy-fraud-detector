import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.paths import PROJECT_ROOT
from common.preprocessing import load_numpy_data

from .model import FraudNet

MODEL_PATH = PROJECT_ROOT / "results" / "pytorch.pt"

# Preprocessing I could easily pull out to common so i did
X_train, X_test, y_train, y_test, scaler = load_numpy_data()

# New thing: torch uses tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Match shape
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = FraudNet(input_size=X_train.shape[1])

# More sophisticated loss function because of class imbalance
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()

pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop

epochs = 50
best_loss = float("inf")
losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Remove accumulated
        logits = model(X_batch)  # But a batch through the model
        loss = criterion(
            logits, y_batch
        )  # compute loss (yes, with logits, because it's more reliable to let the loss function compute the sigmoid)
        loss.backward()  # compute derivatives backward by chain rule
        optimizer.step()  # adjust weights
        total_loss += loss.item()

        losses.append(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch + 1}: {total_loss:.4f}")
print("Training complete.")
