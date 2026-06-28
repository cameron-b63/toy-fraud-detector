import torch

from common.dataset import make_torch_dataloaders
from common.metrics import evaluate_binary_classifier
from common.paths import CSV_PATH, RESULTS_DIR

from .model import FraudNet

MODEL_PATH = RESULTS_DIR / "pytorch.pt"

train_loader, X_test, y_test, _ = make_torch_dataloaders(CSV_PATH)

model = FraudNet(input_size=X_test.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

evaluate_binary_classifier(y_test.numpy(), preds.numpy(), probs.numpy())
