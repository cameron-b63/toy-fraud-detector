# %%
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from common.dataset import make_torch_dataloaders
from pytorch.model import FraudNet

# %%
CSV_PATH = "../data/credit_card_fraud_synthetic.csv"

train_loader, X_test, y_test, _ = make_torch_dataloaders(
    CSV_PATH,
    batch_size=32,
)

print(X_test.shape)
print(y_test.shape)

# %%
MODEL_PATH = "../results/pytorch.pt"

model = FraudNet(input_size=X_test.shape[1])

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

model.eval()

# %%
with torch.no_grad():
    logits = model(X_test)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

# %%
y_true = y_test.numpy()

y_pred = predictions.numpy()

y_prob = probabilities.numpy()

# %%
print(classification_report(y_true, y_pred, digits=4))

# %%
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.show()

# %%
RocCurveDisplay.from_predictions(
    y_true,
    y_prob,
)

plt.show()

# %%

auc = roc_auc_score(y_true, y_prob)

print(f"ROC AUC: {auc:.4f}")

# %% [markdown]
# ## Discussion
#
#
