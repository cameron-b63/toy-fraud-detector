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
# This is the first model I trained and examined, so some observations aren't really specific to pytorch.
#
# The model successfully identified all fraudulent charges in the test set, with a recall of 1.0 for fraudulent charges.
# This is excellent for the business case, because false negatives would be significantly more damaging than false positives in this context.
# A credit card company would likely much rather investigate a few innocuous charges rather than miss real fraud, and would therefore prioritize recall over precision in this application.
# Because of the additional positive weighting information given to the optimizer, the precision of fraudulent labeling was 0.1818, which is quite low.
# Due to the class imbalance present in the training and test data which is representative of the broader use case, these results are actually quite strong.
#
# I want to note that accuracy is a terrible metric for our dataset.
#
# The original test data has 18 fraudulent examples out of 1000, and simply labeling all charges as benign would result in an accuracy of 98.2%, despite catching none of the fraudulent charges.
# Our model has an accuracy of 91%, yet catches all fraudulent transactions in the test set. This is why recall, precision, and the confusion matrix are far more useful indicators than accuracy when dealing with a class imbalance.
#
# I noticed that during training, the downward trend in total loss toward the final epochs was not really slowing down like I would expect.
# I believe that increasing the number of epochs in training could very well reduce the false positive rate without running the risk of overfitting.
