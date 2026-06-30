# %%
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common.metrics import evaluate_binary_classifier
from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data
from jax_m.model import FraudNet

# %%
_, X_test, _, y_test, _ = load_numpy_data()

X_test = jnp.asarray(X_test)

print(f"Test samples: {len(y_test)}")

# %%
model = FraudNet()

with open(RESULTS_DIR / "jax_params.pkl", "rb") as f:
    params = pickle.load(f)

# %%
logits = model.apply(params, X_test)

probabilities = jax.nn.sigmoid(logits)

predictions = (probabilities >= 0.5).astype(jnp.int32)

probabilities = np.asarray(probabilities).flatten()
predictions = np.asarray(predictions).flatten()

# %%
evaluate_binary_classifier(
    y_test,
    predictions,
    probabilities,
)

# %%
cm = confusion_matrix(y_test, predictions)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()

plt.show()

# %%
plt.figure(figsize=(8, 4))

plt.hist(probabilities, bins=20)

plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")

plt.show()

# %%
top_n = 10

indices = np.argsort(probabilities)[::-1][:top_n]

for i in indices:
    print(f"Index {i:3d} | P(Fraud) = {probabilities[i]:.4f} | Actual = {y_test[i]}")

# %% [markdown]
# ## Discussion
#
# JAX offers much more digestible syntax for someone like myself than something like pytorch or tensorflow.
# The functional approach has parity with my understanding of machine learning as a set of transformations instead of treating "The Model" as a magic box.
#
# That said, the results were not terribly impressive. The model labeled a rather large portion of transactions as fraudulent, yet did not recall all instances.
# The distribution of probabilities explains this: the probabilities are centered around the threshold of 0.5, meaning there wasn't a very meaningful result.
#
# This is attributable to the tiny and imbalanced training set, and I'd be very excited to use JAX on a more appropriate data set.
# I don't think more epochs or a different learning rate could fix this. Perhaps a different structure for the neural network could yield something more meaningful, but I am doubtful.
# Additionally, because of the clustering around 0.5, I don't think thresholding could fix the output here.
