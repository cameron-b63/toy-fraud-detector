import pickle

import jax
import jax.numpy as jnp
import numpy as np

from common.metrics import evaluate_binary_classifier
from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data

from .model import FraudNet

_, X_test, _, y_test, _ = load_numpy_data()

X_test = jnp.asarray(X_test)

model = FraudNet()

with open(RESULTS_DIR / "jax_params.pkl", "rb") as f:
    params = pickle.load(f)

logits = model.apply(params, X_test)

probabilities = jax.nn.sigmoid(logits)

predictions = (probabilities >= 0.5).astype(jnp.int32)

probabilities = np.asarray(probabilities).flatten()
predictions = np.asarray(predictions).flatten()

evaluate_binary_classifier(
    y_test,
    predictions,
    probabilities,
)
