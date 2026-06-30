# %% [markdown]
# # TensorFlow Results

# %%
import tensorflow as tf
from tensorflow.keras.models import load_model

from common.metrics import evaluate_binary_classifier
from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data

# %%
X_train, X_test, y_train, y_test, _ = load_numpy_data()

# %%
model = load_model(RESULTS_DIR / "tensorflow.keras", compile=False)

# %%
logits = model.predict(X_test)

probabilities = tf.sigmoid(logits).numpy()

predictions = (probabilities >= 0.5).astype(int)

# %%
evaluate_binary_classifier(
    y_test,
    predictions,
    probabilities,
)
# %% [markdown]
# ## Discussion
#
# Using Keras weighting as advised resulted in less-than-desirable results, with a precision and recall of 0.0 for fraudulent cases after the first training cycle.
# This is because the loss function wasn't sufficient to allow the model to properly learn from the small sample size.
#
# Creating a custom loss function resembling the pytorch implementation resulted in much better outcomes.
# It appears that for a severe class imbalance, tensorflow would require some rather serious finagling to get it working.
#
# Even after accounting for weighting properly, it appears that something is going wrong with the tensorflow version of the pipeline. I'm still seeing a recall of 0, which won't do for a fraud detector.
