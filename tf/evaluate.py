import tensorflow as tf

from common.metrics import evaluate_binary_classifier
from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data

X_train, X_test, y_train, y_test, scaler = load_numpy_data()

model = tf.keras.models.load_model(RESULTS_DIR / "tensorflow.keras", compile=False)

logits = model.predict(X_test)

probabilities = tf.sigmoid(logits).numpy()

predictions = (probabilities >= 0.5).astype(int)

evaluate_binary_classifier(
    y_test,
    predictions,
    probabilities,
)
