import numpy as np
import tensorflow as tf
from tensorflow import keras

from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data

from .model import build_model

X_train, X_test, y_train, y_test, scaler = load_numpy_data()

model = build_model(X_train.shape[1])

num_negative = np.sum(y_train == 0)
num_positive = np.sum(y_train == 1)
pos_weight = num_negative / num_positive

class_weight = {0: 1.0, 1: pos_weight}


# Defining a custom loss function because the default doesn't weigh like pytorch does.
def weighted_bce(pos_weight):
    def loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
                pos_weight=pos_weight,
            )
        )

    return loss


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_bce(pos_weight),
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(RESULTS_DIR / "tensorflow.keras"),
    save_best_only=True,
    monitor="val_loss",
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    # class_weight=class_weight,
    validation_split=0.2,
    verbose=1,
    callbacks=[checkpoint],
)

model.save(RESULTS_DIR / "tensorflow.keras")
