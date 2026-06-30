import pickle

import jax
import jax.numpy as jnp
import optax

from common.paths import RESULTS_DIR
from common.preprocessing import load_numpy_data

from .model import FraudNet

X_train, X_test, y_train, y_test, _ = load_numpy_data()

X_train = jnp.asarray(X_train)
y_train = jnp.asarray(y_train)

model = FraudNet()

num_samples = X_train.shape[0]
batch_size = 32

key = jax.random.PRNGKey(63)

params = model.init(key, jnp.ones((1, X_train.shape[1])))

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()

pos_weight = float(num_negative / num_positive)


# Homerolling a bit to match pytorch's undefeated pos_weight semantics
def weighted_bce_with_logits(logits, labels, pos_weight):
    log_weight = 1.0 + (pos_weight - 1.0) * labels

    loss = (1.0 - labels) * logits + log_weight * jax.nn.softplus(-logits)
    return loss


def loss_fn(params, X, y):
    logits = model.apply(params, X)

    loss = weighted_bce_with_logits(
        logits,
        y,
        pos_weight,
    )

    return jnp.mean(loss)


@jax.jit
def train_step(params, opt_state, X, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params,
    )

    params = optax.apply_updates(
        params,
        updates,
    )

    return params, opt_state, loss


epochs = 50

for epoch in range(epochs):
    key, subkey = jax.random.split(key)

    permutation = jax.random.permutation(
        subkey,
        num_samples,
    )

    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]

    epoch_loss = 0.0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        params, opt_state, loss = train_step(
            params,
            opt_state,
            X_train,
            y_train,
        )

        epoch_loss += float(loss)

    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Epoch {epoch + 1} | Loss = {epoch_loss / num_batches:.4f}")


with open(RESULTS_DIR / "jax_params.pkl", "wb") as f:
    pickle.dump(params, f)

print("Saved model.")
