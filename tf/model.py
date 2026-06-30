from tensorflow import keras


def build_model(input_size: int) -> keras.Model:

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    return model
