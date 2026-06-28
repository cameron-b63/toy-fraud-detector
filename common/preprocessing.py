import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .paths import CSV_PATH


def load_numpy_data(
    csv_path=CSV_PATH,
    test_size=0.2,
    random_state=63,
):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
