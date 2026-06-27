import torch
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing import load_numpy_data


def make_torch_dataloaders(csv_path, batch_size=32):
    X_train, X_test, y_train, y_test, scaler = load_numpy_data(csv_path)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, X_test, y_test, scaler
