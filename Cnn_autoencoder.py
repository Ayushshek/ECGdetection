import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    return df.values.astype(np.float32)


def split_normal(
    data: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_temp = train_test_split(
        data, test_size=0.30, random_state=seed, shuffle=True
    )
    x_val, x_test = train_test_split(
        x_temp, test_size=0.50, random_state=seed, shuffle=True
    )
    return x_train, x_val, x_test


def add_channel(x: np.ndarray) -> np.ndarray:
    return x[..., np.newaxis]


def to_torch_shape(x: np.ndarray) -> np.ndarray:
    # Conv1d expects (N, C, L)
    return np.transpose(x, (0, 2, 1))


class CNNAutoencoder(nn.Module):
    def __init__(self, input_len: int = 187, latent_dim: int = 16):
        super().__init__()
        self.input_len = input_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(64 * input_len, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 64 * input_len)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.latent(x)
        x = self.decoder_fc(x)
        x = x.view(-1, 64, self.input_len)
        x = self.decoder(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
) -> int:
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_pred = model(x_batch)
            loss = criterion(x_pred, x_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x_pred = model(x_batch)
                loss = criterion(x_pred, x_batch)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train:.6f} - val_loss: {avg_val:.6f}")

        if avg_val + 1e-8 < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return epoch


def reconstruction_errors(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_pred = model(x_batch)
            mse = torch.mean((x_batch - x_pred) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors, axis=0)


def save_errors_csv(path: str, errors: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "reconstruction_error": errors,
            "pred_label": preds,
            "true_label": labels,
        }
    )
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN autoencoder for PTBDB anomaly detection")
    parser.add_argument("--normal", default="ptbdb_normal.csv")
    parser.add_argument("--abnormal", default="ptbdb_abnormal.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--plot", action="store_true", help="Save histogram plot")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    normal = load_csv(args.normal)
    abnormal = load_csv(args.abnormal)

    x_train, x_val, x_test_normal = split_normal(normal, seed=args.seed)

    x_train = to_torch_shape(add_channel(x_train))
    x_val = to_torch_shape(add_channel(x_val))
    x_test_normal = to_torch_shape(add_channel(x_test_normal))
    x_test_abnormal = to_torch_shape(add_channel(abnormal))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CNNAutoencoder(input_len=x_train.shape[2], latent_dim=16).to(device)

    epochs_trained = train_model(
        model, train_loader, val_loader, device, args.epochs, patience=10
    )

    # Reconstruction errors
    val_errors = reconstruction_errors(model, x_val, device, args.batch_size)
    threshold = float(val_errors.mean() + 2 * val_errors.std())

    test_normal_errors = reconstruction_errors(
        model, x_test_normal, device, args.batch_size
    )
    test_abnormal_errors = reconstruction_errors(
        model, x_test_abnormal, device, args.batch_size
    )

    pred_normal = (test_normal_errors > threshold).astype(int)
    pred_abnormal = (test_abnormal_errors > threshold).astype(int)

    # Labels for evaluation only
    y_normal = np.zeros_like(pred_normal)
    y_abnormal = np.ones_like(pred_abnormal)

    # Metrics
    mse_test_normal = float(test_normal_errors.mean())
    try:
        y_true = np.concatenate([y_normal, y_abnormal])
        y_scores = np.concatenate([test_normal_errors, test_abnormal_errors])
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "threshold": threshold,
        "mse_test_normal": mse_test_normal,
        "roc_auc": roc_auc,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_normal_samples": int(x_test_normal.shape[0]),
        "test_abnormal_samples": int(x_test_abnormal.shape[0]),
        "epochs_trained": int(epochs_trained),
        "device": str(device),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save errors and predictions
    save_errors_csv(
        os.path.join(args.output_dir, "errors_test_normal.csv"),
        test_normal_errors,
        pred_normal,
        y_normal,
    )
    save_errors_csv(
        os.path.join(args.output_dir, "errors_test_abnormal.csv"),
        test_abnormal_errors,
        pred_abnormal,
        y_abnormal,
    )

    # Optional combined file
    combined_errors = np.concatenate([test_normal_errors, test_abnormal_errors])
    combined_preds = np.concatenate([pred_normal, pred_abnormal])
    combined_labels = np.concatenate([y_normal, y_abnormal])
    save_errors_csv(
        os.path.join(args.output_dir, "errors_test_combined.csv"),
        combined_errors,
        combined_preds,
        combined_labels,
    )

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "autoencoder.pt"))

    # Optional histogram plot
    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(test_normal_errors, bins=50, alpha=0.6, label="Normal")
        plt.hist(test_abnormal_errors, bins=50, alpha=0.6, label="Abnormal")
        plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
        plt.xlabel("Reconstruction error (MSE)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "reconstruction_error_hist.png"))
        plt.close()


if __name__ == "__main__":
    main()
