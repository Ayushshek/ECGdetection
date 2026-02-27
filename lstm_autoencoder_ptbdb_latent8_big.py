import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ecg_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    # Keep only the first 187 columns (drop label or extras if present).
    if df.shape[1] > 187:
        df = df.iloc[:, :187]
    elif df.shape[1] < 187:
        raise ValueError(f"Expected at least 187 columns in {path}, got {df.shape[1]}")
    return df.values.astype(np.float32)


def split_normal_data(data: np.ndarray, seed: int = 42):
    n = len(data)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return data[train_idx], data[val_idx], data[test_idx]


class LSTMAutoencoderBig(nn.Module):
    def __init__(self, timesteps: int = 187):
        super().__init__()
        self.timesteps = timesteps

        # Encoder (as requested)
        self.enc_lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.enc_lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.enc_dense = nn.Linear(64, 8)

        # Decoder (as requested)
        self.dec_lstm1 = nn.LSTM(input_size=8, hidden_size=64, batch_first=True)
        self.dec_lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dec_dense = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.enc_lstm1(x)
        _, (h_n, _) = self.enc_lstm2(x)
        latent = self.enc_dense(h_n[-1])

        x = latent.unsqueeze(1).repeat(1, self.timesteps, 1)
        x, _ = self.dec_lstm1(x)
        x, _ = self.dec_lstm2(x)
        x = self.dec_dense(x)
        return x


def train_model(model, train_loader, val_loader, device, max_epochs=100, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_val = float("inf")
    best_state = None
    patience_left = patience
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        epochs_trained += 1

        print(
            f"Epoch {epoch + 1}/{max_epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return epochs_trained


def reconstruction_errors(model, data, device, batch_size=256):
    loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            mse = torch.mean((batch_x - recon) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    if not errors:
        return np.array([], dtype=np.float32)
    return np.concatenate(errors, axis=0)


def main() -> None:
    set_seeds(42)

    device = torch.device("cpu")

    normal_path = "ptbdb_normal.csv"
    abnormal_path = "ptbdb_abnormal.csv"

    normal = load_ecg_csv(normal_path)
    abnormal = load_ecg_csv(abnormal_path)

    x_train, x_val, x_test = split_normal_data(normal, seed=42)
    print(
        f"Loaded normal={len(normal)}, abnormal={len(abnormal)}. "
        f"Split: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}",
        flush=True,
    )

    # Reshape to (samples, 187, 1)
    x_train = x_train.reshape((-1, 187, 1))
    x_val = x_val.reshape((-1, 187, 1))
    x_test = x_test.reshape((-1, 187, 1))
    x_abnormal = abnormal.reshape((-1, 187, 1))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val)),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = LSTMAutoencoderBig(timesteps=187).to(device)

    epochs_trained = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=100,
        patience=5,
    )

    # Reconstruction errors
    val_errors = reconstruction_errors(model, x_val, device)
    test_errors = reconstruction_errors(model, x_test, device)
    abnormal_errors = reconstruction_errors(model, x_abnormal, device)

    threshold = float(np.mean(val_errors) + 2 * np.std(val_errors))

    # Predictions: True => abnormal
    test_pred_abnormal = test_errors > threshold
    abnormal_pred_abnormal = abnormal_errors > threshold

    # Metrics
    mean_error_normal = float(np.mean(test_errors))
    mean_error_abnormal = float(np.mean(abnormal_errors))

    total_samples_tested = int(len(x_test) + len(x_abnormal))
    total_pred_abnormal = int(test_pred_abnormal.sum() + abnormal_pred_abnormal.sum())

    # Normal test misclassification: predicted abnormal
    normal_misclass_rate = float(test_pred_abnormal.mean()) if len(test_pred_abnormal) else 0.0
    # Abnormal misclassification: predicted normal
    abnormal_misclass_rate = float((~abnormal_pred_abnormal).mean()) if len(abnormal_pred_abnormal) else 0.0

    # Overall error rate across combined datasets
    y_true = np.concatenate([
        np.zeros(len(x_test), dtype=np.int32),
        np.ones(len(x_abnormal), dtype=np.int32),
    ])
    y_pred = np.concatenate([
        test_pred_abnormal.astype(np.int32),
        abnormal_pred_abnormal.astype(np.int32),
    ])
    overall_error_rate = float(np.mean(y_true != y_pred)) if len(y_true) else 0.0

    # Confusion matrix components
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    # Per-dataset prediction counts
    normal_pred_normal = int((~test_pred_abnormal).sum())
    normal_pred_abnormal = int(test_pred_abnormal.sum())
    abnormal_pred_normal = int((~abnormal_pred_abnormal).sum())
    abnormal_pred_abnormal_count = int(abnormal_pred_abnormal.sum())

    # Build results
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    architecture_desc = (
        "Encoder:\n"
        "- LSTM(128, return_sequences=True)\n"
        "- LSTM(64, return_sequences=False)\n"
        "- Dense(8)  # latent\n"
        "Decoder:\n"
        "- RepeatVector(187)\n"
        "- LSTM(64, return_sequences=True)\n"
        "- LSTM(128, return_sequences=True)\n"
        "- TimeDistributed(Dense(1))\n"
    )

    results_lines = []
    results_lines.append("LSTM Autoencoder for PTBDB Anomaly Detection")
    results_lines.append("Configuration: latent=8 with larger LSTM layers (128/64)")
    results_lines.append("Framework: PyTorch")
    results_lines.append(f"Run datetime: {now}")
    results_lines.append("")
    results_lines.append("Model Architecture:")
    results_lines.append(architecture_desc)
    results_lines.append("")
    results_lines.append("Threshold:")
    results_lines.append(f"- threshold = mean(val_error) + 2 * std(val_error)")
    results_lines.append(f"- threshold_value = {threshold:.6f}")
    results_lines.append("")
    results_lines.append("Reconstruction Error (Mean):")
    results_lines.append(f"- normal_test_mean_error = {mean_error_normal:.6f}")
    results_lines.append(f"- abnormal_mean_error    = {mean_error_abnormal:.6f}")
    results_lines.append("")
    results_lines.append("Evaluation Metrics:")
    results_lines.append(f"- total_samples_tested = {total_samples_tested}")
    results_lines.append(f"- number_predicted_abnormal = {total_pred_abnormal}")
    results_lines.append(f"- normal_misclassification_rate = {normal_misclass_rate:.6f}")
    results_lines.append(f"- abnormal_misclassification_rate = {abnormal_misclass_rate:.6f}")
    results_lines.append(f"- overall_error_rate = {overall_error_rate:.6f}")
    results_lines.append("")
    results_lines.append("Predictions vs Real (Counts):")
    results_lines.append("- Normal test set (real = normal)")
    results_lines.append(f"  - predicted_normal = {normal_pred_normal}")
    results_lines.append(f"  - predicted_abnormal = {normal_pred_abnormal}")
    results_lines.append("- Abnormal set (real = abnormal)")
    results_lines.append(f"  - predicted_normal = {abnormal_pred_normal}")
    results_lines.append(f"  - predicted_abnormal = {abnormal_pred_abnormal_count}")
    results_lines.append("")
    results_lines.append("Confusion Matrix (combined):")
    results_lines.append(f"- TN = {tn}")
    results_lines.append(f"- FP = {fp}")
    results_lines.append(f"- FN = {fn}")
    results_lines.append(f"- TP = {tp}")
    results_lines.append("")
    results_lines.append("Training Info:")
    results_lines.append(f"- epochs_trained = {epochs_trained}")

    results_path = "lstm_autoencoder_results_latent8_big.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))


if __name__ == "__main__":
    main()
