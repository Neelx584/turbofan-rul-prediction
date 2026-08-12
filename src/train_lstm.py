import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_all, SUBSETS
from feature_engineering import active_sensors, normalize_by_condition, MULTI_CONDITION_SUBSETS
from sequence_utils import build_train_sequences, build_test_sequences, WINDOW_SIZE
from train_xgb_rf import phm08_score, train_val_split_by_unit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 5  # early stopping just to stop if val loss doesn't improve for this many epochs
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # just want the final timestep, like return_sequences=False
        out = self.dropout2(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


def train_model(model, X_fit, y_fit, X_val, y_val):
    fit_ds = TensorDataset(
        torch.tensor(X_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )
    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in fit_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(fit_ds)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        print(f"    epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.2f}  val_loss={val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"    early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).cpu().numpy()
    return preds


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    score = phm08_score(y_true, y_pred)
    print(f"    {name}: RMSE={rmse:.2f}  PHM08 score={score:.1f}")
    return {"model": name, "rmse": rmse, "phm08_score": score}


def train_and_evaluate_subset(subset: str, data: dict) -> list:
    print(f"\n--- {subset} ---")
    train_df = data[subset]["train"]
    test_df = data[subset]["test"]
    rul_lookup = data[subset]["rul"]

    sensors = active_sensors(subset)

    if subset in MULTI_CONDITION_SUBSETS:
        print("  normalizing per operating condition...")
        train_df, test_df = normalize_by_condition(train_df, test_df, sensors)

    fit_df, val_df = train_val_split_by_unit(train_df, seed=RANDOM_STATE)
    print(f"  {fit_df['unit'].nunique()} fit units, {val_df['unit'].nunique()} val units")

    print("  building sequences...")
    X_fit, y_fit = build_train_sequences(fit_df, sensors, window=WINDOW_SIZE)
    X_val, y_val = build_train_sequences(val_df, sensors, window=WINDOW_SIZE)
    print(f"  fit sequences: {X_fit.shape}, val sequences: {X_val.shape}")

    model = LSTMModel(len(sensors)).to(DEVICE)
    model = train_model(model, X_fit, y_fit, X_val, y_val)

    results = []
    print("  validation:")
    val_pred = predict(model, X_val)
    results.append({"subset": subset, "split": "val", **evaluate("LSTM", y_val, val_pred)})

    X_test, test_units = build_test_sequences(test_df, sensors, window=WINDOW_SIZE)
    y_test = rul_lookup.set_index("unit").loc[test_units, "RUL"].values
    test_pred = predict(model, X_test)
    print("  test:")
    results.append({"subset": subset, "split": "test", **evaluate("LSTM", y_test, test_pred)})

    return results


if __name__ == "__main__":
    print(f"using device: {DEVICE}")
    data = load_all()
    all_results = []

    for subset in SUBSETS:
        all_results.extend(train_and_evaluate_subset(subset, data))

    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "lstm_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (test set)")
    print("=" * 70)
    print(results_df[results_df["split"] == "test"].to_string(index=False))
    print(f"\nfull results saved to {out_path}")
