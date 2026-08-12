"""
Turns the row-per-cycle data into fixed-length sequences for the LSTM.

An LSTM needs a fixed window shape (samples, timesteps, features), but
engines don't all live the same number of cycles, and test trajectories
get cut off at random points. So this builds sliding windows per engine
and left-pads anything shorter than the window by repeating its first
row -- avoids using zeros as padding, which would look like a weird
outlier since the sensors aren't zero-centered for every subset (only
FD002/FD004 get z-scored, FD001/FD003 stay on their raw scale).

For training: generates one window ending at every cycle of every
engine, so a long-lived engine contributes many overlapping samples.
For test: only the LAST window per unit, since that's the only point
with a real ground-truth RUL to check against.
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 30


def _pad_or_trim(values: np.ndarray, window: int) -> np.ndarray:
    """values is (n_cycles, n_features). Returns exactly `window` rows,
    left-padded by repeating the first row if there's not enough history."""
    n = values.shape[0]
    if n >= window:
        return values[-window:]
    pad_rows = np.repeat(values[0:1], window - n, axis=0)
    return np.vstack([pad_rows, values])


def build_train_sequences(df: pd.DataFrame, sensor_cols: list, window: int = WINDOW_SIZE):
    """One window per cycle, per engine. Returns X (n_samples, window,
    n_features) and y (n_samples,) -- the RUL at the end of each window."""
    X, y = [], []
    for unit, unit_df in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = unit_df[sensor_cols].values
        ruls = unit_df["RUL"].values
        for i in range(len(unit_df)):
            window_vals = _pad_or_trim(values[: i + 1], window)
            X.append(window_vals)
            y.append(ruls[i])
    return np.array(X), np.array(y)


def build_test_sequences(df: pd.DataFrame, sensor_cols: list, window: int = WINDOW_SIZE):
    """Last window only, per unit -- matches what actually gets scored.
    Returns X (n_units, window, n_features) and the unit order, so
    predictions can be matched back up to the RUL lookup file."""
    X, units = [], []
    for unit, unit_df in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = unit_df[sensor_cols].values
        X.append(_pad_or_trim(values, window))
        units.append(unit)
    return np.array(X), np.array(units)