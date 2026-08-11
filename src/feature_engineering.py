import pandas as pd
import numpy as np
from pathlib import Path
 
from load_data import load_all, SUBSETS
 
FEATURES_DIR = Path(__file__).resolve().parent.parent / "features"
FEATURES_DIR.mkdir(exist_ok=True)
 
ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
 
# Confirmed flat (std ~0) in EDA for these two subsets -- carry no signal.
FLAT_SENSORS = {
    "FD001": ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
    "FD003": ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
    "FD002": [],  # multi-condition: same sensors carry info once normalized
    "FD004": [],
}
 
MULTI_CONDITION_SUBSETS = {"FD002", "FD004"}
 
ROLLING_WINDOWS = [5, 10, 20]
 
 
def active_sensors(subset: str) -> list:
    """Sensor columns to actually use for this subset."""
    return [s for s in ALL_SENSORS if s not in FLAT_SENSORS[subset]]
 
 
def add_condition_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets rows into their operating condition based on the 3 op
    settings. Rounding to 2 decimals works because the 6 real conditions
    in FD002/FD004 are far apart which is verified in EDA (the
    condition-shift plot showed 6 clean, well-separated bands).
    """
    df = df.copy()
    df["condition"] = (
        df[OP_SETTING_COLS].round(2).astype(str).agg("_".join, axis=1)
    )
    return df
 
 
def normalize_by_condition(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_cols: list):
    """
    Fits mean/std per condition cluster on the TRAINING data only (avoids
    leaking test statistics), then applies those same per-condition
    stats to both train and test. Any test-set condition not seen in
    training falls back to global train mean/std (defensive shouldn't
    happen given the fixed 6 conditions but avoids a crash if it does).
    """
    train_df = add_condition_cluster(train_df)
    test_df = add_condition_cluster(test_df)
 
    cond_stats = train_df.groupby("condition")[sensor_cols].agg(["mean", "std"])
    global_mean = train_df[sensor_cols].mean()
    global_std = train_df[sensor_cols].std()
 
    def apply_norm(df):
        df = df.copy()
        for sensor in sensor_cols:
            means = df["condition"].map(cond_stats[(sensor, "mean")])
            stds = df["condition"].map(cond_stats[(sensor, "std")])
            means = means.fillna(global_mean[sensor])
            stds = stds.fillna(global_std[sensor]).replace(0, 1)  # guard divide-by-zero
            df[sensor] = (df[sensor] - means) / stds
        return df
 
    return apply_norm(train_df), apply_norm(test_df)
 
 
def add_rolling_features(df: pd.DataFrame, sensor_cols: list, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Per unit, per sensor, adds rolling mean / std / slope at each window
    size. min_periods=1 so early cycles (before a full window exists)
    still get a value instead of NaN and uses whatever history is
    available.
 
    Slope is estimated as: current value - value `window` steps ago and
    window; a cheap linear trend estimate, much faster than a full
    regression fit per row and good enough for this purpose.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")
 
    new_cols = {}
    for w in windows:
        for sensor in sensor_cols:
            roll = grouped[sensor].rolling(window=w, min_periods=1)
            new_cols[f"{sensor}_roll{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{sensor}_roll{w}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0)
            new_cols[f"{sensor}_roll{w}_slope"] = (
                grouped[sensor].diff(w).reset_index(level=0, drop=True) / w
            ).fillna(0)
 
    # Build all new columns at once (avoids the fragmentation performance
    # hit of inserting one column at a time on a wide DataFrame).
    feat_df = pd.concat([df] + [pd.Series(v, name=k) for k, v in new_cols.items()], axis=1)
    return feat_df
 
 
def engineer_subset(subset: str, data: dict) -> None:
    print(f"\n--- {subset} ---")
    train_df = data[subset]["train"]
    test_df = data[subset]["test"]
    rul_df = data[subset]["rul"]
 
    sensors = active_sensors(subset)
    print(f"  Using {len(sensors)} sensors: {sensors}")
 
    if subset in MULTI_CONDITION_SUBSETS:
        print("  Multi-condition subset -- normalizing per operating condition...")
        train_df, test_df = normalize_by_condition(train_df, test_df, sensors)
    else:
        train_df = train_df.copy()
        test_df = test_df.copy()
 
    print("  Adding rolling-window features...")
    train_feat = add_rolling_features(train_df, sensors)
    test_feat = add_rolling_features(test_df, sensors)
 
    train_out = FEATURES_DIR / f"{subset}_train_features.csv"
    test_out = FEATURES_DIR / f"{subset}_test_features.csv"
    train_feat.to_csv(train_out, index=False)
    test_feat.to_csv(test_out, index=False)
 
    print(f"  {train_feat.shape[0]} train rows x {train_feat.shape[1]} cols -> {train_out.name}")
    print(f"  {test_feat.shape[0]} test rows x {test_feat.shape[1]} cols -> {test_out.name}")
 
 
if __name__ == "__main__":
    data = load_all()
    for subset in SUBSETS:
        engineer_subset(subset, data)
    print(f"\nAll done. Feature files saved to {FEATURES_DIR}")