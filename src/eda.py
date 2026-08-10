import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from pathlib import Path
 
from load_data import load_all, SUBSETS
 
OUT_DIR = Path(__file__).resolve().parent.parent / "eda_plots"
OUT_DIR.mkdir(exist_ok=True)
 
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
 
N_ENGINES_TO_PLOT = 5
 
 

# 1. Sensor variance check

def sensor_variance_check(data: dict):
    print("\n" + "=" * 70)
    print("SENSOR VARIANCE CHECK (flat sensors = no signal, candidates to drop)")
    print("=" * 70)
 
    for subset in SUBSETS:
        train_df = data[subset]["train"]
        print(f"\n--- {subset} ---")
 
        stats = train_df[SENSOR_COLS].agg(["std", "min", "max"]).T
        stats["range"] = stats["max"] - stats["min"]
        stats = stats.sort_values("std")
 
        flat = stats[stats["std"] < 1e-6]
        if len(flat) > 0:
            print(f"  FLAT (std ~0, carries no info): {list(flat.index)}")
        else:
            print("  No completely flat sensors.")
 
        low_var = stats[(stats["std"] >= 1e-6) & (stats["std"] < 0.01)]
        if len(low_var) > 0:
            print(f"  Very low variance (check manually): {list(low_var.index)}")
 
        stats.to_csv(OUT_DIR / f"{subset}_sensor_variance.csv")
 
    print(f"\nFull variance tables saved to {OUT_DIR}/<subset>_sensor_variance.csv")
 
 
# 2.  Trajectory plots per engine 

def plot_trajectories(data: dict):
    print("\n" + "=" * 70)
    print("TRAJECTORY PLOTS")
    print("=" * 70)
 
    # Pick a handful of informative sensors to plot (commonly used in
    # C-MAPSS literature as visibly degrading): T24, T30, T50, P30, Nf, Ps30
    key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]
 
    for subset in SUBSETS:
        train_df = data[subset]["train"]
        units = train_df["unit"].unique()[:N_ENGINES_TO_PLOT]
 
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
 
        for i, sensor in enumerate(key_sensors):
            ax = axes[i]
            for unit in units:
                unit_df = train_df[train_df["unit"] == unit]
                ax.plot(unit_df["cycle"], unit_df[sensor], alpha=0.7, linewidth=0.8)
            ax.set_title(sensor)
            ax.set_xlabel("cycle")
 
        fig.suptitle(f"{subset}: {N_ENGINES_TO_PLOT} sample engines, key sensors over lifetime")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{subset}_trajectories.png", dpi=120)
        plt.close(fig)
        print(f"  Saved {subset}_trajectories.png")
 
 
# 3. Operating condition shift check (matters most for FD002/FD004)

def operating_condition_check(data: dict):
    print("\n" + "=" * 70)
    print("OPERATING CONDITION SHIFT CHECK")
    print("=" * 70)
 
    for subset in SUBSETS:
        train_df = data[subset]["train"].copy()
        n_unique_settings = train_df[OP_SETTING_COLS].drop_duplicates().shape[0]
        print(f"\n--- {subset} ---")
        print(f"  Distinct operational setting combos found: {n_unique_settings}")
 
        if n_unique_settings <= 1:
            print("  Single condition -- no normalization needed.")
            continue
 
        # Cluster the continuous op-settings into discrete condition buckets
        # (rounding is a simple, standard approach for this dataset since
        # the 6 conditions are actually distinct fixed points with noise)
        train_df["condition"] = (
            train_df[OP_SETTING_COLS].round(2).astype(str).agg("_".join, axis=1)
        )
        n_conditions = train_df["condition"].nunique()
        print(f"  Rounded to {n_conditions} distinct condition clusters.")
 
        # Show how much a couple of sensors shift purely due to condition
        sample_sensors = ["sensor_2", "sensor_4", "sensor_11"]
        summary = train_df.groupby("condition")[sample_sensors].mean()
        print(f"  Sensor means by condition (sample):")
        print(summary.to_string())
 
        summary.to_csv(OUT_DIR / f"{subset}_condition_sensor_means.csv")
 
        # Plot one sensor colored by condition to make the shift visible
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond, grp in train_df.groupby("condition"):
            ax.scatter(grp["cycle"], grp["sensor_2"], s=2, alpha=0.3, label=cond)
        ax.set_title(f"{subset}: sensor_2 colored by operating condition")
        ax.set_xlabel("cycle")
        ax.set_ylabel("sensor_2")
        if n_conditions <= 8:
            ax.legend(markerscale=5, fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{subset}_condition_shift.png", dpi=120)
        plt.close(fig)
        print(f"  Saved {subset}_condition_shift.png")
 
 
if __name__ == "__main__":
    data = load_all()
    sensor_variance_check(data)
    plot_trajectories(data)
    operating_condition_check(data)
    print(f"\nAll done. Check the '{OUT_DIR}' folder for plots and CSVs.")