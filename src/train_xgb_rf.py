"""
Tabular modeling: Random Forest + XGBoost, trained per subset.

Splits off a validation set by ENGINE UNIT rather than by row. If I
split by row instead, cycles from the same engine could end up in both
train and validation, and since consecutive cycles from one engine are
basically the same trajectory, that would make the validation score
look way better than it actually is.

I am tracking two metrics:
  -> RMSE, the usual error measure, in cycles
  -> PHM08 score, the actual scoring function from the original 2008
    competition paper. It punishes late predictions (saying an engine
    has more life left than it really does) way harder than early
    ones, which makes sense for maintenance as missing a failure is
    much worse than servicing something a bit early. A few badly late
    predictions can wreck this score even when RMSE looks fine so
    should be checking both.

For the test set I only score the LAST cycle of each unit's
trajectory because that's the only point the official RUL file actually
gives ground truth(test trajectories are cut off before failure
on purpose).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from load_data import load_all, SUBSETS

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VAL_FRACTION = 0.2  # how much of the training units to hold back for validation
RANDOM_STATE = 42

NON_FEATURE_COLS = {"unit", "cycle", "RUL", "condition",
                     "op_setting_1", "op_setting_2", "op_setting_3"}


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray, a_early: float = 13, a_late: float = 10) -> float:
    """
    The official PHM08 scoring function from the paper (eq. 11).
    d = predicted therefore true. Late predictions (d >= 0, thought there was
    more life left than there was) get the steeper penalty, early ones
    the gentler one.

    Note: the equation as OCR'd out of the paper's PDF and
    the paper's own wording actually contradicts each other on which
    constant (10 or 13) goes with which branch. It is clear that
    late predictions have to be penalized harder, so I picked the
    pairing that actually produces that (a_early=13, a_late=10) amd
    checked it numerically and it matches with how everyone else implements
    this scoring function anyway.
    """
    d = y_pred - y_true
    score = np.where(
        d < 0,
        np.exp(-d / a_early) - 1,
        np.exp(d / a_late) - 1,
    )
    return float(np.sum(score))


def train_val_split_by_unit(train_df: pd.DataFrame, val_fraction=VAL_FRACTION, seed=RANDOM_STATE):
    """Splits by whole engine so the same engine never shows up in both sets."""
    rng = np.random.default_rng(seed)
    units = train_df["unit"].unique()
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])

    val_df = train_df[train_df["unit"].isin(val_units)]
    fit_df = train_df[~train_df["unit"].isin(val_units)]
    return fit_df, val_df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Test trajectories are truncated, so I only score the last cycle
    of each unit. That's the only point RUL_FDxxx.txt actually labels."""
    return df.sort_values("cycle").groupby("unit").tail(1).reset_index(drop=True)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)  # can't have negative RUL
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    score = phm08_score(y_true, y_pred)
    print(f"    {name}: RMSE={rmse:.2f}  PHM08 score={score:.1f}")
    return {"model": name, "rmse": rmse, "phm08_score": score}


def train_and_evaluate_subset(subset: str, rul_lookup: pd.DataFrame) -> list:
    print(f"\n--- {subset} ---")
    train_path = FEATURES_DIR / f"{subset}_train_features.csv"
    test_path = FEATURES_DIR / f"{subset}_test_features.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"  no feature files for {subset} yet, run feature_engineering.py first")
        return []

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    fit_df, val_df = train_val_split_by_unit(train_df)
    feature_cols = get_feature_cols(fit_df)
    print(f"  {len(feature_cols)} features, {fit_df['unit'].nunique()} fit units, "
          f"{val_df['unit'].nunique()} val units")

    X_fit, y_fit = fit_df[feature_cols], fit_df["RUL"]
    X_val, y_val = val_df[feature_cols], val_df["RUL"]

    # only scoring the final cycle per test unit, same as the real challenge
    test_last = last_cycle_per_unit(test_df)
    X_test = test_last[feature_cols]
    y_test = rul_lookup.set_index("unit").loc[test_last["unit"], "RUL"].values

    results = []

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_fit, y_fit)
    print("  validation:")
    results.append({"subset": subset, "split": "val",
                     **evaluate("RandomForest", y_val, rf.predict(X_val))})
    print("  test:")
    results.append({"subset": subset, "split": "test",
                     **evaluate("RandomForest", y_test, rf.predict(X_test))})

    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_fit, y_fit)
    print("  validation:")
    results.append({"subset": subset, "split": "val",
                     **evaluate("XGBoost", y_val, xgb.predict(X_val))})
    print("  test:")
    results.append({"subset": subset, "split": "test",
                     **evaluate("XGBoost", y_test, xgb.predict(X_test))})

    return results


if __name__ == "__main__":
    data = load_all()
    all_results = []

    for subset in SUBSETS:
        rul_lookup = data[subset]["rul"]
        all_results.extend(train_and_evaluate_subset(subset, rul_lookup))

    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "xgb_rf_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (test set -- the real held-out numbers)")
    print("=" * 70)
    test_summary = results_df[results_df["split"] == "test"].pivot(
        index="subset", columns="model", values=["rmse", "phm08_score"]
    )
    print(test_summary.to_string())
    print(f"\nfull results saved to {out_path}")
