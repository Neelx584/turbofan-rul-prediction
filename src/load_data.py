import pandas as pd
from pathlib import Path
 
DATA_DIR = Path(__file__).resolve().parent.parent / "CMAPSSData"  # <- confirmed real path
RUL_CLIP = 125  # standard clip used in most C-MAPSS work (see notes below)
 
COLS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
 
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
 
 
def load_raw(path: Path) -> pd.DataFrame:
    """Load a single train/test txt file into a clean DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df
 
 
def add_train_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """
    Training sets run every unit to failure, so RUL at each row is just
    (max cycle for that unit) - (current cycle). We clip at `clip` because
    early-life degradation is essentially flat/nonexistent in this sim --
    without clipping, models get penalized for not predicting a linear
    decline that doesn't actually exist yet in the sensor signal.
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=clip)
    return df
 
 
def load_subset(subset: str, data_dir: Path = DATA_DIR):
    """
    Returns (train_df, test_df, test_rul) for one subset, e.g. 'FD001'.
    train_df has a per-row RUL column (clipped).
    test_df is the raw truncated trajectories (label it with test_rul
    only at the final cycle per unit, since that's what's scored).
    """
    train_df = load_raw(data_dir / f"train_{subset}.txt")
    test_df = load_raw(data_dir / f"test_{subset}.txt")
    rul_df = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"]
    )
    rul_df["unit"] = rul_df.index + 1  # RUL file is ordered by unit 1..N
 
    train_df = add_train_rul(train_df)
 
    return train_df, test_df, rul_df
 
 
def load_all(data_dir: Path = DATA_DIR):
    """Load all four subsets into a dict keyed by subset name."""
    data = {}
    for subset in SUBSETS:
        train_df, test_df, rul_df = load_subset(subset, data_dir)
        data[subset] = {"train": train_df, "test": test_df, "rul": rul_df}
        print(
            f"{subset}: {train_df['unit'].nunique()} train units "
            f"({len(train_df)} rows), {test_df['unit'].nunique()} test units "
            f"({len(test_df)} rows)"
        )
    return data
 
 
if __name__ == "__main__":
    data = load_all()
 
    # quick sanity check on FD001
    fd001_train = data["FD001"]["train"]
    print("\nFD001 train sample:")
    print(fd001_train.head())
    print("\nRUL distribution (clipped at", RUL_CLIP, "):")
    print(fd001_train["RUL"].describe())