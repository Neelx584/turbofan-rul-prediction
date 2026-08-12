# Turbofan Engine RUL Prediction

Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using
NASA's C-MAPSS degradation simulation dataset (the PHM08 challenge data).

Comparing two modeling approaches:
- Feature engineering + XGBoost / Random Forest
- LSTM sequence model (PyTorch)

across all four C-MAPSS subsets (FD001-FD004), which differ in number of
operating conditions (1 vs 6) and fault modes (1 vs 2).

## Data

Not included in this repo, too large for git. Grab it from NASA directly:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Unzip into a `CMAPSSData/` folder at the project root:

```
CMAPSSData/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  train_FD003.txt  test_FD003.txt  RUL_FD003.txt
  train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

## Setup

```
pip install -r requirements.txt
```

Note: TensorFlow doesn't support Python 3.14 yet, so the LSTM track is
built in PyTorch instead.

## Usage

```
python src/load_data.py            # load + label all 4 subsets
python src/eda.py                  # sensor variance, trajectories, condition shift
python src/feature_engineering.py  # per-condition normalization + rolling features
python src/train_xgb_rf.py         # train + evaluate RF and XGBoost
python src/train_lstm.py           # train + evaluate the LSTM
python src/compare_models.py       # combine results, generate comparison plots
```

## What I found in EDA

- FD001/FD003 (single condition): sensors 1, 5, 10, 16, 18, 19 are
  completely flat, no signal, dropped them.
- FD002/FD004 (six conditions): those same sensors turn out to carry
  real info, but only once you normalize per operating condition first
  -- raw values are dominated by which condition the engine's in, not
  by wear.
- Best degradation sensors overall: 2, 3, 4, 11, 15 trend up over an
  engine's life, sensor 7 trends down.
- FD003 has two different fault modes and you can actually see it --
  different engines show different sensor signatures near failure.

## Results

Test set RMSE (cycles) and PHM08 score, lower is better on both:

| Subset | RandomForest RMSE | XGBoost RMSE | LSTM RMSE | RandomForest PHM08 | XGBoost PHM08 | LSTM PHM08 |
|---|---|---|---|---|---|---|
| FD001 | 19.03 | **17.36** | 42.80 | 1063.4 | **680.4** | 30301.4 |
| FD002 | 28.24 | 27.91 | **26.15** | 11548.6 | 12212.1 | **8587.7** |
| FD003 | 18.68 | **17.77** | 44.56 | 1230.8 | **816.1** | 50840.5 |
| FD004 | 28.98 | 28.49 | **26.72** | 6415.96 | 5450.5 | **4650.5** |

The interesting part isn't the raw numbers, it's the pattern: **the
tabular models (XGBoost especially) win comfortably on FD001/FD003,
and the LSTM wins on FD002/FD004** -- a clean flip, not just noise.

Why this makes sense:

- FD001/FD003 only have 100 training engines each, and a single
  operating condition. There's not much data for an LSTM to learn a
  temporal representation from scratch, whereas the rolling-window
  features (mean/std/slope) hand the tabular models a lot of the
  useful trend information directly.
- FD002/FD004 have 2.5x more training engines (260/249) and a much
  harder input space (6 operating conditions, and for FD004, 2 fault
  modes). With more data and more complexity, the LSTM has enough to
  work with to learn patterns that weren't fully captured by my
  hand-picked rolling features, and it pulls ahead.

Basic takeaway: simple engineered features beat a deep sequence model
on small, simple data. The deep model catches up and wins once there's
more data and the problem gets genuinely harder to hand-engineer for.

See `results/comparison_rmse.html` and `results/comparison_phm08.html`
for interactive plots (open in a browser, hover for exact values), or
`results/combined_test_results.csv` for the raw numbers.

## Status

- [x] Data loading + RUL labeling
- [x] EDA
- [x] Feature engineering (per-condition normalization for FD002/FD004,
      rolling window features)
- [x] XGBoost / Random Forest baseline
- [x] LSTM model (PyTorch)
- [x] Full comparison + writeup