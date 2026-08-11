# Turbofan Engine RUL Prediction

Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using
NASA's C-MAPSS degradation simulation dataset (PHM08 challenge data).

Two modeling approaches, compared:
- Feature engineering + XGBoost / Random Forest
- LSTM sequence model

Across all four C-MAPSS subsets (FD001-FD004), which vary in operating
conditions (1 vs 6) and fault modes (1 vs 2).

## Data

Not included in this repo. Download from the official NASA Website
NASA source:
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Unzip into a `CMAPSSData/` folder in the project root:

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

## Usage

```
python src/load_data.py   # load + label all 4 subsets, sanity check
python src/eda.py         # sensor variance, trajectory plots, condition-shift analysis
```

## Findings so far

-> FD001/FD003 (single operating condition): sensors 1, 5, 10, 16, 18, 19
  are flat and carry no signal, these are dropped for these subsets.
  
-> FD002/FD004 (six operating conditions): those same sensors show
  variance, but it's driven by which condition the engine is in, not
  degradation. Raw sensor values must be normalized per operating
  condition before degradation trends become visible.
  
-> Clearest degradation signal (single condition data): sensors 2, 3, 4,
  11, 15 trend upward over engine life; sensor 7 trends downward.
  
-> FD003 has two distinct fault modes (HPC and fan degradation).
  Different engines show different sensor signatures at failure.

## Status

- [x] Data loading + RUL labeling
- [x] EDA (sensor variance, trajectories, operating condition analysis)
- [x] Feature engineering (per-condition normalization for FD002/FD004)
- [ ] XGBoost/RF model
- [ ] LSTM model
- [ ] PHM08 scoring function + evaluation
