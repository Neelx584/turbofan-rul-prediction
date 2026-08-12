import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
 
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
 
MODEL_ORDER = ["RandomForest", "XGBoost", "LSTM"]
MODEL_COLORS = {"RandomForest": "#4C72B0", "XGBoost": "#DD8452", "LSTM": "#55A868"}
SUBSET_ORDER = ["FD001", "FD002", "FD003", "FD004"]
 
 
def load_combined_test_results() -> pd.DataFrame:
    xgb_rf_path = RESULTS_DIR / "xgb_rf_results.csv"
    lstm_path = RESULTS_DIR / "lstm_results.csv"
 
    if not xgb_rf_path.exists() or not lstm_path.exists():
        raise FileNotFoundError(
            "need both xgb_rf_results.csv and lstm_results.csv in results/ -- "
            "run train_xgb_rf.py and train_lstm.py first"
        )
 
    xgb_rf = pd.read_csv(xgb_rf_path)
    lstm = pd.read_csv(lstm_path)
 
    combined = pd.concat([xgb_rf, lstm], ignore_index=True)
    return combined[combined["split"] == "test"].copy()
 
 
def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, log_scale: bool, out_name: str, title: str):
    fig = go.Figure()
 
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model].set_index("subset")
        values = [model_df.loc[s, metric] if s in model_df.index else None for s in SUBSET_ORDER]
        fig.add_trace(go.Bar(
            name=model,
            x=SUBSET_ORDER,
            y=values,
            marker_color=MODEL_COLORS[model],
            hovertemplate=f"%{{x}}<br>{model}<br>{ylabel}: %{{y:.1f}}<extra></extra>",
        ))
 
    fig.update_layout(
        title=title,
        xaxis_title="Subset",
        yaxis_title=ylabel,
        yaxis_type="log" if log_scale else "linear",
        barmode="group",
        template="plotly_white",
        legend_title="Model",
    )
 
    out_path = RESULTS_DIR / out_name
    fig.write_html(out_path)
    print(f"  saved {out_name}")
 
 
if __name__ == "__main__":
    combined = load_combined_test_results()
 
    print("Combined test results:")
    pivot = combined.pivot(index="subset", columns="model", values=["rmse", "phm08_score"])
    pivot = pivot.reindex(SUBSET_ORDER)
    print(pivot.to_string())
 
    combined.to_csv(RESULTS_DIR / "combined_test_results.csv", index=False)
    print(f"\nsaved combined table to {RESULTS_DIR / 'combined_test_results.csv'}")
 
    print("\ngenerating comparison plots...")
    plot_metric(
        combined, "rmse", "RMSE (cycles)", log_scale=False,
        out_name="comparison_rmse.html", title="RMSE by subset and model (test set)",
    )
    plot_metric(
        combined, "phm08_score", "PHM08 score", log_scale=True,
        out_name="comparison_phm08.html", title="PHM08 score by subset and model (test set)",
    )
 
    print(f"\nall done, check {RESULTS_DIR} -- open the .html files in a browser")