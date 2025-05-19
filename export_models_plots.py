import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Define the models and their output directories
MODEL_OUTPUTS = {
    "GBM": ("gbm_outputs", "gbm_model.joblib"),
    "LightGBM": ("lightgbm_outputs", "lightgbm_model.joblib"),
    "XGBoost": ("xgboost_outputs", "xgboost_model.joblib"),
    "Random Forest": ("randomforest_outputs", "randomforest_model.joblib"),
    "CatBoost": ("catboost_outputs", "catboost_model.joblib"),
}

# Use the same feature names as in your app.py
FEATURE_NAMES = [
    "ACK Flag Count", "Active Max", "Active Mean", "Active Min", "Active Std",
    "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_mean",
    "bwd_pkt_len_mean", "flow_byts_s", "flow_pkts_s",
    "init_win_byts_fwd", "init_win_byts_bwd"
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

for model_name, (subdir, model_file) in MODEL_OUTPUTS.items():
    model_path = os.path.join(MODELS_DIR, subdir, model_file)
    plots_dir = os.path.join(MODELS_DIR, subdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[!] {model_name} model not found at {model_path}")
        continue

    try:
        model = joblib.load(model_path)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Ensure importances and feature names match
            if len(importances) != len(FEATURE_NAMES):
                print(f"[!] {model_name} feature importances length mismatch. Skipping plot.")
                continue
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title(f"{model_name} Feature Importances")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [FEATURE_NAMES[i] for i in indices], rotation=90)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "feature_importance.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"[+] Saved feature importance plot for {model_name} at {plot_path}")
        else:
            print(f"[!] {model_name} does not support feature_importances_. Skipping plot.")
    except Exception as e:
        print(f"[!] Error loading or plotting for {model_name}: {e}") 