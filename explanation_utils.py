import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import streamlit as st

# Path to your dataset for background sampling
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"

# Default feature names (can be overridden by argument)
DEFAULT_FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def align_feature_names(input_data: pd.DataFrame, expected_names: list) -> pd.DataFrame:
    """
    Align input_data columns to match expected_names (case/underscore insensitive).
    Fill missing columns with 0.
    """
    # Build mapping from normalized name to actual input column
    norm = lambda s: s.lower().replace(' ', '').replace('_', '')
    input_map = {norm(col): col for col in input_data.columns}
    aligned = {}
    for name in expected_names:
        n = norm(name)
        if n in input_map:
            aligned[name] = input_data[input_map[n]]
        else:
            aligned[name] = 0  # Fill missing with 0
    return pd.DataFrame([aligned])

@st.cache_data(show_spinner=False)
def load_data_sample(sample_size: int = 100, feature_names=None) -> pd.DataFrame:
    """
    Loads a background sample for SHAP/LIME explanations.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES
    try:
        df = pd.read_csv(DATA_FILE, usecols=feature_names)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return df.sample(n=min(len(df), sample_size), random_state=42)
    except Exception as e:
        st.error(f"Failed to load background data: {e}")
        return pd.DataFrame(columns=feature_names)


def generate_shap_explanation(model, scaler, input_data: pd.DataFrame, feature_names: list) -> plt.Figure:
    """
    Generate a compact horizontal bar chart of SHAP values for the positive class.
    """
    # Align input_data columns to scaler/model expected names
    if hasattr(scaler, 'feature_names_in_'):
        expected_names = list(scaler.feature_names_in_)
    else:
        expected_names = feature_names
    input_data_aligned = align_feature_names(input_data, expected_names)
    class_index = 1
    # Scale and prepare DataFrame
    X_scaled = scaler.transform(input_data_aligned)
    X_df = pd.DataFrame(X_scaled, columns=expected_names)

    # Background for KernelExplainer
    background = load_data_sample(100, expected_names)
    if not background.empty:
        bg_scaled = scaler.transform(background)
        bg_df = pd.DataFrame(bg_scaled, columns=expected_names)
    else:
        bg_df = X_df

    # Compute SHAP values
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_df)
        # shap_vals is list for binary: two arrays
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[class_index])[0]
        else:
            # shap_vals array shape (1, features)
            vals = shap_vals[0]
    except Exception:
        # Fallback to KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(bg_df, 10))
        shap_vals = explainer.shap_values(X_df, nsamples=100)
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[class_index])[0]
        else:
            vals = np.array(shap_vals)[0]

    # Select top features by absolute effect
    max_display = min(len(expected_names), 10)
    inds = np.argsort(np.abs(vals))[-max_display:]
    top_features = [expected_names[i] for i in inds][::-1]  # reverse for largest at top
    top_vals = vals[inds][::-1]

    # Plot horizontal bar
    fig, ax = plt.subplots(figsize=(4, 2), dpi=80)
    ax.barh(top_features, top_vals)
    ax.set_xlabel('SHAP Value')
    plt.tight_layout()
    return fig


def generate_lime_explanation(model, scaler, input_data: pd.DataFrame, feature_names: list,
                            class_names: list = ["Benign", "Attack"]) -> plt.Figure:
    """
    Generate a compact LIME explanation plot for the positive class.
    """
    # Align input_data columns to scaler/model expected names
    if hasattr(scaler, 'feature_names_in_'):
        expected_names = list(scaler.feature_names_in_)
    else:
        expected_names = feature_names
    input_data_aligned = align_feature_names(input_data, expected_names)
    # Scale inputs
    X_scaled = scaler.transform(input_data_aligned)
    instance = X_scaled[0]

    # Background sample
    background = load_data_sample(200, expected_names)
    bg = background if not background.empty else input_data_aligned
    bg_scaled = scaler.transform(bg)

    # Build LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg_scaled,
        feature_names=expected_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=10
    )

    # Compact figure size to match SHAP
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(4, 2)
    fig.set_dpi(80)
    plt.tight_layout()
    return fig
