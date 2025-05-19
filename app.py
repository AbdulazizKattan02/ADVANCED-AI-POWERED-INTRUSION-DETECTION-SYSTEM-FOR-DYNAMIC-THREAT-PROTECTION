import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import base64
from io import BytesIO
import tempfile
import pyshark # Keep for potential future use or if realtime_capture needs it indirectly
import threading
import time
import matplotlib.pyplot as plt # Needed for SHAP/LIME plots
import collections # Import collections for deque check
from collections import Counter # Import Counter for consensus analysis
import plotly.graph_objects as go # Import Plotly for charts
import traceback
import model_loader
import realtime_capture
import explanation_utils

# Set page configuration
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject Custom CSS ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

# Define paths (Updated to relative paths for sandbox compatibility)
PROJECT_DIR = "."
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_FILE = os.path.join(PROJECT_DIR, "results.csv") # Define results file path
CSS_FILE = os.path.join(PROJECT_DIR, "style.css") # Define CSS file path

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define attack types (Keep as is)
ATTACK_TYPES = {
    "benign": "Benign Traffic",
    "dos": "Denial of Service (DoS)",
    "probe": "Probing/Scanning",
    "web": "Web Attack"
}

# Feature names used for training (from retrain_trained_models.py)
FEATURE_NAMES = [
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

# Function to get feature importance (Keep as is, but ensure model compatibility)
def get_feature_importance(model, feature_names):
    """
    Get feature importance from model.

    Args:
        model: Trained model
        feature_names (list): List of feature names

    Returns:
        dict: Dictionary containing feature importance
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Ensure importances length matches feature_names length
            if len(importances) == len(feature_names):
                return {feature: importance for feature, importance in zip(feature_names, importances)}
            else:
                print(f"Warning: Feature importance length ({len(importances)}) doesn't match feature names length ({len(feature_names)}). Returning zeros.")
                return {feature: 0 for feature in feature_names}
        # Add support for coefficients if applicable (e.g., linear trained_models, though not used here)
        elif hasattr(model, "coef_"):
            # Assuming binary classification, take the first set of coefficients
            if model.coef_.ndim > 1:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
            if len(importances) == len(feature_names):
                return {feature: importance for feature, importance in zip(feature_names, importances)}
            else:
                print(f"Warning: Coefficient length ({len(importances)}) doesn't match feature names length ({len(feature_names)}). Returning zeros.")
                return {feature: 0 for feature in feature_names}
        else:
            print(f"Model type {type(model)} does not have feature_importances_ or coef_ attribute. Returning zeros.")
            return {feature: 0 for feature in feature_names}
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return {feature: 0 for feature in feature_names}

# Function to predict with explanation (Update to use FEATURE_NAMES)
def predict_with_explanation(model_dict, input_data, model_name):
    """
    Make prediction with explanation using the globally defined FEATURE_NAMES.

    Args:
        model_dict (dict): Dictionary containing model and scaler
        input_data (pd.DataFrame): Input data for prediction (should have FEATURE_NAMES columns)
        model_name (str): Name of the model

    Returns:
        dict: Dictionary containing prediction results
    """
    try:
        model = model_dict["model"]
        scaler = model_dict["scaler"]

        # Ensure input_data has the correct columns
        if not all(col in input_data.columns for col in FEATURE_NAMES):
            st.error(f"Input data is missing required features. Expected: {FEATURE_NAMES}")
            return None # Or return default error structure

        # Select and order columns
        input_data_ordered = input_data[FEATURE_NAMES]

        # Handle potential negative window sizes (-1 replaced with 0 during training)
        if "init_win_byts_fwd" in input_data_ordered.columns:
            input_data_ordered["init_win_byts_fwd"] = input_data_ordered["init_win_byts_fwd"].replace(-1, 0)
        if "init_win_byts_bwd" in input_data_ordered.columns:
            input_data_ordered["init_win_byts_bwd"] = input_data_ordered["init_win_byts_bwd"].replace(-1, 0)

        # Scale input data
        input_data_scaled = scaler.transform(input_data_ordered)
        # Create DataFrame with feature names for trained_models that require it (like LightGBM warning seen during training)
        input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=FEATURE_NAMES)

        # Make prediction
        prediction = model.predict(input_data_scaled_df)[0]
        prediction_proba = model.predict_proba(input_data_scaled_df)[0]

        return {
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "feature_importance": get_feature_importance(model, FEATURE_NAMES)
        }
    except Exception as e:
        st.error(f"Error in prediction for {model_name}: {e}")
        st.code(traceback.format_exc(), language="python")
        st.text(traceback.format_exc())
        print(f"[{model_name}] Prediction error:\n{traceback.format_exc()}")
        # Return default values or None
        return {
            "prediction": 0, # Default to benign on error?
            "prediction_proba": np.array([1.0, 0.0]),
            "feature_importance": {feature: 0 for feature in FEATURE_NAMES}
        }

# Function to classify attack type (Keep as is)
def classify_attack_type(prediction_proba, threshold=0.5):
    """
    Classify attack type based on prediction probability.

    Args:
        prediction_proba (np.ndarray): Prediction probability [prob_benign, prob_attack]
        threshold (float): Threshold for classifying as benign

    Returns:
        str: Attack type ("benign", "dos", "probe", "web")
    """
    if prediction_proba[0] >= threshold:
        return "benign"

    # Simplified multi-class simulation based on attack probability
    attack_prob = prediction_proba[1]
    if attack_prob < 0.6:
        return "probe"      # Port Scan
    elif attack_prob < 0.8:
        return "dos"        # Denial of Service (DoS)
    else:
        return "web"        # Web Attack

# Simple function to parse binary file (Keep for now, but update features)
def simple_parse_pcap(uploaded_file):
    """
    Simplified function to parse binary file and extract basic network features.
    This is a placeholder and does not accurately reflect real PCAP parsing.
    It generates dummy data matching the FEATURE_NAMES structure.

    Args:
        uploaded_file: Uploaded file object (content not actually used)

    Returns:
        pd.DataFrame: DataFrame containing dummy features for one sample
    """
    st.warning("PCAP parsing is using simplified dummy data generation.")
    try:
        dummy_features = {}
        for feature in FEATURE_NAMES:
            # Assign plausible random values based on feature name
            if "Flag" in feature or "Count" in feature:
                dummy_features[feature] = int(np.random.randint(0, 2))
            elif "Port" in feature:
                dummy_features[feature] = int(np.random.randint(0, 65536))
            elif "Packets" in feature or "Packet" in feature or "Bytes" in feature or "Length" in feature:
                dummy_features[feature] = int(np.random.randint(0, 10000))
            elif ("Duration" in feature or "Mean" in feature or "Std" in feature or "Variance" in feature or 
                "Ratio" in feature or "Rate" in feature or "/s" in feature or "Min" in feature or "Max" in feature):
                dummy_features[feature] = float(np.random.uniform(0, 1000))
            else:
                dummy_features[feature] = float(np.random.uniform(0, 1000))

        df = pd.DataFrame([dummy_features], columns=FEATURE_NAMES)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        return df
    except Exception as e:
        st.error(f"Error generating dummy PCAP data: {e}")
        return None

# Function to generate AI-based sample data (Keep for now, but update features)
def generate_ai_sample_data(sample_type):
    """
    Generate AI-based sample data matching FEATURE_NAMES structure.
    This uses heuristics based on attack types.

    Args:
        sample_type (str): Type of sample data to generate (e.g., "Benign Traffic")

    Returns:
        pd.DataFrame: DataFrame containing generated sample data
    """
    np.random.seed(int(time.time())) # Use time for seed

    # Base values matching FEATURE_NAMES structure
    base_values = {
        "Benign Traffic": {
            "flow_duration": (1.0, 10.0), "tot_fwd_pkts": (2, 20), "tot_bwd_pkts": (2, 20),
            "totlen_fwd_pkts": (500, 5000), "totlen_bwd_pkts": (500, 5000),
            "fwd_pkt_len_mean": (50, 500), "bwd_pkt_len_mean": (50, 500),
            "init_win_byts_fwd": (8192, 65535), "init_win_byts_bwd": (1024, 65535)
        },
        "Denial of Service (DoS)": {
            "flow_duration": (0.01, 1.0), "tot_fwd_pkts": (10, 500), "tot_bwd_pkts": (0, 10),
            "totlen_fwd_pkts": (1000, 50000), "totlen_bwd_pkts": (0, 1000),
            "fwd_pkt_len_mean": (100, 1000), "bwd_pkt_len_mean": (0, 200),
            "init_win_byts_fwd": (0, 1024), "init_win_byts_bwd": (0, 256)
        },
        "Probing/Scanning": {
            "flow_duration": (0.5, 5.0), "tot_fwd_pkts": (1, 5), "tot_bwd_pkts": (0, 3),
            "totlen_fwd_pkts": (0, 500), "totlen_bwd_pkts": (0, 500),
            "fwd_pkt_len_mean": (0, 100), "bwd_pkt_len_mean": (0, 100),
            "init_win_byts_fwd": (1024, 65535), "init_win_byts_bwd": (0, 1024)
        },
        "Web Attack": {
            "flow_duration": (0.1, 5.0), "tot_fwd_pkts": (3, 20), "tot_bwd_pkts": (3, 20),
            "totlen_fwd_pkts": (200, 5000), "totlen_bwd_pkts": (200, 10000),
            "fwd_pkt_len_mean": (50, 300), "bwd_pkt_len_mean": (50, 500),
            "init_win_byts_fwd": (1024, 65535), "init_win_byts_bwd": (256, 65535)
        }
    }

    base = base_values[sample_type]
    jitter = np.random.uniform(0.8, 1.2)
    features = {}

    for key, (low, high) in base.items():
        if isinstance(low, int):
            features[key] = np.random.randint(low, high + 1) * jitter
        else:
            features[key] = np.random.uniform(low, high) * jitter

    # Calculate rates based on generated values
    duration = features["flow_duration"]
    if duration == 0:
        features["flow_byts_s"] = 0
        features["flow_pkts_s"] = 0
    else:
        features["flow_byts_s"] = (features["totlen_fwd_pkts"] + features["totlen_bwd_pkts"]) / duration
        features["flow_pkts_s"] = (features["tot_fwd_pkts"] + features["tot_bwd_pkts"]) / duration

    # Ensure non-negativity for counts and lengths
    for col in ["tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "init_win_byts_fwd", "init_win_byts_bwd"]:
        features[col] = max(0, int(features[col]))
    for col in ["flow_duration", "fwd_pkt_len_mean", "bwd_pkt_len_mean", "flow_byts_s", "flow_pkts_s"]:
        features[col] = max(0, features[col])

    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    df = df.replace([np.inf, -np.inf], 0).fillna(0) # Replace inf/nan

    return df

# Function to get download link for DataFrame
def get_table_download_link(df, filename="results.csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV File</a>'
    return href

# Main function
def main():
    # Inject custom CSS
    load_css(CSS_FILE)

    # --- Session State Initialization ---
    if "capture_running" not in st.session_state:
        st.session_state.capture_running = False
    if "capture_error" not in st.session_state:
        st.session_state.capture_error = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "realtime_results" not in st.session_state:
        st.session_state.realtime_results = collections.deque(maxlen=realtime_capture.MAX_PACKETS_DISPLAY)

    # --- Load trained_models ---
    trained_models = model_loader.load_models()
    if not trained_models:
        st.error("Failed to load trained_models. Application cannot proceed.")
        return

    # --- Sidebar Setup ---
    with st.sidebar:
        st.title("Network Traffic Classifier")
        st.markdown("## 🛡️ Network Security")
        available_trained_models = list(trained_models.keys())
        selected_trained_models = st.multiselect(
            "Select trained_models (Multiple Selection Enabled)", # Updated label
            available_trained_models,
            default=available_trained_models # Default to all trained_models selected as per screenshot
        )
        if not selected_trained_models:
            st.warning("Please select at least one model.")
            st.stop()

        st.markdown("**Input Method**") # Make label bold
        input_method = st.radio(
            "Input Method", # Keep label for internal logic, hide with label_visibility
            ["Manual Input", "Upload CSV", "Upload PCAP", "AI-Generated Sample Data", "Real-time Capture"],
            label_visibility="collapsed" # Hide the default radio label
        )

        # Save Results Button (Appears if there are results)
        if "analysis_results" in st.session_state and len(st.session_state.analysis_results) > 0:
            st.sidebar.markdown("---")
            st.sidebar.markdown("## 💾 Save Results")
            if st.sidebar.button("Save All Results to CSV", key="save_all_results_button_sidebar_enhanced3"): # Unique key
                try:
                    results_df = pd.DataFrame(st.session_state.analysis_results)
                    
                    # More robust column handling for cases where not all trained_models run or produce all outputs
                    all_cols_in_results = set()
                    for record in st.session_state.analysis_results:
                        if isinstance(record, dict): # Check if record is a dictionary
                            all_cols_in_results.update(record.keys())

                    # Start with FEATURE_NAMES that are actually present
                    ordered_columns = [col for col in FEATURE_NAMES if col in all_cols_in_results]
                    
                    # Add model-specific columns that are present
                    model_specific_cols = []
                    # 'available_trained_models' is defined in the 'with st.sidebar:' scope earlier
                    for model_key in available_trained_models: 
                        pred_col = f"{model_key}_Prediction"
                        prob_col = f"{model_key}_AttackProb"
                        # Check if these columns were actually produced and are in results
                        if pred_col in all_cols_in_results: model_specific_cols.append(pred_col)
                        if prob_col in all_cols_in_results: model_specific_cols.append(prob_col)
                    
                    # Add any other columns that might have been added (e.g., from realtime capture, or other metadata)
                    other_present_cols = [col for col in all_cols_in_results if col not in ordered_columns and col not in model_specific_cols]
                    
                    # Construct the final column order: features first, then model predictions/probabilities, then others.
                    final_column_order = ordered_columns + sorted(list(set(model_specific_cols))) + sorted(other_present_cols)
                    
                    results_df = results_df.reindex(columns=final_column_order).fillna("N/A") # Fill missing with N/A for consistency

                    results_df.to_csv(RESULTS_FILE, index=False)
                    st.sidebar.success(f"Results saved to {RESULTS_FILE}")
                    st.sidebar.markdown(get_table_download_link(results_df, "analysis_results.csv"), unsafe_allow_html=True)
                except Exception as e:
                    st.sidebar.error(f"Error saving results: {str(e)[:200]}") # Limit error message length for UI
                    print(f"Error saving results: {traceback.format_exc()}")

    # --- Main Page Title ---
    st.title("ADVANCED AI-POWERED INTRUSION DETECTION SYSTEM FOR DYNAMIC THREAT PROTECTION")
    st.markdown("This application uses machine learning to classify network traffic as Benign or Harmful, and identifies the type of attack if harmful.")

    input_data = None
    input_df_display = None
    analysis_triggered = False # Flag to know when to display results

    # --- Input Section ---
    st.markdown("### Enter Network Traffic Features")

    if input_method == "Manual Input":
        with st.form(key="manual_input_form"):
            col1, col2, col3 = st.columns(3)
            input_values = {}
            # Define typical values for 'Benign' traffic
            benign_defaults = {
                "Flow Duration": (1.0 + 10.0) / 2,
                "Total Fwd Packets": (2 + 20) / 2,
                "Total Backward Packets": (2 + 20) / 2,
                "Total Length of Fwd Packets": (500 + 5000) / 2,
                "Total Length of Bwd Packets": (500 + 5000) / 2,
                "Fwd Packet Length Mean": (50 + 500) / 2,
                "Bwd Packet Length Mean": (50 + 500) / 2,
                "Init_Win_bytes_forward": (8192 + 65535) / 2,
                "Init_Win_bytes_backward": (1024 + 65535) / 2,
            }
            # Calculate rates based on generated values
            benign_defaults["Flow Bytes/s"] = (benign_defaults["Total Length of Fwd Packets"] + benign_defaults["Total Length of Bwd Packets"]) / benign_defaults["Flow Duration"]
            benign_defaults["Flow Packets/s"] = (benign_defaults["Total Fwd Packets"] + benign_defaults["Total Backward Packets"]) / benign_defaults["Flow Duration"]

            cols = st.columns(3)
            for i, feature in enumerate(FEATURE_NAMES):
                with cols[i % 3]:
                    min_val = 0
                    max_val = None
                    # Use benign default if available, else fallback
                    if feature in benign_defaults:
                        default_val = benign_defaults[feature]
                    elif "Flag" in feature or "Count" in feature:
                        default_val = 0
                    elif "Mean" in feature or "Std" in feature or "Variance" in feature:
                        default_val = 10.0
                    elif "Max" in feature:
                        default_val = 1000.0
                    elif "Min" in feature:
                        default_val = 0.0
                    elif "Length" in feature or "Bytes" in feature:
                        default_val = 1000.0
                    elif "/s" in feature or "Rate" in feature:
                        default_val = 100.0
                    else:
                        # Final fallback: use 0 for int, 0.0 for float
                        default_val = 0.0

                    # Set step and format
                    if isinstance(default_val, float):
                        step = 0.1
                        format_str = "%.2f"
                        min_val = float(min_val)
                        max_val = float(max_val) if max_val is not None else None
                    else:
                        step = 1
                        format_str = "%d"
                        min_val = int(min_val)
                        max_val = int(max_val) if max_val is not None else None

                    input_values[feature] = st.number_input(
                        label=feature.replace("_", " ").title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step,
                        format=format_str,
                        key=f"manual_{feature}"
                    )
            submitted = st.form_submit_button("Classify Traffic")
            if submitted:
                input_list = [input_values[feature] for feature in FEATURE_NAMES]
                input_data = pd.DataFrame([input_list], columns=FEATURE_NAMES)
                input_df_display = input_data.copy()
                st.session_state.analysis_results = []
                analysis_triggered = True

    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                if not all(col in input_data.columns for col in FEATURE_NAMES):
                    st.error(f"Uploaded CSV is missing required features. Expected: {FEATURE_NAMES}")
                    input_data = None
                else:
                    input_data = input_data[FEATURE_NAMES].head(1) # Process only the first row
                    input_df_display = input_data.copy()
                    st.session_state.analysis_results = []
                    analysis_triggered = True # Trigger analysis display
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                input_data = None

    elif input_method == "Upload PCAP":
        uploaded_file = st.file_uploader("Choose a PCAP file", type=["pcap", "pcapng"])
        if uploaded_file is not None:
            input_data = simple_parse_pcap(uploaded_file)
            if input_data is not None:
                input_df_display = input_data.copy()
                st.session_state.analysis_results = []
                analysis_triggered = True # Trigger analysis display

    elif input_method == "AI-Generated Sample Data":
        sample_type = st.selectbox("Select Sample Type", list(ATTACK_TYPES.values()))
        if st.button("Generate and Analyze Sample"):
            input_data = generate_ai_sample_data(sample_type)
            if input_data is not None:
                input_df_display = input_data.copy()
                st.session_state.analysis_results = []
                analysis_triggered = True # Trigger analysis display

    elif input_method == "Real-time Capture":
        # (Keep existing Real-time Capture logic - assumes CSS handles styling)
        st.header("Real-time Network Traffic Capture")
        interface = st.text_input("Enter network interface name (e.g., eth0, en0)", value="eth0")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Capture", key="start_capture", disabled=st.session_state.capture_running):
                if interface:
                    st.session_state.capture_error = None
                    st.session_state.realtime_results.clear()
                    st.session_state.capture_thread = threading.Thread(
                        target=realtime_capture.start_capture,
                        args=(interface, realtime_capture.packet_handler, st.session_state),
                        daemon=True
                    )
                    st.session_state.capture_running = True
                    st.session_state.capture_thread.start()
                    st.success(f"Capture started on {interface}...")
                    time.sleep(0.5) # Give thread time to start
                    st.rerun()
                else:
                    st.warning("Please enter a network interface name.")
        with col2:
            if st.button("Stop Capture", key="stop_capture", disabled=not st.session_state.capture_running):
                if st.session_state.capture_running:
                    realtime_capture.stop_capture(st.session_state)
                    st.success("Capture stopped.")
                    time.sleep(0.5) # Give thread time to stop
                    st.rerun()

        results_placeholder = st.empty()
        if st.session_state.capture_error:
            st.error(f"Capture Error: {st.session_state.capture_error}")
            st.session_state.capture_running = False # Ensure state is correct on error

        if st.session_state.capture_running or len(st.session_state.realtime_results) > 0:
            with results_placeholder.container():
                st.subheader("Live Packet Analysis")
                if not st.session_state.realtime_results:
                    st.info("Waiting for packets...")
                else:
                    # Display latest results from the deque
                    results_to_display = list(st.session_state.realtime_results)
                    df_display = pd.DataFrame(results_to_display)
                    st.dataframe(df_display, use_container_width=True)

            # Rerun only if capture is actively running
            if st.session_state.capture_running:
                time.sleep(1) # Refresh interval
                try:
                    st.rerun()
                except Exception as e:
                    # Handle potential errors during rerun if needed
                    print(f"Error during rerun: {e}")
                    st.session_state.capture_running = False # Stop if rerun fails badly
                    st.rerun()

    # --- Analysis Results Section (Displayed only after analysis is triggered for non-realtime) ---
    if analysis_triggered and input_data is not None:
        st.markdown("--- ") # Separator
        st.markdown("### Classification Results")

        # Run analysis for each selected model
        model_results_list = []
        all_feature_importances = {}
        for model_name in selected_trained_models:
            if model_name in trained_models:
                model_dict = trained_models[model_name]
                result = predict_with_explanation(model_dict, input_data, model_name)
                if result:
                    prediction = result["prediction"]
                    prediction_proba = result["prediction_proba"]
                    feature_importance = result["feature_importance"]
                    attack_type = classify_attack_type(prediction_proba)
                    attack_prob = prediction_proba[1]
                    benign_prob = prediction_proba[0]
                    model_results_list.append({
                        "Model": model_name,
                        "Prediction": "Attack" if prediction == 1 else "Benign",
                        "Attack Type": ATTACK_TYPES.get(attack_type, "Unknown") if prediction == 1 else "N/A",
                        "Benign Probability": benign_prob,
                        "Attack Probability": attack_prob,
                        "ResultDict": result
                    })
                    all_feature_importances[model_name] = feature_importance
                else:
                    # Keep error handling for model analysis failure
                    model_results_list.append({
                        "Model": model_name, "Prediction": "Error", "Attack Type": "N/A",
                        "Benign Probability": np.nan, "Attack Probability": np.nan, "ResultDict": None
                    })
            else:
                # Keep error handling for model loading failure
                model_results_list.append({
                    "Model": model_name, "Prediction": "Missing", "Attack Type": "N/A",
                    "Benign Probability": np.nan, "Attack Probability": np.nan, "ResultDict": None
                })

        # --- Display Detailed Model Results in Tabs ---
        if model_results_list:
            # Use model names directly for tabs
            tab_titles = [f"{name}" for name in selected_trained_models]
            model_tabs = st.tabs(tab_titles)

            for i, model_name in enumerate(selected_trained_models):
                with model_tabs[i]:
                    model_result_data = next((item for item in model_results_list if item["Model"] == model_name), None)
                    if model_result_data and model_result_data["ResultDict"]:
                        result = model_result_data["ResultDict"]
                        prediction = result["prediction"]
                        prediction_proba = result["prediction_proba"]
                        feature_importance = result["feature_importance"]
                        attack_type = model_result_data["Attack Type"]
                        attack_prob = model_result_data["Attack Probability"]
                        benign_prob = model_result_data["Benign Probability"]
                        # 🔧 Add this right after
                        attack_prob = np.nan_to_num(attack_prob)
                        benign_prob = np.nan_to_num(benign_prob)

                        col1, col2 = st.columns([2, 1]) # Adjusted ratio closer to screenshot
                        with col1:
                            # Prediction Box (Using Markdown with custom class)
                            st.markdown("**Prediction**")
                            if prediction == 1:
                                # Use markdown with a class for styling via CSS
                                st.markdown(f'<div class="prediction-box-error">Harmful Traffic: {attack_type}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="prediction-box-success">Benign Traffic</div>', unsafe_allow_html=True)
                            # Confidence Metric (Looks like screenshot)
                            st.metric("Confidence", f"{attack_prob*100:.2f}%" if prediction == 1 else f"{benign_prob*100:.2f}%")

                            # Prediction Probabilities Chart (Plotly)
                            st.markdown("**Prediction Probabilities**")
                            # Use a simpler bar chart as in the screenshot
                            fig_single_prob = go.Figure(data=[
                                go.Bar(name='Benign', x=['Benign'], y=[benign_prob], marker_color='#aec7e8'), # Light blue
                                go.Bar(name='Attack', x=['Attack'], y=[attack_prob], marker_color='#1f77b4') # Dark blue
                            ])
                            fig_single_prob.update_layout(
                                title_text='',
                                yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=True),
                                xaxis=dict(showgrid=False, zeroline=False),
                                showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), # Reduced margins
                                height=200,
                                paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                                plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                                font_color='#FAFAFA' # Light text
                            )
                            st.plotly_chart(fig_single_prob, use_container_width=True)

                        with col2:
                            # Feature Values Table
                            st.markdown("**Feature Values**")
                            if input_df_display is not None:
                                # Transpose for better vertical display if many features
                                input_features_df = input_df_display.iloc[0].reset_index()
                                input_features_df.columns = ["Feature", "Value"]
                                # Format values for display
                                input_features_df['Value'] = input_features_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                                st.dataframe(input_features_df, height=350, use_container_width=True, hide_index=True)
                            else:
                                st.write("N/A")

                        st.markdown("--- ") # Separator

                        # Feature Importance
                        if feature_importance:
                            st.markdown("**Feature Importance**")
                            sorted_importance = sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
                            importance_df = pd.DataFrame(sorted_importance, columns=["Feature", "Importance"])
                            importance_df = importance_df.replace([np.inf, -np.inf], np.nan).dropna()
                            # Horizontal Bar Chart (Plotly)
                            fig_imp = go.Figure(go.Bar(
                                x=importance_df["Importance"].head(10),
                                y=importance_df["Feature"].head(10),
                                orientation='h',
                                marker_color='#1f77b4'
                            ))
                            fig_imp.update_layout(
                                title='',
                                xaxis_title="Importance Score",
                                yaxis_title="Feature",
                                yaxis={'categoryorder':'total ascending'},
                                margin=dict(l=10, r=10, t=30, b=10),
                                height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#FAFAFA'
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.markdown("Feature importance not available for this model.")

                        st.markdown("--- ") # Separator

                        # Interpretation (Sync color with prediction)
                        st.markdown("**Interpretation**")
                        interpretation_class = "interpretation-box-success" if prediction == 0 else "interpretation-box-error"
                        interpretation_text = f"The traffic has been classified as {'Benign' if prediction == 0 else 'Harmful'} and identified as a {attack_type if prediction == 1 else 'N/A'} attack by the {model_name} model."
                        st.markdown(f'<div class="{interpretation_class}">{interpretation_text}</div>', unsafe_allow_html=True)

                        # Add more detailed descriptions based on attack type
                        if prediction == 1:
                            if attack_type == ATTACK_TYPES["dos"]:
                                st.markdown("- **Characteristics:** Typically involves overwhelming network resources, high packet rates, and abnormal traffic patterns.")
                            elif attack_type == ATTACK_TYPES["probe"]:
                                st.markdown("- **Characteristics:** Often involves scanning ports or hosts, usually low volume traffic, potentially a precursor to larger attacks.")
                            elif attack_type == ATTACK_TYPES["web"]:
                                st.markdown("- **Characteristics:** Attempts to gain unauthorized local access from a remote machine, often exploiting vulnerabilities in network services.")
                            else:
                                st.markdown("- Specific characteristics depend on the attack subtype.")
                        # No specific characteristics needed for benign
                        st.markdown("**Key Influencing Features**")
                        if feature_importance:
                            sorted_importance = sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
                            if sorted_importance:
                                for feature, importance in sorted_importance[:3]: # Show top 3
                                    value = input_data[feature].iloc[0]
                                    # Format value appropriately before putting in f-string
                                    if isinstance(value, float):
                                        value_str = f"{value:.2f}"
                                    else:
                                        value_str = str(value) # Handle ints and other types
                                    st.markdown(f"- **{feature.replace('_', ' ').title()}**: `{value_str}` (Importance: {importance:.4f})")
                            else: # This else corresponds to 'if sorted_importance:'
                                st.write("No significant features identified.")
                        else: # This else corresponds to 'if feature_importance:'
                            st.write("Feature importance data not available.")                  # Recommendations
                        st.markdown("**Recommendations**")
                        if prediction == 1:
                            st.markdown("- Investigate the source and destination of this traffic.")
                            st.markdown("- Consider blocking the source IP address temporarily or permanently based on investigation.")
                            st.markdown("- Check system logs for any signs of successful exploitation.")
                            st.markdown("- Update firewall rules or Intrusion Detection/Prevention System (IDPS) signatures to detect similar patterns.")
                            st.markdown("- Enhance monitoring for the affected systems or network segments.")
                        else:
                            st.markdown("- Continue routine monitoring of network traffic.")

                        st.markdown("--- ") # Separator

                        # SHAP and LIME Explanations
                        with st.expander(f"Model Explanations (SHAP & LIME) for {model_name}", expanded=False):
                            # 1) Generate both figures
                            shap_fig = explanation_utils.generate_shap_explanation(
                                model_dict["model"], model_dict["scaler"], input_data, FEATURE_NAMES
                            )
                            lime_fig = explanation_utils.generate_lime_explanation(
                                model_dict["model"], model_dict["scaler"], input_data, FEATURE_NAMES
                            )

                            # 2) Two narrow columns
                            col1, col2 = st.columns(2)

                            # 3) SHAP on the left
                            with col1:
                                shap_fig.set_size_inches(4, 2)
                                shap_fig.set_dpi(80)
                                st.pyplot(shap_fig)

                            # 4) LIME on the right
                            with col2:
                                lime_fig.set_size_inches(4, 2)
                                lime_fig.set_dpi(80)
                                st.pyplot(lime_fig)


            # --- Display Model Comparison & Consensus Below Tabs ---
            st.markdown("--- ") # Separator
            st.markdown("### Model Comparison")
            results_df = pd.DataFrame(model_results_list)

            # Error display similar to screenshot (check for specific errors)
            # Example: Check if any model had a prediction error
            if "Error" in results_df["Prediction"].values:
                st.error("An error occurred during prediction for one or more trained_models. Please check logs or input data.")
            # Add more specific error checks if needed, e.g., feature mismatch

            # Prediction Comparison Table
            st.markdown("**Prediction Comparison**")
            display_df = results_df[["Model", "Prediction", "Attack Type", "Benign Probability", "Attack Probability"]].copy()
            display_df[["Benign Probability", "Attack Probability"]] = display_df[["Benign Probability", "Attack Probability"]].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
# Consensus Analysis
            st.markdown("### Consensus Analysis")
            valid_results = [res for res in model_results_list if res["Prediction"] not in ["Error", "Missing"]]
            if valid_results:
                valid_predictions = [res["Prediction"] for res in valid_results]
                prediction_counts = Counter(valid_predictions)
                majority_prediction, majority_count = prediction_counts.most_common(1)[0]
                consensus_level = (majority_count / len(valid_predictions)) * 100

                st.metric("Consensus Level", f"{consensus_level:.1f}%")
                st.write(f"**Majority Prediction:** {majority_prediction} ({majority_count} out of {len(valid_predictions)} trained_models)")

                if majority_prediction == "Attack":
                    attack_types = [res["Attack Type"] for res in valid_results if res["Prediction"] == "Attack"]
                    attack_type_counts = Counter(attack_types)
                    st.write("**Attack Type Distribution (among trained_models predicting Attack):**")
                    if attack_type_counts:
                        for attack_type, count in attack_type_counts.items():
                            st.markdown(f"- {attack_type}: {count} trained_models ({count/len(attack_types)*100:.1f}%)")
                    else:
                        st.write("_No trained_models predicted a specific attack type._")
            else:
                st.warning("No valid predictions available for consensus analysis.")

            # Save results logic (optional, keep if needed)
    if len(st.session_state.analysis_results) > 0:
        st.sidebar.markdown("--- ")
        st.sidebar.markdown("## 💾 Save Results")
        if st.sidebar.button("Save All Results to CSV"):
            try:
                results_df = pd.DataFrame(st.session_state.analysis_results)
                # Ensure consistent column order if possible
                # Define expected columns based on features + model outputs
                expected_cols = FEATURE_NAMES + [f"{m}_{s}" for m in available_trained_models for s in ["Prediction", "AttackProb"]]
                # Reindex DataFrame, adding missing columns with NaN
                results_df = results_df.reindex(columns=expected_cols)
                results_df.to_csv(RESULTS_FILE, index=False)
                st.sidebar.success(f"Results saved to {RESULTS_FILE}")
                # Provide download link
                st.sidebar.markdown(get_table_download_link(results_df, "analysis_results.csv"), unsafe_allow_html=True)
            except Exception as e:
                st.sidebar.error(f"Error saving results: {e}")

        elif analysis_triggered:
            st.warning("No trained_models were selected or results could not be generated.")
if __name__ == "__main__":
    main()