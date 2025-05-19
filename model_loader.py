import os
import joblib
import streamlit as st

# Define the absolute path to the models directory
TRAINED_MODELS_DIR = r"E:\New folder (6)\project\models"

# Mapping of model names to their output subdirectories and file names
MODEL_OUTPUTS = {
    "GBM": ("gbm_outputs", "gbm_model.joblib", "gbm_scaler.joblib"),
    "LightGBM": ("lightgbm_outputs", "lightgbm_model.joblib", "lightgbm_scaler.joblib"),
    "XGBoost": ("xgboost_outputs", "xgboost_model.joblib", "xgboost_scaler.joblib"),
    "Random Forest": ("randomforest_outputs", "randomforest_model.joblib", "randomforest_scaler.joblib"),
    "CatBoost": ("catboost_outputs", "catboost_model.joblib", "catboost_scaler.joblib"),
}
# Note: If a model or scaler file is missing or incompatible, it will be skipped and a warning will be shown.
# The app will support all 5 models if their files are present and compatible.

@st.cache_resource
def load_models():
    """
    Load retrained models and scalers from their respective output directories.
    Returns:
        dict: Dictionary containing loaded models and scalers {model_name: {"model": model, "scaler": scaler}}
    """
    models = {}
    print(f"Loading models from directory: {TRAINED_MODELS_DIR}")
    if not os.path.exists(TRAINED_MODELS_DIR):
        st.error(f"Models directory not found: {TRAINED_MODELS_DIR}")
        print(f"Error: Models directory not found: {TRAINED_MODELS_DIR}")
        return models # Return empty dict

    for model_name, (subdir, model_file, scaler_file) in MODEL_OUTPUTS.items():
        model_path = os.path.join(TRAINED_MODELS_DIR, subdir, model_file)
        scaler_path = os.path.join(TRAINED_MODELS_DIR, subdir, scaler_file)

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                models[model_name] = {
                    "model": model,
                    "scaler": scaler
                }
                print(f"{model_name} model and scaler loaded successfully.")
            except Exception as e:
                st.error(f"Error loading {model_name} model or scaler from {model_path}/{scaler_path}: {e}")
                print(f"Error loading {model_name} model or scaler from {model_path} or {scaler_path}: {e}")
        else:
            st.warning(f"{model_name} model or scaler file not found. Checked paths: {model_path}, {scaler_path}")
            print(f"Warning: {model_name} model or scaler file not found. Checked paths: {model_path}, {scaler_path}")

    if not models:
        st.error("No models were loaded successfully. Please ensure the model files exist and are valid.")
        print("Error: No models were loaded successfully.")

    return models

