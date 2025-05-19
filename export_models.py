# export_models.py

import os
import joblib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# === Output Directory ===
output_dir = r"E:\project\models"
os.makedirs(output_dir, exist_ok=True)

# === Generate Dummy Data (11 features) ===
X, y = make_classification(n_samples=500, n_features=11, random_state=42)

# === Helper function to fit, train, and save model + scaler ===
def export_model_and_scaler(model, model_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    joblib.dump(model, os.path.join(output_dir, f"{model_name.lower()}_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, f"{model_name.lower()}_scaler.joblib"))
    print(f"✅ {model_name} model and scaler saved successfully.")

# === Export All Models ===
export_model_and_scaler(GradientBoostingClassifier(), "GBM")
export_model_and_scaler(LGBMClassifier(), "LightGBM")
export_model_and_scaler(XGBClassifier(), "XGBoost")
export_model_and_scaler(RandomForestClassifier(), "RandomForest")
export_model_and_scaler(CatBoostClassifier(), "CatBoost")

print("\n🎉 All models and scalers exported successfully.")
