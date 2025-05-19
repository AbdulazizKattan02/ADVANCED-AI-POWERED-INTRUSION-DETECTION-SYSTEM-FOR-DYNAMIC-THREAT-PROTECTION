# RandomForest Training Script with Class Grouping and Filtering
# This script trains a RandomForest model after grouping some classes and removing others entirely.
# It includes data loading, class grouping, class filtering, preprocessing, model training,
# evaluation, and saving of the model, scaler, results, and plots.

import pandas as pd
import numpy as np
import time
import os
import sys 
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, roc_curve, auc, precision_recall_curve)

import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"
MODEL_OUTPUT_DIR = "randomforest_outputs" 
PLOTS_DIR = os.path.join(MODEL_OUTPUT_DIR, "plots")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "randomforest_model.joblib")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "randomforest_scaler.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, "randomforest_label_encoder.joblib") 
RESULTS_FILE = os.path.join(MODEL_OUTPUT_DIR, "randomforest_results.csv")

# Create output directories if they don_t exist
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def apply_class_grouping_and_filtering(df, target_column="Label"):
    """Applies class grouping and then filters out specified classes."""
    print("Applying class grouping...")
    class_grouping_map = {
        "DDoS": "DoS_Attacks",
        "DoS GoldenEye": "DoS_Attacks",
        "DoS Hulk": "DoS_Attacks",
        "DoS Slowhttptest": "DoS_Attacks",
        "DoS slowloris": "DoS_Attacks",
        "Web Attack  Brute Force": "Web_Attacks",
        "Web Attack  Sql Injection": "Web_Attacks",
        "Web Attack  XSS": "Web_Attacks",
        "FTP-Patator": "Patator_Attacks", # This group will be removed
        "SSH-Patator": "Patator_Attacks"  # This group will be removed
    }
    df[target_column] = df[target_column].replace(class_grouping_map)
    print(f"Class grouping applied. Unique classes after grouping: {sorted(list(df[target_column].unique()))}")

    classes_to_remove = ["Patator_Attacks", "Bot", "Heartbleed", "Infiltration"]
    
    print(f"Filtering out specified classes: {classes_to_remove}...")
    original_row_count = len(df)
    df = df[~df[target_column].isin(classes_to_remove)]
    filtered_row_count = len(df)
    print(f"Filtered out {original_row_count - filtered_row_count} rows.")
    remaining_classes = sorted(list(df[target_column].unique()))
    print(f"Classes remaining for training: {remaining_classes}")
    
    if df.empty or df[target_column].nunique() < 2:
        print("Error: After filtering, the dataset is empty or has less than two classes. Cannot proceed.")
        sys.exit(1)
    return df

def load_data_group_filter_encode(file_path, target_column="Label"):
    """Loads data, applies class grouping, filtering, and encodes the target variable."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file {file_path} not found. Please ensure it is in the correct path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("Data loaded successfully.")
    print(f"Original shape of the dataframe: {df.shape}")

    df = apply_class_grouping_and_filtering(df, target_column)

    print("Preprocessing data post grouping and filtering...")
    with tqdm(total=3, desc="Data preprocessing") as pbar:
        # Replace infinite values with NaN, then fill NaNs with 0
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        pbar.update(1)
        
        # Select only numeric columns for filling NaNs, to avoid issues with object columns if any exist besides target
        numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols_all] = df[numeric_cols_all].fillna(0)
        pbar.update(1)
        
        if target_column not in df.columns:
            print(f"Error: Target column 	{target_column}	 not found after processing.")
            sys.exit(1)

        y_raw = df[target_column]
        # Drop the original target column. Also drop 'attack_type' if it exists, as it might be a duplicate or source of leakage.
        columns_to_drop = [target_column]
        if "attack_type" in df.columns:
            columns_to_drop.append("attack_type")
        X_raw = df.drop(columns=columns_to_drop, errors='ignore')

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        pbar.update(1)
    
    print(f"Target variable {target_column} (post-grouping & filtering) encoded. Classes: {le.classes_}")
    return X_raw, y_encoded, le

# --- Plotting functions (ensure they are suitable for RandomForest) ---
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(max(8, len(classes)//1.2), max(6, len(classes)//1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()
    print("Confusion matrix plot saved.")

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(feature_names))
        
        plt.figure(figsize=(12, max(6, top_n // 2)))
        plt.title(f"Feature Importances (Top {top_n}) - {model_name}")
        sns.barplot(x=importances[indices][:top_n], y=np.array(feature_names)[indices][:top_n])
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance.png"))
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.pie(importances[indices][:top_n], labels=np.array(feature_names)[indices][:top_n], autopct="%1.1f%%", startangle=90)
        plt.title(f"Feature Importance Distribution (Top {top_n}) - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance_pie.png"))
        plt.close()
        print("Feature importance plots saved.")
    else:
        print(f"Model ({model_name}) has not been trained or does not support feature_importances_.")

def plot_learning_curves(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (F1 Weighted)")
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1_weighted")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, "o-", color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g",
                label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig(os.path.join(PLOTS_DIR, f"{title.lower().replace(' ', '_').replace('-', '_')}_learning_curves.png"))
        print("Learning curves plot saved.")
    except Exception as e:
        print(f"Could not generate learning curves: {e}")
    finally:
        plt.close()

def plot_roc_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(max(8, n_classes*1.0), max(6, n_classes*0.8)))
    if n_classes <= 2 and y_probs.ndim > 1 and y_probs.shape[1] > 1: 
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1]) 
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    elif n_classes > 2 and y_probs.ndim > 1 and y_probs.shape[1] == n_classes: 
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {le.classes_[i]} vs Rest (AUC = {roc_auc:.2f})")
    else:
        print("ROC curve cannot be plotted. y_probs shape or n_classes might be problematic.")
        plt.close()
        return 
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_roc_curve.png"))
    plt.close()
    print("ROC curve plot saved.")

def plot_precision_recall_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(max(8, n_classes*1.0), max(6, n_classes*0.8)))
    if n_classes <= 2 and y_probs.ndim > 1 and y_probs.shape[1] > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        plt.plot(recall, precision, label=f"{model_name}")
    elif n_classes > 2 and y_probs.ndim > 1 and y_probs.shape[1] == n_classes:
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
            plt.plot(recall, precision, label=f"Class {le.classes_[i]} vs Rest")
    else:
        print("Precision-Recall curve cannot be plotted. y_probs shape or n_classes might be problematic.")
        plt.close()
        return
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="best", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_precision_recall_curve.png"))
    plt.close()
    print("Precision-Recall curve plot saved.")

def plot_prediction_distribution(y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(10, 6))
    if y_probs.ndim > 1 and y_probs.shape[1] == n_classes:
        for i in range(n_classes):
            sns.histplot(y_probs[:, i], label=f"Class {le.classes_[i]} Probabilities", kde=True, stat="density", common_norm=False, element="step")
    else:
        print("Prediction distribution plot cannot be generated. y_probs shape or n_classes might be problematic.")
        plt.close()
        return
        
    plt.title(f"Prediction Probabilities Distribution - {model_name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_prediction_distribution.png"))
    plt.close()
    print("Prediction distribution plot saved.")

# Custom tqdm progress bar for RandomForest (since it doesn't have a direct callback for iterations)
# We will wrap the fit method or use verbose mode if appropriate and parse, but direct tqdm on n_estimators is simpler for user feedback.

def main():
    """Main function to orchestrate the training script generation."""
    X_raw, y_encoded, le = load_data_group_filter_encode(DATA_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    print(f"Label encoder for grouped & filtered classes saved to {LABEL_ENCODER_FILE}")

    print("Splitting and preprocessing data...")
    with tqdm(total=3, desc="Data preparation") as pbar:
        stratify_option = y_encoded if len(np.unique(y_encoded)) > 1 and len(y_encoded) > 0 else None
        if len(X_raw) == 0 or len(y_encoded) == 0:
            print("Error: No data available for training after processing. Exiting.")
            sys.exit(1)
        
        # Use the full dataset for training and testing
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_encoded, test_size=0.2, random_state=42, stratify=stratify_option
        )
        pbar.update(1)
        
        # Identify numeric features from the raw training data *before* any transformations that might change column names to non-strings
        numeric_features_original_names = X_train_raw.select_dtypes(include=np.number).columns
        
        # It's crucial that X_train and X_test are DataFrames for the scaler to work correctly with column names.
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()
        pbar.update(1)

        scaler = StandardScaler()
        # Fit scaler ONLY on training data's numeric features
        X_train[numeric_features_original_names] = scaler.fit_transform(X_train_raw[numeric_features_original_names])
        # Transform test data's numeric features using the SAME fitted scaler
        X_test[numeric_features_original_names] = scaler.transform(X_test_raw[numeric_features_original_names])
        pbar.update(1)
    
    print("Numeric features scaled (fitted on train, transformed train/test).")
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    print(f"Data split and preprocessed: X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n--- Random Forest Model (Grouped & Filtered Classes) ---")
    model_name = "RandomForest_Grouped_Filtered"
    
    num_unique_classes = len(le.classes_)
    if num_unique_classes == 0:
        print("Error: No classes left to train the model. Exiting.")
        sys.exit(1)
    elif num_unique_classes == 1:
        print("Warning: Only one class left after processing. Model will predict this single class.")

    # Revert to original RandomForest settings
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample' if num_unique_classes > 1 else None
    )

    # Faster hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],  # Focused range
        'max_depth': [None, 15, 20],      # Reduced options
        'min_samples_split': [2, 5],      # Kept important values
        'min_samples_leaf': [1, 2],       # Kept important values
        'max_features': ['sqrt', 'log2']  # Kept important values
    }
    
    print("\nStarting RandomizedSearchCV for RandomForest hyperparameters...")
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=10,  # Try 10 different parameter combinations
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print("This will try 10 different parameter combinations with 3-fold CV (30 fits total)...")
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    rf_model = random_search.best_estimator_
    print("\nStarting Random Forest model training with best parameters...")
    start_time = time.time()
    
    # RandomForest doesn't have a direct tqdm callback for n_estimators like LightGBM/XGBoost.
    # Training can be long; user will see script running. tqdm is used for data prep.
    # For a very rough progress, one could set model.verbose and parse output, but it's complex.
    # We'll rely on the script execution time for now.
    try:
        rf_model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)
        
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds.")

    # Check if model is trained (RandomForest stores trees in estimators_)
    if not hasattr(rf_model, "estimators_") or not rf_model.estimators_:
        print("\nERROR: Model training appears to have failed. Cannot proceed with evaluation.")
        sys.exit(1)
    
    print("\nEvaluating trained model...")
    y_pred = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    train_time_val = training_time
    model_for_plotting = rf_model

    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")
    print(f"{model_name} Precision: {prec:.4f}")
    print(f"{model_name} Recall: {rec:.4f}")

    results_df = pd.DataFrame([{
        "model": model_name,
        "training_time_seconds": train_time_val,
        "f1_score": f1,
        "accuracy": acc,
        "recall": rec,
        "precision": prec
    }])
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

    joblib.dump(rf_model, MODEL_FILE) 
    print(f"Model saved to {MODEL_FILE}")

    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, le.classes_, model_name)
    plot_feature_importance(model_for_plotting, list(X_train.columns), model_name)
    
    print("Learning curve plotting is commented out by default. Uncomment if needed.")
    # print("Generating learning curves (can be slow)...")
    # estimator_for_lc = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample' if num_unique_classes > 1 else None)
    # plot_learning_curves(estimator_for_lc, 
    #                      f"Learning Curves - {model_name}", X_train, y_train, cv=3, n_jobs=-1)

    plot_roc_curve(y_test, y_probs, model_name, le)
    plot_precision_recall_curve(y_test, y_probs, model_name, le)
    plot_prediction_distribution(y_probs, model_name, le)
    
    print(f"\n{model_name} script execution complete. Outputs are in {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

