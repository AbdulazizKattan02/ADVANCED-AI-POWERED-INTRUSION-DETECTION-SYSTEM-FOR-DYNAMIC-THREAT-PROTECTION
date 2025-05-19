# XGBoost Training Script
# This script is a template for training an XGBoost model.
# It includes data loading, preprocessing, model definition (placeholder for training),
# evaluation, and saving of the model, scaler, results, and plots.
# Ensure you have the necessary libraries installed: pandas, scikit-learn, xgboost, matplotlib, seaborn, joblib, tqdm
# You can typically install them using: pip install pandas scikit-learn xgboost matplotlib seaborn joblib tqdm

import pandas as pd
import numpy as np
import time
import os
import sys 
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb 
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, roc_curve, auc, precision_recall_curve,
                            classification_report, matthews_corrcoef, cohen_kappa_score)

import matplotlib.pyplot as plt
import seaborn as sns

class TQDMProgressBar(xgb.callback.TrainingCallback):
    def __init__(self, tqdm_bar):
        self.tqdm_bar = tqdm_bar

    def after_iteration(self, model, epoch, evals_log):
        self.tqdm_bar.update(1)
        return False  # Return value indicates whether training should stop

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"
MODEL_OUTPUT_DIR = "xgboost_outputs"
PLOTS_DIR = os.path.join(MODEL_OUTPUT_DIR, "plots")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "xgboost_model.joblib")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "xgboost_scaler.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, "xgboost_label_encoder.joblib") # Added for saving label encoder
RESULTS_FILE = os.path.join(MODEL_OUTPUT_DIR, "xgboost_results.csv")

# Create output directories if they don_t exist
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def apply_class_grouping_and_filtering(df, target_column="Label"):
    print("Applying class grouping...")
    
    # First, clean up any encoding issues in the labels
    df[target_column] = df[target_column].str.strip()
    
    # Print unique classes before grouping
    print("\nUnique classes before grouping:")
    print(sorted(df[target_column].unique()))
    
    # Group Web Attacks first using string matching
    df[target_column] = df[target_column].apply(
        lambda x: 'Web_Attacks' if 'Web Attack' in x else x
    )
    
    # Define other attack groups
    dos_attacks = ['DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris']
    patator_attacks = ['FTP-Patator', 'SSH-Patator']
    
    # Create the grouping map for other attacks
    class_grouping_map = {}
    
    # Add DoS attacks
    for attack in dos_attacks:
        class_grouping_map[attack] = 'DoS_Attacks'
    
    # Add Patator attacks
    for attack in patator_attacks:
        class_grouping_map[attack] = 'Patator_Attacks'
    
    # Apply the remaining grouping
    df[target_column] = df[target_column].replace(class_grouping_map)
    
    print("\nClass grouping applied. Unique classes after grouping:")
    print(sorted(list(df[target_column].unique())))
    
    # Filter out specified classes
    classes_to_remove = ["Patator_Attacks", "Bot", "Heartbleed", "Infiltration"]
    print(f"\nFiltering out specified classes: {classes_to_remove}")
    
    original_row_count = len(df)
    df = df[~df[target_column].isin(classes_to_remove)]
    filtered_row_count = len(df)
    
    print(f"Filtered out {original_row_count - filtered_row_count} rows.")
    print(f"\nClasses remaining for training: {sorted(list(df[target_column].unique()))}")
    
    if df.empty or df[target_column].nunique() < 2:
        print("Error: After filtering, the dataset is empty or has less than two classes. Cannot proceed.")
        sys.exit(1)
    
    # Print detailed class distribution
    print("\nDetailed class distribution after grouping and filtering:")
    class_dist = df[target_column].value_counts()
    for class_name, count in class_dist.items():
        print(f"{class_name}: {count} ({count/len(df)*100:.2f}%)")
    
    return df

def load_data_group_filter_encode(file_path, target_column="Label"):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    print("Data loaded successfully.")
    print(f"Original shape of the dataframe: {df.shape}")
    df = apply_class_grouping_and_filtering(df, target_column)
    print("Preprocessing data post grouping and filtering...")
    with tqdm(total=3, desc="Data preprocessing") as pbar:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        pbar.update(1)
        numeric_cols_fill = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols_fill] = df[numeric_cols_fill].fillna(0)
        pbar.update(1)
        if target_column not in df.columns:
            print(f"Error: Target column {target_column} not found after processing.")
            sys.exit(1)
        y_raw = df[target_column]
        columns_to_drop = [target_column]
        if "attack_type" in df.columns:
            columns_to_drop.append("attack_type")
        X_raw = df.drop(columns=columns_to_drop, errors='ignore')
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        pbar.update(1)
    print(f"Target variable {target_column} (post-grouping & filtering) encoded. Classes: {le.classes_}")
    return X_raw, y_encoded, le

def analyze_features(X):
    """Analyze features for potential data leakage and other issues."""
    print("\nFeature Analysis:")
    
    # Check for timestamp-like features
    time_related = [col for col in X.columns if any(term in col.lower() 
                for term in ['time', 'timestamp', 'date', 'hour', 'minute', 'second'])]
    if time_related:
        print("\nPotential temporal features (might leak information):")
        for col in time_related:
            print(f"- {col}")
    
    # Check for ID-like features
    id_related = [col for col in X.columns if any(term in col.lower() 
                for term in ['id', 'index', 'key', 'number', 'num'])]
    if id_related:
        print("\nPotential ID-like features (might leak information):")
        for col in id_related:
            print(f"- {col}")
    
    # Check for features with very high cardinality
    high_cardinality = []
    for col in X.columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio > 0.9:  # More than 90% unique values
            high_cardinality.append((col, unique_ratio))
    if high_cardinality:
        print("\nFeatures with high cardinality (might be too specific):")
        for col, ratio in high_cardinality:
            print(f"- {col}: {ratio:.2%} unique values")
    
    # Check for features with suspicious patterns
    suspicious = []
    for col in X.columns:
        # Check if values are too perfectly separated
        value_counts = X[col].value_counts()
        if len(value_counts) > 1 and value_counts.iloc[0] / len(X) > 0.999:
            suspicious.append((col, "Highly skewed"))
        # Check for perfect increments
        if X[col].dtype in [np.int64, np.float64]:
            diffs = X[col].diff().dropna().unique()
            if len(diffs) == 1:
                suspicious.append((col, "Perfect increments"))
    if suspicious:
        print("\nFeatures with suspicious patterns:")
        for col, reason in suspicious:
            print(f"- {col}: {reason}")

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Plot confusion matrix with better formatting."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Print raw confusion matrix
    print("\nConfusion Matrix (raw counts):")
    print("True \\ Pred", end="\t")
    for c in classes:
        print(f"{c[:10]}", end="\t")
    print()
    for i, row in enumerate(cm):
        print(f"{classes[i][:10]}", end="\t")
        for cell in row:
            print(f"{cell}", end="\t")
        print()
    
    # Print percentage matrix
    print("\nConfusion Matrix (percentages by true class):")
    print("True \\ Pred", end="\t")
    for c in classes:
        print(f"{c[:10]}", end="\t")
    print()
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i, row in enumerate(cm_percentage):
        print(f"{classes[i][:10]}", end="\t")
        for cell in row:
            print(f"{cell:.2%}", end="\t")
        print()
    
    # Plot heatmap
    plt.figure(figsize=(max(8, len(classes)//1.2), max(6, len(classes)//1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()
    print("\nConfusion matrix plot saved.")

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
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        importances_dict = booster.get_score(importance_type="gain")
        importances = np.array([importances_dict.get(f, 0) for f in feature_names])
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
        print(f"Model ({model_name}) does not support feature_importances_ or get_booster().")

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

def remove_redundant_features(X, threshold=0.95):
    """Remove constant and highly correlated features."""
    # Remove constant features
    constant_filter = X.nunique() > 1
    X = X.loc[:, constant_filter]
    
    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X = X.drop(columns=to_drop)
    
    # Add feature selection based on variance threshold
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
    
    print(f"\nRemoved {sum(~constant_filter)} constant features")
    print(f"Removed {len(to_drop)} highly correlated features")
    print(f"Remaining features after variance threshold: {X.shape[1]}")
    return X

def perform_feature_engineering(X):
    """Perform advanced feature engineering."""
    print("\nPerforming feature engineering...")
    
    # Create interaction features for highly correlated pairs
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(i, j) for i, j in zip(*np.where(upper > 0.8))]
    
    for i, j in high_corr_pairs:
        col1, col2 = X.columns[i], X.columns[j]
        X[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
        X[f'{col1}_{col2}_ratio'] = X[col1] / (X[col2] + 1e-6)
    
    # Add polynomial features for important features
    important_features = X.columns[:10]  # Assuming first 10 features are important
    for col in important_features:
        X[f'{col}_squared'] = X[col] ** 2
        X[f'{col}_cubed'] = X[col] ** 3
    
    # Add statistical features
    X['mean'] = X.mean(axis=1)
    X['std'] = X.std(axis=1)
    X['max'] = X.max(axis=1)
    X['min'] = X.min(axis=1)
    
    return X

def select_features(X, y, threshold=0.01):
    """Select features using multiple methods."""
    print("\nPerforming feature selection...")
    
    # Mutual Information
    mi_scores = mutual_info_classif(X, y)
    mi_features = X.columns[mi_scores > threshold]
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_features = X.columns[rf_importance > threshold]
    
    # Combine selected features
    selected_features = list(set(mi_features) | set(rf_features))
    print(f"Selected {len(selected_features)} features")
    
    return X[selected_features]

def optimize_hyperparameters(X_train, y_train):
    """Perform hyperparameter optimization using GridSearchCV with a tqdm timer."""
    print("\nOptimizing hyperparameters...")

    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [500, 1000],
        'min_child_weight': [1, 3],
        'gamma': [0.1, 0.3],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1],
        'reg_lambda': [1.0]
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        tree_method='hist',
        device='cuda',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        n_jobs=1,
        verbose=1
    )

    with tqdm(total=1, desc="GridSearchCV (elapsed time)", bar_format="{l_bar}{bar}| {elapsed} elapsed") as pbar:
        start = time.time()
        grid_search.fit(X_train, y_train)
        pbar.update(1)
        end = time.time()
        print(f"Grid search took {end - start:.2f} seconds.")

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_

def create_sampling_strategy(X, y):
    """Create a balanced sampling strategy."""
    print("\nCreating sampling strategy...")
    
    # Calculate class distribution
    class_counts = np.bincount(y)
    total_samples = len(y)
    target_samples = total_samples / len(class_counts)
    
    # Create sampling strategy
    sampling_strategy = {}
    for i, count in enumerate(class_counts):
        if count < target_samples:
            sampling_strategy[i] = int(target_samples)
        else:
            sampling_strategy[i] = count
    
    return sampling_strategy

def main():
    """Main function to orchestrate the training script generation."""
    X_raw, y_encoded, le = load_data_group_filter_encode(DATA_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    print(f"Label encoder saved to {LABEL_ENCODER_FILE}")

    # Print class distribution
    print("\nClass distribution:")
    class_counts = pd.Series(y_encoded).value_counts()
    for idx, count in class_counts.items():
        print(f"{le.classes_[idx]}: {count} ({count/len(y_encoded)*100:.2f}%)")

    # Analyze features before any preprocessing
    analyze_features(X_raw)
    
    # Perform feature engineering
    X_raw = perform_feature_engineering(X_raw)
    
    # Remove redundant features
    X_raw = remove_redundant_features(X_raw)
    
    # Select features
    X_raw = select_features(X_raw, y_encoded)
    
    print("\nSplitting and preprocessing data...")
    with tqdm(total=3, desc="Data preparation") as pbar:
        stratify_option = y_encoded if len(np.unique(y_encoded)) > 1 else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42, stratify=stratify_option)
        pbar.update(1)
        
        numeric_features = X_train_raw.select_dtypes(include=np.number).columns
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()
        pbar.update(1)

        scaler = RobustScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train_raw[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test_raw[numeric_features])
        pbar.update(1)
    
    print("Numeric features scaled (fitted on train, transformed train/test).")
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    # Create sampling strategy
    sampling_strategy = create_sampling_strategy(X_train, y_train)
    
    # Create sampling pipeline
    sampling_pipeline = Pipeline([
        ('over', SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
    ])
    
    # Apply sampling
    X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train_resampled, y_train_resampled)
    
    # Initialize model with optimized parameters
    xgb_model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        use_label_encoder=False,
        tree_method='hist',
        device='cuda',
        random_state=42,
        callbacks=[TQDMProgressBar(None)]
    )
    
    print("\nTraining final model on resampled data...")
    start_time = time.time()
    
    with tqdm(total=xgb_model.n_estimators, desc="Training XGBoost") as pbar:
        xgb_model.get_params()['callbacks'][0].tqdm_bar = pbar
        xgb_model.fit(
            X_train_resampled, 
            y_train_resampled,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    print("\nEvaluating trained model...")
    y_pred = xgb_model.predict(X_test)
    y_probs = xgb_model.predict_proba(X_test)
    
    # Calculate advanced metrics
    print("\nAdvanced metrics:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\nMatthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"Cohen's Kappa Score: {cohen_kappa_score(y_test, y_pred):.4f}")
    
    # Calculate per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(le.classes_):
        class_mask = y_test == i
        if np.any(class_mask):
            class_pred = y_pred[class_mask]
            print(f"\n{class_name}:")
            print(f"Accuracy: {accuracy_score(y_test[class_mask], class_pred):.4f}")
            print(f"F1-Score: {f1_score(y_test[class_mask], class_pred, average='weighted'):.4f}")
            print(f"Precision: {precision_score(y_test[class_mask], class_pred, average='weighted'):.4f}")
            print(f"Recall: {recall_score(y_test[class_mask], class_pred, average='weighted'):.4f}")

    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"\nOverall metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    results_df = pd.DataFrame([{
        "model": "XGBoost_Enhanced",
        "training_time_seconds": training_time,
        "f1_score": f1,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred)
    }])
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, le.classes_, "XGBoost_Enhanced")
    plot_feature_importance(xgb_model, list(X_train.columns), "XGBoost_Enhanced")
    plot_roc_curve(y_test, y_probs, "XGBoost_Enhanced", le)
    plot_precision_recall_curve(y_test, y_probs, "XGBoost_Enhanced", le)
    plot_prediction_distribution(y_probs, "XGBoost_Enhanced", le)
    print(f"\nEnhanced XGBoost script execution complete. Outputs are in {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

