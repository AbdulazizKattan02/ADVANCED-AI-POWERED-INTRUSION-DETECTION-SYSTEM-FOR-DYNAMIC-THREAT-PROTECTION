import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import os
import joblib
import lightgbm as lgb
import sys

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"
MODEL_OUTPUT_DIR = "lightgbm_outputs"
PLOTS_DIR = os.path.join(MODEL_OUTPUT_DIR, "plots")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "lightgbm_model.joblib")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "lightgbm_scaler.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, "lightgbm_label_encoder.joblib")
RESULTS_FILE = os.path.join(MODEL_OUTPUT_DIR, "lightgbm_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Plotting Functions ---
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(max(8, len(classes)//1.2), max(6, len(classes)//1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance: no attribute")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))
    # Bar plot
    plt.figure(figsize=(12, max(6, top_n // 2)))
    sns.barplot(x=importances[indices][:top_n], y=np.array(feature_names)[indices][:top_n])
    plt.title(f"Feature Importances (Top {top_n}) - {model_name}")
    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.savefig(path1)
    plt.close()
    print(f"Saved plot: {path1}")
    # Pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(importances[indices][:top_n], labels=np.array(feature_names)[indices][:top_n], autopct="%1.1f%%", startangle=90)
    plt.title(f"Feature Importance Distribution (Top {top_n}) - {model_name}")
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance_pie.png")
    plt.savefig(path2)
    plt.close()
    print(f"Saved plot: {path2}")


def plot_learning_curves(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (F1 Weighted)")
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1_weighted")
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Train')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='CV')
        plt.legend()
        path = os.path.join(PLOTS_DIR, f"{title.lower().replace(' ', '_').replace('-', '_')}_learning_curves.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")
    except Exception as e:
        print(f"Learning curve error: {e}")


def plot_roc_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(8, 6))
    if y_probs.ndim == 1 or y_probs.shape[1] < n_classes:
        print("Skipping ROC: insufficient probabilities")
        return
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {le.classes_[i]} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(fontsize='small')
    path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_precision_recall_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(8, 6))
    if y_probs.ndim != 2 or y_probs.shape[1] != n_classes:
        print("Skipping PR: insufficient probabilities")
        return
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        plt.plot(rec, prec, label=f"Class {le.classes_[i]}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {model_name}")
    plt.legend(fontsize='small')
    path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_precision_recall_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_prediction_distribution(y_probs, model_name, le):
    plt.figure(figsize=(8, 6))
    if y_probs.ndim == 2 and y_probs.shape[1] == len(le.classes_):
        for i, cls in enumerate(le.classes_):
            sns.kdeplot(y_probs[:, i], label=str(cls), fill=False)
    else:
        print("Skipping distribution: shape mismatch")
        return
    plt.xlabel("Probability"); plt.ylabel("Density")
    plt.title(f"Prediction Distribution - {model_name}")
    plt.legend(fontsize='small')
    path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_prediction_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")

# --- Main ---
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
    
    print(f"\nRemoved {sum(~constant_filter)} constant features")
    print(f"Removed {len(to_drop)} highly correlated features")
    print(f"Remaining features: {X.shape[1]}")
    return X

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

def main():
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATA_FILE)
    df = apply_class_grouping_and_filtering(df, "Label")
    X_raw = df.drop(columns=["Label"])
    label_enc = LabelEncoder().fit(df["Label"])
    y = label_enc.transform(df["Label"])
    joblib.dump(label_enc, LABEL_ENCODER_FILE)

    print("Removing redundant features...")
    X_clean = remove_redundant_features(X_raw)
    analyze_features(X_clean)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
    joblib.dump(scaler, SCALER_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=None, stratify=y
    )
    counts = np.bincount(y_train);
    cw = {i: len(y_train)/(len(counts)*c) for i,c in enumerate(counts)}
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=None, stratify=y_train
    )

    model_name = "LightGBM_Grouped_Filtered"
    
    # Enhanced model configuration
    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        boosting_type='gbdt',
        n_estimators=3000,  # Increased from 2000
        learning_rate=0.01,  # Increased from 0.005
        num_leaves=127,  # Increased from 63
        max_depth=15,  # Increased from 12
        min_child_samples=20,  # Increased from 10
        min_child_weight=0.001,  # Added to control overfitting
        min_split_gain=0.0,  # Added to control overfitting
        colsample_bytree=0.9,  # Increased from 0.8
        subsample=0.9,  # Increased from 0.8
        subsample_freq=1,  # Changed from 5
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight=cw,
        n_jobs=-1,
        importance_type='gain',
        verbosity=-1,
        random_state=42  # Added for reproducibility
    )

    print("Training LightGBM with cross-validation...")
    start = time.time()
    
    # Add cross-validation
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        print(f"\nTraining fold {fold}/5...")
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric='multi_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )
        
        y_fold_pred = model.predict(X_fold_val)
        cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
        cv_scores['f1'].append(f1_score(y_fold_val, y_fold_pred, average='weighted'))
        cv_scores['precision'].append(precision_score(y_fold_val, y_fold_pred, average='weighted'))
        cv_scores['recall'].append(recall_score(y_fold_val, y_fold_pred, average='weighted'))
    
    print("\nCross-validation scores:")
    for metric, scores in cv_scores.items():
        print(f"{metric.capitalize()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Train final model on full training set
    print("\nTraining final model on full training set...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")

    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    print(f"Results: Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    pd.DataFrame([{
        'Model': model_name, 'Accuracy': acc, 'F1 Score': f1,
        'Precision': prec, 'Recall': rec, 'Training Time': train_time
    }]).to_csv(RESULTS_FILE, index=False)
    joblib.dump(model, MODEL_FILE)

    print("Generating and saving plots...")
    plot_confusion_matrix(y_test, y_pred, label_enc.classes_, model_name)
    plot_feature_importance(model, X_clean.columns, model_name)
    plot_roc_curve(y_test, y_prob, model_name, label_enc)
    plot_precision_recall_curve(y_test, y_prob, model_name, label_enc)
    plot_prediction_distribution(y_prob, model_name, label_enc)
    # learning curves subset
    idx = np.random.choice(len(X_tr), size=min(20000, len(X_tr)), replace=False)
    plot_learning_curves(model, f"Learning Curves - {model_name}", X_tr.iloc[idx], y_tr[idx], cv=3, n_jobs=-1)
    print("All plots saved.")

if __name__ == "__main__":
    main()