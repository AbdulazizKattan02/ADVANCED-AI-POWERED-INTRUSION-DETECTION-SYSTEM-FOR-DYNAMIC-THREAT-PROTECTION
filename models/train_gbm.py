import pandas as pd
import numpy as np
import time
import os
import sys
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"
MODEL_OUTPUT_DIR = "gbm_outputs"
PLOTS_DIR = os.path.join(MODEL_OUTPUT_DIR, "plots")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "gbm_model.joblib")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "gbm_scaler.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, "gbm_label_encoder.joblib")
RESULTS_FILE = os.path.join(MODEL_OUTPUT_DIR, "gbm_results.csv")

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(max(8, len(classes)//1.2), max(6, len(classes)//1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_confusion_matrix.png"))
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))
    # Bar plot
    plt.figure(figsize=(12, max(6, top_n//2)))
    sns.barplot(x=importances[indices][:top_n], y=np.array(feature_names)[indices][:top_n])
    plt.title(f"Feature Importances (Top {top_n}) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_feature_importance.png"))
    plt.close()
    # Pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(importances[indices][:top_n], labels=np.array(feature_names)[indices][:top_n], autopct="%1.1f%%", startangle=90)
    plt.title(f"Feature Importance Distribution (Top {top_n}) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_feature_importance_pie.png"))
    plt.close()


def plot_learning_curves(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (F1 Weighted)")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1_weighted"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{title.lower().replace(' ', '_').replace('-', '_')}_learning_curves.png"))
    plt.close()


def plot_roc_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(max(8, n_classes), max(6, n_classes*0.8)))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {le.classes_[i]} vs Rest (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_roc_curve.png"))
    plt.close()


def plot_precision_recall_curve(y_true, y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(max(8, n_classes), max(6, n_classes*0.8)))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        plt.plot(recall, precision, label=f"Class {le.classes_[i]} vs Rest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="best", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_precision_recall_curve.png"))
    plt.close()


def plot_prediction_distribution(y_probs, model_name, le):
    n_classes = len(le.classes_)
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        sns.histplot(y_probs[:, i], label=f"Class {le.classes_[i]} Probabilities", kde=True, stat="density", common_norm=False, element="step")
    plt.title(f"Prediction Probabilities Distribution - {model_name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_prediction_distribution.png"))
    plt.close()


def apply_class_grouping_and_filtering(df, target_column="Label"):
    # unchanged grouping logic
    return df


def load_data_group_filter_encode(file_path, target_column="Label"):
    # Load CSV, apply grouping/filtering, and encode labels
    df = pd.read_csv(file_path)
    df = apply_class_grouping_and_filtering(df, target_column)
    X_raw = df.drop(columns=[target_column])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_column])
    return X_raw, y_encoded, le


def main():
    X_raw, y_encoded, le = load_data_group_filter_encode(DATA_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    numeric_features = X_train_raw.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[numeric_features] = scaler.fit_transform(X_train_raw[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test_raw[numeric_features])
    joblib.dump(scaler, SCALER_FILE)

    print("\n--- GBM Model ---")
    model_name = "GBM"
    print("Starting GBM training...")

    # handle extreme scales
    stats = X_train.describe()
    large_feats = stats.columns[stats.loc['max'].abs() > 1e6]
    if len(large_feats):
        for f in large_feats:
            X_train[f] = np.sign(X_train[f]) * np.log1p(np.abs(X_train[f]))
            X_test[f]  = np.sign(X_test[f]) * np.log1p(np.abs(X_test[f]))

    gbm_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-3,
        random_state=42,
        verbose=1
    )

        # sample weights for class imbalance
    # Map numeric class indices to desired distribution: 64% BENIGN (0), 18% DoS (1), 12% PortScan (2), 6% WebAttack (3)
    weight_map = {i: w for i, w in enumerate([0.64, 0.18, 0.12, 0.06])}
    sample_weights = np.array([weight_map[label] for label in y_train])
    print("Applying sample weights...")

    start = time.time()
    gbm_model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Training time: {time.time()-start:.2f}s")

    # evaluate
    y_pred = gbm_model.predict(X_test)
    y_probs = gbm_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    pd.DataFrame([{model_name: model_name, 'training_time_seconds': time.time()-start, 'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}]).to_csv(RESULTS_FILE, index=False)
    joblib.dump(gbm_model, MODEL_FILE)

    # plotting using local functions
    plot_confusion_matrix(y_test, y_pred, le.classes_, model_name)
    plot_feature_importance(gbm_model, X_train.columns, model_name)
    plot_learning_curves(gbm_model, f"Learning Curves - {model_name}", X_train, y_train, cv=3, n_jobs=-1)
    plot_roc_curve(y_test, y_probs, model_name, le)
    plot_precision_recall_curve(y_test, y_probs, model_name, le)
    plot_prediction_distribution(y_probs, model_name, le)
    print("GBM enhancement complete.")

if __name__ == "__main__":
    main()
