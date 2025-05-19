import os, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
DATA_FILE = r"E:\New folder (6)\project\data\CIC-IDS-2017.csv"
MODEL_OUTPUT_DIR = "catboost_outputs"
PLOTS_DIR = os.path.join(MODEL_OUTPUT_DIR, "plots")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "catboost_model.joblib")
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, "catboost_scaler.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, "catboost_label_encoder.joblib")
RESULTS_FILE = os.path.join(MODEL_OUTPUT_DIR, "catboost_results.csv")

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)

def apply_class_grouping(df, target_column="Label"):
    df[target_column] = df[target_column].str.strip()
    df[target_column] = df[target_column].apply(lambda x: "Web_Attacks" if "Web Attack" in x else x)
    group_map = {
        "DDoS": "DoS_Attacks", "DoS GoldenEye": "DoS_Attacks",
        "DoS Hulk": "DoS_Attacks", "DoS Slowhttptest": "DoS_Attacks", "DoS slowloris": "DoS_Attacks",
        "FTP-Patator": "Patator_Attacks", "SSH-Patator": "Patator_Attacks"
    }
    df[target_column] = df[target_column].replace(group_map)
    df = df[~df[target_column].isin(["Patator_Attacks", "Bot", "Heartbleed", "Infiltration"])]
    return df

def load_and_preprocess():
    df = pd.read_csv(DATA_FILE)
    df = apply_class_grouping(df, target_column="Label")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    X = df.drop(columns=["Label"])
    feature_names = X.columns
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(le, LABEL_ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return X_scaled, y, le, feature_names

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
    plt.figure(figsize=(12, max(6, top_n//2)))
    sns.barplot(x=importances[indices][:top_n], y=np.array(feature_names)[indices][:top_n])
    plt.title(f"Feature Importances (Top {top_n}) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_feature_importance.png"))
    plt.close()
    plt.figure(figsize=(10, 10))
    plt.pie(importances[indices][:top_n], labels=np.array(feature_names)[indices][:top_n], autopct="%1.1f%%", startangle=90)
    plt.title(f"Feature Importance Distribution (Top {top_n}) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_feature_importance_pie.png"))
    plt.close()

def plot_learning_curves(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 3)):
    if cv is None:
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
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

def train():
    X, y, le, feature_names = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    class_weights = {i: total / c for i, c in class_counts.items()}
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        early_stopping_rounds=50,
        random_seed=42,
        class_weights=class_weights,
        verbose=100,
        train_dir=PLOTS_DIR
    )
    start = time.time()
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    joblib.dump(model, MODEL_FILE)
    y_pred = model.predict(X_test).flatten()
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\nAccuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    pd.DataFrame([{
        "model": "CatBoost",
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "training_time_seconds": training_time
    }]).to_csv(RESULTS_FILE, index=False)
    model_name = "CatBoost"
    plot_confusion_matrix(y_test, y_pred, le.classes_, model_name)
    plot_feature_importance(model, feature_names, model_name)
    # Use a shallow CatBoost for learning curve
    shallow_model = CatBoostClassifier(
        iterations=100,  # much lower for speed
        learning_rate=0.05,
        depth=4,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        early_stopping_rounds=10,
        random_seed=42,
        class_weights=class_weights,
        verbose=0
    )
    plot_learning_curves(
        shallow_model,
        f"Learning Curves - {model_name}",
        X_train, y_train,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        n_jobs=1
    )
    plot_roc_curve(y_test, y_prob, model_name, le)
    plot_precision_recall_curve(y_test, y_prob, model_name, le)
    plot_prediction_distribution(y_prob, model_name, le)
    print("CatBoost training and visualization complete.")

if __name__ == "__main__":
    train()
