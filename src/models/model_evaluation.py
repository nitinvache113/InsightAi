import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "evaluation_outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

from src.features.preprocess import prepare_data


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INSIGHTAI - MODEL EVALUATION")
print("=" * 70)

print("\nLoading dataset...")


(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    train_indices,
    test_indices
) = prepare_data()


print(
    f"Training samples : {len(X_train):,}"
)

print(
    f"Testing samples  : {len(X_test):,}"
)

print(
    f"Features          : {X_train.shape[1]}"
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_tuned.pkl"
)


print(
    "\nLoading tuned XGBoost model..."
)


model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "Generating predictions..."
)


y_probability = model.predict_proba(
    X_test
)[:, 1]


y_prediction = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)


precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    y_probability
)


average_precision = (
    average_precision_score(
        y_test,
        y_probability
    )
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Recall             : {recall:.4f}"
)

print(
    f"F1 Score           : {f1:.4f}"
)

print(
    f"ROC-AUC            : {roc_auc:.4f}"
)

print(
    f"Average Precision  : {average_precision:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_prediction
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_prediction,
    target_names=[
        "No Churn",
        "Churn"
    ],
    zero_division=0
)


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# Save report
with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Average Precision"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            average_precision
        ]
    }
)


metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    ),
    index=False
)


# ============================================================
# CONFUSION MATRIX DATA
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Churn",
        "Actual Churn"
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn"
    ]
)


cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "InsightAI - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(
    tick_marks,
    [
        "No Churn",
        "Churn"
    ]
)

plt.yticks(
    tick_marks,
    [
        "No Churn",
        "Churn"
    ]
)


threshold = cm.max() / 2


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black",
            fontsize=14
        )


plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.3f})"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - InsightAI"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    recall_values,
    precision_values,
    label=f"XGBoost (AP = {average_precision:.3f})"
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - InsightAI"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PREDICTION DATA
# ============================================================

prediction_df = pd.DataFrame(
    {
        "Actual_Churn": y_test,
        "Predicted_Churn": y_prediction,
        "Churn_Probability": y_probability
    }
)


prediction_df[
    "Risk_Level"
] = pd.cut(
    prediction_df[
        "Churn_Probability"
    ],
    bins=[
        -np.inf,
        0.40,
        0.70,
        np.inf
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

risk_distribution = (
    prediction_df[
        "Risk_Level"
    ]
    .value_counts()
    .reset_index()
)


risk_distribution.columns = [
    "Risk_Level",
    "Customers"
]


risk_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "risk_distribution.csv"
    ),
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    "\nGenerated files:"
)

print(
    "1. model_metrics.csv"
)

print(
    "2. classification_report.txt"
)

print(
    "3. confusion_matrix.csv"
)

print(
    "4. confusion_matrix.png"
)

print(
    "5. roc_curve.png"
)

print(
    "6. precision_recall_curve.png"
)

print(
    "7. test_predictions.csv"
)

print(
    "8. risk_distribution.csv"
)

print("\n")