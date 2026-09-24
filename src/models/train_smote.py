from pathlib import Path
import sys

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import prepare_data


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }


def train_model(name, model, X_train, y_train, X_test, y_test):

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    result = evaluate_model(
        name,
        model,
        X_test,
        y_test
    )

    print(f"Accuracy : {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall   : {result['Recall']:.4f}")
    print(f"F1 Score : {result['F1 Score']:.4f}")
    print(f"ROC-AUC  : {result['ROC-AUC']:.4f}")

    return model, result


def main():

    print("=" * 70)
    print("INSIGHTAI - SMOTE EXPERIMENT")
    print("=" * 70)

    # -----------------------------------------
    # Prepare original train/test data
    # -----------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    print("\nOriginal training distribution:")
    print(pd.Series(y_train).value_counts())

    # -----------------------------------------
    # Apply SMOTE ONLY to training data
    # -----------------------------------------

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # -----------------------------------------
    # Logistic Regression
    # -----------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model, logistic_result = train_model(
        "Logistic Regression + SMOTE",
        logistic_model,
        X_train_smote,
        y_train_smote,
        X_test,
        y_test
    )

    # -----------------------------------------
    # Random Forest
    # -----------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf_model, rf_result = train_model(
        "Random Forest + SMOTE",
        rf_model,
        X_train_smote,
        y_train_smote,
        X_test,
        y_test
    )

    # -----------------------------------------
    # XGBoost
    # -----------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb_model, xgb_result = train_model(
        "XGBoost + SMOTE",
        xgb_model,
        X_train_smote,
        y_train_smote,
        X_test,
        y_test
    )

    # -----------------------------------------
    # Comparison
    # -----------------------------------------

    results = pd.DataFrame([
        logistic_result,
        rf_result,
        xgb_result
    ])

    print("\n")
    print("=" * 70)
    print("SMOTE MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    results_path = (
        PROJECT_ROOT
        / "models"
        / "smote_model_comparison.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # -----------------------------------------
    # Save models
    # -----------------------------------------

    model_dir = PROJECT_ROOT / "models"

    joblib.dump(
        logistic_model,
        model_dir / "logistic_smote.pkl"
    )

    joblib.dump(
        rf_model,
        model_dir / "random_forest_smote.pkl"
    )

    joblib.dump(
        xgb_model,
        model_dir / "xgboost_smote.pkl"
    )

    print("\nSMOTE experiment completed.")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()