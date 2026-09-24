from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


def main():

    print("=" * 60)
    print("INSIGHTAI - MACHINE LEARNING EXPERIMENT")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    # ------------------------------------------------
    # Model 1: Logistic Regression
    # ------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # Model 2: Random Forest
    # ------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # Model 3: XGBoost
    # ------------------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_results = evaluate_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # Model comparison
    # ------------------------------------------------

    results = pd.DataFrame([
        logistic_results,
        rf_results,
        xgb_results
    ])

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False
        )
    )

    # Save results

    results.to_csv(
        PROJECT_ROOT
        / "models"
        / "model_comparison.csv",
        index=False
    )

    # Save models

    joblib.dump(
        logistic_model,
        MODEL_DIR / "logistic_regression.pkl"
    )

    joblib.dump(
        random_forest_model,
        MODEL_DIR / "random_forest.pkl"
    )

    joblib.dump(
        xgb_model,
        MODEL_DIR / "xgboost.pkl"
    )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl"
    )

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()