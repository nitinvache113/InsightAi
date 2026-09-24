from pathlib import Path
import sys

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import prepare_data


def main():

    print("=" * 70)
    print("INSIGHTAI - XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load and preprocess data
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    print("\nOriginal training distribution:")
    print(pd.Series(y_train).value_counts())

    # --------------------------------------------------
    # Apply SMOTE ONLY to training data
    # --------------------------------------------------

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # --------------------------------------------------
    # Base XGBoost
    # --------------------------------------------------

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------
    # Hyperparameter search space
    # --------------------------------------------------

    param_distributions = {

        "n_estimators": [
            100,
            200,
            300,
            500
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            8
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
            0.2
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "min_child_weight": [
            1,
            3,
            5,
            7
        ],

        "gamma": [
            0,
            0.1,
            0.2,
            0.5
        ]
    }

    # --------------------------------------------------
    # Randomized search
    # --------------------------------------------------

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("\nStarting hyperparameter optimization...")

    search.fit(
        X_train_smote,
        y_train_smote
    )

    print("\nOptimization completed.")

    # --------------------------------------------------
    # Best parameters
    # --------------------------------------------------

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Cross-Validation ROC-AUC:")
    print(
        round(
            search.best_score_,
            4
        )
    )

    # --------------------------------------------------
    # Evaluate on untouched test set
    # --------------------------------------------------

    best_model = search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

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

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TUNED XGBOOST TEST RESULTS")
    print("=" * 70)

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

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_dir = PROJECT_ROOT / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        model_dir / "xgboost_tuned.pkl"
    )

    joblib.dump(
        preprocessor,
        model_dir / "preprocessor.pkl"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = pd.DataFrame([
        {
            "Model": "Tuned XGBoost + SMOTE",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    ])

    results.to_csv(
        model_dir / "xgboost_tuned_results.csv",
        index=False
    )

    print("\nTuned model saved successfully.")

    print(
        "\nModel location:"
    )

    print(
        model_dir / "xgboost_tuned.pkl"
    )


if __name__ == "__main__":
    main()