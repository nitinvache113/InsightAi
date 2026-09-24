import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# INSIGHTAI - DATA PREPROCESSING
# ============================================================

DATA_PATH = "data/processed/telco_clean.csv"


def prepare_data():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Convert TotalCharges
    # --------------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Churn"]
    )

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # --------------------------------------------------------
    # Drop customer ID
    # --------------------------------------------------------

    X = X.drop(
        columns=["customerID"]
    )

    # --------------------------------------------------------
    # Define feature types
    # --------------------------------------------------------

    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Numeric pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # --------------------------------------------------------
    # Return original indices too
    # --------------------------------------------------------

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        X_train.index,
        X_test.index
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        train_indices,
        test_indices
    ) = prepare_data()

    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    print(
        "Training features:",
        X_train.shape[1]
    )

    print(
        "Testing features:",
        X_test.shape[1]
    )