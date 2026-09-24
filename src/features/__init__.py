from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_clean.csv"
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    # Remove customer ID because it has no predictive meaning
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert target variable
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def create_preprocessor(X):

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


def prepare_data():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    print("=" * 60)
    print("INSIGHTAI - FEATURE ENGINEERING")
    print("=" * 60)

    print("\nOriginal training samples:")
    print(len(y_train))

    print("\nOriginal testing samples:")
    print(len(y_test))

    print("\nProcessed training shape:")
    print(X_train.shape)

    print("\nProcessed testing shape:")
    print(X_test.shape)

    print("\nChurn distribution in training data:")
    print(y_train.value_counts())

    print("\nChurn distribution in testing data:")
    print(y_test.value_counts())

    print("\nPreprocessing completed successfully!")