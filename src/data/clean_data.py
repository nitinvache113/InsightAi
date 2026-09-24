from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"


def clean_dataset(df):
    df = df.copy()

    # Convert TotalCharges from text to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove records without churn information
    df = df.dropna(subset=["Churn"])

    return df


df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)

df_clean = clean_dataset(df)

print("Cleaned dataset shape:", df_clean.shape)

PROCESSED_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_clean.to_csv(
    PROCESSED_PATH,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(PROCESSED_PATH)