from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("INSIGHTAI - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe().T)

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    (df["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)

# -----------------------------
# Churn Distribution Chart
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_distribution.png",
    dpi=200
)

plt.close()

# -----------------------------
# Contract vs Churn
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "contract_vs_churn.png",
    dpi=200
)

plt.close()

# -----------------------------
# Tenure vs Churn
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "tenure_vs_churn.png",
    dpi=200
)

plt.close()

# -----------------------------
# Monthly Charges vs Churn
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_charges_vs_churn.png",
    dpi=200
)

plt.close()

print("\nEDA completed successfully!")

print("\nCharts saved in:")
print(OUTPUT_DIR)