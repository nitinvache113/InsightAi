import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.features.preprocess import prepare_data


# ============================================================
# INSIGHTAI - CUSTOMER INTELLIGENCE ENGINE
# ============================================================

print("=" * 70)
print("INSIGHTAI - CUSTOMER INTELLIGENCE ENGINE")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/telco_clean.csv"

XGB_MODEL_PATH = "models/xgboost_tuned.pkl"

SEGMENT_MODEL_PATH = (
    "data/processed/segmentation_outputs/kmeans_model.pkl"
)

SEGMENT_SCALER_PATH = (
    "data/processed/segmentation_outputs/segmentation_scaler.pkl"
)

OUTPUT_DIR = (
    "data/processed/intelligence_outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("\nLoading models...")

model = joblib.load(
    XGB_MODEL_PATH
)

kmeans = joblib.load(
    SEGMENT_MODEL_PATH
)

segment_scaler = joblib.load(
    SEGMENT_SCALER_PATH
)

print("Models loaded successfully.")


# ============================================================
# 3. LOAD ORIGINAL CUSTOMER DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("\nCustomer dataset loaded.")

print(
    "Total customers:",
    len(df)
)


# ============================================================
# 4. PREPARE CHURN DATA
# ============================================================

print("\nPreparing data for churn prediction...")

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    train_indices,
    test_indices
) = prepare_data()


print("Data prepared successfully.")

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print(
    "Features:",
    X_test.shape[1]
)


# ============================================================
# 5. CHURN PREDICTION
# ============================================================

print("\nGenerating churn predictions...")


X_test_processed = pd.DataFrame(
    X_test
)


# Churn probability
churn_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# Churn prediction
churn_prediction = (
    churn_probability >= 0.50
).astype(int)


print(
    "Churn predictions generated."
)


# ============================================================
# 6. CONNECT PREDICTIONS TO ORIGINAL CUSTOMERS
# ============================================================

print("\nConnecting predictions to customer records...")


customer_results = df.loc[
    test_indices
].copy()


print(
    "Customer records:",
    len(customer_results)
)

print(
    "Predictions:",
    len(churn_probability)
)


# Reset index so everything aligns cleanly
customer_results = customer_results.reset_index(
    drop=True
)


# Store original dataset index
customer_results[
    "Original_Index"
] = list(test_indices)


# ============================================================
# 7. ADD CHURN INFORMATION
# ============================================================

customer_results[
    "Churn_Probability"
] = churn_probability


customer_results[
    "Predicted_Churn"
] = churn_prediction


# ============================================================
# 8. RISK LEVEL
# ============================================================

def assign_risk(probability):

    if probability >= 0.70:

        return "High"

    elif probability >= 0.40:

        return "Medium"

    else:

        return "Low"


customer_results[
    "Risk_Level"
] = customer_results[
    "Churn_Probability"
].apply(
    assign_risk
)


print(
    "Risk levels generated."
)


# ============================================================
# 9. CUSTOMER SEGMENTATION
# ============================================================

print("\nGenerating customer segments...")


segmentation_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


segment_data = customer_results[
    segmentation_features
].copy()


# Convert values to numeric
for column in segmentation_features:

    segment_data[column] = pd.to_numeric(
        segment_data[column],
        errors="coerce"
    )


# Identify valid rows
valid_mask = ~segment_data.isna().any(
    axis=1
)


segment_data_valid = segment_data[
    valid_mask
]


print(
    "Valid segmentation records:",
    len(segment_data_valid)
)


# Standardize
segment_scaled = segment_scaler.transform(
    segment_data_valid
)


# Predict clusters
segments = kmeans.predict(
    segment_scaled
)


# Create empty segment column
customer_results[
    "Customer_Segment"
] = pd.Series(
    pd.NA,
    index=customer_results.index,
    dtype="Int64"
)


# Assign cluster labels
customer_results.loc[
    segment_data_valid.index,
    "Customer_Segment"
] = segments


print(
    "Customer segments generated."
)


# ============================================================
# 10. RETENTION RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(row):

    risk = row.get(
        "Risk_Level",
        ""
    )

    contract = row.get(
        "Contract",
        ""
    )

    tech_support = row.get(
        "TechSupport",
        ""
    )

    online_security = row.get(
        "OnlineSecurity",
        ""
    )

    online_backup = row.get(
        "OnlineBackup",
        ""
    )

    internet_service = row.get(
        "InternetService",
        ""
    )

    monthly_charges = row.get(
        "MonthlyCharges",
        0
    )

    tenure = row.get(
        "tenure",
        0
    )


    recommendations = []


    # --------------------------------------------------------
    # High risk
    # --------------------------------------------------------

    if risk == "High":

        recommendations.append(
            "Priority retention outreach"
        )


    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    if contract == "Month-to-month":

        recommendations.append(
            "Offer annual or long-term contract incentive"
        )


    # --------------------------------------------------------
    # Technical support
    # --------------------------------------------------------

    if tech_support == "No":

        recommendations.append(
            "Offer technical support assistance"
        )


    # --------------------------------------------------------
    # Online security
    # --------------------------------------------------------

    if online_security == "No":

        recommendations.append(
            "Promote online security service"
        )


    # --------------------------------------------------------
    # Online backup
    # --------------------------------------------------------

    if online_backup == "No":

        recommendations.append(
            "Offer online backup package"
        )


    # --------------------------------------------------------
    # High monthly charges
    # --------------------------------------------------------

    if monthly_charges >= 80:

        recommendations.append(
            "Review pricing and provide personalized offer"
        )


    # --------------------------------------------------------
    # New customer
    # --------------------------------------------------------

    if tenure <= 6:

        recommendations.append(
            "Provide early-stage customer engagement"
        )


    # --------------------------------------------------------
    # Fiber customer
    # --------------------------------------------------------

    if internet_service == "Fiber optic":

        recommendations.append(
            "Offer fiber service support or loyalty benefit"
        )


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            "Continue regular customer engagement"
        )


    return "; ".join(
        recommendations
    )


customer_results[
    "Retention_Recommendation"
] = customer_results.apply(
    generate_recommendation,
    axis=1
)


print(
    "Retention recommendations generated."
)


# ============================================================
# 11. CUSTOMER SEGMENT PROFILE
# ============================================================

print("\nGenerating segment profiles...")


segment_profile = (
    customer_results
    .dropna(
        subset=["Customer_Segment"]
    )
    .groupby(
        "Customer_Segment"
    )
    .agg(

        Customer_Count=(
            "customerID",
            "count"
        ),

        Avg_Tenure=(
            "tenure",
            "mean"
        ),

        Avg_MonthlyCharges=(
            "MonthlyCharges",
            "mean"
        ),

        Avg_TotalCharges=(
            "TotalCharges",
            "mean"
        ),

        Avg_Churn_Probability=(
            "Churn_Probability",
            "mean"
        ),

        High_Risk_Customers=(
            "Risk_Level",
            lambda x: (
                x == "High"
            ).sum()
        )

    )
)


# ============================================================
# 12. SEGMENT CHURN RATE
# ============================================================

segment_churn = (
    customer_results
    .dropna(
        subset=["Customer_Segment"]
    )
    .groupby(
        "Customer_Segment"
    )[
        "Predicted_Churn"
    ]
    .mean()
    .mul(100)
    .rename(
        "Predicted_Churn_Rate_Percent"
    )
)


segment_profile = segment_profile.join(
    segment_churn
)


# ============================================================
# 13. ROUND NUMERIC VALUES
# ============================================================

numeric_columns = [
    "Avg_Tenure",
    "Avg_MonthlyCharges",
    "Avg_TotalCharges",
    "Avg_Churn_Probability",
    "Predicted_Churn_Rate_Percent"
]


for column in numeric_columns:

    if column in segment_profile.columns:

        segment_profile[column] = (
            segment_profile[column]
            .round(2)
        )


# ============================================================
# 14. DISPLAY RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)


risk_distribution = (
    customer_results[
        "Risk_Level"
    ]
    .value_counts()
)


print(
    risk_distribution
)


# ============================================================
# 15. DISPLAY SEGMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SEGMENT DISTRIBUTION")
print("=" * 70)


segment_distribution = (
    customer_results[
        "Customer_Segment"
    ]
    .value_counts()
    .sort_index()
)


print(
    segment_distribution
)


# ============================================================
# 16. DISPLAY SEGMENT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENT PROFILE")
print("=" * 70)


print(
    segment_profile
)


# ============================================================
# 17. SAVE CUSTOMER INTELLIGENCE
# ============================================================

customer_intelligence_path = (
    f"{OUTPUT_DIR}/customer_intelligence.csv"
)


customer_results.to_csv(
    customer_intelligence_path,
    index=False
)


print(
    "\nSaved:",
    customer_intelligence_path
)


# ============================================================
# 18. SAVE SEGMENT PROFILE
# ============================================================

segment_profile_path = (
    f"{OUTPUT_DIR}/segment_profile.csv"
)


segment_profile.to_csv(
    segment_profile_path
)


print(
    "Saved:",
    segment_profile_path
)


# ============================================================
# 19. HIGH-RISK CUSTOMERS
# ============================================================

high_risk = customer_results[
    customer_results[
        "Risk_Level"
    ] == "High"
].copy()


high_risk = high_risk.sort_values(
    "Churn_Probability",
    ascending=False
)


high_risk_path = (
    f"{OUTPUT_DIR}/high_risk_customers.csv"
)


high_risk.to_csv(
    high_risk_path,
    index=False
)


print(
    "Saved:",
    high_risk_path
)


# ============================================================
# 20. PRIORITY CUSTOMERS
# ============================================================

priority_customers = customer_results[
    (
        customer_results["Risk_Level"]
        == "High"
    )
    &
    (
        customer_results["Churn_Probability"]
        >= 0.80
    )
].copy()


priority_customers = (
    priority_customers
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
)


priority_path = (
    f"{OUTPUT_DIR}/priority_customers.csv"
)


priority_customers.to_csv(
    priority_path,
    index=False
)


print(
    "Saved:",
    priority_path
)


# ============================================================
# 21. TOP PRIORITY CUSTOMER PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("TOP PRIORITY CUSTOMERS")
print("=" * 70)


preview_columns = [
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "Churn_Probability",
    "Risk_Level",
    "Customer_Segment",
    "Retention_Recommendation"
]


available_preview_columns = [
    column
    for column in preview_columns
    if column in priority_customers.columns
]


if len(priority_customers) > 0:

    print(
        priority_customers[
            available_preview_columns
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "No customers currently meet "
        "the priority threshold."
    )


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER INTELLIGENCE ENGINE COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "1. customer_intelligence.csv"
)

print(
    "2. segment_profile.csv"
)

print(
    "3. high_risk_customers.csv"
)

print(
    "4. priority_customers.csv"
)

print("\nOutput directory:")

print(
    OUTPUT_DIR
)

print("=" * 70)