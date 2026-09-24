import os
import sys
import joblib
import shap
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.features.preprocess import prepare_data


# ============================================================
# INSIGHTAI - FINAL CUSTOMER INSIGHT ENGINE
# ============================================================

print("=" * 70)
print("INSIGHTAI - FINAL CUSTOMER INSIGHT ENGINE")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/telco_clean.csv"

MODEL_PATH = "models/xgboost_tuned.pkl"

OUTPUT_DIR = (
    "data/processed/intelligence_outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

print("\nXGBoost model loaded.")


# ============================================================
# 3. PREPARE DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    train_indices,
    test_indices
) = prepare_data()


print("Data prepared.")


# ============================================================
# 4. FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


X_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)


# ============================================================
# 5. SHAP
# ============================================================

print("\nCalculating SHAP explanations...")

explainer = shap.TreeExplainer(
    model
)

shap_values = (
    explainer
    .shap_values(X_test_df)
)

print("SHAP values calculated.")


# ============================================================
# 6. CUSTOMER DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

customers = df.loc[
    test_indices
].copy()

customers = customers.reset_index(
    drop=True
)


# ============================================================
# 7. CHURN PREDICTION
# ============================================================

probabilities = (
    model
    .predict_proba(X_test_df)[:, 1]
)


customers[
    "Churn_Probability"
] = probabilities


customers[
    "Predicted_Churn"
] = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# 8. RISK LEVEL
# ============================================================

def risk_level(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    return "Low"


customers[
    "Risk_Level"
] = customers[
    "Churn_Probability"
].apply(
    risk_level
)


# ============================================================
# 9. LOAD SEGMENT INFORMATION
# ============================================================

segment_file = (
    "data/processed/intelligence_outputs/"
    "customer_intelligence.csv"
)

segment_df = pd.read_csv(
    segment_file
)


customers[
    "Customer_Segment"
] = segment_df[
    "Customer_Segment"
].values


customers[
    "Retention_Priority"
] = segment_df[
    "Risk_Level"
].map({
        "High": "Critical",
        "Medium": "Medium",
        "Low": "Low"
    })


# ============================================================
# 10. SHAP EXPLANATION
# ============================================================

def get_top_shap_features(
    customer_index,
    top_n=5
):

    values = shap_values[
        customer_index
    ]

    explanation = pd.DataFrame({

        "feature":
            feature_names,

        "shap_value":
            values,

        "absolute_value":
            np.abs(values)

    })


    explanation = (
        explanation
        .sort_values(
            "absolute_value",
            ascending=False
        )
        .head(top_n)
    )


    return explanation


# ============================================================
# 11. HUMAN-READABLE FEATURE NAMES
# ============================================================

def clean_feature_name(
    feature
):

    feature = feature.replace(
        "categorical__",
        ""
    )

    feature = feature.replace(
        "numerical__",
        ""
    )

    feature = feature.replace(
        "_",
        " "
    )

    return feature


# ============================================================
# 12. CREATE TOP SHAP DRIVERS
# ============================================================

top_driver_list = []

top_driver_values = []

top_driver_directions = []


for index in range(
    len(customers)
):

    explanation = (
        get_top_shap_features(
            index,
            top_n=5
        )
    )


    drivers = []

    values = []

    directions = []


    for _, row in explanation.iterrows():

        feature = clean_feature_name(
            row["feature"]
        )

        value = row[
            "shap_value"
        ]


        direction = (
            "Increases churn risk"
            if value > 0
            else "Reduces churn risk"
        )


        drivers.append(
            feature
        )

        values.append(
            round(float(value), 4)
        )

        directions.append(
            direction
        )


    top_driver_list.append(
        " | ".join(drivers)
    )

    top_driver_values.append(
        " | ".join(
            map(str, values)
        )
    )

    top_driver_directions.append(
        " | ".join(directions)
    )


customers[
    "Top_SHAP_Drivers"
] = top_driver_list


customers[
    "SHAP_Values"
] = top_driver_values


customers[
    "SHAP_Directions"
] = top_driver_directions


# ============================================================
# 13. FINAL RECOMMENDATION
# ============================================================

def final_recommendation(
    row
):

    recommendations = []


    probability = row[
        "Churn_Probability"
    ]

    contract = row.get(
        "Contract",
        ""
    )

    tenure = row.get(
        "tenure",
        0
    )

    security = row.get(
        "OnlineSecurity",
        ""
    )

    support = row.get(
        "TechSupport",
        ""
    )

    monthly = row.get(
        "MonthlyCharges",
        0
    )


    if probability >= 0.80:

        recommendations.append(
            "Immediate personal outreach"
        )

    elif probability >= 0.70:

        recommendations.append(
            "Priority retention outreach"
        )

    elif probability >= 0.40:

        recommendations.append(
            "Proactive retention communication"
        )

    else:

        recommendations.append(
            "Maintain regular engagement"
        )


    if contract == "Month-to-month":

        recommendations.append(
            "Offer long-term contract incentive"
        )


    if tenure <= 6:

        recommendations.append(
            "Provide new-customer support"
        )


    if security == "No":

        recommendations.append(
            "Promote online security"
        )


    if support == "No":

        recommendations.append(
            "Offer technical support"
        )


    if monthly >= 80:

        recommendations.append(
            "Review pricing"
        )


    return "; ".join(
        recommendations
    )


customers[
    "Final_Retention_Action"
] = customers.apply(
    final_recommendation,
    axis=1
)


# ============================================================
# 14. SAVE FINAL DATASET
# ============================================================

OUTPUT_PATH = (
    f"{OUTPUT_DIR}/final_customer_insights.csv"
)


customers.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nFinal customer intelligence saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 15. DISPLAY EXAMPLE CUSTOMERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 HIGHEST-RISK CUSTOMERS")
print("=" * 70)


preview_columns = [
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "Churn_Probability",
    "Risk_Level",
    "Customer_Segment",
    "Top_SHAP_Drivers",
    "Final_Retention_Action"
]


preview = (
    customers
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
    .head(10)
)


print(
    preview[
        preview_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 16. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("FINAL CUSTOMER INTELLIGENCE COMPLETED")
print("=" * 70)