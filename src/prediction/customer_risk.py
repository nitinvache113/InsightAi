import os
import sys
import joblib
import shap
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.features.preprocess import prepare_data


# ============================================================
# INSIGHTAI - CUSTOMER RISK ANALYSIS
# ============================================================

print("=" * 70)
print("INSIGHTAI - CUSTOMER LEVEL CHURN EXPLANATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------

MODEL_PATH = "models/xgboost_tuned.pkl"

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully.")


# ------------------------------------------------------------
# 2. Prepare dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test, preprocessor = prepare_data()

print("Data prepared successfully.")


# ------------------------------------------------------------
# 3. Feature names
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)


# ------------------------------------------------------------
# 4. Create SHAP explainer
# ------------------------------------------------------------

explainer = shap.TreeExplainer(model)


# ------------------------------------------------------------
# 5. Calculate SHAP values
# ------------------------------------------------------------

shap_values = explainer.shap_values(X_test_df)


# ------------------------------------------------------------
# 6. Select one customer
# ------------------------------------------------------------

customer_index = 0

customer = X_test_df.iloc[[customer_index]]

customer_shap = shap_values[customer_index]


# ------------------------------------------------------------
# 7. Churn probability
# ------------------------------------------------------------

probability = model.predict_proba(customer)[0][1]

percentage = probability * 100


# ------------------------------------------------------------
# 8. Risk category
# ------------------------------------------------------------

if probability >= 0.70:

    risk_level = "HIGH"

elif probability >= 0.40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ------------------------------------------------------------
# 9. Create explanation dataframe
# ------------------------------------------------------------

explanation = pd.DataFrame({
    "feature": feature_names,
    "shap_value": customer_shap,
    "abs_shap": np.abs(customer_shap)
})

explanation = explanation.sort_values(
    "abs_shap",
    ascending=False
)


# ------------------------------------------------------------
# 10. Display customer result
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CUSTOMER CHURN RISK")
print("=" * 70)

print(f"\nCustomer Test Index : {customer_index}")

print(
    f"Churn Probability   : {percentage:.2f}%"
)

print(
    f"Risk Level          : {risk_level}"
)


# ------------------------------------------------------------
# 11. Top factors
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TOP FACTORS INFLUENCING THIS PREDICTION")
print("-" * 70)


top_factors = explanation.head(10)

for _, row in top_factors.iterrows():

    direction = (
        "INCREASES CHURN RISK"
        if row["shap_value"] > 0
        else "REDUCES CHURN RISK"
    )

    print(
        f"{row['feature']:<50} "
        f"{row['shap_value']:.4f} "
        f"-> {direction}"
    )


print("\n" + "=" * 70)
print("CUSTOMER ANALYSIS COMPLETED")
print("=" * 70)