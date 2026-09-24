import os
import sys
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from src.features.preprocess import prepare_data


# ============================================================
# INSIGHTAI - SHAP EXPLAINABLE AI ANALYSIS
# ============================================================

print("=" * 70)
print("INSIGHTAI - SHAP EXPLAINABLE AI ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load trained XGBoost model
# ------------------------------------------------------------

MODEL_PATH = "models/xgboost_tuned.pkl"

model = joblib.load(MODEL_PATH)

print("\nTuned XGBoost model loaded successfully.")


# ------------------------------------------------------------
# 2. Prepare data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test, preprocessor = prepare_data()

print("\nData prepared successfully.")

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ------------------------------------------------------------
# 3. IMPORTANT:
# prepare_data() already returns transformed data.
# DO NOT call preprocessor.transform() again.
# ------------------------------------------------------------

X_test_df = pd.DataFrame(X_test)

print("\nSHAP input shape:", X_test_df.shape)


# ------------------------------------------------------------
# 4. Get feature names
# ------------------------------------------------------------

try:
    feature_names = preprocessor.get_feature_names_out()

    # Make sure number of names matches transformed data
    if len(feature_names) == X_test_df.shape[1]:
        X_test_df.columns = feature_names
    else:
        print(
            "\nWarning: Feature name count does not match "
            "transformed data."
        )

except Exception as e:

    print("\nCould not obtain feature names:", e)

    feature_names = [
        f"feature_{i}"
        for i in range(X_test_df.shape[1])
    ]

    X_test_df.columns = feature_names


print("Number of features:", X_test_df.shape[1])


# ------------------------------------------------------------
# 5. Create SHAP TreeExplainer
# ------------------------------------------------------------

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_test_df)

print("SHAP values calculated successfully.")


# ------------------------------------------------------------
# 6. Output directory
# ------------------------------------------------------------

OUTPUT_DIR = "data/processed/shap_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 7. Global SHAP Summary Plot
# ------------------------------------------------------------

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_df,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary.png")


# ------------------------------------------------------------
# 8. SHAP Feature Importance Bar Plot
# ------------------------------------------------------------

print("\nGenerating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_df,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_feature_importance.png")


# ------------------------------------------------------------
# 9. Save SHAP values
# ------------------------------------------------------------

shap_df = pd.DataFrame(
    shap_values,
    columns=X_test_df.columns
)

shap_df.to_csv(
    f"{OUTPUT_DIR}/shap_values.csv",
    index=False
)

print("Saved: shap_values.csv")


# ------------------------------------------------------------
# 10. Calculate global feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": X_test_df.columns,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)


# ------------------------------------------------------------
# 11. Save Top 20 Features
# ------------------------------------------------------------

importance.head(20).to_csv(
    f"{OUTPUT_DIR}/top_20_features.csv",
    index=False
)


# ------------------------------------------------------------
# 12. Display Top 20
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 SHAP FEATURES")
print("=" * 70)

print(
    importance.head(20).to_string(index=False)
)


print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)