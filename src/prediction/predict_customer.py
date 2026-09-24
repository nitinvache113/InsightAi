import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MODEL PATHS
# ============================================================

XGBOOST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_tuned.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "preprocessor.pkl"
)

KMEANS_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "segmentation_outputs",
    "kmeans_model.pkl"
)

SEGMENT_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "segmentation_outputs",
    "segmentation_scaler.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

xgb_model = joblib.load(
    XGBOOST_MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

kmeans_model = joblib.load(
    KMEANS_MODEL_PATH
)

segment_scaler = joblib.load(
    SEGMENT_SCALER_PATH
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

from src.features.preprocess import prepare_data


(
    X_train,
    X_test,
    y_train,
    y_test,
    loaded_preprocessor,
    train_indices,
    test_indices
) = prepare_data()


# ============================================================
# FEATURE NAMES
# ============================================================

try:

    feature_names = (
        loaded_preprocessor
        .get_feature_names_out()
    )

except Exception:

    feature_names = [
        f"Feature_{i}"
        for i in range(
            X_train.shape[1]
        )
    ]


# ============================================================
# CLEAN FEATURE NAME
# ============================================================

def clean_feature_name(
    feature_name
):

    feature_name = str(
        feature_name
    )

    feature_name = (
        feature_name
        .replace(
            "categorical__",
            ""
        )
        .replace(
            "numerical__",
            ""
        )
    )

    feature_name = (
        feature_name
        .replace(
            "_",
            " "
        )
    )

    return feature_name


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability
):

    if probability >= 0.70:

        return "High"

    elif probability >= 0.40:

        return "Medium"

    return "Low"


# ============================================================
# RETENTION RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    customer,
    probability,
    risk
):

    recommendations = []


    # High churn risk
    if probability >= 0.70:

        recommendations.append(
            "Immediate personal outreach"
        )


    # Medium risk
    elif probability >= 0.40:

        recommendations.append(
            "Proactive retention follow-up"
        )


    # Contract
    if customer.get(
        "Contract"
    ) == "Month-to-month":

        recommendations.append(
            "Offer long-term contract incentive"
        )


    # New customer
    if customer.get(
        "tenure",
        0
    ) <= 6:

        recommendations.append(
            "Provide new-customer engagement and onboarding support"
        )


    # Security
    if customer.get(
        "OnlineSecurity"
    ) == "No":

        recommendations.append(
            "Promote online security service"
        )


    # Technical support
    if customer.get(
        "TechSupport"
    ) == "No":

        recommendations.append(
            "Offer technical support assistance"
        )


    # Charges
    monthly_charges = float(
        customer.get(
            "MonthlyCharges",
            0
        )
    )

    if monthly_charges >= 80:

        recommendations.append(
            "Review pricing and provide a suitable retention offer"
        )


    # Fiber
    if customer.get(
        "InternetService"
    ) == "Fiber optic":

        recommendations.append(
            "Provide fiber service quality and support follow-up"
        )


    # Fallback
    if not recommendations:

        recommendations.append(
            "Continue regular customer engagement"
        )


    return recommendations


# ============================================================
# CUSTOMER SEGMENT PREDICTION
# ============================================================

def predict_segment(
    customer
):

    segment_features = pd.DataFrame(
        {
            "tenure": [
                float(
                    customer.get(
                        "tenure",
                        0
                    )
                )
            ],

            "MonthlyCharges": [
                float(
                    customer.get(
                        "MonthlyCharges",
                        0
                    )
                )
            ],

            "TotalCharges": [
                float(
                    customer.get(
                        "TotalCharges",
                        0
                    )
                )
            ]
        }
    )


    # Scale using the same scaler
    # used during K-Means training.

    scaled_features = (
        segment_scaler.transform(
            segment_features
        )
    )


    segment = kmeans_model.predict(
        scaled_features
    )[0]


    return int(segment)


# ============================================================
# MAIN CUSTOMER PREDICTION
# ============================================================

def predict_customer(
    customer_data
):

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )


    # --------------------------------------------------------
    # Customer ID is not used by model
    # --------------------------------------------------------

    customer_id = customer_df[
        "customerID"
    ].iloc[0] if "customerID" in customer_df.columns else "NEW-CUSTOMER"


    if "customerID" in customer_df.columns:

        customer_df = customer_df.drop(
            columns=[
                "customerID"
            ]
        )


    # --------------------------------------------------------
    # TotalCharges conversion
    # --------------------------------------------------------

    if "TotalCharges" in customer_df.columns:

        customer_df[
            "TotalCharges"
        ] = pd.to_numeric(
            customer_df[
                "TotalCharges"
            ],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_customer = preprocessor.transform(
        customer_df
    )


    # --------------------------------------------------------
    # CHURN PREDICTION
    # --------------------------------------------------------

    churn_probability = float(
        xgb_model.predict_proba(
            X_customer
        )[0][1]
    )


    predicted_churn = int(
        churn_probability >= 0.50
    )


    risk_level = get_risk_level(
        churn_probability
    )


    # --------------------------------------------------------
    # K-MEANS SEGMENT
    # --------------------------------------------------------

    try:

        customer_segment = predict_segment(
            customer_data
        )

    except Exception as error:

        print(
            "Segment prediction warning:",
            error
        )

        customer_segment = None


    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        xgb_model
    )


    shap_values = explainer.shap_values(
        X_customer
    )


    # Handle different SHAP output formats

    if isinstance(
        shap_values,
        list
    ):

        shap_row = np.array(
            shap_values[-1][0]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            shap_row = shap_array[
                0,
                :,
                -1
            ]

        else:

            shap_row = shap_array[
                0
            ]


    # --------------------------------------------------------
    # TOP DRIVERS
    # --------------------------------------------------------

    driver_data = []


    for index, value in enumerate(
        shap_row
    ):

        driver_data.append(
            {
                "feature":
                    clean_feature_name(
                        feature_names[index]
                    ),

                "shap_value":
                    float(value),

                "absolute_value":
                    abs(
                        float(value)
                    )
            }
        )


    driver_data = sorted(
        driver_data,
        key=lambda x:
            x["absolute_value"],
        reverse=True
    )


    top_drivers = []


    for driver in driver_data[:5]:

        if driver[
            "shap_value"
        ] >= 0:

            direction = (
                "Increases churn"
            )

        else:

            direction = (
                "Reduces churn"
            )


        top_drivers.append(
            {
                "feature":
                    driver["feature"],

                "shap_value":
                    round(
                        driver["shap_value"],
                        4
                    ),

                "direction":
                    direction
            }
        )


    # --------------------------------------------------------
    # RETENTION RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            customer_data,
            churn_probability,
            risk_level
        )
    )


    # --------------------------------------------------------
    # RETENTION PRIORITY
    # --------------------------------------------------------

    if churn_probability >= 0.80:

        retention_priority = (
            "Critical"
        )

    elif churn_probability >= 0.70:

        retention_priority = (
            "High"
        )

    elif churn_probability >= 0.40:

        retention_priority = (
            "Medium"
        )

    else:

        retention_priority = (
            "Low"
        )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "customer_id":
            customer_id,

        "churn_probability":
            churn_probability,

        "predicted_churn":
            predicted_churn,

        "risk_level":
            risk_level,

        "customer_segment":
            customer_segment,

        "retention_priority":
            retention_priority,

        "top_drivers":
            top_drivers,

        "recommendations":
            recommendations
    }