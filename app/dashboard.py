# ============================================================
# InsightAI - Explainable Customer Intelligence Platform
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
import sys
import ast
import json

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightAI | Customer Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

INTELLIGENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intelligence_outputs"
)

SEGMENTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "segmentation_outputs"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_outputs"
)

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap_outputs"
)

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    """Safely load a CSV file."""

    if path.exists():

        try:
            return pd.read_csv(path)

        except Exception as e:

            st.error(
                f"Unable to read {path.name}: {e}"
            )

            return pd.DataFrame()

    return pd.DataFrame()


def format_percentage(value):
    """Format decimal as percentage."""

    try:
        return f"{float(value) * 100:.2f}%"

    except Exception:
        return "N/A"


def normalize_columns(df):
    """Normalize commonly used column names."""

    df = df.copy()

    if "Customer_Segment" in df.columns:
        df["Segment"] = df["Customer_Segment"]

    elif "Cluster" in df.columns:
        df["Segment"] = df["Cluster"]

    if "Churn_Probability" in df.columns:
        df["Churn Probability"] = df["Churn_Probability"]

    if "Risk_Level" in df.columns:
        df["Risk"] = df["Risk_Level"]

    return df


def safe_numeric(df, column):

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def parse_list(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    text = str(value).strip()

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    if "|" in text:
        return [
            x.strip()
            for x in text.split("|")
            if x.strip()
        ]

    if "," in text:
        return [
            x.strip()
            for x in text.split(",")
            if x.strip()
        ]

    return [text] if text else []


def risk_order():

    return [
        "Low",
        "Medium",
        "High"
    ]


def priority_order():

    return [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]


# ============================================================
# LOAD DATA
# ============================================================

final_df = load_csv(
    INTELLIGENCE_DIR
    / "final_customer_insights.csv"
)

intelligence_df = load_csv(
    INTELLIGENCE_DIR
    / "customer_intelligence.csv"
)

retention_df = load_csv(
    INTELLIGENCE_DIR
    / "retention_strategy.csv"
)

segment_profile_df = load_csv(
    INTELLIGENCE_DIR
    / "segment_profile.csv"
)

segments_df = load_csv(
    SEGMENTATION_DIR
    / "customer_segments.csv"
)

metrics_df = load_csv(
    EVALUATION_DIR
    / "model_metrics.csv"
)

classification_report_file = (
    EVALUATION_DIR
    / "classification_report.txt"
)

confusion_matrix_file = (
    EVALUATION_DIR
    / "confusion_matrix.csv"
)

predictions_file = (
    EVALUATION_DIR
    / "test_predictions.csv"
)

risk_distribution_file = (
    EVALUATION_DIR
    / "risk_distribution.csv"
)


# ============================================================
# NORMALIZE DATA
# ============================================================

final_df = normalize_columns(final_df)
intelligence_df = normalize_columns(intelligence_df)
retention_df = normalize_columns(retention_df)
segments_df = normalize_columns(segments_df)
segment_profile_df = normalize_columns(segment_profile_df)

if not final_df.empty:

    if "Churn_Probability" in final_df.columns:

        final_df = safe_numeric(
            final_df,
            "Churn_Probability"
        )

    if "MonthlyCharges" in final_df.columns:

        final_df = safe_numeric(
            final_df,
            "MonthlyCharges"
        )

    if "tenure" in final_df.columns:

        final_df = safe_numeric(
            final_df,
            "tenure"
        )

    if "TotalCharges" in final_df.columns:

        final_df = safe_numeric(
            final_df,
            "TotalCharges"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 InsightAI")

st.sidebar.caption(
    "Explainable AI Customer Intelligence Platform"
)

st.sidebar.divider()

pages = [
    "📊 Executive Overview",
    "🔎 Customer Explorer",
    "⚠️ Risk Intelligence",
    "👥 Customer Segments",
    "🎯 Retention Center",
    "🔮 Predict New Customer",
    "📈 Model Performance"
]

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.divider()

st.sidebar.markdown(
    """
### Platform Modules

🧠 Churn Prediction  
👥 Customer Segmentation  
🔍 Explainable AI  
🎯 Retention Intelligence  
📊 Analytics Dashboard  
📈 Model Evaluation
"""
)

st.sidebar.divider()

st.sidebar.caption(
    "InsightAI • Academic / Research Project"
)


# ============================================================
# CHECK MAIN DATA
# ============================================================

if final_df.empty:

    st.error(
        """
        Customer intelligence data was not found.

        Please run the complete InsightAI pipeline first.
        """
    )

    st.stop()


# ============================================================
# PAGE 1
# EXECUTIVE OVERVIEW
# ============================================================

if page == "📊 Executive Overview":

    st.title("📊 Executive Overview")

    st.caption(
        "AI-powered overview of customer churn, risk and retention intelligence."
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_customers = len(final_df)

    churned_customers = 0

    if "Predicted_Churn" in final_df.columns:

        churned_customers = int(
            pd.to_numeric(
                final_df["Predicted_Churn"],
                errors="coerce"
            ).fillna(0).sum()
        )

    elif "Churn" in final_df.columns:

        churned_customers = int(
            (
                final_df["Churn"]
                .astype(str)
                .str.lower()
                .eq("yes")
            ).sum()
        )

    high_risk = 0

    if "Risk_Level" in final_df.columns:

        high_risk = int(
            final_df["Risk_Level"]
            .astype(str)
            .str.lower()
            .eq("high")
            .sum()
        )

    critical_customers = 0

    if "Retention_Priority" in final_df.columns:

        critical_customers = int(
            final_df["Retention_Priority"]
            .astype(str)
            .str.lower()
            .eq("critical")
            .sum()
        )

    average_probability = 0

    if "Churn_Probability" in final_df.columns:

        average_probability = (
            final_df["Churn_Probability"]
            .mean()
        )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Customers Evaluated",
        f"{total_customers:,}"
    )

    c2.metric(
        "Predicted Churn",
        f"{churned_customers:,}"
    )

    c3.metric(
        "High Risk",
        f"{high_risk:,}"
    )

    c4.metric(
        "Critical Priority",
        f"{critical_customers:,}"
    )

    c5.metric(
        "Avg Churn Probability",
        format_percentage(average_probability)
    )

    st.divider()

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("🔥 Risk Distribution")

        if "Risk_Level" in final_df.columns:

            risk_counts = (
                final_df["Risk_Level"]
                .value_counts()
                .reindex(
                    risk_order(),
                    fill_value=0
                )
                .reset_index()
            )

            risk_counts.columns = [
                "Risk Level",
                "Customers"
            ]

            fig = px.bar(
                risk_counts,
                x="Risk Level",
                y="Customers",
                text="Customers",
                title="Customer Risk Levels"
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader("👥 Customer Segments")

        if "Segment" in final_df.columns:

            segment_counts = (
                final_df["Segment"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            segment_counts.columns = [
                "Segment",
                "Customers"
            ]

            segment_counts["Segment"] = (
                segment_counts["Segment"]
                .astype(str)
            )

            fig = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                hole=0.45,
                title="Customer Segment Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # CHURN PROBABILITY
    # --------------------------------------------------------

    st.subheader("📈 Churn Probability Distribution")

    if "Churn_Probability" in final_df.columns:

        fig = px.histogram(
            final_df,
            x="Churn_Probability",
            nbins=30,
            title="Predicted Churn Probability",
            labels={
                "Churn_Probability":
                "Churn Probability"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TOP RISK CUSTOMERS
    # --------------------------------------------------------

    st.subheader("🚨 Highest-Risk Customers")

    if "Churn_Probability" in final_df.columns:

        top_risk = (
            final_df
            .sort_values(
                "Churn_Probability",
                ascending=False
            )
            .head(10)
            .copy()
        )

        display_columns = [
            col for col in [
                "customerID",
                "Churn_Probability",
                "Risk_Level",
                "Customer_Segment",
                "Retention_Priority",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "Final_Retention_Action"
            ]
            if col in top_risk.columns
        ]

        st.dataframe(
            top_risk[display_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 2
# CUSTOMER EXPLORER
# ============================================================

elif page == "🔎 Customer Explorer":

    st.title("🔎 Customer Explorer")

    st.caption(
        "Explore individual customers, churn probability, SHAP drivers and retention actions."
    )

    customer_column = None

    if "customerID" in final_df.columns:
        customer_column = "customerID"

    elif "CustomerID" in final_df.columns:
        customer_column = "CustomerID"

    if customer_column is None:

        st.warning(
            "Customer ID column not available."
        )

        st.stop()

    customer_ids = (
        final_df[customer_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids
    )

    customer_data = final_df[
        final_df[customer_column].astype(str)
        == selected_customer
    ]

    if customer_data.empty:
        st.warning("Customer not found.")
        st.stop()

    customer = customer_data.iloc[0]

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER KPI
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    probability = customer.get(
        "Churn_Probability",
        np.nan
    )

    risk = customer.get(
        "Risk_Level",
        "N/A"
    )

    segment = customer.get(
        "Customer_Segment",
        customer.get("Segment", "N/A")
    )

    priority = customer.get(
        "Retention_Priority",
        "N/A"
    )

    c1.metric(
        "Customer ID",
        str(selected_customer)
    )

    c2.metric(
        "Churn Probability",
        format_percentage(probability)
    )

    c3.metric(
        "Risk Level",
        str(risk)
    )

    c4.metric(
        "Customer Segment",
        str(segment)
    )

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.subheader("👤 Customer Profile")

    profile_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    available_profile = [
        col
        for col in profile_columns
        if col in customer.index
    ]

    if available_profile:

        profile_df = pd.DataFrame(
            {
                "Attribute": available_profile,
                "Value": [
                    customer[col]
                    for col in available_profile
                ]
            }
        )

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # SHAP DRIVERS
    # --------------------------------------------------------

    st.subheader("🧠 Explainable AI — Churn Drivers")

    top_drivers = customer.get(
        "Top_SHAP_Drivers",
        ""
    )

    shap_values = customer.get(
        "SHAP_Values",
        ""
    )

    shap_directions = customer.get(
        "SHAP_Directions",
        ""
    )

    drivers = parse_list(top_drivers)
    values = parse_list(shap_values)
    directions = parse_list(shap_directions)

    if drivers:

        shap_records = []

        for i, driver in enumerate(drivers):

            value = np.nan

            direction = ""

            if i < len(values):

                try:
                    value = float(values[i])
                except Exception:
                    pass

            if i < len(directions):

                direction = str(
                    directions[i]
                )

            shap_records.append(
                {
                    "Feature": driver,
                    "SHAP Value": value,
                    "Direction": direction
                }
            )

        shap_df = pd.DataFrame(
            shap_records
        )

        if not shap_df.empty:

            fig = px.bar(
                shap_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title="Top Churn Drivers"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                shap_df,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "SHAP driver information is not available for this customer."
        )

    # --------------------------------------------------------
    # RETENTION ACTION
    # --------------------------------------------------------

    st.subheader("🎯 Recommended Retention Action")

    action = customer.get(
        "Final_Retention_Action",
        "No recommendation available."
    )

    st.info(
        str(action)
    )

    st.metric(
        "Retention Priority",
        str(priority)
    )


# ============================================================
# PAGE 3
# RISK INTELLIGENCE
# ============================================================

elif page == "⚠️ Risk Intelligence":

    st.title("⚠️ Risk Intelligence")

    st.caption(
        "Identify and analyze customers with elevated churn risk."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        selected_risks = st.multiselect(
            "Risk Level",
            risk_order(),
            default=risk_order()
        )

    with c2:

        min_probability = st.slider(
            "Minimum Churn Probability",
            0.0,
            1.0,
            0.0,
            0.05
        )

    risk_df = final_df.copy()

    if "Risk_Level" in risk_df.columns:

        risk_df = risk_df[
            risk_df["Risk_Level"]
            .isin(selected_risks)
        ]

    if "Churn_Probability" in risk_df.columns:

        risk_df = risk_df[
            risk_df["Churn_Probability"]
            >= min_probability
        ]

    # --------------------------------------------------------
    # RISK KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Filtered Customers",
        f"{len(risk_df):,}"
    )

    if "Risk_Level" in risk_df.columns:

        c2.metric(
            "High Risk",
            str(
                (
                    risk_df["Risk_Level"]
                    == "High"
                ).sum()
            )
        )

        c3.metric(
            "Medium Risk",
            str(
                (
                    risk_df["Risk_Level"]
                    == "Medium"
                ).sum()
            )
        )

        c4.metric(
            "Low Risk",
            str(
                (
                    risk_df["Risk_Level"]
                    == "Low"
                ).sum()
            )
        )

    st.divider()

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    if "Risk_Level" in risk_df.columns:

        counts = (
            risk_df["Risk_Level"]
            .value_counts()
            .reindex(
                risk_order(),
                fill_value=0
            )
            .reset_index()
        )

        counts.columns = [
            "Risk Level",
            "Customers"
        ]

        fig = px.bar(
            counts,
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="Filtered Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TOP HIGH RISK
    # --------------------------------------------------------

    st.subheader("🚨 Priority Risk Customers")

    if "Churn_Probability" in risk_df.columns:

        top_risk = (
            risk_df
            .sort_values(
                "Churn_Probability",
                ascending=False
            )
            .head(50)
        )

        display_columns = [
            col for col in [
                "customerID",
                "Churn_Probability",
                "Risk_Level",
                "Customer_Segment",
                "Retention_Priority",
                "Contract",
                "tenure",
                "MonthlyCharges"
            ]
            if col in top_risk.columns
        ]

        st.dataframe(
            top_risk[display_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = top_risk.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Risk Customers",
            data=csv_data,
            file_name="risk_customers.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 4
# CUSTOMER SEGMENTS
# ============================================================

elif page == "👥 Customer Segments":

    st.title("👥 Customer Segmentation")

    st.caption(
        "K-Means based customer segmentation combined with churn intelligence."
    )

    if final_df.empty:

        st.warning(
            "Customer data unavailable."
        )

        st.stop()

    # --------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # --------------------------------------------------------

    if "Segment" in final_df.columns:

        segment_counts = (
            final_df["Segment"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers"
        ]

        segment_counts["Segment"] = (
            segment_counts["Segment"]
            .astype(str)
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                text="Customers",
                title="Customers per Segment"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                hole=0.45,
                title="Segment Share"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # SEGMENT PROFILE
    # --------------------------------------------------------

    st.subheader("📋 Segment Profiles")

    if not segment_profile_df.empty:

        st.dataframe(
            segment_profile_df,
            use_container_width=True,
            hide_index=True
        )

    elif not segments_df.empty:

        numeric_columns = [
            col
            for col in [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Churn"
            ]
            if col in segments_df.columns
        ]

        if numeric_columns:

            profile = (
                segments_df
                .groupby("Segment")[numeric_columns]
                .mean()
                .reset_index()
            )

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # SEGMENT VS CHURN
    # --------------------------------------------------------

    if (
        "Segment" in final_df.columns
        and "Churn_Probability" in final_df.columns
    ):

        st.subheader(
            "📈 Segment vs Churn Probability"
        )

        segment_probability = (
            final_df
            .groupby("Segment")[
                "Churn_Probability"
            ]
            .mean()
            .reset_index()
        )

        segment_probability.columns = [
            "Segment",
            "Average Churn Probability"
        ]

        segment_probability[
            "Segment"
        ] = segment_probability[
            "Segment"
        ].astype(str)

        fig = px.bar(
            segment_probability,
            x="Segment",
            y="Average Churn Probability",
            text="Average Churn Probability",
            title="Average Churn Probability by Segment"
        )

        fig.update_traces(
            texttemplate="%{text:.2%}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TENURE VS MONTHLY CHARGES
    # --------------------------------------------------------

    if (
        "tenure" in final_df.columns
        and "MonthlyCharges" in final_df.columns
        and "Segment" in final_df.columns
    ):

        st.subheader(
            "🔬 Customer Distribution by Tenure and Charges"
        )

        plot_df = final_df.dropna(
            subset=[
                "tenure",
                "MonthlyCharges",
                "Segment"
            ]
        )

        fig = px.scatter(
            plot_df,
            x="tenure",
            y="MonthlyCharges",
            color="Segment",
            hover_data=[
                "customerID"
            ]
            if "customerID" in plot_df.columns
            else None,
            title="Tenure vs Monthly Charges"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 5
# RETENTION CENTER
# ============================================================

elif page == "🎯 Retention Center":

    st.title("🎯 Retention Intelligence")

    st.caption(
        "Prioritize customers and generate data-driven retention actions."
    )

    # --------------------------------------------------------
    # USE RETENTION DATA IF AVAILABLE
    # --------------------------------------------------------

    if retention_df.empty:

        retention_view = final_df.copy()

    else:

        retention_view = retention_df.copy()

        # Merge final insights where required
        if (
            "customerID" in retention_view.columns
            and "customerID" in final_df.columns
        ):

            extra_columns = [
                col
                for col in [
                    "Churn_Probability",
                    "Risk_Level",
                    "Customer_Segment"
                ]
                if col in final_df.columns
            ]

            if extra_columns:

                merge_df = final_df[
                    ["customerID"]
                    + extra_columns
                ].drop_duplicates(
                    "customerID"
                )

                retention_view = retention_view.merge(
                    merge_df,
                    on="customerID",
                    how="left",
                    suffixes=("", "_insight")
                )

    # --------------------------------------------------------
    # PRIORITY COUNTS
    # --------------------------------------------------------

    priority_column = None

    if "Retention_Priority" in retention_view.columns:

        priority_column = "Retention_Priority"

    elif "Priority" in retention_view.columns:

        priority_column = "Priority"

    if priority_column:

        counts = (
            retention_view[priority_column]
            .astype(str)
            .value_counts()
            .reindex(
                priority_order(),
                fill_value=0
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Critical",
            f"{counts.get('Critical', 0):,}"
        )

        c2.metric(
            "High",
            f"{counts.get('High', 0):,}"
        )

        c3.metric(
            "Medium",
            f"{counts.get('Medium', 0):,}"
        )

        c4.metric(
            "Low",
            f"{counts.get('Low', 0):,}"
        )

        st.divider()

        priority_chart = (
            counts
            .reset_index()
        )

        priority_chart.columns = [
            "Priority",
            "Customers"
        ]

        fig = px.bar(
            priority_chart,
            x="Priority",
            y="Customers",
            text="Customers",
            title="Retention Priority Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CAMPAIGN DISTRIBUTION
    # --------------------------------------------------------

    campaign_column = None

    for column in [
        "Recommended_Campaign",
        "Campaign",
        "Retention_Campaign"
    ]:

        if column in retention_view.columns:

            campaign_column = column
            break

    if campaign_column:

        st.subheader(
            "📣 Recommended Retention Campaigns"
        )

        campaigns = (
            retention_view[campaign_column]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        campaigns.columns = [
            "Campaign",
            "Customers"
        ]

        fig = px.bar(
            campaigns,
            x="Customers",
            y="Campaign",
            orientation="h",
            title="Recommended Campaign Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CRITICAL CUSTOMERS
    # --------------------------------------------------------

    st.subheader(
        "🚨 Customers Requiring Immediate Attention"
    )

    critical_df = retention_view.copy()

    if priority_column:

        critical_df = critical_df[
            critical_df[priority_column]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "critical",
                    "high"
                ]
            )
        ]

    if "Churn_Probability" in critical_df.columns:

        critical_df = (
            critical_df
            .sort_values(
                "Churn_Probability",
                ascending=False
            )
        )

    display_columns = [
        col
        for col in [
            "customerID",
            "Churn_Probability",
            "Risk_Level",
            "Customer_Segment",
            "Retention_Priority",
            "Recommended_Campaign",
            "Final_Retention_Action",
            "Contract",
            "tenure",
            "MonthlyCharges"
        ]
        if col in critical_df.columns
    ]

    if display_columns:

        st.dataframe(
            critical_df[
                display_columns
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        csv_data = critical_df[
            display_columns
        ].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Retention Priority List",
            data=csv_data,
            file_name="retention_priority_customers.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 6
# PREDICT NEW CUSTOMER
# ============================================================

elif page == "🔮 Predict New Customer":

    st.title("🔮 Predict New Customer")

    st.caption(
        "Estimate churn probability and generate explainable retention recommendations."
    )

    st.info(
        "Enter customer information below and use the trained InsightAI model to estimate churn risk."
    )

    # --------------------------------------------------------
    # IMPORT MODEL
    # --------------------------------------------------------

    try:

        from src.prediction.predict_customer import (
            predict_customer
        )

        prediction_available = True

    except Exception as e:

        prediction_available = False

        st.warning(
            f"Prediction module could not be loaded: {e}"
        )

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    with st.form("prediction_form"):

        st.subheader(
            "👤 Customer Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            customer_id = st.text_input(
                "Customer ID",
                "NEW-CUSTOMER-001"
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

        with c2:

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            online_security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with c3:

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=monthly_charges * tenure
            )

        submitted = st.form_submit_button(
            "🚀 Predict Customer Risk",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        if not prediction_available:

            st.error(
                "Prediction module is unavailable."
            )

            st.stop()

        customer_data = {

            "customerID":
                customer_id,

            "gender":
                gender,

            "SeniorCitizen":
                senior_citizen,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "tenure":
                tenure,

            "PhoneService":
                phone_service,

            "MultipleLines":
                multiple_lines,

            "InternetService":
                internet_service,

            "OnlineSecurity":
                online_security,

            "OnlineBackup":
                online_backup,

            "DeviceProtection":
                device_protection,

            "TechSupport":
                tech_support,

            "StreamingTV":
                streaming_tv,

            "StreamingMovies":
                streaming_movies,

            "Contract":
                contract,

            "PaperlessBilling":
                paperless_billing,

            "PaymentMethod":
                payment_method,

            "MonthlyCharges":
                monthly_charges,

            "TotalCharges":
                total_charges
        }

        try:

            with st.spinner(
                "Running InsightAI prediction..."
            ):

                result = predict_customer(
                    customer_data
                )

            st.success(
                "Prediction completed successfully."
            )

            st.divider()

            # ------------------------------------------------
            # RESULT KPI
            # ------------------------------------------------

            probability = result.get(
                "churn_probability",
                result.get(
                    "Churn_Probability",
                    None
                )
            )

            risk = result.get(
                "risk_level",
                result.get(
                    "Risk_Level",
                    "N/A"
                )
            )

            segment = result.get(
                "customer_segment",
                result.get(
                    "Customer_Segment",
                    "N/A"
                )
            )

            priority = result.get(
                "retention_priority",
                result.get(
                    "Retention_Priority",
                    "N/A"
                )
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Churn Probability",
                format_percentage(probability)
            )

            c2.metric(
                "Risk Level",
                str(risk)
            )

            c3.metric(
                "Customer Segment",
                str(segment)
            )

            c4.metric(
                "Retention Priority",
                str(priority)
            )

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            if probability is not None:

                st.subheader(
                    "🎯 Churn Risk Score"
                )

                try:

                    probability_value = float(
                        probability
                    )

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability_value * 100,
                            number={
                                "suffix": "%"
                            },
                            title={
                                "text":
                                "Predicted Churn Probability"
                            },
                            gauge={
                                "axis": {
                                    "range": [
                                        0,
                                        100
                                    ]
                                },
                                "threshold": {
                                    "line": {
                                        "width": 4
                                    },
                                    "value":
                                    probability_value * 100
                                }
                            }
                        )
                    )

                    gauge.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True
                    )

                except Exception:
                    pass

            # ------------------------------------------------
            # TOP DRIVERS
            # ------------------------------------------------

            st.subheader(
                "🧠 Top Churn Drivers"
            )

            drivers = result.get(
                "top_drivers",
                result.get(
                    "Top_SHAP_Drivers",
                    []
                )
            )

            if isinstance(
                drivers,
                str
            ):

                drivers = parse_list(
                    drivers
                )

            if drivers:

                driver_records = []

                for item in drivers:

                    if isinstance(
                        item,
                        dict
                    ):

                        driver_records.append(
                            item
                        )

                    else:

                        driver_records.append(
                            {
                                "Feature":
                                str(item)
                            }
                        )

                drivers_df = pd.DataFrame(
                    driver_records
                )

                st.dataframe(
                    drivers_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # RETENTION RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "🎯 Retention Recommendations"
            )

            recommendations = result.get(
                "recommendations",
                result.get(
                    "Retention_Recommendations",
                    []
                )
            )

            if isinstance(
                recommendations,
                str
            ):

                recommendations = [
                    recommendations
                ]

            if recommendations:

                for recommendation in recommendations:

                    st.success(
                        f"• {recommendation}"
                    )

            else:

                st.info(
                    "No retention recommendations returned."
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# PAGE 7
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.caption(
        "Evaluation of the churn prediction model using an independent test dataset."
    )

    # --------------------------------------------------------
    # CHECK EVALUATION FILES
    # --------------------------------------------------------

    if not EVALUATION_DIR.exists():

        st.warning(
            """
            Evaluation results are not available yet.

            Run:

            python src/models/model_evaluation.py
            """
        )

        st.stop()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Evaluation Metrics"
    )

    # Always initialize this variable so the dashboard remains
    # stable even when model_metrics.csv is missing or empty.
    selected_metrics = metrics_df.copy() if not metrics_df.empty else pd.DataFrame()

    if metrics_df.empty:

        st.warning(
            "model_metrics.csv was not found or is empty. Showing the recorded InsightAI baseline/tuning results below."
        )

        selected_metrics = pd.DataFrame([
            {
                "Model": "Logistic Regression",
                "Accuracy": 0.805536,
                "Precision": 0.657233,
                "Recall": 0.558824,
                "F1 Score": 0.604046,
                "ROC-AUC": 0.841861,
            },
            {
                "Model": "Random Forest",
                "Accuracy": 0.765791,
                "Precision": 0.552133,
                "Recall": 0.622995,
                "F1 Score": 0.585427,
                "ROC-AUC": 0.821759,
            },
            {
                "Model": "XGBoost",
                "Accuracy": 0.789212,
                "Precision": 0.626230,
                "Recall": 0.510695,
                "F1 Score": 0.562592,
                "ROC-AUC": 0.837162,
            },
            {
                "Model": "Tuned XGBoost (Test)",
                "Accuracy": 0.7786,
                "Precision": 0.5807,
                "Recall": 0.5963,
                "F1 Score": 0.5884,
                "ROC-AUC": 0.8280,
            },
        ])

    else:

        # Find tuned XGBoost row if available
        selected_metrics = metrics_df.copy()

        model_column = None

        for column in [
            "Model",
            "model",
            "Model_Name"
        ]:

            if column in selected_metrics.columns:

                model_column = column
                break

        if model_column:

            tuned_rows = selected_metrics[
                selected_metrics[
                    model_column
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    "tuned"
                )
            ]

            if not tuned_rows.empty:

                selected_metrics = (
                    tuned_rows.iloc[
                        [0]
                    ]
                )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    # --------------------------------------------------------
    # FIND AVAILABLE METRIC COLUMNS
    # --------------------------------------------------------

    metric_columns = [
        col
        for col in metric_names
        if col in selected_metrics.columns
    ]

    # --------------------------------------------------------
    # DISPLAY METRICS SAFELY
    # --------------------------------------------------------

    if metric_columns and not selected_metrics.empty:

        cols = st.columns(
            min(len(metric_columns), 5)
        )

        for col, metric_name in zip(
            cols,
            metric_columns
        ):

            value = pd.to_numeric(
                selected_metrics.iloc[0][metric_name],
                errors="coerce"
            )

            if pd.notna(value):

                col.metric(
                    metric_name,
                    f"{float(value):.2%}"
                )

    else:

        st.warning(
            "No standard evaluation metrics were found in model_metrics.csv."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(selected_metrics.columns)
        )

        # --------------------------------------------------------
        # METRICS TABLE
        # --------------------------------------------------------

        st.subheader(
            "📋 Detailed Model Metrics"
        )

        if not metrics_df.empty:

            display_metrics = (
                metrics_df.copy()
            )

            numeric_columns = (
                display_metrics
                .select_dtypes(
                    include="number"
                )
                .columns
            )

            for column in numeric_columns:

                display_metrics[column] = (
                    display_metrics[column]
                    .round(4)
                )

            st.dataframe(
                display_metrics,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------------

        st.subheader(
            "🎯 Confusion Matrix"
        )

        confusion_image = (
            EVALUATION_DIR
            / "confusion_matrix.png"
        )

        if confusion_image.exists():

            st.image(
                str(confusion_image),
                caption="Confusion Matrix — Independent Test Dataset",
                width="stretch"
            )

        if confusion_matrix_file.exists():

            with st.expander(
                "View Confusion Matrix Values"
            ):

                cm_df = load_csv(
                    confusion_matrix_file
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True,
                    hide_index=True
                )

        # --------------------------------------------------------
        # ROC CURVE
        # --------------------------------------------------------

        st.subheader(
            "📈 ROC Curve"
        )

        roc_image = (
            EVALUATION_DIR
            / "roc_curve.png"
        )

        if roc_image.exists():

            st.image(
                str(roc_image),
                caption="Receiver Operating Characteristic Curve",
                width="stretch"
            )

        else:

            st.info(
                "ROC curve image not available."
            )

        # --------------------------------------------------------
        # PRECISION RECALL
        # --------------------------------------------------------

        st.subheader(
            "🎯 Precision–Recall Curve"
        )

        pr_image = (
            EVALUATION_DIR
            / "precision_recall_curve.png"
        )

        if pr_image.exists():

            st.image(
                str(pr_image),
                caption="Precision–Recall Curve",
                width="stretch"
            )

        else:

            st.info(
                "Precision–Recall curve image not available."
            )

        st.info(
            """
            Precision–Recall analysis is particularly useful for churn
            prediction because the dataset contains fewer churn customers
            than non-churn customers.
            """
        )

        # --------------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------------

        st.subheader(
            "📝 Classification Report"
        )

        if classification_report_file.exists():

            with open(
                classification_report_file,
                "r",
                encoding="utf-8"
            ) as file:

                report = file.read()

            st.code(
                report,
                language="text"
            )

        else:

            st.info(
                "Classification report is not available."
            )

        # --------------------------------------------------------
        # RISK DISTRIBUTION
        # --------------------------------------------------------

        st.subheader(
            "⚠️ Test Dataset Risk Distribution"
        )

        risk_distribution_df = load_csv(
            risk_distribution_file
        )

        if not risk_distribution_df.empty:

            st.dataframe(
                risk_distribution_df,
                use_container_width=True,
                hide_index=True
            )

            if "Risk_Level" in risk_distribution_df.columns:

                risk_counts = (
                    risk_distribution_df[
                        "Risk_Level"
                    ]
                    .value_counts()
                    .reindex(
                        risk_order(),
                        fill_value=0
                    )
                    .reset_index()
                )

                risk_counts.columns = [
                    "Risk Level",
                    "Customers"
                ]

                fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Customers",
                    text="Customers",
                    title="Predicted Risk Levels on Test Dataset"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # --------------------------------------------------------
        # TEST PREDICTION SUMMARY
        # --------------------------------------------------------

        st.subheader(
            "🔍 Test Prediction Summary"
        )

        predictions_df = load_csv(
            predictions_file
        )

        if not predictions_df.empty:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Test Customers",
                f"{len(predictions_df):,}"
            )

            probability_column = None

            for column in [
                "Predicted_Probability",
                "Churn_Probability",
                "Probability"
            ]:

                if column in predictions_df.columns:

                    probability_column = column
                    break

            if probability_column:

                probabilities = pd.to_numeric(
                    predictions_df[
                        probability_column
                    ],
                    errors="coerce"
                )

                c2.metric(
                    "Average Churn Probability",
                    format_percentage(
                        probabilities.mean()
                    )
                )

            churn_column = None

            for column in [
                "Predicted_Churn",
                "Prediction"
            ]:

                if column in predictions_df.columns:

                    churn_column = column
                    break

            if churn_column:

                churn_values = pd.to_numeric(
                    predictions_df[
                        churn_column
                    ],
                    errors="coerce"
                ).fillna(0)

                c3.metric(
                    "Predicted Churn Customers",
                    f"{int(churn_values.sum()):,}"
                )

        # --------------------------------------------------------
        # RESEARCH INTERPRETATION
        # --------------------------------------------------------

        st.subheader(
            "🔬 Research Interpretation"
        )

        st.markdown(
            """
            ### Model Evaluation

            The churn prediction model is evaluated using an independent
            test dataset that was not used during model training.

            Multiple evaluation metrics are reported because accuracy alone
            does not fully describe churn prediction performance.

            **Accuracy**

            Measures the overall proportion of correctly classified
            customers.

            **Precision**

            Measures how many customers predicted as churn actually belong
            to the churn class.

            **Recall**

            Measures how many actual churn customers were successfully
            identified.

            **F1 Score**

            Provides a balance between precision and recall.

            **ROC-AUC**

            Measures the model's ability to distinguish churn customers
            from non-churn customers across classification thresholds.

            The evaluation therefore considers both predictive performance
            and the practical requirement of identifying customers who may
            require retention intervention.
            """
        )

        # --------------------------------------------------------
        # CROSS VALIDATION NOTE
        # --------------------------------------------------------

        st.subheader(
            "📌 Evaluation Note"
        )

        st.warning(
            """
            During hyperparameter tuning, the model achieved a higher
            cross-validation ROC-AUC than its independent test ROC-AUC.

            The independent test result should be used when reporting
            final generalization performance because the test set was
            kept separate from model selection.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "InsightAI | Explainable AI Customer Intelligence"
)
