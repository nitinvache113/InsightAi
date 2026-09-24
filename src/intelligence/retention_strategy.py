import os
import pandas as pd


# ============================================================
# INSIGHTAI - RETENTION STRATEGY ENGINE
# ============================================================

print("=" * 70)
print("INSIGHTAI - RETENTION STRATEGY ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD CUSTOMER INTELLIGENCE DATA
# ============================================================

INPUT_PATH = (
    "data/processed/intelligence_outputs/"
    "customer_intelligence.csv"
)

OUTPUT_DIR = (
    "data/processed/intelligence_outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df = pd.read_csv(
    INPUT_PATH
)


print("\nCustomer intelligence data loaded.")

print(
    "Customers:",
    len(df)
)


# ============================================================
# 2. RETENTION STRATEGY FUNCTION
# ============================================================

def generate_strategy(row):

    risk = row.get(
        "Risk_Level",
        ""
    )

    segment = row.get(
        "Customer_Segment",
        -1
    )

    contract = row.get(
        "Contract",
        ""
    )

    tenure = row.get(
        "tenure",
        0
    )

    monthly_charges = row.get(
        "MonthlyCharges",
        0
    )

    tech_support = row.get(
        "TechSupport",
        ""
    )

    online_security = row.get(
        "OnlineSecurity",
        ""
    )

    internet_service = row.get(
        "InternetService",
        ""
    )


    strategies = []


    # ========================================================
    # HIGH RISK
    # ========================================================

    if risk == "High":

        strategies.append(
            "Immediate retention outreach"
        )


    # ========================================================
    # MEDIUM RISK
    # ========================================================

    elif risk == "Medium":

        strategies.append(
            "Proactive retention communication"
        )


    # ========================================================
    # LOW RISK
    # ========================================================

    else:

        strategies.append(
            "Maintain regular engagement"
        )


    # ========================================================
    # SHORT TENURE
    # ========================================================

    if tenure <= 6:

        strategies.append(
            "Early customer onboarding support"
        )


    # ========================================================
    # CONTRACT
    # ========================================================

    if contract == "Month-to-month":

        strategies.append(
            "Long-term contract conversion offer"
        )


    # ========================================================
    # TECHNICAL SUPPORT
    # ========================================================

    if tech_support == "No":

        strategies.append(
            "Technical support promotion"
        )


    # ========================================================
    # ONLINE SECURITY
    # ========================================================

    if online_security == "No":

        strategies.append(
            "Online security package promotion"
        )


    # ========================================================
    # HIGH MONTHLY CHARGES
    # ========================================================

    if monthly_charges >= 80:

        strategies.append(
            "Personalized pricing review"
        )


    # ========================================================
    # FIBER OPTIC
    # ========================================================

    if internet_service == "Fiber optic":

        strategies.append(
            "Fiber service experience review"
        )


    # ========================================================
    # SEGMENT-SPECIFIC STRATEGY
    # ========================================================

    if segment == 0:

        strategies.append(
            "Loyalty and long-term customer engagement"
        )

    elif segment == 1:

        strategies.append(
            "Early intervention and retention campaign"
        )


    return "; ".join(
        strategies
    )


# ============================================================
# 3. APPLY STRATEGY
# ============================================================

df[
    "Retention_Strategy"
] = df.apply(
    generate_strategy,
    axis=1
)


print(
    "\nRetention strategies generated."
)


# ============================================================
# 4. PRIORITY LEVEL
# ============================================================

def assign_priority(row):

    probability = row[
        "Churn_Probability"
    ]

    if probability >= 0.80:

        return "Critical"

    elif probability >= 0.70:

        return "High"

    elif probability >= 0.40:

        return "Medium"

    else:

        return "Low"


df[
    "Retention_Priority"
] = df.apply(
    assign_priority,
    axis=1
)


# ============================================================
# 5. CAMPAIGN TYPE
# ============================================================

def assign_campaign(row):

    priority = row[
        "Retention_Priority"
    ]

    contract = row.get(
        "Contract",
        ""
    )

    tenure = row.get(
        "tenure",
        0
    )


    if priority == "Critical":

        return "Immediate Personal Outreach"

    elif priority == "High":

        return "Retention Offer Campaign"

    elif contract == "Month-to-month":

        return "Contract Upgrade Campaign"

    elif tenure <= 6:

        return "New Customer Engagement Campaign"

    else:

        return "Customer Loyalty Campaign"


df[
    "Recommended_Campaign"
] = df.apply(
    assign_campaign,
    axis=1
)


# ============================================================
# 6. SAVE UPDATED DATASET
# ============================================================

OUTPUT_PATH = (
    f"{OUTPUT_DIR}/retention_strategy.csv"
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_PATH
)


# ============================================================
# 7. PRIORITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RETENTION PRIORITY DISTRIBUTION")
print("=" * 70)


print(
    df[
        "Retention_Priority"
    ].value_counts()
)


# ============================================================
# 8. CAMPAIGN DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDED CAMPAIGNS")
print("=" * 70)


print(
    df[
        "Recommended_Campaign"
    ].value_counts()
)


# ============================================================
# 9. CRITICAL CUSTOMERS
# ============================================================

critical_customers = df[
    df[
        "Retention_Priority"
    ] == "Critical"
].copy()


critical_customers = (
    critical_customers
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
)


critical_path = (
    f"{OUTPUT_DIR}/critical_retention_customers.csv"
)


critical_customers.to_csv(
    critical_path,
    index=False
)


print(
    "\nCritical customers:",
    len(critical_customers)
)


print(
    "Saved:",
    critical_path
)


# ============================================================
# 10. RETENTION SUMMARY BY SEGMENT
# ============================================================

segment_summary = (
    df
    .groupby(
        "Customer_Segment"
    )
    .agg(

        Customer_Count=(
            "customerID",
            "count"
        ),

        Average_Churn_Probability=(
            "Churn_Probability",
            "mean"
        ),

        Critical_Customers=(
            "Retention_Priority",
            lambda x:
            (x == "Critical").sum()
        ),

        High_Priority_Customers=(
            "Retention_Priority",
            lambda x:
            (x == "High").sum()
        )

    )
)


segment_summary[
    "Average_Churn_Probability"
] = (
    segment_summary[
        "Average_Churn_Probability"
    ]
    .round(4)
)


segment_summary_path = (
    f"{OUTPUT_DIR}/retention_segment_summary.csv"
)


segment_summary.to_csv(
    segment_summary_path
)


print(
    "\nSaved:",
    segment_summary_path
)


# ============================================================
# 11. TOP 20 PRIORITY CUSTOMERS
# ============================================================

top_priority = (
    df
    .sort_values(
        "Churn_Probability",
        ascending=False
    )
    .head(20)
)


preview_columns = [
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "Churn_Probability",
    "Risk_Level",
    "Customer_Segment",
    "Retention_Priority",
    "Recommended_Campaign"
]


available_columns = [
    column
    for column in preview_columns
    if column in top_priority.columns
]


print("\n" + "=" * 70)
print("TOP 20 RETENTION PRIORITY CUSTOMERS")
print("=" * 70)


print(
    top_priority[
        available_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 12. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("RETENTION STRATEGY ENGINE COMPLETED")
print("=" * 70)