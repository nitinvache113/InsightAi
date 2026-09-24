from predict_customer import predict_customer


customer = {
    "customerID": "TEST-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0
}

result = predict_customer(customer)

print("=" * 60)
print("INSIGHTAI - CUSTOMER PREDICTION")
print("=" * 60)

print("Customer ID:", customer["customerID"])
print("Churn Probability:", round(result["churn_probability"] * 100, 2), "%")
print("Risk Level:", result["risk_level"])

print()
print("TOP CHURN DRIVERS")
print("-" * 40)

for driver in result["top_drivers"]:
    print(
        driver["feature"],
        "| SHAP:",
        round(driver["shap_value"], 4),
        "| Direction:",
        driver["direction"]
    )

print()
print("RETENTION RECOMMENDATIONS")
print("-" * 40)

for recommendation in result["recommendations"]:
    print("-", recommendation)

print("=" * 60)