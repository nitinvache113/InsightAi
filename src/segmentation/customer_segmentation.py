import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, os.path.abspath("."))


# ============================================================
# INSIGHTAI - CUSTOMER SEGMENTATION
# ============================================================

print("=" * 70)
print("INSIGHTAI - CUSTOMER SEGMENTATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load cleaned customer data
# ------------------------------------------------------------

DATA_PATH = "data/processed/telco_clean.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. Select segmentation features
# ------------------------------------------------------------

segmentation_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


print("\nSegmentation features:")

for feature in segmentation_features:
    print("-", feature)


# ------------------------------------------------------------
# 3. Prepare data
# ------------------------------------------------------------

X = df[segmentation_features].copy()


# Convert possible numeric columns
for column in segmentation_features:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# Remove missing values
X = X.dropna()


print("\nSegmentation data shape:", X.shape)


# ------------------------------------------------------------
# 4. Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")


# ------------------------------------------------------------
# 5. Find optimal number of clusters
# ------------------------------------------------------------

print("\nCalculating Elbow Method and Silhouette Scores...")

inertia = []
silhouette_scores = []

K_RANGE = range(2, 9)


for k in K_RANGE:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K={k} | "
        f"Inertia={kmeans.inertia_:.2f} | "
        f"Silhouette={score:.4f}"
    )


# ------------------------------------------------------------
# 6. Create output directory
# ------------------------------------------------------------

OUTPUT_DIR = "data/processed/segmentation_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# 7. Elbow plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_RANGE),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.title(
    "InsightAI - Elbow Method"
)

plt.xticks(list(K_RANGE))

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved: elbow_method.png")


# ------------------------------------------------------------
# 8. Silhouette score plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_RANGE),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.title(
    "InsightAI - Silhouette Score"
)

plt.xticks(list(K_RANGE))

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved: silhouette_scores.png")


# ------------------------------------------------------------
# 9. Find best silhouette score
# ------------------------------------------------------------

best_index = np.argmax(
    silhouette_scores
)

best_k = list(K_RANGE)[best_index]

best_score = silhouette_scores[best_index]


print("\n" + "=" * 70)

print(
    f"Best K according to Silhouette Score: {best_k}"
)

print(
    f"Best Silhouette Score: {best_score:.4f}"
)

print("=" * 70)


# ------------------------------------------------------------
# 10. Train final K-Means model
# ------------------------------------------------------------

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans_final.fit_predict(
    X_scaled
)


# ------------------------------------------------------------
# 11. Add clusters to dataset
# ------------------------------------------------------------

segmented_df = df.loc[
    X.index
].copy()

segmented_df["Cluster"] = cluster_labels


# ------------------------------------------------------------
# 12. Save segmented dataset
# ------------------------------------------------------------

segmented_df.to_csv(
    f"{OUTPUT_DIR}/customer_segments.csv",
    index=False
)


print("\nSaved: customer_segments.csv")


# ------------------------------------------------------------
# 13. Cluster profile
# ------------------------------------------------------------

cluster_profile = segmented_df.groupby(
    "Cluster"
)[segmentation_features].mean()


cluster_profile["Customer_Count"] = (
    segmented_df
    .groupby("Cluster")
    .size()
)


cluster_profile = cluster_profile[
    [
        "Customer_Count",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]


print("\n" + "=" * 70)

print("CUSTOMER SEGMENT PROFILES")

print("=" * 70)

print(
    cluster_profile.round(2)
)


# ------------------------------------------------------------
# 14. Save cluster profile
# ------------------------------------------------------------

cluster_profile.to_csv(
    f"{OUTPUT_DIR}/cluster_profile.csv"
)


print(
    "\nSaved: cluster_profile.csv"
)


# ------------------------------------------------------------
# 15. Save K-Means model and scaler
# ------------------------------------------------------------

joblib.dump(
    kmeans_final,
    f"{OUTPUT_DIR}/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}/segmentation_scaler.pkl"
)


print("\nK-Means model saved.")

print("Scaler saved.")


# ------------------------------------------------------------
# 16. Completion
# ------------------------------------------------------------

print("\n" + "=" * 70)

print("CUSTOMER SEGMENTATION COMPLETED")

print("=" * 70)