import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Hierarchical Clustering", layout="centered")
st.title("🌳 Hierarchical Clustering (User-defined Clusters)")

# Load dataset
df = pd.read_csv("dataset.csv")
numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

# Load scaler & features
bundle = joblib.load("model.pkl")
scaler = bundle["scaler"]
features = bundle["features"]

# Scale data
scaled_data = scaler.transform(numeric_df)

# -----------------------------
# User input: number of clusters
# -----------------------------
st.sidebar.header("⚙️ Clustering Settings")
n_clusters = st.sidebar.slider(
    "Select number of clusters",
    min_value=2,
    max_value=10,
    value=3
)

# -----------------------------
# Dendrogram
# -----------------------------
st.subheader("📌 Dendrogram")

linked = linkage(scaled_data, method="ward")

plt.figure(figsize=(8, 4))
dendrogram(linked, truncate_mode="level", p=5)
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.title("Hierarchical Dendrogram")
st.pyplot(plt)

# -----------------------------
# Hierarchical clustering
# -----------------------------
model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

labels = model.fit_predict(scaled_data)

# -----------------------------
# Cluster visualization
# -----------------------------
st.subheader("📊 Cluster Visualization (First 2 Features)")

plt.figure(figsize=(6, 4))
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=labels,
    cmap="viridis"
)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title(f"Hierarchical Clustering (k = {n_clusters})")
st.pyplot(plt)
