import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Page Configuration
st.set_page_config(
    page_title="Wine Clustering",
    layout="wide"
)

st.title("Wine Classification using Hierarchical Clustering")

from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "wine_clustering.csv")

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature Selection
X = df.copy()

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dendrogram
st.subheader("Hierarchical Clustering Dendrogram")

linked = linkage(X_scaled, method='ward')

fig1, ax1 = plt.subplots(figsize=(12, 6))

dendrogram(
    linked,
    truncate_mode='level',
    p=5,
    ax=ax1
)

ax1.set_title("Dendrogram")
ax1.set_xlabel("Wine Samples")
ax1.set_ylabel("Euclidean Distance")

st.pyplot(fig1)

# Cluster Selection
st.subheader("Choose Number of Clusters")

n_clusters = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=3
)

# Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# Clustered Data
st.subheader("Clustered Dataset")
st.dataframe(df.head())

# PCA Visualization
pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

fig2, ax2 = plt.subplots(figsize=(8, 6))

scatter = ax2.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=clusters,
    cmap='rainbow',
    s=60
)

ax2.set_title("Wine Clusters using PCA")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")

plt.colorbar(scatter)

st.pyplot(fig2)

# Cluster Distribution
st.subheader("Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_counts)

# Cluster Summary
st.subheader("Cluster Summary")

summary = df.groupby("Cluster").mean(numeric_only=True)

st.dataframe(summary)

# Download Results
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="wine_clustered.csv",
    mime="text/csv"
)
