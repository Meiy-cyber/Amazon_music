import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "duration_ms",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "valence",
]


@st.cache_data
def load_and_clean_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    available_features = [c for c in AUDIO_FEATURES if c in df.columns]
    if len(available_features) < 2:
        st.error(
            "Expected audio feature columns not found in CSV. "
            f"Looking for: {AUDIO_FEATURES}"
        )
        st.stop()

    df = df.drop_duplicates().dropna(subset=available_features).reset_index(drop=True)
    return df, available_features


st.title("🎵 Amazon Music Clustering Explorer")
st.caption(
    "Unsupervised clustering on audio features using K-Means or DBSCAN, "
    "with Silhouette Score evaluation and PCA 2D visualization."
)

# Sidebar controls
st.sidebar.header("1. Data & Model Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload single_genre_artists.csv", type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload your `single_genre_artists.csv` file from the sidebar to start."
    )
    st.stop()

df, feature_cols = load_and_clean_data(uploaded_file)

selected_features = st.sidebar.multiselect(
    "Audio Features for Clustering",
    options=feature_cols,
    default=feature_cols,
)

if len(selected_features) < 2:
    st.warning("Select at least 2 audio features.")
    st.stop()

scaler_choice = st.sidebar.selectbox(
    "Feature Scaler", ["StandardScaler", "MinMaxScaler"]
)
algorithm = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN"])

if scaler_choice == "StandardScaler":
    scaler = StandardScaler()
else:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(df[selected_features])

# Fit chosen algorithm
if algorithm == "K-Means":
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=12, value=4)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    model = KMeans(n_clusters=k, n_init=10, random_state=int(random_state))
    labels = model.fit_predict(X_scaled)
else:
    eps = st.sidebar.slider("DBSCAN eps", min_value=0.2, max_value=5.0, value=1.5, step=0.1)
    min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=2, max_value=30, value=5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

df_clustered = df.copy()
df_clustered["cluster"] = labels.astype(str)

# Metrics row (valid only when >= 2 clusters exist; for DBSCAN exclude noise if only 1 real cluster)
unique_clusters = set(labels)
n_clusters = len(unique_clusters - {-1}) if algorithm == "DBSCAN" else len(unique_clusters)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracks Analyzed", f"{len(df_clustered):,}")
col2.metric("Features Used", len(selected_features))
col3.metric("Clusters Found", n_clusters)

if n_clusters >= 2 and len(set(labels)) >= 2:
    # Subsample if dataset is huge so silhouette_score stays fast
    sample_size = min(len(X_scaled), 10000)
    sil_score = silhouette_score(
        X_scaled, labels, sample_size=sample_size, random_state=42
    )
    db_score = davies_bouldin_score(X_scaled, labels)
    col4.metric("Silhouette Score", f"{sil_score:.3f}")
    st.caption(f"Davies-Bouldin Index (lower is better): **{db_score:.3f}**")
else:
    col4.metric("Silhouette Score", "N/A")
    st.warning(
        "Need at least 2 clusters to compute Silhouette Score. "
        "Try adjusting K or DBSCAN `eps` / `min_samples`."
    )

# Optional K-sweep chart for K-Means
if algorithm == "K-Means":
    with st.expander("K-Means K Selection (Silhouette Score across K = 2..8)"):
        k_scores = []
        sample_n = min(len(X_scaled), 8000)
        for cand_k in range(2, 9):
            km = KMeans(n_clusters=cand_k, n_init=10, random_state=42)
            lbl = km.fit_predict(X_scaled)
            sc = silhouette_score(
                X_scaled, lbl, sample_size=sample_n, random_state=42
            )
            k_scores.append({"K": cand_k, "Silhouette Score": sc, "Inertia": km.inertia_})
        k_df = pd.DataFrame(k_scores)
        fig_k = px.line(
            k_df,
            x="K",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score vs. Number of Clusters (K)",
        )
        st.plotly_chart(fig_k, use_container_width=True)

# 2D PCA Visualization (for visual exploration only)
st.subheader("2D Cluster Projection (PCA — Visual Only)")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
plot_df = df_clustered.copy()
plot_df["PC1"] = coords[:, 0]
plot_df["PC2"] = coords[:, 1]

hover_cols = [c for c in ["track_name", "artist_name", "danceability", "energy", "acousticness"] if c in plot_df.columns]
fig_pca = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="cluster",
    hover_data=hover_cols,
    title=f"PCA Scatter ({pca.explained_variance_ratio_.sum() * 100:.1f}% variance explained)",
    opacity=0.75,
)
st.plotly_chart(fig_pca, use_container_width=True)

# Cluster Profiles & Heatmap
st.subheader("Cluster Audio Profiles (Mean Values)")
profile_df = df_clustered.groupby("cluster")[selected_features].mean()
st.dataframe(profile_df.round(3), use_container_width=True)

# Normalized heatmap for easy cross-feature comparison
norm_profile = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min() + 1e-9)
fig_hm = px.imshow(
    norm_profile,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Blues",
    title="Normalized Mean Feature Heatmap Across Clusters",
)
st.plotly_chart(fig_hm, use_container_width=True)

# Sample tracks & CSV export
st.subheader("Clustered Tracks & Export")
cluster_filter = st.selectbox(
    "Filter by Cluster", ["All"] + sorted(df_clustered["cluster"].unique().tolist())
)
view_df = (
    df_clustered
    if cluster_filter == "All"
    else df_clustered[df_clustered["cluster"] == cluster_filter]
)
st.dataframe(view_df.head(100), use_container_width=True)

csv_bytes = df_clustered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Labeled CSV",
    data=csv_bytes,
    file_name="amazon_music_clustered.csv",
    mime="text/csv",
)