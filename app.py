import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛒",
    layout="wide"
)

# ── Load all artifacts ────────────────────────────────────
@st.cache_resource
def load_models():
    P1 = "/kaggle/input/notebooks/vishesh806/amazon-hybrid-recsys-phase1-eda/"
    P2 = "/kaggle/input/notebooks/vishesh806/amazon-hybrid-recsys-phase2-svd/"
    P3 = "/kaggle/input/notebooks/vishesh806/amazon-hybrid-recsys-phase3-content-based/"

    train            = pd.read_csv(P1 + "train.csv")
    product_profiles = pd.read_csv(P3 + "product_profiles.csv")
    tfidf_matrix     = sp.load_npz(P3 + "tfidf_matrix.npz")

    with open(P2 + "svd_model.pkl", "rb") as f:
        svd = pickle.load(f)

    product_ids = product_profiles["ProductId"].tolist()
    product_idx = {pid: idx for idx, pid in enumerate(product_ids)}

    return train, product_profiles, tfidf_matrix, svd, product_ids, product_idx

train, product_profiles, tfidf_matrix, svd, product_ids, product_idx = load_models()

# ── Recommendation functions ──────────────────────────────
def get_cb_recs(product_id, n=10):
    if product_id not in product_idx:
        return []
    idx = product_idx[product_id]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    scores[idx] = -1
    top = np.argsort(scores)[::-1][:n]
    return [(product_ids[i], round(scores[i], 4)) for i in top]

def get_hybrid_recs(user_id, seed_product, n=10, alpha=0.6):
    if seed_product not in product_idx:
        return []
    idx = product_idx[seed_product]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    scores[idx] = -1
    top = np.argsort(scores)[::-1][:n*3]
    candidates = [(product_ids[i], scores[i]) for i in top]
    hybrid = []
    for pid, cb_score in candidates:
        svd_pred = svd.predict(user_id, pid).est
        svd_norm = (svd_pred - 1) / 4
        h_score  = alpha * svd_norm + (1 - alpha) * cb_score
        hybrid.append((pid, round(h_score, 4), round(svd_norm, 4), round(cb_score, 4)))
    hybrid.sort(key=lambda x: x[1], reverse=True)
    return hybrid[:n]

# ── UI ────────────────────────────────────────────────────
st.title("🛒 Amazon Product Recommendation System")
st.markdown("**Hybrid model: SVD (Collaborative Filtering) + TF-IDF (Content-Based)**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
mode  = st.sidebar.radio("Recommendation Mode", ["Content-Based Only", "Hybrid (SVD + TF-IDF)"])
alpha = st.sidebar.slider("Alpha (SVD weight)", 0.0, 1.0, 0.6, 0.1,
                           help="Higher = more SVD influence. Optimal = 0.6")
top_k = st.sidebar.slider("Number of recommendations", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""
| Model | Precision@10 |
|-------|-------------|
| TF-IDF | 0.0096 |
| SVD | 0.0180 |
| **Hybrid** | **0.0203** |
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Dataset Stats")
st.sidebar.markdown("""
- Reviews: 2,19,515
- Users: 23,261
- Products: 17,538
- Sparsity: 99.94%
""")

# Main inputs
col1, col2 = st.columns(2)

with col1:
    all_users = train["UserId"].unique().tolist()
    user_id   = st.selectbox("Select User ID", all_users[:200])

with col2:
    user_products = train[train["UserId"] == user_id]["ProductId"].tolist()
    if user_products:
        seed_product = st.selectbox("Seed Product (last interacted)", user_products)
    else:
        seed_product = st.selectbox("Seed Product", product_ids[:50])

# User interaction count indicator
n_interactions = len(user_products)
if n_interactions >= 5:
    st.info(f"👤 User has **{n_interactions}** interactions in training data. ✅ Sufficient for SVD")
else:
    st.warning(f"👤 User has **{n_interactions}** interactions. ⚠️ Sparse user — Content-Based will dominate")

# Generate recommendations
if st.button("🔍 Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):

        if mode == "Content-Based Only":
            recs = get_cb_recs(seed_product, n=top_k)
            if recs:
                df_recs = pd.DataFrame(recs, columns=["ProductId", "Similarity Score"])
                df_recs.index += 1
                st.success(f"Top {top_k} content-based recommendations for seed product: `{seed_product}`")
                st.dataframe(df_recs, use_container_width=True)
            else:
                st.error("Product not found in index.")

        else:
            recs = get_hybrid_recs(user_id, seed_product, n=top_k, alpha=alpha)
            if recs:
                df_recs = pd.DataFrame(recs, columns=["ProductId", "Hybrid Score", "SVD Score", "CB Score"])
                df_recs.index += 1
                st.success(f"Top {top_k} hybrid recommendations (alpha={alpha})")
                st.dataframe(df_recs, use_container_width=True)

                st.markdown("### 📊 Score Breakdown (SVD vs Content-Based)")
                chart_data = df_recs[["SVD Score", "CB Score"]].head(10)
                chart_data.index = df_recs["ProductId"].head(10)
                st.bar_chart(chart_data)
            else:
                st.error("Product not found in index.")

st.markdown("---")
st.markdown("Built by Vishesh | IIT Kharagpur | Amazon Product Reviews Dataset")
