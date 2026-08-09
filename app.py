"""
app.py
-------
Streamlit frontend for the AI Book Recommendation System.

Theme concept: "Reading Lamp" -- a deep ink-navy background (like a study
at night), a warm brass/gold accent (the lamp light), and a burgundy
secondary accent (leather book spines). Serif display type for titles,
clean sans body type for everything else.

Tabs:
  1. Recommend    -- single-book "more like this" OR multi-book taste
                      profile blending, with a "why this book" explanation
  2. Browse        -- filter the library by genre/rating
  3. Insights       -- evaluation metrics, training curves, and a 2D
                      embedding map proving the model learned something
                      real (not just asserting it)

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from recommend import (
    load_recommender_data,
    load_metrics,
    recommend_by_title,
    recommend_by_profile,
    explain_recommendation,
    recommend_by_genre_and_rating,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Inkwell — AI Book Recommender",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Theme: custom CSS (dark, "reading lamp" palette)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #10121A;
    --bg-elevated: #191C28;
    --card: #1D2030;
    --card-border: #2C3046;
    --gold: #E8B75D;
    --gold-soft: rgba(232, 183, 93, 0.15);
    --burgundy: #8C3A4A;
    --text: #EDE9DE;
    --text-muted: #9691A8;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #171A26 0%, #10121A 55%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: var(--bg-elevated);
    border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
    letter-spacing: 0.3px;
}

.hero {
    padding: 2.2rem 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(232,183,93,0.10), rgba(140,58,74,0.10));
    border: 1px solid var(--card-border);
    margin-bottom: 1.8rem;
}
.hero h1 { font-size: 2.6rem; margin: 0 0 0.3rem 0; }
.hero .accent { color: var(--gold); }
.hero p { color: var(--text-muted); font-size: 1.05rem; margin: 0; }

.book-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-left: 4px solid var(--gold);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.book-card:hover { transform: translateY(-3px); border-left-color: var(--burgundy); }
.book-title { font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.15rem; }
.book-author { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 0.55rem; font-style: italic; }
.badge-row { margin-bottom: 0.5rem; }
.badge {
    display: inline-block; background: var(--gold-soft); color: var(--gold);
    border: 1px solid rgba(232,183,93,0.35); border-radius: 999px;
    padding: 0.15rem 0.65rem; font-size: 0.75rem; margin-right: 0.35rem; margin-bottom: 0.3rem;
}
.badge-rating { background: rgba(140,58,74,0.18); color: #E8949F; border: 1px solid rgba(140,58,74,0.4); }
.book-desc { color: var(--text-muted); font-size: 0.85rem; line-height: 1.4; max-height: 4.2em; overflow: hidden; }
.sim-bar-track { background: #2C3046; border-radius: 999px; height: 6px; margin-top: 0.6rem; overflow: hidden; }
.sim-bar-fill { background: linear-gradient(90deg, var(--gold), var(--burgundy)); height: 100%; }
.why-box {
    margin-top: 0.6rem; padding: 0.5rem 0.7rem; background: rgba(232,183,93,0.06);
    border: 1px dashed rgba(232,183,93,0.3); border-radius: 8px; font-size: 0.78rem; color: var(--text-muted);
}
.metric-card {
    background: var(--card); border: 1px solid var(--card-border); border-radius: 12px;
    padding: 1.1rem 1.3rem; text-align: center;
}
.metric-value { font-family: 'Playfair Display', serif; font-size: 2rem; color: var(--gold); font-weight: 700; }
.metric-label { color: var(--text-muted); font-size: 0.82rem; margin-top: 0.2rem; }

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stSlider {
    background: var(--card) !important; color: var(--text) !important; border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, var(--gold), #C9954A); color: #1a1204; font-weight: 600;
    border: none; border-radius: 8px; padding: 0.5rem 1.4rem;
}
.stButton button:hover { background: linear-gradient(135deg, #f0c777, var(--gold)); color: #1a1204; }

button[data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'Inter'; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data loading (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def get_data():
    return load_recommender_data()

@st.cache_resource
def get_metrics():
    return load_metrics()

@st.cache_resource
def get_2d_projection(_embeddings):
    """PCA-projects the high-dim embeddings down to 2D for visualization."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_embeddings)
    return coords, pca.explained_variance_ratio_

try:
    df, embeddings = get_data()
except FileNotFoundError:
    st.error(
        "No trained artifacts found. Please run the pipeline first:\n\n"
        "1. `python download_data.py`\n"
        "2. `python preprocess.py`\n"
        "3. `python train.py`\n\n"
        "Then relaunch `streamlit run app.py`."
    )
    st.stop()

metrics = get_metrics()


# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📖 Inkwell <span class="accent">— AI Book Recommender</span></h1>
    <p>A hybrid feedforward neural network learns book embeddings from genre, author, and description —
    then blends your favorites into a taste profile and explains every match.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Card rendering helper
# ----------------------------------------------------------------------
def render_book_card(row, show_similarity=False, explanation=None):
    genre_text = str(row.get("genre", ""))[:60]
    rating = row.get("rating", 0)
    desc = str(row.get("description", ""))[:180]
    sim_html = ""
    if show_similarity and "similarity" in row:
        pct = max(0, min(100, round(float(row["similarity"]) * 100)))
        sim_html = f"""
        <div class="sim-bar-track"><div class="sim-bar-fill" style="width:{pct}%;"></div></div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">{pct}% match</div>
        """

    why_html = ""
    if explanation is not None:
        bits = []
        if explanation.get("shared_genres"):
            bits.append("shares genres: <b>" + ", ".join(explanation["shared_genres"][:3]) + "</b>")
        if explanation.get("same_author"):
            bits.append("same author")
        if bits:
            why_html = f'<div class="why-box">💡 Why this: {" · ".join(bits)}</div>'

    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">{row['title']}</div>
        <div class="book-author">by {row['author']}</div>
        <div class="badge-row">
            <span class="badge badge-rating">★ {rating:.2f}</span>
            <span class="badge">{genre_text if genre_text else 'Uncategorized'}</span>
        </div>
        <div class="book-desc">{desc}{'...' if len(str(row.get('description', ''))) > 180 else ''}</div>
        {sim_html}
        {why_html}
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕯️ How it works")
    st.write(
        "A hybrid feedforward network shares one encoder between two "
        "tasks: reconstructing a book's features, and predicting its "
        "average rating. That second task forces the embedding to "
        "capture something real about reader reception, not just "
        "compress numbers."
    )
    st.markdown("---")
    st.markdown(f"**Books in library:** {len(df):,}")
    st.markdown(f"**Embedding size:** {embeddings.shape[1]} dimensions")
    if metrics:
        st.markdown(f"**Test MAE (rating):** {metrics['test_mae']:.3f}")
        st.markdown(f"**Precision@{metrics['precision_at_k']['k']}:** {metrics['precision_at_k']['precision_at_k']:.2f}")
    st.markdown("---")
    st.caption("Built with TensorFlow/Keras + Streamlit")


# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮  Recommend", "🔎  Browse by Genre", "📊  Model Insights"])

# ======================== TAB 1: RECOMMEND ========================
with tab1:
    mode = st.radio(
        "Recommendation mode",
        ["Single book", "Taste profile (2-5 books)"],
        horizontal=True,
    )
    title_list = sorted(df["title"].unique().tolist())

    if mode == "Single book":
        st.markdown("#### Pick a book you love")
        selected_title = st.selectbox("Search for a title", title_list, index=0)
        n_recs = st.slider("Number of recommendations", 3, 15, 8)

        if st.button("✨ Recommend similar books"):
            with st.spinner("Consulting the neural network..."):
                recs = recommend_by_title(selected_title, df, embeddings, top_n=n_recs)

            if recs.empty:
                st.warning("Couldn't find that title in the dataset.")
            else:
                st.markdown(f"#### Because you liked *{selected_title}*")
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    explanation = explain_recommendation(selected_title, row, df)
                    with cols[i % 2]:
                        render_book_card(row, show_similarity=True, explanation=explanation)

    else:
        st.markdown("#### Build your taste profile")
        st.caption("Pick 2-5 books you love — the network blends their embeddings into one combined taste vector.")
        selected_titles = st.multiselect(
            "Your favorite books", title_list, max_selections=5,
            placeholder="Start typing a title...",
        )
        n_recs = st.slider("Number of recommendations", 3, 15, 8, key="profile_n")

        if st.button("✨ Recommend from my profile"):
            if len(selected_titles) < 2:
                st.warning("Pick at least 2 books to build a profile.")
            else:
                with st.spinner("Blending your taste profile..."):
                    recs = recommend_by_profile(selected_titles, df, embeddings, top_n=n_recs)

                if recs.empty:
                    st.warning("Couldn't build a profile from those titles.")
                else:
                    st.markdown(f"#### Based on your {len(selected_titles)}-book profile")
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(recs.iterrows()):
                        # Explain relative to the closest seed book by genre overlap
                        explanation = explain_recommendation(selected_titles[0], row, df)
                        with cols[i % 2]:
                            render_book_card(row, show_similarity=True, explanation=explanation)

# ======================== TAB 2: BROWSE ========================
with tab2:
    st.markdown("#### Browse the library")
    c1, c2 = st.columns([2, 1])
    with c1:
        genre_query = st.text_input("Genre contains...", value="Fantasy")
    with c2:
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 4.0, 0.1)

    results = recommend_by_genre_and_rating(df, genre_query, min_rating, top_n=12)
    st.markdown(f"**{len(results)} books found**")
    cols = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 2]:
            render_book_card(row, show_similarity=False)

# ======================== TAB 3: MODEL INSIGHTS ========================
with tab3:
    st.markdown("#### Does this model actually work?")
    st.caption("Real evaluation numbers from a held-out test set the model never trained on — not just an assertion.")

    if metrics is None:
        st.info("No metrics found yet. Run `python train.py` to generate evaluation metrics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics['test_mae']:.3f}</div>
                <div class="metric-label">Test MAE<br>(rating prediction)</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics['test_rmse']:.3f}</div>
                <div class="metric-label">Test RMSE<br>(rating prediction)</div></div>""", unsafe_allow_html=True)
        with c3:
            pk = metrics['precision_at_k']
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{pk['precision_at_k']:.2f}</div>
                <div class="metric-label">Precision@{pk['k']}<br>(genre match rate)</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{pk['random_baseline']:.2f}</div>
                <div class="metric-label">Random baseline<br>Precision@{pk['k']}</div></div>""", unsafe_allow_html=True)

        lift = pk['precision_at_k'] / pk['random_baseline'] if pk['random_baseline'] > 0 else float('inf')
        st.markdown(
            f"<p style='color:var(--text-muted); margin-top:1rem;'>The model's recommendations are "
            f"<b style='color:var(--gold);'>{lift:.1f}× more likely</b> to share a genre with the seed book "
            f"than a random recommendation — evaluated on {pk['n_evaluated']} held-out books.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Training curves")
        hist = metrics.get("loss_history", {})
        if hist.get("train_loss"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist["train_loss"], name="Train loss", line=dict(color="#E8B75D")))
            fig.add_trace(go.Scatter(y=hist["val_loss"], name="Validation loss", line=dict(color="#8C3A4A")))
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EDE9DE", family="Inter"),
                xaxis_title="Epoch", yaxis_title="Combined loss (reconstruction + rating)",
                legend=dict(orientation="h", y=1.1), height=380, margin=dict(t=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Embedding space (2D projection)")
        st.caption("Every book's 32-dimensional embedding projected to 2D via PCA. Books that cluster together were judged similar by the network — color by genre to see if genre clusters visually separate.")

        coords, var_ratio = get_2d_projection(embeddings)
        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["primary_genre"] = plot_df["genre"].fillna("Unknown").str.split(",").str[0].str.strip()

        top_genres = plot_df["primary_genre"].value_counts().head(10).index.tolist()
        plot_df["genre_group"] = plot_df["primary_genre"].where(plot_df["primary_genre"].isin(top_genres), "Other")

        fig2 = go.Figure()
        palette = ["#E8B75D", "#8C3A4A", "#6C8EBF", "#82B366", "#D6B656",
                   "#B85450", "#9673A6", "#D79B00", "#6699CC", "#999999", "#4A4E69"]
        for i, g in enumerate(plot_df["genre_group"].unique()):
            sub = plot_df[plot_df["genre_group"] == g]
            fig2.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers", name=g,
                marker=dict(size=6, color=palette[i % len(palette)], opacity=0.75),
                text=sub["title"], hovertemplate="<b>%{text}</b><extra>" + g + "</extra>",
            ))
        fig2.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EDE9DE", family="Inter"),
            xaxis_title=f"PC1 ({var_ratio[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({var_ratio[1]*100:.1f}% variance)",
            legend=dict(orientation="v"), height=550, margin=dict(t=30),
        )
        st.plotly_chart(fig2, use_container_width=True)
