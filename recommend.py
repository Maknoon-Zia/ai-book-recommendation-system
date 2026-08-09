"""
recommend.py
-------------
Loads the pre-computed book embeddings (from train.py) and cleaned
metadata, and provides:

  - recommend_by_title():       single-book "more like this"
  - recommend_by_profile():     blends 2-5 books into one taste profile
                                  and recommends from that (this is closer
                                  to how real recommenders work -- nobody
                                  has just ONE favorite book)
  - explain_recommendation():   returns the shared genre/author words that
                                  justify why a given book was recommended,
                                  so the app can show "why" not just "what"
  - recommend_by_genre_and_rating(): metadata filter for the Browse tab
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = "artifacts"


def load_recommender_data():
    """Loads the cleaned dataframe and the trained embeddings from disk."""
    df = pd.read_pickle(os.path.join(ARTIFACTS_DIR, "books_clean.pkl"))
    embeddings = np.load(os.path.join(ARTIFACTS_DIR, "embeddings.npy"))
    return df, embeddings


def load_metrics():
    """Loads the evaluation metrics saved by train.py, if present."""
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path) as f:
        return json.load(f)


def _title_to_index(title: str, df: pd.DataFrame):
    matches = df.index[df["title"] == title].tolist()
    return matches[0] if matches else None


def recommend_by_title(title: str, df: pd.DataFrame, embeddings: np.ndarray, top_n: int = 8):
    """Given an exact book title, returns the top_n most similar books."""
    idx = _title_to_index(title, df)
    if idx is None:
        return pd.DataFrame()

    target_vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, embeddings)[0]

    result = df.copy()
    result["similarity"] = sims
    result = result.drop(index=idx).sort_values("similarity", ascending=False).head(top_n)
    return result


def recommend_by_profile(titles: list, df: pd.DataFrame, embeddings: np.ndarray, top_n: int = 10):
    """
    Blends 2-5 seed books into a single taste profile by averaging their
    embeddings, then recommends the books closest to that blended point.

    This is meaningfully different from single-book similarity: the
    average of "The Hobbit" + "Dune" embeddings lands somewhere between
    epic fantasy and sci-fi -- capturing a *combined* taste rather than
    just cloning one book.
    """
    indices = [i for i in (_title_to_index(t, df) for t in titles) if i is not None]
    if not indices:
        return pd.DataFrame()

    profile_vec = embeddings[indices].mean(axis=0, keepdims=True)
    sims = cosine_similarity(profile_vec, embeddings)[0]

    result = df.copy()
    result["similarity"] = sims
    result = result.drop(index=indices).sort_values("similarity", ascending=False).head(top_n)
    return result


def explain_recommendation(seed_title: str, rec_row: pd.Series, df: pd.DataFrame) -> dict:
    """
    Returns a small, human-readable explanation for why `rec_row` was
    recommended alongside `seed_title` -- shared genre tags and whether
    the author matches. This doesn't require re-running the network; it's
    a lightweight metadata comparison on top of the embedding-based match,
    which is enough to make the recommendation feel justified rather than
    a black box.
    """
    seed_matches = df.index[df["title"] == seed_title].tolist()
    if not seed_matches:
        return {"shared_genres": [], "same_author": False}

    seed_row = df.loc[seed_matches[0]]

    def genre_words(genre_str):
        return set(w.strip().lower() for w in str(genre_str).replace("/", ",").split(",") if w.strip())

    seed_genres = genre_words(seed_row.get("genre", ""))
    rec_genres = genre_words(rec_row.get("genre", ""))
    shared = sorted(seed_genres & rec_genres)

    same_author = str(seed_row.get("author", "")).strip().lower() == str(rec_row.get("author", "")).strip().lower()

    return {"shared_genres": shared, "same_author": same_author}


def recommend_by_genre_and_rating(df: pd.DataFrame, genre_query: str, min_rating: float, top_n: int = 12):
    """Simple metadata filter+sort for the Browse tab (no network needed)."""
    filtered = df[df["genre"].str.contains(genre_query, case=False, na=False)]
    filtered = filtered[filtered["rating"] >= min_rating]
    return filtered.sort_values("rating", ascending=False).head(top_n)
