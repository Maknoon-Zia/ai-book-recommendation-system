"""
preprocess.py
--------------
Turns the raw books.csv into a numeric feature matrix that a neural network
can consume, and saves everything needed for inference later (vectorizers,
scalers, the cleaned dataframe).

Why these features?
- Title/Author/Genre/Description -> combined into one text blob -> TF-IDF
  -> compressed with TruncatedSVD (this is basically how we turn "content"
  into numbers a feedforward network can read).
- Rating / number of ratings / pages -> numeric columns, scaled to [0, 1].

The dataset's exact column names can vary slightly between versions, so
this script auto-detects the right column for each field using a list of
likely aliases.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

RAW_CSV = "data/books.csv"
ARTIFACTS_DIR = "artifacts"

# How many dimensions to compress the TF-IDF text matrix down to.
# Keeps the network's input size manageable (10k books x huge sparse TF-IDF
# would be slow / memory heavy, so we reduce it first).
SVD_COMPONENTS = 100

# Candidate column names we might see in this dataset (Kaggle CSVs vary).
COLUMN_ALIASES = {
    "title": ["title", "book", "book_name", "name", "book title"],
    "author": ["author", "authors", "book_author", "writer"],
    "genre": ["genre", "genres", "category", "categories", "tags"],
    "description": ["description", "desc", "summary", "book_description", "blurb"],
    "rating": ["rating", "avg_rating", "average_rating", "book_rating", "rating_score"],
    "num_ratings": ["num_ratings", "ratings_count", "total_ratings", "number_of_ratings"],
    "pages": ["pages", "num_pages", "page_count"],
    "image": ["image", "image_url", "cover", "cover_url", "img", "url_img"],
}


def find_column(df: pd.DataFrame, aliases: list) -> str | None:
    """Case-insensitively match a real column name against a list of aliases."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    # fallback: partial match (e.g. "Book Genres" contains "genre")
    for col_lower, col_real in lower_map.items():
        for alias in aliases:
            if alias in col_lower:
                return col_real
    return None


def clean_text(text) -> str:
    """Lowercase, strip HTML/punctuation noise from free text fields."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_and_standardize() -> pd.DataFrame:
    """Load the raw CSV and rename whichever columns it has into a standard schema."""
    df = pd.read_csv(RAW_CSV)

    resolved = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        col = find_column(df, aliases)
        resolved[standard_name] = col

    print("Detected column mapping:")
    for k, v in resolved.items():
        print(f"  {k:>12} -> {v}")

    # Build a clean, standardized dataframe. Missing optional columns become
    # empty/NaN so the rest of the pipeline still works.
    clean_df = pd.DataFrame()
    clean_df["title"] = df[resolved["title"]] if resolved["title"] else "Unknown Title"
    clean_df["author"] = df[resolved["author"]] if resolved["author"] else "Unknown Author"
    clean_df["genre"] = df[resolved["genre"]] if resolved["genre"] else ""
    clean_df["description"] = df[resolved["description"]] if resolved["description"] else ""
    def to_numeric_safe(series):
        """
        Converts a column to numeric, first stripping thousands-separator
        commas (e.g. this dataset stores Num_Ratings as "5,691,311", a
        string -- pd.to_numeric would silently turn every value into NaN
        without this cleanup).

        Checks `dtype == object` alone is NOT enough: pandas >= 2.x/3.x
        can represent text columns as a native "str" dtype instead of
        "object" (depending on pandas version and settings), which would
        make that check silently skip the comma-cleanup and corrupt this
        entire column to NaN -> 0. We instead just always run the
        string-clean step through .astype(str), which is safe whether
        the column started numeric or text.
        """
        series = series.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(series, errors="coerce")

    clean_df["rating"] = to_numeric_safe(df[resolved["rating"]]) if resolved["rating"] else np.nan
    clean_df["num_ratings"] = to_numeric_safe(df[resolved["num_ratings"]]) if resolved["num_ratings"] else np.nan
    clean_df["pages"] = to_numeric_safe(df[resolved["pages"]]) if resolved["pages"] else np.nan
    clean_df["image"] = df[resolved["image"]] if resolved["image"] else ""

    # Drop rows with no title/author at all (junk rows), fill the rest.
    clean_df = clean_df.dropna(subset=["title"]).reset_index(drop=True)
    clean_df["author"] = clean_df["author"].fillna("Unknown Author")
    clean_df["genre"] = clean_df["genre"].fillna("")
    clean_df["description"] = clean_df["description"].fillna("")
    # median() on an all-NaN column (e.g. this dataset has no "pages"
    # column at all) returns NaN itself, which would leave the column
    # empty and break the scaler downstream -- fall back to a sane
    # default (0) whenever the column is entirely missing.
    rating_median = clean_df["rating"].median()
    clean_df["rating"] = clean_df["rating"].fillna(rating_median if pd.notna(rating_median) else 0)
    clean_df["num_ratings"] = clean_df["num_ratings"].fillna(0)
    pages_median = clean_df["pages"].median()
    clean_df["pages"] = clean_df["pages"].fillna(pages_median if pd.notna(pages_median) else 0)
    clean_df["image"] = clean_df["image"].fillna("")

    # De-duplicate on title+author
    clean_df = clean_df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)

    return clean_df


def build_features(df: pd.DataFrame):
    """
    Builds the numeric input matrix X for the neural network:
      [ TF-IDF(text) reduced by SVD  |  scaled numeric columns ]
    """
    # 1) Combine text fields (genre repeated 2x so it weighs a bit more,
    #    since genre similarity matters a lot for "you might also like").
    text_blob = (
        df["genre"].apply(clean_text) + " " + df["genre"].apply(clean_text) + " " +
        df["author"].apply(clean_text) + " " +
        df["description"].apply(clean_text)
    )

    tfidf = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2)
    tfidf_matrix = tfidf.fit_transform(text_blob)

    n_components = min(SVD_COMPONENTS, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_features = svd.fit_transform(tfidf_matrix)

    # 2) Numeric features, scaled to [0, 1] so they're on the same footing
    #    as the (already fairly small-magnitude) SVD text features.
    numeric_cols = df[["rating", "num_ratings", "pages"]].values
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(numeric_cols)

    # 3) Concatenate into the final input matrix for the network
    X = np.hstack([text_features, numeric_features]).astype("float32")

    return X, tfidf, svd, scaler


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading and standardizing dataset...")
    df = load_and_standardize()
    print(f"Cleaned dataset: {df.shape[0]} books")

    print("Building TF-IDF + SVD + numeric feature matrix...")
    X, tfidf, svd, scaler = build_features(df)
    print(f"Final feature matrix shape: {X.shape}")

    # Save everything needed later for training and inference.
    df.to_pickle(os.path.join(ARTIFACTS_DIR, "books_clean.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "features.npy"), X)
    joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.joblib"))
    joblib.dump(svd, os.path.join(ARTIFACTS_DIR, "svd.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w") as f:
        json.dump({"n_books": len(df), "input_dim": int(X.shape[1])}, f)

    print(f"\nSaved artifacts to '{ARTIFACTS_DIR}/':")
    print("  books_clean.pkl, features.npy, tfidf.joblib, svd.joblib, scaler.joblib, meta.json")


if __name__ == "__main__":
    main()
