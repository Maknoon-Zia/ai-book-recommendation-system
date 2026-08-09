"""
train.py
---------
Trains the HYBRID feedforward network (shared encoder + reconstruction
head + rating-prediction head) on the book feature matrix produced by
preprocess.py.

After training, this script does three things:
  1. Generates an embedding for every book (via the trained encoder) --
     these are what the app uses for similarity search.
  2. Evaluates the model with real metrics: rating-prediction MAE/RMSE on
     a held-out test set, and Precision@K for the recommender itself
     (does "similar books" actually share genres with the seed book?).
  3. Saves everything -- model, embeddings, metrics -- to artifacts/ so
     the Streamlit app can display proof the system works, not just
     assert it.

Run:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from model import build_hybrid_model

ARTIFACTS_DIR = "artifacts"
EMBEDDING_DIM = 32
EPOCHS = 60
BATCH_SIZE = 64
PRECISION_AT_K = 10          # how many recommendations to check per book
N_EVAL_BOOKS = 300           # how many books to sample when computing Precision@K (keeps eval fast)


def compute_precision_at_k(df, embeddings, k=10, n_sample=300, seed=42):
    """
    A recommender-specific metric, not just a generic ML metric.

    For a sample of books, we find their top-K nearest neighbors in
    embedding space, then check what fraction of those neighbors share at
    least one genre word with the seed book. This is a reasonable proxy
    for "are these actually good recommendations?" given we don't have
    real user click/purchase data to validate against.

    Returns the mean precision@k across the sampled books, plus a random
    baseline (precision@k if we recommended random books instead) so the
    number has real context -- "0.62" means nothing on its own, but
    "0.62 vs a 0.11 random baseline" proves the model is doing real work.
    """
    rng = np.random.default_rng(seed)
    n_books = len(df)
    sample_idx = rng.choice(n_books, size=min(n_sample, n_books), replace=False)

    genre_sets = df["genre"].fillna("").str.lower().apply(
        lambda g: set(w.strip() for w in g.replace("/", ",").split(",") if w.strip())
    ).values

    sims = cosine_similarity(embeddings)
    precisions, random_precisions = [], []

    for idx in sample_idx:
        seed_genres = genre_sets[idx]
        if not seed_genres:
            continue

        # Model-based top-K neighbors (excluding the book itself)
        order = np.argsort(-sims[idx])
        order = order[order != idx][:k]
        hits = sum(1 for j in order if genre_sets[j] & seed_genres)
        precisions.append(hits / k)

        # Random baseline: k random other books
        rand_idx = rng.choice([i for i in range(n_books) if i != idx], size=k, replace=False)
        rand_hits = sum(1 for j in rand_idx if genre_sets[j] & seed_genres)
        random_precisions.append(rand_hits / k)

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "random_baseline": float(np.mean(random_precisions)) if random_precisions else 0.0,
        "k": k,
        "n_evaluated": len(precisions),
    }


def main():
    # ---- Load feature matrix + metadata ----
    X = np.load(os.path.join(ARTIFACTS_DIR, "features.npy"))
    with open(os.path.join(ARTIFACTS_DIR, "meta.json")) as f:
        meta = json.load(f)
    df = pd.read_pickle(os.path.join(ARTIFACTS_DIR, "books_clean.pkl"))

    input_dim = meta["input_dim"]
    y_rating = df["rating"].values.astype("float32")
    print(f"Loaded feature matrix: {X.shape} (input_dim={input_dim})")

    # ---- Train / val / test split ----
    # We hold out a TEST set (never seen during training or early stopping)
    # specifically so the reported metrics are honest, not just training
    # curve numbers.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_rating, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ---- Build + train the hybrid model ----
    model, encoder = build_hybrid_model(input_dim, embedding_dim=EMBEDDING_DIM)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    print("\nTraining hybrid network (reconstruction + rating prediction)...")
    history = model.fit(
        X_train,
        {"reconstruction": X_train, "rating_pred": y_train},
        validation_data=(X_val, {"reconstruction": X_val, "rating_pred": y_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---- Evaluate on the held-out TEST set (honest, unseen numbers) ----
    print("\nEvaluating on held-out test set...")
    _, rating_preds = model.predict(X_test, verbose=0)
    rating_preds = rating_preds.flatten()
    test_mae = float(mean_absolute_error(y_test, rating_preds))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, rating_preds)))
    print(f"Test MAE (rating prediction): {test_mae:.4f}")
    print(f"Test RMSE (rating prediction): {test_rmse:.4f}")

    # ---- Generate embeddings for every book using the trained encoder ----
    print("\nGenerating embeddings for all books...")
    all_embeddings = encoder.predict(X, batch_size=256, verbose=0)
    print("Embeddings shape:", all_embeddings.shape)

    # ---- Recommender-specific evaluation: Precision@K vs random baseline ----
    print(f"\nComputing Precision@{PRECISION_AT_K} (this validates the RECOMMENDATIONS, not just the model)...")
    prec_metrics = compute_precision_at_k(df, all_embeddings, k=PRECISION_AT_K, n_sample=N_EVAL_BOOKS)
    print(f"Precision@{PRECISION_AT_K}: {prec_metrics['precision_at_k']:.3f}  "
          f"(random baseline: {prec_metrics['random_baseline']:.3f})")

    # ---- Save everything ----
    encoder.save(os.path.join(ARTIFACTS_DIR, "encoder.keras"))
    model.save(os.path.join(ARTIFACTS_DIR, "hybrid_model.keras"))
    np.save(os.path.join(ARTIFACTS_DIR, "embeddings.npy"), all_embeddings)

    metrics = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "precision_at_k": prec_metrics,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_history": {
            "train_loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "train_rating_mae": [float(v) for v in history.history.get("rating_pred_mae", [])],
            "val_rating_mae": [float(v) for v in history.history.get("val_rating_pred_mae", [])],
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved trained model -> {ARTIFACTS_DIR}/hybrid_model.keras")
    print(f"Saved encoder -> {ARTIFACTS_DIR}/encoder.keras")
    print(f"Saved embeddings -> {ARTIFACTS_DIR}/embeddings.npy")
    print(f"Saved metrics -> {ARTIFACTS_DIR}/metrics.json")
    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
