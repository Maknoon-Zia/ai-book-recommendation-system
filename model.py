"""
model.py
---------
The neural network itself: a feedforward AUTOENCODER.

Why an autoencoder for a recommendation system?
We don't have user-item interaction data (this dataset is book metadata,
not "which user rated which book"), so a classic collaborative-filtering
network doesn't apply here. Instead we use a CONTENT-BASED approach:

  1. Each book is already represented as a numeric vector (from
     preprocess.py: TF-IDF/SVD of its text + scaled numeric stats).
  2. We train a feedforward network to compress that vector down to a
     small "embedding" and then reconstruct the original vector from it
     (encoder -> bottleneck -> decoder). This forces the bottleneck layer
     to learn a dense representation that captures what makes each book
     similar to others.
  3. At inference time we only use the ENCODER half. Feeding a book's
     features through it gives a short embedding vector. Books with
     similar embeddings (measured by cosine similarity) are recommended
     to each other.

This is a legitimate, standard feedforward (fully-connected) neural
network architecture -- every layer is Dense, activations flow strictly
forward, no recurrence/attention.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_hybrid_model(input_dim: int, embedding_dim: int = 32):
    """
    Builds a HYBRID feedforward network with two heads sharing one encoder:

                          ┌─> Decoder ─> reconstruction (unsupervised)
      Input ─> Encoder ─> embedding
                          └─> Rating head ─> predicted rating (supervised)

    Why hybrid instead of a plain autoencoder?
    A pure autoencoder only learns to compress-and-reconstruct a book's
    features -- nothing forces the embedding to actually relate to quality
    or reader reception. By adding a second head that predicts the book's
    real average rating from the SAME embedding, we push the encoder to
    learn a representation that is useful for a real downstream task
    (rating prediction), not just for copying its input back out. This is
    a standard multi-task learning setup and it's the difference between
    "the network learned to compress numbers" and "the network learned
    something about what these books have in common."

    Returns (model, encoder):
      - model: full network with both heads, used for training
      - encoder: bottleneck-only sub-model, used to generate embeddings
        for similarity search at inference time
    """

    inputs = keras.Input(shape=(input_dim,), name="book_features")

    # ---- Shared encoder: progressively compress the feature vector ----
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)

    # ---- Head 1: Decoder (unsupervised) — reconstruct original features ----
    d = layers.Dense(128, activation="relu")(embedding)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(256, activation="relu")(d)
    reconstruction = layers.Dense(input_dim, activation="linear", name="reconstruction")(d)

    # ---- Head 2: Rating predictor (supervised) — predicts avg rating ----
    r = layers.Dense(32, activation="relu")(embedding)
    r = layers.Dropout(0.15)(r)
    rating_pred = layers.Dense(1, activation="linear", name="rating_pred")(r)

    model = keras.Model(inputs, [reconstruction, rating_pred], name="hybrid_book_model")
    encoder = keras.Model(inputs, embedding, name="book_encoder")

    # Both heads trained jointly. reconstruction loss is naturally larger in
    # scale (input_dim outputs) than rating loss (1 output), so we weight
    # the rating head up to keep its gradient signal meaningful.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"reconstruction": "mse", "rating_pred": "mse"},
        loss_weights={"reconstruction": 1.0, "rating_pred": 2.0},
        metrics={"rating_pred": "mae"},
    )

    return model, encoder


# Kept for backwards compatibility / simpler experiments if you ever want
# to go back to a plain (non-hybrid) autoencoder.
def build_autoencoder(input_dim: int, embedding_dim: int = 32):
    inputs = keras.Input(shape=(input_dim,), name="book_features")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = layers.Dense(128, activation="relu")(embedding)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs, outputs, name="book_autoencoder")
    encoder = keras.Model(inputs, embedding, name="book_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return autoencoder, encoder
