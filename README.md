# 📖 Inkwell — AI Book Recommendation System

A content-based book recommender built with a **hybrid feedforward neural
network** in TensorFlow/Keras, served through a dark-themed **Streamlit**
frontend. Uses the Kaggle dataset
[*Best Books 10k Multi Genre Data*](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data),
downloaded **directly via the Kaggle API** — no manual clicking or zip
extraction required.

**Why this project is designed the way it is:** this dataset contains
book *metadata* (title, genre, author, description, rating) — it does
**not** contain individual user ratings. Rather than fabricating fake
user IDs and synthetic ratings to force a collaborative-filtering model
onto data that doesn't support it, this project uses a **content-based
hybrid network**: one encoder, two training signals (reconstruction +
real rating prediction), producing embeddings that are validated against
held-out metrics — not just asserted to work.

### Highlights
- 🧠 **Hybrid multi-task network**: one shared encoder trained on two
  objectives at once (autoencoder reconstruction + rating-prediction
  regression), which forces the embedding to encode something
  meaningful about reader reception, not just compress numbers.
- 🎯 **Taste-profile blending**: pick 2–5 favorite books and the app
  averages their embeddings into a single combined taste vector, instead
  of only supporting "more like this one book."
- 💡 **Explainable recommendations**: every suggestion shows *why* it
  was picked (shared genres, matching author) — not a black-box score.
- 📊 **Real evaluation, not vibes**: a held-out test set reports rating-
  prediction MAE/RMSE, and a genre-based Precision@K metric compares the
  model's recommendations against a random baseline so the numbers have
  context. A 2D PCA projection of the embedding space is plotted so you
  can visually confirm genre clusters actually separate.
- 🎨 Dark "reading lamp" themed UI (ink navy + brass gold + burgundy).


## How it works (the short version)

1. **Download**: `kagglehub` pulls the dataset straight from Kaggle's servers.
2. **Preprocess**: each book's genre/author/description is turned into a
   TF-IDF vector, compressed with SVD, and combined with scaled numeric
   fields (rating, ratings count, pages) into one feature vector per book.
3. **Train**: a hybrid feedforward network shares one encoder
   (`Input -> 256 -> 128 -> 32 (embedding)`) between two heads:
   - a **decoder** (`128 -> 256 -> Output`) that reconstructs the
     original feature vector (unsupervised signal)
   - a **rating predictor** (`32 -> 32 -> 1`) that predicts the book's
     real average rating from the same embedding (supervised signal)

   Training both jointly means the embedding has to be useful for an
   actual downstream task, not just good at copying its input back out.
4. **Evaluate**: MAE/RMSE on a held-out test set for rating prediction,
   plus Precision@K (do the top-K nearest neighbors in embedding space
   actually share genres with the seed book?) against a random baseline.
5. **Recommend**: cosine similarity between embeddings — either from one
   seed book, or from the average of several books blended into a taste
   profile.

---

## Step-by-step setup

### 1. Get a Kaggle API key (one-time, ~2 minutes)
1. Create a free account at [kaggle.com](https://www.kaggle.com) if you
   don't have one.
2. Go to **kaggle.com/settings → API → "Create New Token"**. This
   downloads `kaggle.json`.
3. Place the file at:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`

   (Alternatively, set environment variables `KAGGLE_USERNAME` and
   `KAGGLE_KEY` instead of using a file.)

That's the only manual step — after this, everything downloads
programmatically.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset (no manual download needed)
```bash
python download_data.py
```
This fetches the dataset via the Kaggle API, caches it locally, and
copies the CSV to `data/books.csv`. It also prints the real column names
so you can sanity-check them.

### 4. Preprocess the data
```bash
python preprocess.py
```
Cleans the data and builds the feature matrix used to train the network.
Saves everything to `artifacts/`.

### 5. Train the neural network
```bash
python train.py
```
Trains the hybrid network (reconstruction + rating-prediction heads) on
a proper train/val/test split, evaluates it on the held-out test set,
computes Precision@K against a random baseline, and saves the trained
model, encoder, book embeddings, and `metrics.json` to `artifacts/`.
You'll see console output like:

```
Test MAE (rating prediction): 0.19
Test RMSE (rating prediction): 0.27
Precision@10: 0.58  (random baseline: 0.13)
```

### 6. Launch the app locally
```bash
streamlit run app.py
```
Opens the dark-themed "Inkwell" frontend in your browser at
`http://localhost:8501`.


---

## Project structure

```
book_recommender/
├── download_data.py         # Pulls dataset from Kaggle via kagglehub  (local-only step)
├── preprocess.py             # Cleans data, builds TF-IDF/SVD + numeric features (local-only)
├── model.py                   # Hybrid feedforward network (reconstruction + rating heads)
├── train.py                    # Trains, evaluates (MAE/RMSE/Precision@K)  (local-only step)
├── recommend.py             # Similarity search, profile blending, explainability
├── app.py                        # Streamlit dark-theme frontend — THIS is the deploy entry point
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml      # Native Streamlit dark theme
├── data/                        # Downloaded CSV (gitignored — regenerate via download_data.py)
└── artifacts/
    ├── books_clean.pkl          # ✅ committed — read by app.py
    ├── embeddings.npy           # ✅ committed — read by app.py
    ├── metrics.json             # ✅ committed — read by app.py
    ├── hybrid_model.keras       # ❌ gitignored — training artifact only
    ├── encoder.keras            # ❌ gitignored — training artifact only
    ├── features.npy             # ❌ gitignored — training artifact only
    ├── tfidf.joblib             # ❌ gitignored — training artifact only
    ├── svd.joblib                # ❌ gitignored — training artifact only
    └── scaler.joblib             # ❌ gitignored — training artifact only
```

## App tabs

- **🔮 Recommend** — single-book "more like this," or switch to **taste
  profile mode** and pick 2–5 favorites to get blended recommendations.
  Each result shows a match %, shared genres, and whether the author
  matches.
- **🔎 Browse by Genre** — straightforward metadata filter, no network
  needed, for exploring the library directly.
- **📊 Model Insights** — the proof-of-work tab: test MAE/RMSE, Precision@K
  vs. random baseline, training loss curves, and an interactive 2D map of
  the embedding space colored by genre.

---

## Notes & things you may want to tweak

- **Column names**: Kaggle datasets occasionally change column names
  between versions. `preprocess.py` auto-detects common aliases
  (title/book/name, genre/genres/category, etc.) — check the printed
  "Detected column mapping" the first time you run it to confirm it
  matched correctly. If a field wasn't found, add its real column name
  to the `COLUMN_ALIASES` dict at the top of `preprocess.py`.
- **Embedding size**: `EMBEDDING_DIM = 32` in `train.py` — increase for
  more nuanced similarity, decrease for speed on very large datasets.
- **Re-running**: if you tweak preprocessing or the model, just re-run
  `preprocess.py` and/or `train.py` — the app always reads the latest
  files in `artifacts/`.
- **No GPU needed**: 10k rows trains in well under a minute on CPU.
