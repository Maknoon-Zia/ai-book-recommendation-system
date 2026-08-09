"""
download_data.py
-----------------
Downloads the "Best Books 10k Multi Genre Data" dataset directly from Kaggle
using the `kagglehub` library -- no manual download / zip extraction needed.

SETUP (one-time, only step that requires manual action):
1. Create a free Kaggle account: https://www.kaggle.com
2. Go to: https://www.kaggle.com/settings -> "API" section -> "Create New Token"
   This downloads a file called `kaggle.json`.
3. Place it here:
      Linux/Mac:  ~/.kaggle/kaggle.json
      Windows:    C:\\Users\\<you>\\.kaggle\\kaggle.json
   (kagglehub will also pick up KAGGLE_USERNAME / KAGGLE_KEY env vars if you
   prefer setting environment variables instead of a file.)

That's it -- after this, running this script will pull the dataset straight
from Kaggle's servers, cache it locally, and return the folder path. You
never have to click "Download" on the Kaggle website.

Run:
    python download_data.py
"""

import os
import shutil
import pandas as pd
import kagglehub

# The Kaggle dataset handle is the part of the URL after kaggle.com/datasets/
DATASET_HANDLE = "ishikajohari/best-books-10k-multi-genre-data"

# Where we want a clean, stable copy of the CSV for the rest of the pipeline
LOCAL_DATA_DIR = "data"
LOCAL_CSV_PATH = os.path.join(LOCAL_DATA_DIR, "books.csv")


def download_dataset() -> str:
    """
    Downloads (or reuses a cached copy of) the dataset from Kaggle and
    returns the local folder path kagglehub stored it in.
    """
    print(f"Downloading '{DATASET_HANDLE}' from Kaggle (or using local cache)...")
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    print(f"Dataset files are stored at: {dataset_path}")
    return dataset_path


def find_csv(dataset_path: str) -> str:
    """
    The dataset folder may contain one or more files. This finds the first
    .csv file inside it, since that's what we need for this project.
    """
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError("No CSV file found inside the downloaded dataset.")


def main():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    dataset_path = download_dataset()
    csv_path = find_csv(dataset_path)

    # Copy it into our project's data/ folder so every other script has a
    # single, predictable path to read from.
    shutil.copy(csv_path, LOCAL_CSV_PATH)
    print(f"Copied dataset CSV to: {LOCAL_CSV_PATH}")

    # Quick sanity check / preview so you can confirm the real column names
    # (Kaggle datasets sometimes tweak column names between versions).
    df = pd.read_csv(LOCAL_CSV_PATH)
    print("\nShape:", df.shape)
    print("\nColumns found in the dataset:")
    print(list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))


if __name__ == "__main__":
    main()
