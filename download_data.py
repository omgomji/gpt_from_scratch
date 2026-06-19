"""
download_data.py
----------------
Downloads the Tiny Shakespeare corpus from Karpathy's char-rnn repository
and saves it to data/input.txt.

Usage:
    python download_data.py
"""

import os
import urllib.request

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")


def download(url: str = DATASET_URL, dest: str = OUTPUT_PATH) -> None:
    """Download the corpus file if it does not already exist."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"[INFO] Dataset already exists at '{dest}' ({size_mb:.2f} MB). Skipping download.")
        return

    print(f"[INFO] Downloading Tiny Shakespeare from:\n       {url}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1_048_576
    print(f"[INFO] Saved to '{dest}' ({size_mb:.2f} MB).")


if __name__ == "__main__":
    download()
