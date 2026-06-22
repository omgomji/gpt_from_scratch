"""
download_data.py
----------------
Downloads training corpora for GPT-from-Scratch.

Usage:
    python download_data.py --corpus shakespeare
    python download_data.py --corpus gutenberg
    python download_data.py --corpus wikitext
    python download_data.py --corpus all
"""

import os
import argparse
import urllib.request

# Each corpus key maps to one or more URLs.
# If a list is given, all files are downloaded and concatenated.
CORPORA = {
    "shakespeare": [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    ],
    "wikitext": [
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    ],
    "gutenberg": [
        # Alice in Wonderland (~148 KB)
        "https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt",
        # Pride and Prejudice (~724 KB)
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        # Adventures of Sherlock Holmes (~594 KB)
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        # Moby Dick (~1.3 MB)
        "https://www.gutenberg.org/files/2701/2701-0.txt",
        # A Tale of Two Cities (~804 KB)
        "https://www.gutenberg.org/files/98/98-0.txt",
    ],
}


def download_file(url: str, dest: str) -> None:
    """Download a single file to dest if it does not already exist."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"[INFO] File already exists at '{dest}' ({size_mb:.2f} MB).")
        return

    print(f"[INFO] Downloading from:\n       {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"[INFO] Saved to '{dest}' ({size_mb:.2f} MB).")
    except Exception as e:
        print(f"[WARN] Failed to download '{url}': {e}")
        if os.path.exists(dest):
            os.remove(dest)


def download_corpus(name: str, urls: list, data_dir: str) -> list:
    """Download all URLs for a corpus and return paths of successfully downloaded files."""
    downloaded = []
    if len(urls) == 1:
        dest = os.path.join(data_dir, f"{name}.txt")
        download_file(urls[0], dest)
        if os.path.exists(dest):
            downloaded.append(dest)
    else:
        # Multiple files: save each with an index suffix, then combine
        parts = []
        for i, url in enumerate(urls):
            dest = os.path.join(data_dir, f"{name}_part{i}.txt")
            download_file(url, dest)
            if os.path.exists(dest):
                parts.append(dest)

        if parts:
            combined = os.path.join(data_dir, f"{name}.txt")
            print(f"[INFO] Combining {len(parts)} files into '{combined}'...")
            with open(combined, "w", encoding="utf-8") as outfile:
                for p in parts:
                    with open(p, "r", encoding="utf-8", errors="replace") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")
            size_mb = os.path.getsize(combined) / 1_048_576
            print(f"[INFO] Combined corpus: '{combined}' ({size_mb:.2f} MB).")
            downloaded.append(combined)

    return downloaded


def combine_all(dest: str, corpus_files: list) -> None:
    """Concatenate all given corpus files into one master file."""
    print(f"[INFO] Combining all corpora into '{dest}'...")
    with open(dest, "w", encoding="utf-8") as outfile:
        for f in corpus_files:
            if not os.path.exists(f):
                continue
            with open(f, "r", encoding="utf-8", errors="replace") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")
    size_mb = os.path.getsize(dest) / 1_048_576
    print(f"[INFO] Created master corpus '{dest}' ({size_mb:.2f} MB).")


def main():
    parser = argparse.ArgumentParser(description="Download training data for GPT-from-Scratch.")
    parser.add_argument(
        "--corpus",
        choices=list(CORPORA.keys()) + ["all"],
        default="shakespeare",
        help="Which corpus to download.",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.corpus == "all":
        all_paths = []
        for name, urls in CORPORA.items():
            paths = download_corpus(name, urls, data_dir)
            all_paths.extend(paths)
        combine_all(os.path.join(data_dir, "all.txt"), all_paths)
    else:
        urls = CORPORA[args.corpus]
        paths = download_corpus(args.corpus, urls, data_dir)

        # Backwards compatibility: mirror to data/input.txt
        if paths:
            input_txt = os.path.join(data_dir, "input.txt")
            src = paths[0]  # combined file for multi-URL corpus
            print(f"[INFO] Mirroring to '{input_txt}'...")
            with open(input_txt, "w", encoding="utf-8") as outfile:
                with open(src, "r", encoding="utf-8", errors="replace") as infile:
                    outfile.write(infile.read())


if __name__ == "__main__":
    main()
