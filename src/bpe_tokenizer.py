"""
bpe_tokenizer.py
----------------
Byte-Pair Encoding (BPE) tokenizer from scratch.

Satisfies: V2 Upgrade (BPE Tokenizer)
"""

import json
from typing import List, Dict, Tuple


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Return counts of consecutive pairs in the list of ints."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace all consecutive occurrences of `pair` with `idx` in `ids`."""
    newids = []
    i = 0
    while i < len(ids):
        # If we are not at the very last position and the pair matches
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.
    
    Attributes
    ----------
    merges : dict[tuple[int, int], int]
        Mapping from a pair of token IDs to the new merged token ID.
    vocab : dict[int, bytes]
        Mapping from token ID to its underlying byte sequence.
    """

    def __init__(self) -> None:
        # Base vocabulary: 0-255 map to their raw byte values
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)

    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        """Train the BPE tokenizer on the provided text.
        
        Parameters
        ----------
        text : str
            The training text.
        vocab_size : int
            Target vocabulary size (must be >= 256).
        verbose : bool
            If True, print progress during training.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        
        # Convert text to raw bytes, then to list of ints
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        if verbose:
            print(f"Training BPE: text length={len(ids)}, target vocab_size={vocab_size}, merges={num_merges}")

        # Perform merges
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            
            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Save the merge and update vocab
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Apply merge to the IDs
            ids = merge(ids, pair, idx)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i+1}/{num_merges}: {pair} -> {idx}")

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs.
        
        Parameters
        ----------
        text : str
            The string to encode.
            
        Returns
        -------
        list[int]
            The list of token IDs.
        """
        # Convert to raw bytes
        ids = list(text.encode("utf-8"))
        
        # Iteratively apply merges
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair that has the lowest merge index (i.e. was merged earliest in training)
            # If no pairs in `stats` are in `self.merges`, we're done.
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a string.
        
        Parameters
        ----------
        ids : list[int]
            The list of token IDs to decode.
            
        Returns
        -------
        str
            The decoded string.
        """
        # Concatenate the bytes for each token ID
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Decode as utf-8, replacing errors
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save the tokenizer merges to a JSON file.
        
        Parameters
        ----------
        path : str
            Path to the output JSON file.
        """
        # JSON only supports string keys, so we convert the tuple pairs to strings
        data = {
            f"{p0},{p1}": idx
            for (p0, p1), idx in self.merges.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load tokenizer merges from a JSON file.
        
        Parameters
        ----------
        path : str
            Path to the input JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Restore merges and rebuild vocab
        for pair_str, idx in data.items():
            p0, p1 = map(int, pair_str.split(","))
            pair = (p0, p1)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
