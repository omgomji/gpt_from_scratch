"""
tokenizer.py
------------
Character-level tokenizer (Module 1 from SDD §5.1).

Satisfies: FR-1 (read corpus), FR-2 (build vocab), FR-3 (encode), FR-4 (decode)

The tokenizer is intentionally simple — every unique character in the corpus
becomes one token. This is the approach used by Karpathy's nanoGPT tutorial
and keeps the architecture as transparent as possible for educational purposes.

Usage
-----
    from src.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)

    ids   = tokenizer.encode("Hello World")
    text  = tokenizer.decode(ids)
    assert text == "Hello World"        # round-trip guarantee (T-1)
"""

from __future__ import annotations

from typing import List


class CharTokenizer:
    """Character-level vocabulary builder and encoder/decoder.

    Attributes
    ----------
    stoi : dict[str, int]
        Maps each character to its integer token ID.
    itos : dict[int, str]
        Inverse map — integer token ID to character.
    vocab_size : int
        Number of unique characters in the vocabulary.
        Set to 0 until :meth:`build_vocab` is called.

    Notes
    -----
    * Vocabulary is derived purely from observed characters in the corpus —
      no special tokens are added by default (SDD §5.1, Module 1).
    * Character ordering in the vocabulary is deterministic: sorted by Unicode
      code-point, which ensures reproducibility across runs (NFR-1).
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocab(self, text: str) -> None:
        """Derive the character vocabulary from *text*.

        Scans *text* for every unique character, sorts them by Unicode
        code-point for determinism, and builds the ``stoi`` / ``itos``
        lookup tables.

        Parameters
        ----------
        text : str
            Raw UTF-8 corpus text (FR-1 / FR-2).
        """
        chars: List[str] = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of integer token IDs (FR-3).

        Parameters
        ----------
        text : str
            Any string drawn from the same character set as the corpus.

        Returns
        -------
        List[int]
            Token-ID sequence, one integer per character.

        Raises
        ------
        RuntimeError
            If :meth:`build_vocab` has not been called yet.
        KeyError
            If *text* contains a character absent from the vocabulary.
        """
        self._check_vocab()
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of integer token IDs back to a string (FR-4).

        Parameters
        ----------
        ids : List[int]
            Sequence of token IDs produced by :meth:`encode`.

        Returns
        -------
        str
            Reconstructed text.

        Raises
        ------
        RuntimeError
            If :meth:`build_vocab` has not been called yet.
        KeyError
            If any ID is outside the known vocabulary range.
        """
        self._check_vocab()
        return "".join(self.itos[i] for i in ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of unique characters in the vocabulary (0 before build_vocab)."""
        return len(self.stoi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_vocab(self) -> None:
        if not self.stoi:
            raise RuntimeError(
                "Vocabulary is empty. Call CharTokenizer.build_vocab(text) first."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"CharTokenizer(vocab_size={self.vocab_size})"
