"""Byte-level tokenizer: each byte is a token, plus one beginning-of-sequence (BOS) token.

You may replace this with any tokenizer, provided that:
  - decode(encode(text)) == text exactly (eval.py checks this),
  - encode returns a numpy array of ints < vocab_size, and there is a `bos` token id,
  - any files it needs live in out/ (they count towards the size limit).
"""
import numpy as np


class ByteTokenizer:
    vocab_size = 257
    bos = 256

    def encode(self, text: bytes) -> np.ndarray:
        return np.frombuffer(text, dtype=np.uint8).astype(np.uint16)

    def decode(self, ids) -> bytes:
        return np.asarray(ids).astype(np.uint8).tobytes()


def load_tokenizer(out_dir):
    return ByteTokenizer()
