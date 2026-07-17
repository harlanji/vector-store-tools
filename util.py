from typing import BinaryIO, Iterator, Optional, Any

from bson import decode_file_iter  # for potential future use

import numpy as np  # ChromaDB already depends on numpy

# ============================================================
# BINARY PACKING / UNPACKING (int32 words)
# ============================================================

def pack_embeddings_to_int32(
    embeddings: list[list[float]] | np.ndarray,
    threshold: float = 0.0
) -> list[list[int]]:
    """Convert float embeddings to list of packed int32 words (one int32 = 32 bits)."""
    if len(embeddings) == 0:
        return []

    arr = np.asarray(embeddings, dtype=np.float32)
    n, dim = arr.shape
    num_words = (dim + 31) // 32

    # Binarize
    bits = (arr > threshold).astype(np.uint32)

    packed = np.zeros((n, num_words), dtype=np.uint32)
    for i in range(dim):
        word_idx = i // 32
        bit_idx = i % 32
        packed[:, word_idx] |= (bits[:, i] << bit_idx)

    return packed.tolist()  # BSON-friendly list of lists of ints


def unpack_int32_to_floats(
    packed_embeddings: list[list[int]] | np.ndarray,
    dim: int
) -> list[list[float]]:
    """Unpack int32 words back to float vectors (0.0 / 1.0). Lossy but fast."""
    if len(packed_embeddings) == 0:
        return []

    arr = np.asarray(packed_embeddings, dtype=np.uint32)
    n, num_words = arr.shape

    floats = np.zeros((n, dim), dtype=np.float32)
    for i in range(dim):
        word_idx = i // 32
        bit_idx = i % 32
        floats[:, i] = ((arr[:, word_idx] >> bit_idx) & 1).astype(np.float32)

    return floats.tolist()

# ============================================================
# READ: BSON Stream → Records (with optional decompression)
# ============================================================

def iter_bson_records(
    stream: BinaryIO,
    decompress_binary: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Stream records from a BSON file.
    If decompress_binary=True, binary-packed embeddings are converted back to floats.
    """
    for record in decode_file_iter(stream):
        if (
            decompress_binary
            and record.get("is_binary_embedding")
            and record.get("embedding") is not None
            and record.get("embedding_dim")
        ):
            dim = record["embedding_dim"]
            record["embedding"] = unpack_int32_to_floats([record["embedding"]], dim)[0]
            record["is_binary_embedding"] = False
        yield record


