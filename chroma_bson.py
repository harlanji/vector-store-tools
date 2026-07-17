from __future__ import annotations
import io
from typing import BinaryIO, Iterator, Optional, Any
import chromadb
from bson import encode, decode_file_iter

from util import *

# ============================================================
# DUMP: ChromaDB Collection → BSON Stream
# ============================================================

def dump_collection_to_bson(
    stream: BinaryIO,
    collection: chromadb.Collection,
    chunk_size: int = 1000,
    binary_compress: bool = False,
    threshold: float = 0.0,
    include: list[str] | None = None,
    start_offset: int = 0, # TODO
    limit: int = 1000, # TODO
    no_text = False,
    
) -> None:
    """
    Stream a ChromaDB collection to BSON (concatenated documents).

    Each record written:
        {
            "id": str,
            "document": str | None,
            "metadata": dict | None,
            "embedding": list[float] | list[int],   # packed ints if binary_compress
            "is_binary_embedding": bool,
            "embedding_dim": int
        }
    """
    if include is None:
        include = ["embeddings", "metadatas", "documents"]

    total = collection.count()
    if total == 0:
        return

    for offset in range(0, total, chunk_size):
        results = collection.get(
            limit=chunk_size,
            offset=offset,
            include=include
        )
        
        print(f"processing offset={offset} / {total}")

        ids = results.get("ids", [])
        embeddings = results.get("embeddings")
        metadatas = results.get("metadatas")
        documents = results.get("documents")

        #if binary_compress and embeddings:
        dim = len(embeddings[0]) # if the collection is empty then it will crash.
        packed = pack_embeddings_to_int32(embeddings, threshold=threshold)
        embeddings_out = packed
        is_binary = True
        #else:
        #    embeddings_out = embeddings or [None] * len(ids)
        #    is_binary = False
        #    dim = len(embeddings[0]) if embeddings else 0

        for i, _id in enumerate(ids):
            record = {
                "id": _id,
                "document": documents[i] if not no_text and documents else None,
                "metadata": metadatas[i] if metadatas else None,
                "embedding": embeddings_out[i] if embeddings_out else None,
                "is_binary_embedding": is_binary,
                "embedding_dim": dim,
            }
            stream.write(encode(record))



# ============================================================
# Load BSON Stream back into a ChromaDB Collection
# ============================================================

def load_bson_to_chromadb(
    stream: BinaryIO,
    collection: chromadb.Collection,
    chunk_size: int = 1000,
    decompress_binary: bool = False,
    no_text = False,
) -> None:
    """Stream BSON records into a ChromaDB collection (batched adds)."""
    batch_ids = []
    batch_embeddings = []
    batch_metadatas = []
    batch_documents = []

    for offset, record in enumerate(iter_bson_records(stream, decompress_binary=decompress_binary)):
        batch_ids.append(record["id"])
        batch_embeddings.append(record.get("embedding"))
        batch_metadatas.append(record.get("metadata"))
        batch_documents.append(record.get("document") if not no_text else None)
        
        if len(batch_ids) >= chunk_size:
            _add_batch(collection, batch_ids, batch_embeddings, batch_metadatas, batch_documents)
            print(f"loaded to offset={offset} / ??")
            batch_ids.clear()
            batch_embeddings.clear()
            batch_metadatas.clear()
            batch_documents.clear()

    if batch_ids:
        _add_batch(collection, batch_ids, batch_embeddings, batch_metadatas, batch_documents)
        print(f"loaded to offset={offset} / ??")
        

def _add_batch(collection, ids, embeddings, metadatas, documents):
    kwargs = {"ids": ids}
    if any(e is not None for e in embeddings):
        kwargs["embeddings"] = embeddings
    if any(m is not None for m in metadatas):
        kwargs["metadatas"] = metadatas
    if any(d is not None for d in documents):
        kwargs["documents"] = documents
    res = collection.add(**kwargs)
    print(res)


# ============================================================
# USAGE EXAMPLES
# ============================================================

#!/usr/bin/env python3
"""
ChromaDB <-> BSON command-line tool
"""

import argparse
import sys
from pathlib import Path
import chromadb

# Paste the helper functions here (pack/unpack, dump, iter, load)
# ... (copy all the functions from the previous response: pack_embeddings_to_int32, unpack_int32_to_floats,
# dump_collection_to_bson, iter_bson_records, load_bson_to_chromadb, _add_batch)

# For brevity I'll assume they are defined above this main block.
# In a real file, put all helpers first.

def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB collection <-> BSON stream tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--action",
        choices=["dump", "load"],
        required=True,
        help="Action to perform: dump collection to BSON or load BSON into collection",
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the .bson file (will be created/overwritten on dump)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of documents per chunk/batch",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=False,
        help="Enable binary compression (pack floats to int32 words)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        default=False,
        help="Omit text",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binarization (used only with --binary)",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        default=False,
        help="When loading, decompress binary embeddings back to floats",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="ChromaDB HTTP host (if using persistent or server mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="ChromaDB HTTP port",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Persistent directory for ChromaDB (local mode)",
    )

    args = parser.parse_args()

    # Initialize ChromaDB client
    if args.persist_dir:
        client = chromadb.PersistentClient(path=str(args.persist_dir))
    elif args.host and args.port:
        client = chromadb.HttpClient(host=args.host, port=args.port)
    else:
        client = chromadb.Client()  # in-memory default

    collection = client.get_or_create_collection(name=args.collection, configuration={
                    "hnsw": {
                        "space": "cosine",
                        "batch_size": 25,
                        "sync_threshold": 100
                        }
                })

    if args.action == "dump":
        mode = "wb"
        print(f"Dumping collection '{args.collection}' → {args.file}")
        with open(args.file, mode) as f:
            dump_collection_to_bson(
                f,
                collection,
                chunk_size=args.chunk_size,
                binary_compress=args.binary,
                threshold=args.threshold,
                no_text=args.no_text,
            )
        print("Dump completed successfully.")

    elif args.action == "load":
        mode = "rb"
        print(f"Loading {args.file} → collection '{args.collection}'")
        with open(args.file, mode) as f:
            load_bson_to_chromadb(
                f,
                collection,
                chunk_size=args.chunk_size,
                decompress_binary=args.decompress,
                no_text=args.no_text,
            )
        print("Load completed successfully.")

    else:
        print("Invalid action", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

