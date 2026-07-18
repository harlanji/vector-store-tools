#!/usr/bin/env python3
"""
ChromaDB Collection Sync Script
Copies (or incrementally syncs) a source collection to a target collection
that uses a different embedding function (e.g. a Binary Embedding Function).

Usage (CLI):
    python sync_chroma.py \
        --source-collection my_docs \
        --target-collection my_docs_binary \
        --persist-directory ./chroma_db \ 
        --batch-size 2000 \
        --change-detection-key updated_at

Or import and use programmatically:
    from sync_chroma import sync_collections
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    source = client.get_collection("my_docs")
    target = client.get_collection("my_docs_binary")  # created with your binary EF
    sync_collections(source, target, change_detection_key="updated_at")
"""

from __future__ import annotations

import argparse
from typing import Any, Generator, Optional

import chromadb



def get_records_in_batches(
    collection: Any,
    batch_size: int = 1000,
    include: list[str] | None = None,
    offset: int = 0
) -> Generator[dict[str, list], None, None]:
    """Paginate through a collection using limit + offset."""
    if include is None:
        include = ["documents", "metadatas"]

    while True:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=include,
        )
        ids: list[str] = results.get("ids", [])
        if not ids:
            break

        documents = results.get("documents", [None] * len(ids))
        metadatas = results.get("metadatas", [None] * len(ids))
        embeddings = results.get("embeddings", [None] * len(ids))

        yield {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
            "offset": offset
        }
        offset += batch_size


def sync_collections(
    source_collection: Any,
    target_collection: Any,
    batch_size: int = 1000,
    change_detection_key: Optional[str] = None,
    new_ef: Any = None,
    dry_run = False,
    start_offset = 0,
) -> None:
    """
    Sync source collection into target collection.

    - Target must be created with your desired embedding_function (e.g. binary).
    - We never pass embeddings → target EF generates them automatically.
    - New records: any ID not present in target.
    - Changed records: if change_detection_key is provided and the value differs
      (or is missing in target).
    - If change_detection_key is None: upsert every record from source (simple & robust).
    """
    print(f"Source collection: {source_collection.name} (count: {source_collection.count()})")
    print(f"Target collection: {target_collection.name} (count: {target_collection.count()})")

    total_upserted = 0
    
    include = ["documents", "metadatas"]
    if new_ef or True:
        include.append("embeddings")

    for batch_no, batch in enumerate(get_records_in_batches(
        source_collection, batch_size=batch_size, include=include, offset = start_offset
    )):
        batch_ids = batch["ids"]
        batch_docs = batch["documents"]
        batch_metas = batch["metadatas"]
        if new_ef:
            batch_embs = batch["embeddings"]

        if not batch_ids:
            continue

        to_upsert_ids: list[str] = []
        to_upsert_docs: list[Any] = []
        to_upsert_metas: list[Any] = []
        if new_ef:
            to_upsert_embs: list[Any] = []

        if change_detection_key is None:
            # Simple mode: upsert everything (new + overwrite existing)
            to_upsert_ids = batch_ids
            to_upsert_docs = batch_docs
            to_upsert_metas = batch_metas
            if new_ef:
                to_upsert_embs = batch_embs
        else:
            # Selective mode: only new or changed (based on metadata key)
            target_results = target_collection.get(
                ids=batch_ids,
                include=["metadatas", "embeddings"],
            )
            target_existing = set(target_results.get("ids", []))
            target_metas_dict = dict(
                zip(
                    target_results.get("ids", []),
                    target_results.get("metadatas") or [None] * len(target_results.get("ids", [])),
                )
            )

            for i, sid in enumerate(batch_ids):
                sdoc = batch_docs[i]
                smeta = batch_metas[i]
                

                if change_detection_key == 'x_zero_emb':
                    batch_embs = batch.get("embeddings")
                    semb = batch_embs[i]
                    if semb.tolist() == [0] * len(semb):
                        #print("zero emb...")
                        to_upsert_ids.append(sid)
                        to_upsert_docs.append(sdoc)
                        to_upsert_metas.append(smeta)
                    else:
                        continue
                elif sid not in target_existing:
                    # New record
                    to_upsert_ids.append(sid)
                    to_upsert_docs.append(sdoc)
                    to_upsert_metas.append(smeta)
                     
                    if new_ef:
                        semb = batch_embs[i]
                        #tembs = new_ef([sdoc], embeddings=[semb]) # we might want to bring in the conditional kwarg caller
                        tembs = binary_embeds([semb])
                        to_upsert_embs.append(tembs[0])
                else:
                    # Existing → check change via metadata key
                    if smeta and change_detection_key in smeta:
                        s_val = smeta[change_detection_key]
                        tmeta = target_metas_dict.get(sid) or {}
                        t_val = tmeta.get(change_detection_key)
                        if t_val is None or s_val != t_val:
                            if new_ef:
                                semb = batch_embs[i]
                                temb = new_ef(sdoc, semb=semb) # we might want to bring in the conditional kwarg caller
                                to_upsert_embs.append(temb)
                            print(f"upsert doc: {sid}")
                            to_upsert_ids.append(sid)
                            to_upsert_docs.append(sdoc)
                            to_upsert_metas.append(smeta)

        if to_upsert_ids:
            upsert_kwargs = dict(
                ids=to_upsert_ids,
                documents=to_upsert_docs,
                metadatas=to_upsert_metas,
                # Do NOT pass embeddings — let target's embedding_function run
            )
            
            if new_ef:
                upsert_kwargs['embeddings'] = to_upsert_embs
            
            if not dry_run:
                target_collection.upsert(**upsert_kwargs)

            total_upserted += len(to_upsert_ids)
            print(f"  Upserted {len(to_upsert_ids)} records (batch progress, offset=#{batch['offset']}, run batch={batch_no})")

    print(f"\nSync complete. Total records upserted: {total_upserted}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync a ChromaDB collection to another that uses a different "
                    "(e.g. Binary) embedding function."
    )
    parser.add_argument(
        "--source-collection",
        required=True,
        help="Name of the source collection",
    )
    parser.add_argument(
        "--target-collection",
        required=True,
        help="Name of the target collection (must already exist and be created "
             "with your BinaryEmbeddingFunction)",
    )
    parser.add_argument(
        "--persist-directory",
        default="./chroma_db",
        help="ChromaDB persist directory (same client for source & target)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to process per batch",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Number of records to process per batch",
    )
    parser.add_argument(
        "--change-detection-key",
        default=None,
        help="Metadata key used to detect changes (e.g. 'updated_at', 'version'). "
             "If omitted, performs full upsert of all records.",
    )
    parser.add_argument(
        "--embedding-fn",
        default=None,
        help="Embedding function to use when moving to new collection. Options: default, binary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embedding function to use when moving to new collection. Options: default, binary",
    )

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_directory)

    new_ef = None
    if args.embedding_fn and args.embedding_fn != 'default':
        if args.embedding_fn == 'binary':
            new_ef = BinaryEmbeddingFunction(model_name=None)
        else:
            raise Exception('invalid embedding function')
    

    try:
        source = client.get_collection(args.source_collection)
        target_kwargs = dict(
            name=args.target_collection,
            configuration={
                "hnsw": {
                        "space": "cosine",
                        "batch_size": 25,
                        "sync_threshold": 100,
                        }
            }
        )
        
        if new_ef:
            target_kwargs['embedding_function'] = new_ef
        
        target = client.get_or_create_collection(**target_kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        print(
            "Make sure both collections exist.\n"
            "The target collection should be created like this:\n"
            "  target = client.get_or_create_collection(\n"
            "      name='your_target_name',\n"
            "      embedding_function=YourBinaryEmbeddingFunction()\n"
            "  )"
        )
        return
    
    sync_kwargs = dict(
        source_collection=source,
        target_collection=target,
        batch_size=args.batch_size,
        change_detection_key=args.change_detection_key,
        start_offset = args.start_offset
    )
    
    if new_ef:
        sync_kwargs['new_ef'] = new_ef
    
    sync_collections(dry_run=args.dry_run, **sync_kwargs)



###

from typing import List
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings

class BinaryEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Custom binary embedding function for ChromaDB.
    - Embeds text with a SentenceTransformer model.
    - Binarizes to ±1 (sign function).
    - Returns as Python lists (required by ChromaDB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        if model_name:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded model: {model_name} for binary embeddings")

    def __call__(self, input: Documents, embeddings = None) -> Embeddings:
        # Get float embeddings
        if not embeddings:
            embeddings = self.model.encode(input, convert_to_numpy=True)  # shape: (batch, dim)
        
        return binary_embeds(embeddings)


def binary_embeds(embeddings):
        # Binarize: sign() → -1 or +1 (or 0 for exact zero, rare)
        binary_embeddings = np.sign(embeddings).astype(np.int8)

        # Convert to list of lists (ChromaDB requirement)
        return binary_embeddings.tolist()

###

if __name__ == "__main__":
    main()
