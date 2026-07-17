#!/usr/bin/env python3
"""
SQLite + sqlite-vec Collection CLI (ChromaDB-compatible)
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.append('/media/harlanji/Work/p/Dev Practice/2026-07-01/chroma-doc-tools')
from migrate_embeddings import BinaryEmbeddingFunction

from sqlite_vec_chroma_api import *

def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB-style operations on SQLite + sqlite-vec database"
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to .db file")
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Output format"
    )
    parser.add_argument(
        "--embedding-function", type=str, default=None,
        help="Import path to embedding function (e.g. 'module:fn')"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # === COUNT ===
    subparsers.add_parser("count", help="Return total number of documents")

    # === PEEK ===
    peek_p = subparsers.add_parser("peek", help="Return first N items")
    peek_p.add_argument("--limit", type=int, default=5)

    # === GET ===
    get_p = subparsers.add_parser("get", help="Get documents by ID or filter")
    get_p.add_argument("--ids", type=str, help="Comma-separated list of IDs")
    get_p.add_argument("--where", type=str, help="JSON metadata filter, e.g. '{\"source\": \"wiki\"}'")
    get_p.add_argument("--limit", type=int, default=100)
    get_p.add_argument("--offset", type=int, default=0)
    get_p.add_argument("--include", type=str, default="metadatas,documents",
                       help="Comma-separated: metadatas,documents,embeddings")

    # === QUERY ===
    query_p = subparsers.add_parser("query", help="Vector / semantic search")
    query_p.add_argument("--texts", type=str, help="Comma-separated query texts")
    query_p.add_argument("--n-results", type=int, default=10, dest="n_results")
    query_p.add_argument("--where", type=str, help="JSON metadata filter")
    query_p.add_argument("--distance", choices=["hamming", "cosine", "l2"], default="hamming")

    args = parser.parse_args()

    # Load embedding function if provided
    embedding_fn = None
    if args.embedding_function:
        module_name, fn_name = args.embedding_function.split(":")
        mod = __import__(module_name, fromlist=[fn_name])
        embedding_fn = getattr(mod, fn_name)
    else:
        embedding_fn = BinaryEmbeddingFunction()
        
    coll = SQLiteVecCollection(
        str(args.db),
        embedding_function=embedding_fn,
        distance_metric=args.distance if hasattr(args, "distance") else "hamming"
    )

    # === Dispatch ===
    if args.command == "count":
        result = {"count": coll.count()}

    elif args.command == "peek":
        result = coll.peek(limit=args.limit)

    elif args.command == "get":
        ids = args.ids.split(",") if args.ids else None
        where = json.loads(args.where) if args.where else None
        include = args.include.split(",") if args.include else None
        result = coll.get(
            ids=ids,
            where=where,
            limit=args.limit,
            offset=args.offset,
            include=include
        )

    elif args.command == "query":
        texts = args.texts.split(",") if args.texts else None
        where = json.loads(args.where) if args.where else None
        result = coll.query(
            query_texts=texts,
            n_results=args.n_results,
            where=where
        )

    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)

    # === Output ===
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Simple CSV output (flattens first level for query/get)
        if isinstance(result, dict) and "ids" in result:
            # Handle Chroma-style nested lists
            keys = list(result.keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=keys)
            writer.writeheader()
            # Write first row for simplicity (extend if needed)
            row = {k: result[k][0] if isinstance(result[k], list) else result[k] for k in keys}
            writer.writerow(row)
        else:
            print(json.dumps(result))  # fallback


if __name__ == "__main__":
    main()