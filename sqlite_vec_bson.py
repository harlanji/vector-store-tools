import sqlite3
import json
from pathlib import Path
from typing import Optional

import sqlite_vec  # pip install sqlite-vec

from util import *

import traceback

def create_sqlite_vec_tables(db: sqlite3.Connection, dim: int) -> None:
    """Tables with integer rowid (compatible with FTS5 contentless)."""
    # Metadata (JSON)
    db.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            original_id TEXT,
            metadata JSON
        )
    """)

    # Documents
    db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            original_id TEXT,
            content TEXT
        )
    """)

    # FTS5 contentless (rowid-based, no contentless_delete issues)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
            content,
            content=''
        )
    """)

    # sqlite-vec bit vectors (integer id)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(
            id INTEGER PRIMARY KEY,
            embedding bit[{dim}]
        )
    """)


def pack_to_bit_blob(packed_int32: list[int], dim: int) -> bytes:
    """Convert our int32-packed list to a compact bit blob for sqlite-vec."""
    # packed_int32: list of 32-bit words
    num_bits = dim
    num_bytes = (num_bits + 7) // 8
    blob = bytearray(num_bytes)

    bit_pos = 0
    for word in packed_int32:
        for b in range(32):
            if bit_pos >= num_bits:
                break
            if word & (1 << b):
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                blob[byte_idx] |= (1 << bit_in_byte)
            bit_pos += 1
        if bit_pos >= num_bits:
            break
    return bytes(blob)

def pack_to_bit_list(packed_int32: list[int], dim: int) -> list[int]:
    """Convert our int32-packed list to a compact bit blob for sqlite-vec."""
    # packed_int32: list of 32-bit words
    num_bits = dim
    num_bytes = (num_bits + 7) // 8
    bit_list = []

    bit_pos = 0
    for word in packed_int32:
        for b in range(32):
            if bit_pos >= num_bits:
                break
            if word & (1 << b):
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                bit = (1 << bit_in_byte)
                bit_list.append(1.0)
            else:
                bit_list.append(-1.0)
            bit_pos += 1
        if bit_pos >= num_bits:
            break
    return bit_list

def load_bson_to_sqlite_vec(
    bson_path: Path | str,
    db_path: Path | str = "vec.db",
    chunk_size: int = 1000,
    auto_create: bool = True,
    no_text: bool = False,
) -> None:
    """Load with integer rowid sequence + contentless FTS5."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("SELECT vec_version()")  # sanity check

    dim = None
    next_id = 1  # Our own auto-increment sequence

    with open(bson_path, "rb") as f:
        for record in iter_bson_records(f, decompress_binary=False):
            if dim is None and record.get("is_binary_embedding"):
                dim = record.get("embedding_dim")
                if auto_create and dim:
                    create_sqlite_vec_tables(conn, dim)

            if dim is None:
                raise ValueError("Could not determine embedding dimension")

            orig_id = record["id"]
            content = record.get("document") or ""
            meta = record.get("metadata") or {}

            if record.get("is_binary_embedding"):
                bit_list = pack_to_bit_list(record["embedding"], dim)
            else:
                raise("Not supported.")

            # Insert with our sequential integer ID
            conn.execute(
                "INSERT INTO documents (id, original_id, content) VALUES (?,?,?)",
                (next_id, orig_id, content if not no_text else None)
            )
            conn.execute(
                "INSERT INTO metadata (id, original_id, metadata) VALUES (?,?,?)",
                (next_id, orig_id, json.dumps(meta))
            )
            if not no_text:
                conn.execute(
                    "INSERT INTO fts_docs(rowid, content) VALUES (?,?)",
                    (next_id, content)
                )
            #print(f"bit_list={bit_list}")
            conn.execute(
                "INSERT INTO vec_docs (id, embedding) values(?, vec_quantize_binary(?))",
                (next_id, str(bit_list))
            )

            next_id += 1

            # Optional batch commit for performance
            if next_id % chunk_size == 0:
                conn.commit()

    conn.commit()
    conn.close()
    print(f"Loaded {next_id-1} records into {db_path} (vector dim={dim})")
    
    



#!/usr/bin/env python3
"""
BSON → SQLite + sqlite-vec + FTS5 loader (bit vectors)
"""

import argparse
import sys
from pathlib import Path

# Paste these helpers here (from previous response):
# create_sqlite_vec_tables, pack_to_bit_blob, load_bson_to_sqlite_vec, iter_bson_records


def main():
    parser = argparse.ArgumentParser(
        description="Load BSON (with binary embeddings) into SQLite + sqlite-vec + FTS5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bson-file",
        type=Path,
        required=True,
        help="Input BSON file (from Chroma dump)",
    )
    parser.add_argument(
        "--db-file",
        type=Path,
        default=Path("vec.db"),
        help="Output SQLite database file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Batch size for inserts",
    )
    parser.add_argument(
        "--no-auto-create",
        action="store_true",
        default=False,
        help="Disable automatic table creation (tables must exist)",
    )
    parser.add_argument(
        "--table-prefix",
        type=str,
        default="",
        help="Optional prefix for table names (e.g. 'myapp_')",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        default=False,
        help="Omit text",
    )

    args = parser.parse_args()

    if not args.bson_file.exists():
        print(f"Error: BSON file not found: {args.bson_file}", file=sys.stderr)
        sys.exit(1)

    try:
        load_bson_to_sqlite_vec(
            bson_path=args.bson_file,
            db_path=args.db_file,
            chunk_size=args.chunk_size,
            auto_create=not args.no_auto_create,
            no_text=args.no_text,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(e)
        sys.exit(1)


if __name__ == "__main__":
    main()