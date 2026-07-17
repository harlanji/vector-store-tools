import sys
import sqlite3
import json
from typing import List, Dict, Any, Optional, Literal
import sqlite_vec


class SQLiteVecCollection:
    """ChromaDB-compatible read-only collection for SQLite + sqlite-vec."""

    def __init__(
        self,
        db_path: str,
        embedding_function: Optional[Any] = None,   # e.g. Chroma-style embedder
        name = 'dummy',
        schema = None,
        configuration = None,
        distance_metric: Literal["cosine", "l2", "hamming"] = "hamming",
    ):
        self.db_path = db_path
        self.name = name
        self.embedding_function = embedding_function
        self.distance_metric = distance_metric
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.row_factory = sqlite3.Row  # dict-like rows
        return self._conn

    def count(self) -> int:
        """Return total number of items."""
        row = self._get_conn().execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0]

    def peek(self, limit: int = 5) -> Dict:
        """Return first N items (Chroma-style)."""
        return self.get(limit=limit)

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,        # simple metadata filter support
        limit: int = 100,
        offset: int = 0,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Chroma-style .get()"""
        if include is None:
            include = ["metadatas", "documents"]

        conn = self._get_conn()
        params = []
        sql = "SELECT d.id as rowid, d.original_id, d.content, m.metadata"

        if ids:
            sql += " FROM documents d LEFT JOIN metadata m ON d.id = m.id WHERE d.original_id IN (" + ",".join("?" * len(ids)) + ")"
            params.extend(ids)
        else:
            sql += " FROM documents d LEFT JOIN metadata m ON d.id = m.id"
            if where:
                # Very basic metadata JSON filtering (extend as needed)
                for k, v in where.items():
                    sql += f" AND json_extract(m.metadata, '$.{k}') = ?"
                    params.append(v)

        sql += f" LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()

        result = {
            "ids": [r["original_id"] for r in rows],
            "documents": [r["content"] for r in rows] if "documents" in include else None,
            "metadatas": [json.loads(r["metadata"]) if r["metadata"] else None for r in rows] if "metadatas" in include else None,
            "embeddings": None,   # optional, expensive to return
        }
        return {k: v for k, v in result.items() if v is not None}

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Chroma-style .query() — vector search (with optional FTS rerank)."""
        if query_texts and self.embedding_function:
            query_embeddings = self.embedding_function(query_texts)

        if not query_embeddings:
            raise ValueError("Provide query_texts + embedding_function or query_embeddings")

        # Take first query for simplicity (extend for multi-query later)
        q_emb = query_embeddings[0]

        conn = self._get_conn()

        # Vector search (Hamming if bit vectors)
        sql = """
            SELECT 
                d.original_id as id,
                d.content,
                m.metadata,
                vec_distance_hamming(v.embedding, vec_quantize_binary(?))/200 as distance
            FROM vec_docs v
            JOIN documents d ON v.id = d.id
            LEFT JOIN metadata m ON d.id = m.id
        """
        params = [sqlite_vec.serialize_float32(q_emb) if self.distance_metric != "hamming" else q_emb]

        if where:
            # Add simple JSON filtering...
            pass

        sql += " ORDER BY distance LIMIT ?"
        params.append(n_results)
        
        print(sql)
        print(params)

        rows = conn.execute(sql, params).fetchall()

        return {
            "ids": [[r["id"] for r in rows]],
            "distances": [[r["distance"] for r in rows]],
            "documents": [[r["content"] for r in rows]],
            "metadatas": [[json.loads(r["metadata"]) if r["metadata"] else None for r in rows]],
        }

