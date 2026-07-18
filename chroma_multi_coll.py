from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from collections import defaultdict


def query_collections(
    collections: List[Any],
    query_texts: Optional[Union[str, List[str]]] = None,
    query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    coll_weights: Optional[Dict[str, float]] = None,
    rrf_k: int = 60,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the same query across multiple ChromaDB collections and fuse results
    using Weighted Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    collections : list of chromadb.Collection
        The collections to query.
    coll_weights : dict, optional
        Mapping of collection.name -> weight (e.g. {"my_coll": 0.8}).
        Collections not in the dict get weight=1.0.
    rrf_k : int
        Constant used in RRF formula (default 60 is the classic value).
    All other parameters are passed directly to collection.query().

    Returns
    -------
    dict in the same shape as collection.query(), plus an extra
    "collections": List[List[str]] field (parallel to "ids").
    """
    if not collections:
        return {"ids": [[]], "collections": [[]]}

    if include is None:
        include = ["metadatas", "documents", "distances"]

    if coll_weights is None:
        coll_weights = {}

    # Build weights (default = 1.0)
    weights: Dict[int, float] = {
        i: coll_weights.get(i, 1.0) for i, coll in enumerate(collections)
    }

    # Query every collection
    results_per_coll: Dict[str, Dict[str, Any]] = {}
    for i, coll in enumerate(collections):
        try:
            res = coll.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
                **kwargs,
            )
            if 'metadatas' in res:
                for md in res['metadatas'][0]:
                    #print(res['metadatas'])
                    md['_coll_name'] = coll.name
            results_per_coll[i] = {"result": res, "weight": weights[i]}
        except Exception as e:
            print(f"[query_collections] Error querying '{coll.name}' ({i}): {e}")
            continue

    if not results_per_coll:
        return {"ids": [[]], "collections": [[]]}

    # Determine number of queries (batch support)
    first_res = next(iter(results_per_coll.values()))["result"]
    num_queries = len(first_res.get("ids", []))

    # Decide which fields will be in the final output
    output_keys = ["ids", "collections"]
    for key in ["distances", "documents", "metadatas", "embeddings"]:
        if key in first_res and first_res.get(key) is not None:
            output_keys.append(key)

    final_result: Dict[str, List] = {k: [] for k in output_keys}

    for q_idx in range(num_queries):
        doc_scores: Dict[tuple, float] = defaultdict(float)
        hit_data: Dict[tuple, Dict[str, Any]] = {}

        for i, coll_data in results_per_coll.items():
            res = coll_data["result"]
            weight = coll_data["weight"]

            if not res.get("ids") or len(res["ids"]) <= q_idx:
                continue

            ids_list = res["ids"][q_idx]

            # Safely get parallel lists (handle missing include fields)
            def _get_list(key: str, default_len: int) -> List[Any]:
                if key in res and res[key] is not None and len(res[key]) > q_idx:
                    return res[key][q_idx]
                return [None] * default_len

            distances_list = _get_list("distances", len(ids_list))
            documents_list = _get_list("documents", len(ids_list))
            metadatas_list = _get_list("metadatas", len(ids_list))
            embeddings_list = _get_list("embeddings", len(ids_list))

            for rank, doc_id in enumerate(ids_list):
                key = (i, doc_id)
                rank_1based = rank + 1
                doc_scores[key] += weight / (rrf_k + rank_1based)

                if key not in hit_data:
                    hit_data[key] = {
                        "id": doc_id,
                        "collection_idx": i,
                        "collection": collections[i].name,
                        "distance": distances_list[rank] if rank < len(distances_list) else None,
                        "document": documents_list[rank] if rank < len(documents_list) else None,
                        "metadata": metadatas_list[rank] if rank < len(metadatas_list) else None,
                        "embedding": embeddings_list[rank] if rank < len(embeddings_list) else None,
                    }

        # Sort by weighted RRF score (descending)
        sorted_keys = sorted(
            doc_scores.keys(),
            key=lambda k: doc_scores[k],
            reverse=True,
        ) #[:n_results]

        # Build output lists for this query
        q_ids: List[str] = []
        q_collections: List[str] = []
        q_distances: Optional[List] = [] if "distances" in output_keys else None
        q_documents: Optional[List] = [] if "documents" in output_keys else None
        q_metadatas: Optional[List] = [] if "metadatas" in output_keys else None
        q_embeddings: Optional[List] = [] if "embeddings" in output_keys else None

        for key in sorted_keys:
            hit = hit_data[key]
            q_ids.append(hit["id"])
            q_collections.append(hit["collection"])

            if q_distances is not None:
                q_distances.append(hit["distance"])
            if q_documents is not None:
                q_documents.append(hit["document"])
            if q_metadatas is not None:
                q_metadatas.append(hit["metadata"])
            if q_embeddings is not None:
                q_embeddings.append(hit["embedding"])

        final_result["ids"].append(q_ids)
        final_result["collections"].append(q_collections)

        if q_distances is not None:
            final_result["distances"].append(q_distances)
        if q_documents is not None:
            final_result["documents"].append(q_documents)
        if q_metadatas is not None:
            final_result["metadatas"].append(q_metadatas)
        if q_embeddings is not None:
            final_result["embeddings"].append(q_embeddings)

    return final_result



