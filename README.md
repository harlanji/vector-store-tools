Tools for Vector Store Portability via BSON.
Dump a ChromaDB store to BSON, close to maximally compact, and load
it into a SQLite-vec + FTS store, and use it in a way API-compatible
with Chroma.

## Storage Comparison

On disk, the binary quantized corpus of my channel transcripts is (all-MiniLM-L6-v2 / dim=384):

* bson: 91mb (packed int32 array)

* SQLite-vec + FTS: 109mb (bit[] + contentless)

* ChromaDB: 476m (floats, not sure we can't improve this, maybe int8).

* Source: https://x.com/harlanji/status/2077906175723815040

## Performance

Yet to eval hamming vs. cosine rigorously but it looks good.


Good initial result on Hamming vs. Cosine.

'yt-vid-st' is the ChromaDB collection, and 'dummy' the SQLite-vec + hamming.

Top results basically same rank, very close distance (ChromaDB is approximate, given their algorithm. Hamming is /200).

Meets expectations.

* Source: https://x.com/harlanji/status/2077935278900343164

## Design

* Public: https://x.com/i/grok/share/7a71c76da2ed4eaca94e6a9056b91db0

Covers design and use of the following commands. They also have --help (via argparse).

* chroma_bson.py - load and dump bson + chroma
* sqlite_vec_chroma_api.py - chroma-compatible API for sqlite-vec + fts
* sqlite_vec_chroma.py - chroma query CLI (only works with sqlite3 wrapper rn)
* util.py - functions for bson + int packing

## License

MIT License

Copyright 2026 Harlan J. Iverson
