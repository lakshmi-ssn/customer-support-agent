"""Searching the handbook.

Given a customer's question, find the few pieces of the handbook most likely to
answer it. Whatever this returns is all your agent will know, so if the right
piece is missing, no amount of clever prompting later can save the answer.

One kind of search already works: `dense`, which searches by meaning. TODO 2 adds
the rest. You wrote all of it in the Lecture 6 notebook, so this is mostly copying
your own code across and then measuring whether it actually helps here.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from . import config, index, llm


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict


# --------------------------------------------------------------------------- #
# tokenizing for BM25
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text):
    """Lowercase + split into tokens for BM25.

    Handles the ERR-4021 vs ERR 4021 problem: a hyphenated code like
    "ERR-4021" is kept whole as one token ("err-4021"), but we ALSO emit its
    parts ("err", "4021") and the squashed form ("err4021"). That way a query
    typed as "ERR 4021" (two plain tokens: "err", "4021") still overlaps with
    a corpus token that was written "ERR-4021", because both produce "err"
    and "4021" as tokens.
    """
    tokens = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if "-" in match:
            parts = match.split("-")
            tokens.extend(parts)
            tokens.append("".join(parts))
    return tokens


class Retriever:
    def __init__(self, mode=None, top_k=None):
        self.mode = mode or config.RETRIEVAL_MODE
        self.top_k = top_k or config.TOP_K
        self.collection, self.chunks = index.load()
        self._bm25 = None

    def _hit(self, chunk, score):
        return Hit(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, title=chunk.title,
                   text=chunk.text, score=score, metadata=dict(chunk.metadata))

    # ---------------------------------------------------------------- dense --
    def dense(self, query, k=None):
        """Embedding search over the Chroma index."""
        k = k or config.CANDIDATE_K
        res = self.collection.query(query_texts=[query], n_results=min(k, len(self.chunks)))
        hits = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(chunk_id=cid, doc_id=meta.get("doc_id", cid.split("#")[0]),
                            title=meta.get("title", ""), text=text,
                            score=1.0 / (1.0 + float(dist)), metadata=dict(meta)))
        return hits

    # -------------------------------------------------------------- lexical --
    def lexical(self, query, k=None):
        """BM25 keyword search.

        `dense` searches by meaning; this searches by exact/overlapping
        tokens, which is what wins for things like error codes that have no
        real "meaning" for an embedding to latch onto.
        """
        k = k or config.CANDIDATE_K

        if self._bm25 is None:
            corpus_tokens = [_tokenize(c.text) for c in self.chunks]
            self._bm25 = BM25Okapi(corpus_tokens)

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            hits.append(self._hit(self.chunks[i], float(scores[i])))
        return hits

    def rrf(self, *rankings, k=None, top=None):
        """Reciprocal Rank Fusion — merge any number of ranked Hit lists.

        Scores from different search methods aren't comparable (dense might
        say 0.4, BM25 might say 12.7), so we throw scores away and combine by
        *rank position* instead: a hit ranked r in a list contributes
        1/(k + r). Contributions from every list a hit appears in are summed,
        so a hit that shows up near the top of two lists beats one that's #1
        in only one list.
        """
        k = k or config.RRF_K

        fused_scores = defaultdict(float)
        hit_by_id = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                fused_scores[hit.chunk_id] += 1.0 / (k + rank)
                # Keep the first-seen Hit object for each id as the representative.
                hit_by_id.setdefault(hit.chunk_id, hit)

        merged = []
        for chunk_id, score in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True):
            original = hit_by_id[chunk_id]
            merged.append(Hit(chunk_id=original.chunk_id, doc_id=original.doc_id,
                              title=original.title, text=original.text,
                              score=score, metadata=original.metadata))

        return merged[:top] if top else merged

    def hybrid(self, query, k=None):
        """Run dense + lexical search, merge with RRF.

        Compare against dense alone with:

            RETRIEVAL_MODE=dense  python scripts/evaluate_dev.py --retrieval-only
            RETRIEVAL_MODE=hybrid python scripts/evaluate_dev.py --retrieval-only

        On a 36-chunk handbook the gain may be small or even negative — report
        the actual number rather than assuming hybrid is better.
        """
        k = k or config.CANDIDATE_K
        dense_hits = self.dense(query, k)
        lexical_hits = self.lexical(query, k)
        return self.rrf(dense_hits, lexical_hits, top=k)

    # ------------------------------------------------------------- optional --
    def rerank(self, query, candidates, top_n=None):
      
        if not candidates:
            return []
        top_n = top_n or len(candidates)
        return sorted(candidates, key=lambda h: h.score, reverse=True)[:top_n]

    def translate_query(self, query):
        """OPTIONAL — rewrite the question before searching. See docstring."""
        raise NotImplementedError("query translation is optional")

    # ----------------------------------------------------------------- entry --
    def search(self, query, k=None):
        k = k or self.top_k
        if self.mode == "dense":
            hits = self.dense(query, config.CANDIDATE_K)
        elif self.mode == "hybrid":
            hits = self.hybrid(query, config.CANDIDATE_K)
        elif self.mode == "hybrid_rerank":
            # Optional branch: keep it working without forcing an extra model call.
            hits = self.rerank(query, self.hybrid(query, config.CANDIDATE_K), top_n=k)
        else:
            raise ValueError(f"unknown RETRIEVAL_MODE {self.mode!r}")
        return self.postprocess(hits)[:k]

    def postprocess(self, hits):
        """Handle the two trap sections: archive_returns_2024 (superseded)
        and community (untrusted, contains a prompt-injection attempt).

        Strategy chosen here: DROP them from results entirely. This is the safest
        behaviour for a support agent because those sections are intentionally
        marked as untrusted in the handbook metadata and should never be used as a
        source of an answer.

        The exact section IDs are confirmed in the Handbook metadata: they are
        `community` and `archive_returns_2024`. We use both the chunk's `doc_id`
        and its metadata fallback to avoid silently dropping results if the stored
        IDs are reformatted in a future change.
        """
        bad_ids = {str(x).strip() for x in (config.UNTRUSTED_DOCS | config.SUPERSEDED_DOCS)}

        trusted = []
        for hit in hits:
            doc_id = str(hit.doc_id or hit.metadata.get("doc_id", "")).strip()
            meta_doc_id = str(hit.metadata.get("doc_id", "")).strip()
            if doc_id in bad_ids or meta_doc_id in bad_ids:
                continue
            trusted.append(hit)
        return trusted


_RETRIEVER = None


def get_retriever():
    """Process-wide singleton — loading the index per query is slow."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def format_context(hits):
    """Lay the retrieved passages out for the model.

    Each block is labelled with its section id, because that id is exactly what
    we ask the model to give back as its source. An earlier version numbered the
    blocks `[1] [2] [3]` as well, and the model dutifully cited "2" — if you show
    a model two identifiers it will sometimes pick the wrong one. One identifier,
    one meaning.
    """
    lines = []
    for h in hits:
        status = h.metadata.get("status", "current")
        flag = "" if status == "current" else f"  (WARNING: status={status})"
        lines.append(f"[section: {h.doc_id}]{flag}\n{h.title}\n{h.text.strip()}")
    return "\n\n".join(lines) if lines else "(nothing retrieved)"