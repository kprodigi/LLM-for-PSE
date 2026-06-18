# -*- coding: utf-8 -*-
"""
Optional DENSE (embedding) retrieval channel for the RAG layer.

This makes the framework's "dense" retrieval claim literally true: a real
embedding model produces vector representations that are retrieved by cosine
similarity, complementing the lexical (TF-IDF / BM25) channel.

Design notes
------------
* Pluggable backend, selected by EMBED_BACKEND (default "fastembed"):
    - "fastembed" : ONNX embeddings (no torch); offline after a one-time model
                    download. Reproducible (pinned model id).
    - "voyage"    : hosted Voyage embeddings (needs VOYAGE_API_KEY + internet).
    - "none"/off  : disable -> lexical-only (zero new dependencies).
* Fails SOFT: if the chosen backend cannot load, the dense channel is simply
  unavailable and the caller falls back to lexical retrieval. The dense channel
  is an UPGRADE, never a hard requirement, so the lexical study still runs.
* No vector DB: the curated corpora are tens of chunks, so an in-memory matrix
  with a single cosine product is exact and faster than any index.
* The ACTIVE backend string is exposed (`backend_name`) so the manuscript and
  the run manifest can state exactly what produced the dense channel.
"""
from __future__ import annotations

import os

import numpy as np

# Pinned default model: small, CPU-friendly, 384-dim, widely supported by fastembed.
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _l2norm(x):
    """Row-wise L2 normalization so a dot product equals cosine similarity."""
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _load_backend(backend, model_name):
    """Return (backend_name, embed_fn) where embed_fn: list[str] -> ndarray (n, d).

    Never raises: an unavailable backend returns (reason_string, None) so the
    dense channel degrades to lexical-only instead of breaking retrieval.
    """
    choice = (backend or os.environ.get("EMBED_BACKEND", "fastembed")).strip().lower()
    if choice in ("none", "off", ""):
        return "none (lexical-only)", None

    if choice == "fastembed":
        try:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name=model_name)

            def embed(texts):
                return np.array(list(model.embed(list(texts))), dtype=float)

            return f"fastembed:{model_name}", embed
        except Exception as exc:  # missing dep, download failure, etc.
            return f"none (fastembed unavailable: {exc})", None

    if choice == "voyage":
        try:
            import voyageai
            client = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
            vmodel = os.environ.get("VOYAGE_MODEL", "voyage-3")

            def embed(texts):
                resp = client.embed(list(texts), model=vmodel, input_type="document")
                return np.array(resp.embeddings, dtype=float)

            return f"voyage:{vmodel}", embed
        except Exception as exc:
            return f"none (voyage unavailable: {exc})", None

    return f"none (unknown EMBED_BACKEND={choice!r})", None


class DenseIndex:
    """In-memory dense retriever over a small curated corpus.

    Embeds the corpus once at construction; `query` embeds the query and returns
    the best-matching corpus indices by cosine similarity. If no embedding
    backend is available, `available` is False and `query` returns [] so the
    caller transparently falls back to its lexical channel.
    """

    def __init__(self, texts, backend=None, model_name=None):
        self.texts = list(texts)
        self.model_name = model_name or os.environ.get("EMBED_MODEL", _DEFAULT_MODEL)
        self.backend_name, self._embed = _load_backend(backend, self.model_name)
        if self._embed is not None and self.texts:
            try:
                self.mat = _l2norm(self._embed(self.texts))  # (n, d), unit rows
            except Exception as exc:  # embedding the corpus failed -> degrade
                self.backend_name = f"none (corpus embed failed: {exc})"
                self.mat = None
        else:
            self.mat = None

    @property
    def available(self):
        return self.mat is not None

    def query(self, text, top_k=4):
        """Return [(corpus_index, cosine_score)] best-first, or [] if no backend."""
        if not self.available:
            return []
        q = _l2norm(self._embed([text]))          # (1, d)
        sims = self.mat @ q[0]                     # (n,) cosine (rows are unit-norm)
        order = np.argsort(sims)[::-1][: int(top_k)]
        return [(int(i), float(sims[i])) for i in order]


# Pinned default reranker: a small CPU cross-encoder, ONNX (no torch), via fastembed.
# jina-reranker-v1-tiny-en is fast and discriminates the terse technical passages from
# the admin/maintenance distractors better than the generic MS-MARCO model; whichever
# model is chosen, it is only ENABLED for a run after the held-out guard confirms it does
# not degrade retrieval (see the reranker_guard helpers in the retrieval-eval modules).
_DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-tiny-en"


class CrossEncoderReranker:
    """Optional cross-encoder reranker (fastembed TextCrossEncoder).

    A cross-encoder scores each (query, passage) pair jointly, so it re-ranks a
    fused candidate pool more precisely than the bag-of-words / bi-encoder channels.
    Fails SOFT exactly like DenseIndex: if the model cannot load, `available` is
    False and `rerank` returns None, so retrieval transparently keeps the fused
    ranking. Enabled by the caller (RAG_RERANK=1); model id overridable via
    EMBED_RERANK_MODEL. The active backend string is exposed for the run manifest.
    """

    def __init__(self, model_name=None):
        self.model_name = (model_name
                           or os.environ.get("EMBED_RERANK_MODEL", _DEFAULT_RERANK_MODEL))
        self._enc = None
        self.backend_name = "off"
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._enc = TextCrossEncoder(model_name=self.model_name)
            self.backend_name = f"fastembed-rerank:{self.model_name}"
        except Exception as exc:  # missing dep, download failure, etc.
            self.backend_name = f"off (reranker unavailable: {exc})"

    @property
    def available(self):
        return self._enc is not None

    def rerank(self, query, docs):
        """Return a cross-encoder relevance score per doc (higher = better), or
        None if the reranker is unavailable / errored (caller keeps fused order)."""
        if not self.available:
            return None
        try:
            return [float(s) for s in self._enc.rerank(str(query), [str(d) for d in docs])]
        except Exception:
            return None
