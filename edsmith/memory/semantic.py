from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from edsmith.data.parser import COMPONENT_HEADINGS


@dataclass
class SemanticExample:
    id: str
    question: str
    essay: str
    component: str
    feedback_text: str
    score: float | None
    band: float


class SemanticMemory:
    """Local ChromaDB-backed few-shot retrieval store.

    Essays are the embedded documents — similarity search finds essays close to
    the query essay, then returns their feedback as few-shot context.

    Two collections: training (read/write) and test (read-only at inference;
    never exposed at individual-record level to the reflection stage).

    An optional cross-encoder reranker can be supplied; when present, retrieval
    fetches k * rerank_factor candidates then reranks to the top k.
    """

    def __init__(
        self,
        drive_path: str | Path,
        collection_train: str,
        collection_test: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str | None = None,
        rerank_factor: int = 3,
    ) -> None:
        persist_dir = str(Path(drive_path) / "chromadb")
        self._client = chromadb.PersistentClient(path=persist_dir)

        ef = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            model_kwargs={"trust_remote_code": True},
        )
        self._train = self._client.get_or_create_collection(collection_train, embedding_function=ef)
        self._test = self._client.get_or_create_collection(collection_test, embedding_function=ef)

        self._reranker = None
        self._rerank_factor = rerank_factor
        if reranker_model:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(reranker_model, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_train(self, examples: list[SemanticExample]) -> None:
        _upsert(self._train, examples)

    def add_test(self, examples: list[SemanticExample]) -> None:
        _upsert(self._test, examples)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_train(
        self,
        query_essay: str,
        component: str,
        k: int = 4,
        exclude_id: str | None = None,
    ) -> list[SemanticExample]:
        return self._retrieve(self._train, query_essay, component, k, exclude_id)

    def retrieve_test(
        self,
        query_essay: str,
        component: str,
        k: int = 4,
        exclude_id: str | None = None,
    ) -> list[SemanticExample]:
        return self._retrieve(self._test, query_essay, component, k, exclude_id)

    def _retrieve(
        self,
        collection,
        query_essay: str,
        component: str,
        k: int,
        exclude_id: str | None,
    ) -> list[SemanticExample]:
        if component not in COMPONENT_HEADINGS:
            raise ValueError(
                f"Unknown component: {component!r}. Must be one of {list(COMPONENT_HEADINGS)}"
            )

        fetch_n = k * self._rerank_factor + 1 if self._reranker else k + 1

        results = collection.query(
            query_texts=[query_essay],
            n_results=fetch_n,
            where={"component": component},
        )

        candidates: list[SemanticExample] = []
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id == exclude_id:
                continue
            meta = results["metadatas"][0][i]
            candidates.append(
                SemanticExample(
                    id=doc_id,
                    essay=results["documents"][0][i],
                    question=meta["question"],
                    component=meta["component"],
                    feedback_text=meta["feedback_text"],
                    score=meta["score"] if meta["score"] >= 0 else None,
                    band=meta["band"],
                )
            )

        if self._reranker and len(candidates) > k:
            pairs = [(query_essay, ex.essay) for ex in candidates]
            scores = self._reranker.predict(pairs)
            candidates = [
                ex for ex, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            ]

        return candidates[:k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def train_count(self) -> int:
        return self._train.count()

    @property
    def test_count(self) -> int:
        return self._test.count()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _upsert(collection, examples: list[SemanticExample]) -> None:
    if not examples:
        return
    collection.upsert(
        ids=[ex.id for ex in examples],
        documents=[ex.essay for ex in examples],
        metadatas=[_metadata(ex) for ex in examples],
    )


def _metadata(ex: SemanticExample) -> dict:
    return {
        "question": ex.question,
        "component": ex.component,
        "feedback_text": ex.feedback_text,
        "band": ex.band,
        "score": ex.score if ex.score is not None else -1.0,
    }
