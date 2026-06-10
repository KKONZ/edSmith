from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from edsmith.data.parser import COMPONENT_HEADINGS, split_evaluation


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

    One document per essay; all evaluation sections are stored as metadata
    (eval_task_response, eval_coherence, eval_lexical, eval_grammar, etc.).
    Retrieval queries by essay similarity then pulls the requested component
    feedback from the result metadata.
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
    # Build from DataFrame (matches Colab build_semantic_memory pattern)
    # ------------------------------------------------------------------

    def build_train_from_df(self, df: pd.DataFrame, batch_size: int = 100) -> None:
        """Populate the training collection from a raw IELTS DataFrame."""
        _build_semantic_memory(df, self._train, id_prefix="train_", batch_size=batch_size)

    def build_test_from_df(self, df: pd.DataFrame, batch_size: int = 100) -> None:
        """Populate the test collection from a raw IELTS DataFrame."""
        _build_semantic_memory(df, self._test, id_prefix="test_", batch_size=batch_size)

    # ------------------------------------------------------------------
    # Write (individual examples, used by orchestrator after Phase 1)
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
        )

        candidates: list[SemanticExample] = []
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id == exclude_id:
                continue
            meta = results["metadatas"][0][i]
            feedback_text = meta.get(f"eval_{component}", "")
            raw_score = meta.get("score", -1)
            candidates.append(
                SemanticExample(
                    id=doc_id,
                    essay=results["documents"][0][i],
                    question=meta.get("question", ""),
                    component=component,
                    feedback_text=feedback_text,
                    score=float(raw_score) if raw_score >= 0 else None,
                    band=float(meta.get("band", 0)),
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
# Build helper — mirrors Colab build_semantic_memory verbatim
# ------------------------------------------------------------------

def _build_semantic_memory(
    df: pd.DataFrame,
    collection,
    id_prefix: str = "",
    additional_metadatas: dict | None = None,
    dynamic_metadatas_from_cols: list[str] | None = None,
    batch_size: int = 100,
) -> None:
    ids = []
    documents = []
    metadatas = []

    if dynamic_metadatas_from_cols is None:
        dynamic_metadatas_from_cols = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        current_metadatas: dict = {}
        if additional_metadatas:
            current_metadatas.update(additional_metadatas)

        for meta_key in dynamic_metadatas_from_cols:
            if meta_key in row and pd.notna(row[meta_key]):
                current_metadatas[meta_key] = str(row[meta_key])

        if "evaluation" in row and pd.notna(row["evaluation"]):
            parsed_eval_sections = split_evaluation(str(row["evaluation"]))
            for section_name, section_text in parsed_eval_sections.items():
                current_metadatas[f"eval_{section_name}"] = section_text

        content_column = "essay"
        if content_column not in row or pd.isna(row[content_column]):
            continue
        document_content = str(row[content_column])

        # Store common lookup fields as top-level metadata
        if "question" in row and pd.notna(row["question"]):
            current_metadatas["question"] = str(row["question"])
        if "band" in row and pd.notna(row["band"]):
            current_metadatas["band"] = float(row["band"])
            current_metadatas["score"] = float(row["band"])

        ids.append(f"{id_prefix}{idx}")
        documents.append(document_content)
        metadatas.append(current_metadatas)

        if len(ids) >= batch_size:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            ids = []
            documents = []
            metadatas = []

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)


# ------------------------------------------------------------------
# Helpers for add_train / add_test (individual SemanticExample path)
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
        f"eval_{ex.component}": ex.feedback_text,
        "band": ex.band,
        "score": ex.score if ex.score is not None else -1.0,
    }
