from __future__ import annotations

import spacy

from edsmith.tools import ToolResult

_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _dep_depth(token: spacy.tokens.Token) -> int:
    depth = 0
    current = token
    while current.head is not current:
        current = current.head
        depth += 1
    return depth


def complexity_stats(text: str) -> ToolResult:
    if not text.strip():
        return ToolResult(tool="complexity", count=0, details=[], summary="No text provided")

    doc = _get_nlp()(text)
    sentences = list(doc.sents)
    content_tokens = [t for t in doc if not t.is_space and not t.is_punct]

    details = []
    for sent in sentences:
        sent_tokens = [t for t in sent if not t.is_space and not t.is_punct]
        max_depth = max((_dep_depth(t) for t in sent_tokens), default=0)
        details.append({"length": len(sent_tokens), "dep_depth": max_depth})

    words = [t.lower_ for t in content_tokens]
    ttr = len(set(words)) / len(words) if words else 0.0
    avg_len = sum(d["length"] for d in details) / len(details) if details else 0.0
    avg_depth = sum(d["dep_depth"] for d in details) / len(details) if details else 0.0

    return ToolResult(
        tool="complexity",
        count=len(sentences),
        details=details,
        summary=(
            f"{len(sentences)} sentence(s); avg length {avg_len:.1f} words; "
            f"TTR {ttr:.2f}; avg dep depth {avg_depth:.1f}"
        ),
    )
