from __future__ import annotations

from edsmith.tools import ToolResult


def pos_tag(text: str) -> ToolResult:
    """POS-tag text with spaCy, returning per-token annotations and sentence-level stats."""
    from edsmith.tools._spacy import get_nlp

    nlp = get_nlp()
    doc = nlp(text)

    details = []
    for token in doc:
        if token.is_space:
            continue
        details.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,       # coarse: NOUN, VERB, ADJ, ADV, …
            "tag": token.tag_,       # fine-grained: NN, VBZ, JJ, …
            "dep": token.dep_,       # syntactic dependency relation
            "is_stop": token.is_stop,
            "sentence": token.sent.text[:80],
        })

    pos_counts: dict[str, int] = {}
    for d in details:
        pos_counts[d["pos"]] = pos_counts.get(d["pos"], 0) + 1

    n_tokens = len(details)
    n_sents = len(list(doc.sents))
    stats: dict[str, float] = {
        "n_tokens": float(n_tokens),
        "n_sentences": float(n_sents),
        "tokens_per_sentence": round(n_tokens / n_sents, 2) if n_sents else 0.0,
        **{f"pct_{pos.lower()}": round(count / n_tokens * 100, 1) if n_tokens else 0.0
           for pos, count in pos_counts.items()},
    }

    top_pos = sorted(pos_counts.items(), key=lambda x: -x[1])[:5]
    summary = (
        f"{n_tokens} tokens across {n_sents} sentence(s); "
        + ", ".join(f"{pos}={cnt}" for pos, cnt in top_pos)
    )

    return ToolResult(
        tool="pos_tag",
        count=n_tokens,
        stats=stats,
        details=details,
        summary=summary,
    )
