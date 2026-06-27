from __future__ import annotations

import pandas as pd
import spacy

from edsmith.tools import ToolResult
from edsmith.tools._spacy import get_nlp
from edsmith.tools.aoa import _aoa_entry

_SUBORD_DEPS = {"advcl", "relcl", "acl", "ccomp", "xcomp"}
_PASSIVE_DEPS = {"nsubjpass", "auxpass"}
_NOMINALIZATION_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism")


def _dep_depth(token: spacy.tokens.Token) -> int:
    depth = 0
    current = token
    seen = set()
    while current.head is not current:
        if current.i in seen:
            break  # cycle guard
        seen.add(current.i)
        current = current.head
        depth += 1
    return depth


def complexity_stats(text: str) -> ToolResult:
    if not text.strip():
        return ToolResult(
            tool="complexity", count=0, stats={}, details=[], summary="No text provided"
        )

    doc = get_nlp()(text)
    sentences = list(doc.sents)

    all_aoa: list[float] = []
    nominalization_count = 0
    total_content = 0
    details = []

    for sent in sentences:
        content = [t for t in sent if not t.is_space and not t.is_punct]
        max_depth = max((_dep_depth(t) for t in content), default=0)
        is_passive = any(t.dep_ in _PASSIVE_DEPS for t in sent)
        has_subordinate = any(t.dep_ in _SUBORD_DEPS for t in sent)

        sent_aoa: list[float] = []
        sent_nsyll: list[int] = []
        sent_nom = 0
        for t in content:
            if t.pos_ == "NOUN" and t.lower_.endswith(_NOMINALIZATION_SUFFIXES):
                sent_nom += 1
            lookup_form = t.lemma_.lower() if t.lemma_ else t.lower_
            entry = _aoa_entry(lookup_form) or _aoa_entry(t.lower_)
            if entry:
                sent_aoa.append(entry["aoa"])
                if entry.get("nsyll") is not None:
                    sent_nsyll.append(entry["nsyll"])

        nominalization_count += sent_nom
        total_content += len(content)
        all_aoa.extend(sent_aoa)

        details.append({
            "length": len(content),
            "dep_depth": max_depth,
            "is_passive": is_passive,
            "has_subordinate": has_subordinate,
            "nominalization_count": sent_nom,
            "mean_aoa": sum(sent_aoa) / len(sent_aoa) if sent_aoa else None,
            "mean_nsyll": sum(sent_nsyll) / len(sent_nsyll) if sent_nsyll else None,
        })

    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    words = [t.lower_ for t in tokens]
    ttr = len(set(words)) / len(words) if words else 0.0

    sent_lens = pd.Series([d["length"] for d in details])
    dep_depths = pd.Series([d["dep_depth"] for d in details])
    passive_ratio = sum(1 for d in details if d["is_passive"]) / len(details) if details else 0.0
    subordinate_ratio = sum(1 for d in details if d["has_subordinate"]) / len(details) if details else 0.0
    nominalization_ratio = nominalization_count / total_content if total_content else 0.0

    stats: dict[str, float] = {
        "sentence_count": float(len(sentences)),
        "ttr": ttr,
        "sent_len_mean": sent_lens.mean(),
        "sent_len_std": sent_lens.std(),
        "dep_depth_mean": dep_depths.mean(),
        "dep_depth_std": dep_depths.std(),
        "dep_depth_kurtosis": dep_depths.kurtosis(),
        "passive_ratio": passive_ratio,
        "subordinate_ratio": subordinate_ratio,
        "nominalization_ratio": nominalization_ratio,
    }

    if all_aoa:
        aoa_s = pd.Series(all_aoa)
        stats.update({
            "aoa_mean": aoa_s.mean(),
            "aoa_std": aoa_s.std(),
            "aoa_skew": aoa_s.skew(),
            "aoa_kurtosis": aoa_s.kurtosis(),
            "pct_late": float((aoa_s >= 10).mean() * 100),
        })

    return ToolResult(
        tool="complexity",
        count=len(sentences),
        stats=stats,
        details=[],
        summary=(
            f"{len(sentences)} sentence(s); "
            f"avg length {stats['sent_len_mean']:.1f} words; "
            f"TTR {ttr:.2f}; passive {passive_ratio:.0%}; "
            f"subordinate {subordinate_ratio:.0%}; "
            f"nominalization {nominalization_ratio:.0%}"
        ),
    )
