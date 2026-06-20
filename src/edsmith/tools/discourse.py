from __future__ import annotations

import re

from edsmith.tools import ToolResult
from edsmith.tools._spacy import get_nlp

# ---------------------------------------------------------------------------
# Transition word lexicon by category.
# NOTE: This list is intentionally non-exhaustive — it captures common
# examples in each category. The POS-based connective cross-reference below
# surfaces additional connectives the wordlist does not enumerate.
# ---------------------------------------------------------------------------

TRANSITION_WORDS: dict[str, list[str]] = {
    "additive": [
        "furthermore", "moreover", "in addition", "additionally", "also",
        "besides", "likewise", "similarly", "equally", "what is more",
        "not only", "as well as", "not to mention", "coupled with",
        "on top of that",
    ],
    "adversative": [
        "however", "nevertheless", "nonetheless", "on the other hand",
        "in contrast", "although", "though", "yet", "despite",
        "in spite of", "even though", "whereas", "while", "conversely",
        "on the contrary", "admittedly", "granted", "that said", "even so",
        "all the same",
    ],
    "causal": [
        "therefore", "consequently", "as a result", "thus", "hence",
        "because", "since", "due to", "owing to", "for this reason",
        "as a consequence", "accordingly", "as such", "given that",
        "it follows that",
    ],
    "sequential": [
        "firstly", "secondly", "thirdly", "finally", "subsequently",
        "then", "next", "afterwards", "to begin with", "to start with",
        "last but not least", "initially", "prior to", "following this",
        "at the same time", "simultaneously",
    ],
    "exemplification": [
        "for example", "for instance", "such as", "namely", "specifically",
        "in particular", "to illustrate", "as an illustration",
        "as a case in point",
    ],
    "conclusion": [
        "in conclusion", "to conclude", "to summarize", "in summary",
        "overall", "in brief", "to sum up", "all in all", "in short",
        "ultimately", "on balance", "by and large",
    ],
    "hedging": [
        "arguably", "it could be argued", "one might suggest",
        "it is possible that", "it seems that", "it appears that",
        "to some extent", "in many cases",
    ],
}

_INTRO_MARKERS = [
    "in recent years", "nowadays", "it is widely believed",
    "it is often argued", "many people believe", "it has been suggested",
    "there is growing", "this essay", "this report",
]

_CONCLUSION_MARKERS = list(TRANSITION_WORDS["conclusion"])

# Flat set of single-word entries for POS cross-reference
_WORDLIST_SINGLE_WORDS: set[str] = {
    phrase for phrases in TRANSITION_WORDS.values()
    for phrase in phrases
    if " " not in phrase
}

# SpaCy POS tags that indicate connective / discourse-linking words
_CONNECTIVE_POS = {"SCONJ", "CCONJ"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras or [text.strip()]


def _para_spans(text: str, paras: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for para in paras:
        idx = text.find(para, pos)
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(para)))
        pos = idx + len(para)
    return spans


def _find_para(char_pos: int, spans: list[tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(spans):
        if start <= char_pos < end:
            return i
    return len(spans) - 1


def _detect_transitions(text_lower: str) -> dict[str, list[str]]:
    return {
        cat: [p for p in phrases if p in text_lower]
        for cat, phrases in TRANSITION_WORDS.items()
        if any(p in text_lower for p in phrases)
    }

# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

def discourse_analysis(text: str) -> ToolResult:
    """Analyse essay discourse structure.

    Transition word detection uses a non-exhaustive example list organised
    by category. The connective_tokens field cross-references this with
    spaCy POS tags (SCONJ/CCONJ), surfacing connective words the wordlist
    does not enumerate.
    """
    if not text.strip():
        return ToolResult(
            tool="discourse", count=0, stats={}, details=[], summary="No text provided"
        )

    nlp = get_nlp()
    doc = nlp(text)

    paras = _split_paragraphs(text)
    spans = _para_spans(text, paras)

    # Group tokens by paragraph
    para_sents: list[list] = [[] for _ in paras]
    para_tokens: list[list] = [[] for _ in paras]
    for sent in doc.sents:
        para_sents[_find_para(sent.start_char, spans)].append(sent)
    for token in doc:
        if not token.is_space and not token.is_punct:
            para_tokens[_find_para(token.idx, spans)].append(token)

    all_content_lemmas: list[str] = []
    details = []

    for i, para in enumerate(paras):
        role = "introduction" if i == 0 else ("conclusion" if i == len(paras) - 1 else "body")
        para_lower = para.lower()

        transitions = _detect_transitions(para_lower)
        has_intro_marker = i == 0 and any(m in para_lower for m in _INTRO_MARKERS)
        has_conclusion_marker = (
            i == len(paras) - 1 and any(m in para_lower for m in _CONCLUSION_MARKERS)
        )

        content = [t for t in para_tokens[i] if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
        content_lemmas = [t.lemma_.lower() for t in content]
        pronouns = [t for t in para_tokens[i] if t.pos_ == "PRON"]

        # POS-detected connectives, cross-referenced against wordlist
        connective_tokens = [
            {
                "word": t.lower_,
                "pos": t.pos_,
                "in_wordlist": t.lower_ in _WORDLIST_SINGLE_WORDS,
            }
            for t in para_tokens[i]
            if t.pos_ in _CONNECTIVE_POS
        ]

        # Cross-paragraph content repetition
        prev_set = set(all_content_lemmas)
        rep_count = sum(1 for l in content_lemmas if l in prev_set)
        repetition_rate = rep_count / len(content_lemmas) if content_lemmas else 0.0
        all_content_lemmas.extend(content_lemmas)

        details.append({
            "index": i,
            "role": role,
            "sentence_count": len(para_sents[i]),
            "transitions": transitions,
            "connective_tokens": connective_tokens,
            "pronoun_count": len(pronouns),
            "repetition_rate": repetition_rate,
            "has_intro_marker": has_intro_marker,
            "has_conclusion_marker": has_conclusion_marker,
        })

    # Global stats
    all_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    pronoun_ratio = sum(1 for t in all_tokens if t.pos_ == "PRON") / len(all_tokens) if all_tokens else 0.0

    content_all = [t for t in all_tokens if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lemma_counts: dict[str, int] = {}
    for t in content_all:
        lemma_counts[t.lemma_.lower()] = lemma_counts.get(t.lemma_.lower(), 0) + 1
    repeated_tokens = sum(c for c in lemma_counts.values() if c > 1)
    repetition_rate_global = repeated_tokens / len(content_all) if content_all else 0.0

    # Aggregate transition stats
    transition_totals = {cat: 0 for cat in TRANSITION_WORDS}
    for d in details:
        for cat, hits in d["transitions"].items():
            transition_totals[cat] += len(hits)
    total_transitions_wordlist = sum(transition_totals.values())

    # Aggregate POS connective stats
    all_connectives = [c for d in details for c in d["connective_tokens"]]
    total_connectives_pos = len(all_connectives)
    uncategorized = [c["word"] for c in all_connectives if not c["in_wordlist"]]
    wordlist_coverage = (
        total_transitions_wordlist / total_connectives_pos
        if total_connectives_pos > 0 else 0.0
    )

    stats: dict[str, float] = {
        "paragraph_count": float(len(paras)),
        "has_introduction_marker": float(details[0]["has_intro_marker"]),
        "has_conclusion_marker": float(details[-1]["has_conclusion_marker"]),
        "total_transitions_wordlist": float(total_transitions_wordlist),
        "total_connectives_pos": float(total_connectives_pos),
        "wordlist_coverage_ratio": wordlist_coverage,
        "pronoun_ratio": pronoun_ratio,
        "lexical_repetition_rate": repetition_rate_global,
        **{f"transitions_{cat}": float(n) for cat, n in transition_totals.items()},
    }

    return ToolResult(
        tool="discourse",
        count=len(paras),
        stats=stats,
        details=details,
        summary=(
            f"{len(paras)} paragraph(s); "
            f"{total_transitions_wordlist} wordlist transition(s) "
            f"({total_connectives_pos} total connectives by POS, "
            f"{len(uncategorized)} not in wordlist: {', '.join(sorted(set(uncategorized))) or 'none'}); "
            f"pronoun ratio {pronoun_ratio:.2f}; "
            f"lexical repetition {repetition_rate_global:.2f}"
        ),
    )
