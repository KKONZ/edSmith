from __future__ import annotations

import spacy

# NER is unused by any tool in this package — disable it for speed.
_DISABLED = ["ner"]
_nlp: spacy.language.Language | None = None


def get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=_DISABLED)
    return _nlp
