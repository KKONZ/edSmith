import pytest


@pytest.fixture(scope="session")
def _blank_nlp():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


@pytest.fixture(autouse=True)
def stub_spacy_nlp(_blank_nlp, monkeypatch):
    """Patch the spaCy model cache with a blank+sentencizer model to avoid loading en_core_web_sm."""
    import edsmith.tools._spacy as _spacy_mod
    monkeypatch.setattr(_spacy_mod, "_nlp", _blank_nlp)
