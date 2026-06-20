import pytest

pytest.importorskip("spacy")

from edsmith.tools.discourse import discourse_analysis

_ESSAY = """In recent years, the rise of social media has transformed communication.
Many people believe this is largely a positive development.

However, there are significant drawbacks to consider. For example, misinformation spreads rapidly online.
Furthermore, excessive use can harm mental health.

In conclusion, while social media offers benefits, its risks must be carefully managed."""


class TestDiscourseAnalysis:
    def test_result_shape(self):
        result = discourse_analysis(_ESSAY)
        assert set(result.keys()) == {"tool", "count", "stats", "details", "summary"}
        assert result["tool"] == "discourse"

    def test_paragraph_count(self):
        result = discourse_analysis(_ESSAY)
        assert result["count"] == 3

    def test_roles_assigned(self):
        result = discourse_analysis(_ESSAY)
        roles = [d["role"] for d in result["details"]]
        assert roles[0] == "introduction"
        assert roles[-1] == "conclusion"
        assert all(r == "body" for r in roles[1:-1])

    def test_details_keys(self):
        result = discourse_analysis(_ESSAY)
        detail = result["details"][0]
        for key in ("index", "role", "sentence_count", "transitions",
                    "pronoun_count", "repetition_rate",
                    "has_intro_marker", "has_conclusion_marker"):
            assert key in detail

    def test_intro_marker_detected(self):
        result = discourse_analysis(_ESSAY)
        assert result["details"][0]["has_intro_marker"] is True

    def test_conclusion_marker_detected(self):
        result = discourse_analysis(_ESSAY)
        assert result["details"][-1]["has_conclusion_marker"] is True

    def test_transition_words_found(self):
        result = discourse_analysis(_ESSAY)
        assert result["stats"]["total_transitions"] > 0

    def test_adversative_transitions(self):
        result = discourse_analysis(_ESSAY)
        assert result["stats"]["transitions_adversative"] > 0

    def test_exemplification_transitions(self):
        result = discourse_analysis(_ESSAY)
        assert result["stats"]["transitions_exemplification"] > 0

    def test_stats_keys(self):
        result = discourse_analysis(_ESSAY)
        for key in ("paragraph_count", "total_transitions", "pronoun_ratio",
                    "lexical_repetition_rate", "has_introduction_marker",
                    "has_conclusion_marker"):
            assert key in result["stats"]

    def test_repetition_higher_for_repetitive_text(self):
        repetitive = "The cat is good. The cat is nice.\n\nThe cat is great. The cat is fine."
        varied = "The cat sat quietly.\n\nShe explored the garden with curiosity."
        rep = discourse_analysis(repetitive)["stats"]["lexical_repetition_rate"]
        var = discourse_analysis(varied)["stats"]["lexical_repetition_rate"]
        assert rep > var

    def test_empty_string(self):
        result = discourse_analysis("")
        assert result["count"] == 0
        assert result["stats"] == {}
