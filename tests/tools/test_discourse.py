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
                    "connective_tokens", "pronoun_count", "repetition_rate",
                    "has_intro_marker", "has_conclusion_marker"):
            assert key in detail

    def test_intro_marker_detected(self):
        result = discourse_analysis(_ESSAY)
        assert result["details"][0]["has_intro_marker"] is True

    def test_conclusion_marker_detected(self):
        result = discourse_analysis(_ESSAY)
        assert result["details"][-1]["has_conclusion_marker"] is True

    def test_wordlist_transitions_found(self):
        result = discourse_analysis(_ESSAY)
        assert result["stats"]["total_transitions_wordlist"] > 0

    def test_pos_connectives_found(self):
        result = discourse_analysis(_ESSAY)
        assert result["stats"]["total_connectives_pos"] > 0

    def test_wordlist_coverage_ratio_present(self):
        result = discourse_analysis(_ESSAY)
        ratio = result["stats"]["wordlist_coverage_ratio"]
        assert 0.0 <= ratio <= 1.0

    def test_connective_tokens_shape(self):
        result = discourse_analysis(_ESSAY)
        all_connectives = [c for d in result["details"] for c in d["connective_tokens"]]
        assert len(all_connectives) > 0
        for c in all_connectives:
            assert "word" in c
            assert "pos" in c
            assert "in_wordlist" in c
            assert c["pos"] in ("SCONJ", "CCONJ")

    def test_known_wordlist_word_marked_covered(self):
        result = discourse_analysis("However, this is not the case. Although it seems so.")
        all_connectives = [c for d in result["details"] for c in d["connective_tokens"]]
        covered = {c["word"] for c in all_connectives if c["in_wordlist"]}
        assert len(covered) > 0

    def test_stats_keys(self):
        result = discourse_analysis(_ESSAY)
        for key in ("paragraph_count", "total_transitions_wordlist", "total_connectives_pos",
                    "wordlist_coverage_ratio", "pronoun_ratio", "lexical_repetition_rate"):
            assert key in result["stats"]

    def test_repetition_higher_for_repetitive_text(self):
        rep = discourse_analysis(
            "The cat is good. The cat is nice.\n\nThe cat is great. The cat is fine."
        )["stats"]["lexical_repetition_rate"]
        var = discourse_analysis(
            "The cat sat quietly.\n\nShe explored the garden with curiosity."
        )["stats"]["lexical_repetition_rate"]
        assert rep > var

    def test_empty_string(self):
        result = discourse_analysis("")
        assert result["count"] == 0
        assert result["stats"] == {}
