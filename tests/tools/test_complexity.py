import pytest

pytest.importorskip("spacy")

from edsmith.tools.complexity import complexity_stats


class TestComplexityStats:
    def test_result_shape(self):
        result = complexity_stats("The cat sat on the mat. It was a sunny day.")
        assert set(result.keys()) == {"tool", "count", "stats", "details", "summary"}
        assert result["tool"] == "complexity"

    def test_sentence_count(self):
        result = complexity_stats("First sentence. Second sentence. Third sentence.")
        assert result["count"] == 3
        assert len(result["details"]) == 3

    def test_details_keys(self):
        result = complexity_stats("The cat sat on the mat.")
        detail = result["details"][0]
        for key in ("length", "dep_depth", "is_passive", "has_subordinate",
                    "nominalization_count", "mean_aoa", "mean_nsyll"):
            assert key in detail

    def test_stats_keys(self):
        result = complexity_stats("The cat sat on the mat. It was sunny.")
        for key in ("ttr", "sent_len_mean", "dep_depth_mean",
                    "passive_ratio", "subordinate_ratio", "nominalization_ratio"):
            assert key in result["stats"]

    def test_passive_detected(self):
        result = complexity_stats("The paper was written by the student.")
        assert result["stats"]["passive_ratio"] > 0
        assert result["details"][0]["is_passive"] is True

    def test_active_not_passive(self):
        result = complexity_stats("The student wrote the paper.")
        assert result["details"][0]["is_passive"] is False

    def test_subordinate_detected(self):
        result = complexity_stats("Although it was raining, she went outside.")
        assert result["stats"]["subordinate_ratio"] > 0
        assert result["details"][0]["has_subordinate"] is True

    def test_nominalization_detected(self):
        result = complexity_stats("The government made a consideration of the situation.")
        assert result["stats"]["nominalization_ratio"] > 0

    def test_aoa_populated_in_stats(self):
        result = complexity_stats("The cat sat on the mat. It was a sunny day.")
        assert "aoa_mean" in result["stats"]
        assert result["stats"]["aoa_mean"] > 0

    def test_diverse_text_higher_ttr(self):
        rep = complexity_stats("The cat sat. The cat ran. The cat slept.")["stats"]["ttr"]
        div = complexity_stats("Urbanisation accelerates productivity. Biodiversity requires intervention.")["stats"]["ttr"]
        assert div > rep

    def test_empty_string(self):
        result = complexity_stats("")
        assert result["count"] == 0
        assert result["stats"] == {}
