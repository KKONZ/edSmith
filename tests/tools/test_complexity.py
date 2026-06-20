import pytest

pytest.importorskip("spacy")

from edsmith.tools.complexity import complexity_stats


class TestComplexityStats:
    def test_result_shape(self):
        result = complexity_stats("The cat sat on the mat. It was a sunny day.")
        assert set(result.keys()) == {"tool", "count", "details", "summary"}
        assert result["tool"] == "complexity"
        assert isinstance(result["count"], int)
        assert isinstance(result["details"], list)
        assert isinstance(result["summary"], str)

    def test_sentence_count(self):
        result = complexity_stats("First sentence. Second sentence. Third sentence.")
        assert result["count"] == 3
        assert len(result["details"]) == 3

    def test_details_keys(self):
        result = complexity_stats("The quick brown fox jumps over the lazy dog.")
        assert result["count"] > 0
        detail = result["details"][0]
        assert "length" in detail
        assert "dep_depth" in detail
        assert isinstance(detail["length"], int)
        assert isinstance(detail["dep_depth"], int)

    def test_diverse_text_higher_ttr(self):
        repetitive = "The cat sat. The cat ran. The cat slept."
        diverse = "Urbanisation accelerates economic productivity. Biodiversity conservation requires systematic intervention."
        r_rep = complexity_stats(repetitive)
        r_div = complexity_stats(diverse)
        # extract TTR from summary string
        ttr_rep = float(r_rep["summary"].split("TTR ")[1].split(";")[0])
        ttr_div = float(r_div["summary"].split("TTR ")[1].split(";")[0])
        assert ttr_div > ttr_rep

    def test_empty_string(self):
        result = complexity_stats("")
        assert result["tool"] == "complexity"
        assert result["count"] == 0
        assert result["details"] == []
