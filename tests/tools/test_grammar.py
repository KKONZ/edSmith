import pytest

pytest.importorskip("language_tool_python")

from edsmith.tools.grammar import grammar_check


class TestGrammarCheck:
    def test_result_shape(self):
        result = grammar_check("The cat sat on the mat.")
        assert set(result.keys()) == {"tool", "count", "stats", "details", "summary"}
        assert result["tool"] == "grammar"
        assert isinstance(result["count"], int)
        assert isinstance(result["stats"], dict)
        assert isinstance(result["details"], list)

    def test_clean_text_zero_errors(self):
        result = grammar_check("The cat sat on the mat.")
        assert result["count"] == 0
        assert result["details"] == []
        assert result["stats"]["error_count"] == 0

    def test_error_text_positive_count(self):
        result = grammar_check("He go to school every day.")
        assert result["count"] > 0
        assert len(result["details"]) == result["count"]

    def test_details_keys(self):
        result = grammar_check("This are wrong.")
        assert result["count"] > 0
        detail = result["details"][0]
        for key in ("message", "word", "offset", "length", "replacements", "rule_id", "aoa", "nsyll", "freq_pm"):
            assert key in detail

    def test_aoa_populated_for_known_word(self):
        result = grammar_check("He go to school every day.")
        # "go" is a very common word and should be in the AoA lookup
        matched = [d for d in result["details"] if d["aoa"] is not None]
        assert len(matched) > 0

    def test_aoa_stats_present_when_errors_found(self):
        result = grammar_check("He go to school every day.")
        assert "mean_aoa_of_errors" in result["stats"]
        assert "pct_errors_basic" in result["stats"]
        assert "pct_errors_advanced" in result["stats"]

    def test_empty_string(self):
        result = grammar_check("")
        assert result["count"] == 0
        assert result["details"] == []
