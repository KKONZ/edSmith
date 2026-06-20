import pytest

pytest.importorskip("language_tool_python")

from edsmith.tools import ToolResult
from edsmith.tools.grammar import grammar_check


class TestGrammarCheck:
    def test_result_shape(self):
        result = grammar_check("The cat sat on the mat.")
        assert set(result.keys()) == {"tool", "count", "details", "summary"}
        assert result["tool"] == "grammar"
        assert isinstance(result["count"], int)
        assert isinstance(result["details"], list)
        assert isinstance(result["summary"], str)

    def test_clean_text_zero_count(self):
        result = grammar_check("The cat sat on the mat.")
        assert result["count"] == 0
        assert result["details"] == []

    def test_error_text_positive_count(self):
        result = grammar_check("He go to school every day.")
        assert result["count"] > 0
        assert len(result["details"]) == result["count"]

    def test_details_keys(self):
        result = grammar_check("This are wrong.")
        assert result["count"] > 0
        detail = result["details"][0]
        assert "message" in detail
        assert "offset" in detail
        assert "length" in detail
        assert "replacements" in detail
        assert "rule_id" in detail

    def test_empty_string(self):
        result = grammar_check("")
        assert result["tool"] == "grammar"
        assert result["count"] == 0
        assert result["details"] == []
