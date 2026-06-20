from edsmith.tools.aoa import compute_aoa_stats


class TestComputeAoaStats:
    def test_result_shape(self):
        result = compute_aoa_stats("The cat sat on the mat.")
        assert set(result.keys()) == {"tool", "count", "stats", "details", "summary"}
        assert result["tool"] == "aoa"

    def test_known_words_matched(self):
        result = compute_aoa_stats("cat dog house")
        assert result["count"] > 0

    def test_details_keys(self):
        result = compute_aoa_stats("cat dog house")
        detail = result["details"][0]
        for key in ("word", "aoa", "freq_pm", "nsyll", "nphon", "nletters", "pos", "perc_known"):
            assert key in detail

    def test_stats_keys(self):
        result = compute_aoa_stats("The cat sat on the mat.")
        for key in ("coverage", "aoa_mean", "aoa_std", "aoa_median", "aoa_skew", "aoa_kurtosis", "pct_early", "pct_late"):
            assert key in result["stats"]

    def test_early_words_low_aoa(self):
        # "cat", "dog", "ball" are acquired very early
        result = compute_aoa_stats("cat dog ball")
        assert result["stats"]["aoa_mean"] < 7

    def test_late_words_high_aoa(self):
        # sophisticated vocabulary acquired later
        result = compute_aoa_stats("ubiquitous quintessential ephemeral")
        assert result["stats"]["aoa_mean"] > 9

    def test_pct_early_and_late_sum_within_range(self):
        result = compute_aoa_stats("The cat sat on the mat.")
        assert 0 <= result["stats"]["pct_early"] <= 100
        assert 0 <= result["stats"]["pct_late"] <= 100

    def test_empty_string(self):
        result = compute_aoa_stats("")
        assert result["count"] == 0
        assert result["details"] == []
        assert result["stats"] == {}
