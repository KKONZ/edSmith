import numpy as np
import pytest

from edsmith.metrics import (
    accuracy,
    adjacent_accuracy,
    compute_all,
    quadratic_weighted_kappa,
    standardized_mean_difference,
)

BANDS = [b / 2 for b in range(2, 19)]  # 1.0 .. 9.0 in 0.5 steps


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([5.0, 6.0, 7.0], [5.0, 6.0, 7.0]) == 1.0

    def test_all_wrong(self):
        assert accuracy([5.0, 6.0], [6.0, 5.0]) == 0.0

    def test_partial(self):
        assert accuracy([5.0, 6.0, 7.0], [5.0, 6.5, 7.0]) == pytest.approx(2 / 3)

    def test_single_correct(self):
        assert accuracy([4.5], [4.5]) == 1.0

    def test_single_wrong(self):
        assert accuracy([4.5], [5.0]) == 0.0


class TestAdjacentAccuracy:
    def test_exact_match(self):
        assert adjacent_accuracy([5.0], [5.0], bands=BANDS) == 1.0

    def test_within_one_step(self):
        # 0.5-step bands → tolerance = 0.5; 5.0 vs 5.5 is within tolerance
        assert adjacent_accuracy([5.0, 6.0], [5.5, 6.5], bands=BANDS) == 1.0

    def test_two_steps_misses(self):
        assert adjacent_accuracy([5.0], [6.0], bands=BANDS) == 0.0

    def test_mixed(self):
        # first pair within tolerance, second not
        result = adjacent_accuracy([5.0, 5.0], [5.5, 7.0], bands=BANDS)
        assert result == pytest.approx(0.5)

    def test_no_bands_default_tolerance_one(self):
        assert adjacent_accuracy([5.0], [5.5]) == 1.0   # 0.5 ≤ 1.0
        assert adjacent_accuracy([5.0], [6.5]) == 0.0   # 1.5 > 1.0


class TestSMD:
    def test_zero_std_returns_zero(self):
        # constant y_true → std = 0 → returns 0.0 without division error
        assert standardized_mean_difference([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]) == 0.0

    def test_over_prediction_positive(self):
        y_true = [4.0, 5.0, 6.0, 7.0]
        y_pred = [4.5, 5.5, 6.5, 7.5]
        expected = 0.5 / float(np.std(y_true))
        assert standardized_mean_difference(y_true, y_pred) == pytest.approx(expected, abs=1e-6)

    def test_under_prediction_negative(self):
        y_true = [5.0, 6.0, 7.0, 8.0]
        y_pred = [4.5, 5.5, 6.5, 7.5]
        assert standardized_mean_difference(y_true, y_pred) < 0

    def test_unbiased_zero(self):
        y = [4.0, 5.0, 6.0, 7.0]
        assert standardized_mean_difference(y, y) == pytest.approx(0.0, abs=1e-9)


class TestQWK:
    def test_perfect_agreement(self):
        assert quadratic_weighted_kappa([5.0, 6.0, 7.0], [5.0, 6.0, 7.0]) == pytest.approx(1.0)

    def test_returns_float(self):
        result = quadratic_weighted_kappa([5.0, 6.0, 7.0], [5.0, 6.0, 8.0])
        assert isinstance(result, float)


class TestComputeAll:
    def test_returns_all_keys(self):
        result = compute_all([5.0, 6.0, 7.0], [5.0, 6.0, 7.0])
        assert set(result.keys()) == {"accuracy", "adjacent_accuracy", "qwk", "smd"}

    def test_perfect_predictions(self):
        result = compute_all([5.0, 6.0, 7.0], [5.0, 6.0, 7.0])
        assert result["accuracy"] == 1.0
        assert result["qwk"] == pytest.approx(1.0)
        assert result["smd"] == pytest.approx(0.0, abs=1e-9)

    def test_bands_forwarded_to_adjacent(self):
        # With IELTS bands, 5.0 vs 5.5 should be adjacent; without bands it also is.
        result_with = compute_all([5.0], [5.5], bands=BANDS)
        result_without = compute_all([5.0], [5.5])
        assert result_with["adjacent_accuracy"] == result_without["adjacent_accuracy"] == 1.0
