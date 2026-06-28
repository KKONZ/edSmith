"""Tests for scorer loss internals — no GPU or API key required."""

import pytest
import torch

from edsmith.training.scorer import (
    _BANDS,
    _BAND_TO_IDX,
    _CornLoss,
    _ScoringTrainer,
    _build_dataset,
    _corn_logits_from_vocab,
)

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

# Fixed token IDs — all must be < _VOCAB_SIZE
_VOCAB_SIZE = 200
_THINK_ID = 50
_END_ID = 51
_INT_TOKS = {i: 60 + i for i in range(1, 10)}             # "1"→61 … "9"→69
_BAND_TOKS = {b: 100 + idx for idx, b in enumerate(_BANDS)}  # 100..116


class _StubTok:
    """Minimal tokenizer stub — no model weights needed."""

    def encode(self, text, add_special_tokens=False):
        if text == "<think>":
            return [_THINK_ID]
        if text == "</think>":
            return [_END_ID]
        try:
            n = int(text)
            if 1 <= n <= 9:
                return [_INT_TOKS[n]]
        except ValueError:
            pass
        try:
            b = float(text)
            if b in _BAND_TOKS:
                return [_BAND_TOKS[b]]
        except ValueError:
            pass
        return [199]  # unknown token, within vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        parts = [f"[{m['role']}]{m['content']}" for m in messages]
        result = "".join(parts)
        if add_generation_prompt:
            result += "[assistant]"
        return result


def _make_logits(B, L, hot_tok=None, hot_val=5.0):
    """Uniform logits; optionally boost one token to make it dominant."""
    logits = torch.zeros(B, L, _VOCAB_SIZE)
    if hot_tok is not None:
        logits[:, :, hot_tok] = hot_val
    return logits


class _StubModel:
    """Returns fixed logits regardless of input."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self._logits)


class _BareTrainer(_ScoringTrainer):
    """Instantiable subclass — SFTTrainer __init__ is skipped."""
    def __init__(self):
        pass  # bypass SFTTrainer setup


@pytest.fixture
def trainer():
    t = _BareTrainer()
    t._init_scoring(_StubTok(), think_weight=0.1, score_weight=5.0)
    return t


def _inputs(ids, labels):
    return {
        "input_ids": torch.tensor([ids]),
        "labels": torch.tensor([labels]),
    }


# ---------------------------------------------------------------------------
# _CornLoss
# ---------------------------------------------------------------------------

class TestCornLoss:
    def test_loss_is_positive(self):
        corn = _CornLoss(num_classes=5)
        logits = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 2, 3, 0, 1])
        assert corn(logits, targets).item() > 0

    def test_perfect_separation_gives_low_loss(self):
        corn = _CornLoss(num_classes=5)
        # logit[k] >> 0 iff target > k → near-zero BCE per task
        logits = torch.tensor([
            [-10., -10., -10., -10.],  # target 0: never > any threshold
            [ 10., -10., -10., -10.],  # target 1: > 0 only
            [ 10.,  10., -10., -10.],  # target 2: > 0,1 only
            [ 10.,  10.,  10., -10.],  # target 3: > 0,1,2 only
            [ 10.,  10.,  10.,  10.],  # target 4: > all
        ])
        targets = torch.tensor([0, 1, 2, 3, 4])
        assert corn(logits, targets).item() < 0.05

    def test_conditional_subsetting_skips_empty_tasks(self):
        # Only target=0 examples; tasks 1..3 have empty subsets and must not error
        corn = _CornLoss(num_classes=4)
        logits = torch.zeros(2, 3)
        targets = torch.tensor([0, 0])
        loss = corn(logits, targets)
        assert torch.isfinite(loss)

    def test_all_max_class(self):
        corn = _CornLoss(num_classes=4)
        logits = torch.zeros(3, 3)
        targets = torch.tensor([3, 3, 3])
        loss = corn(logits, targets)
        assert torch.isfinite(loss)

    def test_normalisaton_by_total_subset_examples(self):
        # Manually verify the denominator is total examples across tasks, not per-task.
        corn = _CornLoss(num_classes=3)
        # targets=[0,1,2]: task0 has 3, task1 has 2, task2 has 1 → denom=6
        logits = torch.zeros(3, 2)
        targets = torch.tensor([0, 1, 2])
        loss = corn(logits, targets)
        # Expected: BCE(0.0, label) for each example in each subset / 6
        # BCE(0.0, label) = log(2) ≈ 0.693 regardless of label (sigmoid(0)=0.5)
        import math
        assert abs(loss.item() - math.log(2)) < 1e-4

    def test_gradient_flows(self):
        corn = _CornLoss(num_classes=4)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        loss = corn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# _corn_logits_from_vocab
# ---------------------------------------------------------------------------

class TestCornLogitsFromVocab:
    # Use the same int-band token IDs as the stub tokenizer
    CLASS_TOKS = list(_INT_TOKS.values())  # [61..69]

    def test_output_shape(self):
        logits = torch.zeros(_VOCAB_SIZE)
        out = _corn_logits_from_vocab(logits, self.CLASS_TOKS)
        assert out.shape == (8,)  # 9 classes → 8 CORN logits

    def test_all_finite(self):
        logits = torch.randn(_VOCAB_SIZE)
        out = _corn_logits_from_vocab(logits, self.CLASS_TOKS)
        assert torch.all(torch.isfinite(out))

    def test_gradient_flows_through_vocab_logits(self):
        logits = torch.randn(_VOCAB_SIZE, requires_grad=True)
        out = _corn_logits_from_vocab(logits, self.CLASS_TOKS)
        out.sum().backward()
        # Only the class token positions should have non-zero grad
        assert logits.grad[self.CLASS_TOKS].abs().sum().item() > 0
        non_class = [t for t in range(_VOCAB_SIZE) if t not in self.CLASS_TOKS]
        assert logits.grad[non_class].abs().sum().item() == 0

    def test_mass_on_highest_class_raises_all_logits(self):
        # All prob on the highest class token → P(class > k | class >= k) ≈ 1 for all k
        logits = torch.full((_VOCAB_SIZE,), -1e6)
        logits[self.CLASS_TOKS[-1]] = 1e6
        out = _corn_logits_from_vocab(logits, self.CLASS_TOKS)
        assert torch.all(out > 5.0)

    def test_mass_on_lowest_class_lowers_all_logits(self):
        # All prob on the lowest class token → P(class > k | class >= k) ≈ 0 for all k
        logits = torch.full((_VOCAB_SIZE,), -1e6)
        logits[self.CLASS_TOKS[0]] = 1e6
        out = _corn_logits_from_vocab(logits, self.CLASS_TOKS)
        assert torch.all(out < -5.0)


# ---------------------------------------------------------------------------
# _ScoringTrainer.compute_loss
# ---------------------------------------------------------------------------

class TestScoringTrainerComputeLoss:
    def _seq(self, *toks):
        """Build a sequence where user token is masked and the rest are labeled."""
        user_tok = 10  # within vocab, masked via -100 in labels
        ids = [user_tok] + list(toks)
        labels = [-100] + list(toks)
        return ids, labels

    def test_basic_loss_is_finite(self, trainer):
        SCORE = _BAND_TOKS[7.0]
        ids, labels = self._seq(_THINK_ID, 20, _END_ID, SCORE, 1)
        inputs = _inputs(ids, labels)
        model = _StubModel(_make_logits(1, len(ids)))
        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)

    def test_think_weight_affects_loss(self, trainer):
        # Sequence with many thinking tokens; changing think_weight should change loss.
        SCORE = _BAND_TOKS[5.0]
        ids, labels = self._seq(_THINK_ID, 20, 21, 22, 23, _END_ID, SCORE, 1)
        inputs = _inputs(ids, labels)
        # hot_tok=1 means the model strongly predicts token 1 everywhere.
        # Thinking tokens (20-23) have high CE; eos (1) has near-zero CE.
        # With uniform logits the weighted avg cancels out, so we need non-uniform CE.
        model = _StubModel(_make_logits(1, len(ids), hot_tok=1, hot_val=20.0))

        trainer._think_weight = 1.0
        loss_full = trainer.compute_loss(model, inputs).item()

        trainer._think_weight = 0.0
        loss_zero = trainer.compute_loss(model, inputs).item()

        assert loss_full != pytest.approx(loss_zero)

    def test_corn_weight_affects_loss(self, trainer):
        # Turning score_weight on/off should change loss when a score token is found.
        SCORE = _BAND_TOKS[7.0]
        ids, labels = self._seq(_THINK_ID, 20, _END_ID, SCORE, 1)
        inputs = _inputs(ids, labels)
        # Make integer "7" token dominant so CORN logit is non-trivial
        model = _StubModel(_make_logits(1, len(ids), hot_tok=_INT_TOKS[7]))

        trainer._score_weight = 0.0
        loss_no_corn = trainer.compute_loss(model, inputs).item()

        trainer._score_weight = 5.0
        loss_with_corn = trainer.compute_loss(model, inputs).item()

        assert loss_no_corn != pytest.approx(loss_with_corn)

    def test_no_think_block_still_finite(self, trainer):
        # Sequence without any thinking tags should still work.
        SCORE = _BAND_TOKS[5.0]
        ids, labels = self._seq(SCORE, 1)
        inputs = _inputs(ids, labels)
        model = _StubModel(_make_logits(1, len(ids)))
        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)

    def test_return_outputs_tuple(self, trainer):
        SCORE = _BAND_TOKS[6.0]
        ids, labels = self._seq(_THINK_ID, 20, _END_ID, SCORE, 1)
        inputs = _inputs(ids, labels)
        model = _StubModel(_make_logits(1, len(ids)))
        result = trainer.compute_loss(model, inputs, return_outputs=True)
        assert isinstance(result, tuple) and len(result) == 2
        loss, outputs = result
        assert torch.isfinite(loss)
        assert hasattr(outputs, "logits")

    def test_all_masked_labels_no_crash(self, trainer):
        # All labels -100 — no CE contribution, no score found, should return 0.
        ids = [500, 600, 700]
        labels = [-100, -100, -100]
        inputs = _inputs(ids, labels)
        model = _StubModel(_make_logits(1, len(ids)))
        loss = trainer.compute_loss(model, inputs)
        assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _build_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def _row(self, score=7.0, feedback=None):
        row = {
            "question": "Discuss the advantages of X.",
            "essay": "X has many advantages.",
            "component": "grammar",
            "score": score,
            "label": _BAND_TO_IDX[score],
        }
        if feedback is not None:
            row["feedback_text"] = feedback
        return row

    def test_feedback_placed_in_think_block(self):
        import pandas as pd
        df = pd.DataFrame([self._row(feedback="Good grammar overall.")])
        ds = _build_dataset(df, _StubTok())
        text = ds["text"][0]
        assert "<think>" in text
        assert "Good grammar overall." in text
        assert "</think>" in text

    def test_score_follows_think_block(self):
        import pandas as pd
        df = pd.DataFrame([self._row(score=7.0, feedback="Some feedback.")])
        ds = _build_dataset(df, _StubTok())
        text = ds["text"][0]
        assert text.index("</think>") < text.index("7.0")

    def test_no_feedback_no_think_block(self):
        import pandas as pd
        df = pd.DataFrame([self._row()])  # no feedback_text column
        ds = _build_dataset(df, _StubTok())
        text = ds["text"][0]
        assert "<think>" not in text

    def test_empty_feedback_no_think_block(self):
        import pandas as pd
        df = pd.DataFrame([self._row(feedback="")])
        ds = _build_dataset(df, _StubTok())
        text = ds["text"][0]
        assert "<think>" not in text

    def test_label_column_preserved(self):
        import pandas as pd
        df = pd.DataFrame([self._row(score=5.0), self._row(score=7.0)])
        ds = _build_dataset(df, _StubTok())
        assert ds["label"] == [_BAND_TO_IDX[5.0], _BAND_TO_IDX[7.0]]
