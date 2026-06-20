---
name: interpret-metrics
description: Use when reading evaluation metrics after Scorer training to decide whether the iteration improved, diagnose failure modes, and inform the Chief Examiner what to focus on.
---

# Interpret Metrics

## Overview

Four metrics are computed after each Scorer evaluation: `accuracy`, `adjacent_accuracy`, `qwk`, and `smd`. They measure different aspects of Scorer performance and must be read together — any single metric in isolation is misleading.

**Core principle:** QWK is the primary standard for ordinal scoring tasks. Accuracy alone rewards a model for getting lucky on the most common band. SMD reveals the direction of bias that accuracy hides.

## When to Use

- After `evaluate_scorer` returns metrics for validation and test splits
- When comparing the current iteration to previous iterations
- When briefing the Chief Examiner on what to investigate
- When deciding whether to approve a proposal or push back

## The Four Metrics

| Metric | What it measures | Target | Warning threshold |
|---|---|---|---|
| `accuracy` | Exact band match rate | > 0.50 | < 0.35 |
| `adjacent_accuracy` | Within 0.5 bands | > 0.80 | < 0.65 |
| `qwk` | Quadratic weighted kappa (primary) | > 0.70 | < 0.50 |
| `smd` | Standardized mean difference (bias) | −0.3 to +0.3 | \|smd\| > 0.5 |

SMD sign convention: **positive = over-prediction** (Scorer awards higher bands than ground truth), **negative = under-prediction**.

## Steps

### 1. Check for Regression

Compare to the previous iteration:

```
Δaccuracy = current.accuracy - prev.accuracy
Δqwk      = current.qwk - prev.qwk
```

A pass is a regression if QWK dropped by more than 0.03, even if accuracy held steady. QWK penalises large misses quadratically — a small accuracy change can mask a large QWK drop when the model starts making farther-off predictions.

### 2. Assess the Accuracy / Adjacent Gap

```
gap = adjacent_accuracy - accuracy
```

| Gap | Interpretation |
|---|---|
| < 0.15 | Scorer is decisive — misses tend to be large when they happen |
| 0.15–0.30 | Normal — most errors are single-band misses |
| > 0.30 | Scorer is systematically close but consistently off by 0.5 — likely a bias issue, check SMD |

### 3. Read the SMD

SMD tells you where the bias is pointing:

| SMD | Meaning | Likely cause |
|---|---|---|
| > +0.5 | Strong over-prediction | Feedback too generous; Examiner not penalising errors enough |
| +0.1 to +0.5 | Mild over-prediction | Check `task_response` and `lexical` components specifically |
| −0.1 to +0.1 | Balanced | No systematic bias |
| −0.1 to −0.5 | Mild under-prediction | Feedback too critical; check `grammar` component |
| < −0.5 | Strong under-prediction | Scorer failing on high-band essays; check training data distribution |

Pass the SMD reading and its direction to the Chief Examiner as part of the briefing.

### 4. Check Val vs. Test Divergence

If `val.accuracy` is significantly higher than `test.accuracy` (> 0.08 gap):

- The Scorer may be overfitting to the validation distribution
- Or the test set is genuinely harder (different question types)
- Do not tune PromptPolicy based on test set individual records — see ADR 0006

### 5. Check for Plateau

If accuracy has not improved by > 0.02 over the last 2 consecutive iterations:

- The Feedback quality may have hit a ceiling with the current PromptPolicy
- Or the Scorer training config needs adjustment (more steps, different LR)
- Flag this for the Chief Examiner to investigate via diagnostic

### 6. Summarise for Chief Examiner

Produce a brief in this format before calling `run_chief_examiner`:

```
Iteration N metrics (val | test):
  accuracy:          0.48 | 0.44   (Δ+0.05 | +0.04 vs prev)
  adjacent_accuracy: 0.81 | 0.78
  qwk:               0.71 | 0.68
  smd:              +0.21 | +0.18   (mild over-prediction)

Assessment: improvement on all metrics; mild positive SMD suggests
            feedback is slightly generous on Lexical Resource.
            Recommend Chief Examiner audit lexical Feedback.
```

## Red Flags — Stop and Investigate

- QWK dropping while accuracy holds or improves — the Scorer is getting luckier on common bands but making larger errors elsewhere
- `|smd| > 0.5` from iteration 1 — the Feedback is systematically biased from the start; fix the PromptPolicy before training further
- `adjacent_accuracy < accuracy` — impossible under normal conditions; indicates a data loading or metric computation bug
- All metrics identical to the previous iteration — the training run may not have completed, or the model path in state.json was not updated

## Quick Reference

```
Primary metric:   QWK > 0.70 (good), > 0.80 (strong), < 0.50 (investigate)
Bias check:       |SMD| < 0.30 balanced; positive = over-predicting
Plateau signal:   < 0.02 accuracy gain for 2+ iterations
Val/test gap:     > 0.08 suggests overfitting or distribution shift
```
