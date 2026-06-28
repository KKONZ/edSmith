"""Scorer training and evaluation — Qwen3 + LoRA + CORN loss.

GPU-only. Install [training] extras before importing:
    pip install -e ".[training]"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from edsmith.data.parser import COMPONENT_HEADINGS

_BANDS: list[float] = [b / 2 for b in range(2, 19)]  # 1.0 … 9.0  (17 values)
_BAND_TO_IDX: dict[float, int] = {b: i for i, b in enumerate(_BANDS)}
_EVAL_BATCH_SIZE = 8

# Components trained as separate models in multi-component mode.
# task_response is excluded: its score correlates directly with the overall band.
_SCORER_COMPONENTS = ["coherence", "lexical", "grammar"]


def _log(msg: str) -> None:
    print(f"[edsmith-train] {msg}", flush=True, file=sys.stderr)


def _default_config() -> dict:
    return {
        "model_name": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "lora_r": 16,
        "lora_alpha": 16,
        "max_steps": 100,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 5,
        "component": None,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train(session_id: str, iteration: int, drive_path: Path, cfg: dict | None = None) -> dict[str, str]:
    """Fine-tune Scorer on feedback for this iteration.

    Single-component mode (cfg["component"] set): trains one model.
    Multi-component mode (cfg["component"] is None): trains one model per
    component in _SCORER_COMPONENTS (coherence, lexical, grammar).

    Reads:  {drive_path}/sessions/{session_id}/feedback_iter{n}.parquet
            {drive_path}/sessions/{session_id}/scorer_config.json
    Writes: {drive_path}/sessions/{session_id}/models/iter{n}/{component}/
    Returns dict mapping component name → model path.
    """
    session_dir = drive_path / "sessions" / session_id
    feedback_path = session_dir / f"feedback_iter{iteration}.parquet"

    if cfg is None:
        scorer_cfg_path = session_dir / "scorer_config.json"
        if scorer_cfg_path.exists():
            loaded = json.loads(scorer_cfg_path.read_text())
            cfg = {**_default_config(), **{k: v for k, v in loaded.items() if v is not None}}
        else:
            cfg = _default_config()

    component = cfg.get("component")
    components_to_train = [component] if component else _SCORER_COMPONENTS

    model_paths: dict[str, str] = {}
    for comp in components_to_train:
        output_dir = str(session_dir / "models" / f"iter{iteration}" / comp)
        comp_cfg = {**cfg, "component": comp}
        model_paths[comp] = _train(str(feedback_path), comp_cfg, output_dir)

    return model_paths


def evaluate(
    session_id: str,
    iteration: int,
    split: str,
    drive_path: Path,
) -> tuple[list[float], list[float]]:
    """Evaluate Scorer on val or test split.

    Auto-detects single vs multi-component from the models directory structure.
    Multi-component: loads each component model, averages predictions for band.

    Reads:  {drive_path}/sessions/{session_id}/models/iter{n}/{component}/
            {drive_path}/sessions/{session_id}/data/{split}.parquet
    Returns (y_true, y_pred).
    """
    session_dir = drive_path / "sessions" / session_id
    models_dir = session_dir / "models" / f"iter{iteration}"
    data_path = str(session_dir / "data" / f"{split}.parquet")

    # Detect component subdirectories
    component_dirs = {
        d.name: str(d)
        for d in sorted(models_dir.iterdir())
        if d.is_dir() and d.name in (*_SCORER_COMPONENTS, "task_response")
    } if models_dir.exists() else {}

    if not component_dirs:
        # Legacy: single model at iter root
        return _evaluate(str(models_dir), data_path)

    if len(component_dirs) == 1:
        comp, path = next(iter(component_dirs.items()))
        return _evaluate(path, data_path, component=comp)

    # Multi-component: evaluate each model, average predictions per essay
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    valid_rows = [row for _, row in df.iterrows() if _parse_band_val(row.get("band")) is not None]
    y_true = [_parse_band_val(row["band"]) for row in valid_rows]

    per_component_preds: dict[str, list[float]] = {}
    for comp, path in component_dirs.items():
        _, comp_preds = _evaluate(path, data_path, component=comp)
        per_component_preds[comp] = comp_preds

    # Average across components per essay
    n = min(len(v) for v in per_component_preds.values())
    y_pred = [
        round(float(np.mean([per_component_preds[c][i] for c in component_dirs])) * 2) / 2
        for i in range(n)
    ]
    y_true = y_true[:n]
    return y_true, y_pred


def _parse_band_val(val) -> float | None:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------
# Custom loss: weighted CE for thinking tokens + headless CORN for score
# ------------------------------------------------------------------

def _corn_logits_from_vocab(logits_at_pos, class_tok_ids: list[int]):
    """Derive CORN logits from vocabulary logits — no projection head.

    Extracts the softmax distribution over `class_tok_ids` tokens, then
    converts to conditional log-odds P(class > k | class >= k) for each k.
    Gradient flows directly through the vocabulary logits; no new parameters.
    """
    import torch
    p = torch.softmax(logits_at_pos[class_tok_ids], dim=0)   # [C]
    cumsum = p.cumsum(0)                                       # P(class <= k)
    p_gt = 1.0 - cumsum[:-1]                                  # P(class > k), [C-1]
    p_ge = torch.cat([p.new_ones(1), 1.0 - cumsum[:-2]])     # P(class >= k), [C-1]
    cond = (p_gt / p_ge.clamp(min=1e-7)).clamp(1e-7, 1.0 - 1e-7)
    return torch.log(cond) - torch.log(1.0 - cond)           # [C-1]


class _CornLoss:
    """CORN: Conditional Ordinal Regression for Neural networks.

    Mirrors coral_pytorch.losses.corn_loss — conditional subsets per task,
    BCE aggregated and normalised by total examples across tasks.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        # Build (mask, binary_label) pairs for each ordinal threshold task.
        # Task i keeps examples where class >= i (y > i-1) and labels whether class > i.
        sets = []
        for i in range(self.num_classes - 1):
            label_mask = targets > i - 1
            label_tensor = (targets[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.0
        for task_index, (train_examples, train_labels) in enumerate(sets):
            if len(train_labels) < 1:
                continue
            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]
            log_sigmoid = F.logsigmoid(pred)
            losses += -torch.sum(
                log_sigmoid * train_labels + (log_sigmoid - pred) * (1 - train_labels)
            )

        return losses / num_examples


class _ScoringTrainer:
    """Mixin — provides compute_loss with thinking down-weighting + headless CORN.

    Usage: subclass SFTTrainer with this mixin, or swap in after importing.
    """

    _think_weight: float
    _score_weight: float
    _think_id: int
    _end_id: int
    _int_band_tok_ids: list[int]          # token IDs for "1".."9"
    _band_seq_to_int_class: dict          # tuple(tok_ids) → int class 0–8
    _corn: _CornLoss

    def _init_scoring(self, tokenizer, think_weight: float, score_weight: float) -> None:
        import torch.nn.functional as F  # noqa: F401 — trigger import check
        self._think_weight = think_weight
        self._score_weight = score_weight
        self._corn = _CornLoss(num_classes=9)  # 9 integer bands: 1..9

        think_ids = tokenizer.encode("<think>", add_special_tokens=False)
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        assert len(think_ids) == 1 and len(end_ids) == 1, (
            "<think> and </think> must each be a single token in this tokenizer"
        )
        self._think_id = think_ids[0]
        self._end_id = end_ids[0]

        self._int_band_tok_ids = [
            tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)
        ]
        # Map integer score strings "1"-"9" → CORN class 0–8.
        # Training uses int(band) so "6" covers both 6.0 and 6.5 labels.
        self._band_seq_to_int_class = {
            tuple(tokenizer.encode(str(i), add_special_tokens=False)): i - 1
            for i in range(1, 10)
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch
        import torch.nn.functional as F

        labels = inputs["labels"]      # [B, L]
        input_ids = inputs["input_ids"]  # [B, L]
        B, L = labels.shape

        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
        logits = outputs.logits  # [B, L, V]

        think_mask = torch.zeros(B, L, device=labels.device)  # 1.0 for think-block tokens only
        corn_logit_list: list = []
        corn_target_list: list[int] = []
        n_labeled_total = 0
        n_think_labeled = 0
        n_score_found = 0

        for b in range(B):
            ids = input_ids[b].tolist()
            lbs = labels[b].tolist()
            in_think = False
            found_score = False
            i = 0
            while i < L:
                tid = ids[i]
                if tid == self._think_id:
                    in_think = True
                elif tid == self._end_id:
                    in_think = False
                    i += 1
                    continue

                if lbs[i] != -100:
                    n_labeled_total += 1
                    if in_think:
                        think_mask[b, i] = 1.0
                        n_think_labeled += 1

                if not in_think and not found_score and lbs[i] != -100:
                    for seq, int_class in self._band_seq_to_int_class.items():
                        end = i + len(seq)
                        if end <= L and tuple(ids[i:end]) == seq:
                            if i > 0:
                                corn_logit_list.append(
                                    _corn_logits_from_vocab(logits[b, i - 1], self._int_band_tok_ids)
                                )
                                corn_target_list.append(int_class)
                                n_score_found += 1
                            found_score = True
                            break
                i += 1

        # CE — think tokens only, scaled by think_weight.
        # shift_think[b, j] = think_mask[b, j+1]: was the predicted token inside <think>?
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_think = think_mask[:, 1:].contiguous()

        ce_loss = logits.new_zeros(())
        if self._think_weight > 0:
            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(B, L - 1)
            n_think = shift_think.sum().clamp(min=1)
            ce_loss = self._think_weight * (ce_per_token * shift_think).sum() / n_think

        # CORN — ordinal regression on score token positions only
        corn_loss = logits.new_zeros(())
        if corn_logit_list:
            corn_logits = torch.stack(corn_logit_list)  # [N, 8]
            corn_targets = torch.tensor(corn_target_list, device=labels.device)
            corn_loss = self._corn(corn_logits, corn_targets)

        total = ce_loss + self._score_weight * corn_loss

        step = getattr(self, "state", None)
        step_n = step.global_step if step is not None else -1
        if step_n % 20 == 0:
            print(
                f"[loss@{step_n}] CE={ce_loss.item():.4f} (think_w={self._think_weight})  "
                f"CORN={corn_loss.item():.4f} (×{self._score_weight})  "
                f"total={total.item():.4f}  "
                f"labeled={n_labeled_total} think={n_think_labeled} scores_found={n_score_found}",
                flush=True,
            )

        return (total, outputs) if return_outputs else total


# ------------------------------------------------------------------
# Training implementation
# ------------------------------------------------------------------

def _train(feedback_path: str, cfg: dict, output_dir: str) -> str:
    import pandas as pd
    import torch
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    _log(f"Python {sys.version}")
    _log(f"Config: {cfg}")
    _log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _log(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    _log("Loading feedback data …")
    df = pd.read_parquet(feedback_path)
    df = df.dropna(subset=["score"])
    if cfg.get("component"):
        df = df[df["component"] == cfg["component"]]
        _log(f"Filtered to component '{cfg['component']}': {len(df)} rows")
    df["label"] = df["score"].map(_BAND_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    _log(f"Training rows: {len(df)}  label range: {df['label'].min()}–{df['label'].max()}")

    _log(f"Loading model {cfg['model_name']} (4bit) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        load_in_4bit=True,
    )
    _log("Model loaded. Applying LoRA …")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    _log(f"LoRA applied. hidden_size={model.config.hidden_size}")

    dataset = _build_dataset(df, tokenizer)
    _log(f"Dataset ready ({len(dataset)} items). Building trainer …")
    print(f"\n[train sample — full chat string]\n{dataset[0]['text']}\n{'─'*60}", flush=True)

    think_weight = cfg.get("think_weight", 0.0)
    score_weight = cfg.get("score_weight", 1.0)
    _log(f"Loss weights — think: {think_weight}  score: {score_weight}")

    from unsloth.chat_templates import train_on_responses_only

    class _Trainer(_ScoringTrainer, SFTTrainer):
        pass

    trainer = _Trainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=cfg["max_steps"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps", 5),
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=dataset,
        dataset_text_field="text",
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer._init_scoring(tokenizer, think_weight=think_weight, score_weight=score_weight)
    _log("Starting training …")
    trainer.train()
    _log("Training complete. Saving model …")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "edsmith_config.json").write_text(json.dumps({"component": cfg.get("component")}))
    _log(f"Saved to {out}")
    return str(out)


# ------------------------------------------------------------------
# Evaluation implementation
# ------------------------------------------------------------------

def _evaluate(model_path: str, eval_data_path: str, component: str | None = None) -> tuple[list[float], list[float]]:
    import numpy as np
    from unsloth import FastLanguageModel
    import pandas as pd
    import torch

    _log(f"Evaluating model at {model_path} …")
    edsmith_cfg_path = Path(model_path) / "edsmith_config.json"
    if component is None and edsmith_cfg_path.exists():
        component = json.loads(edsmith_cfg_path.read_text()).get("component")
    _log(f"Component: {component or 'all'}")
    df = pd.read_parquet(eval_data_path)
    _log(f"Eval rows: {len(df)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    target_components = [component] if component else list(COMPONENT_HEADINGS.keys())
    n_components = len(target_components)
    _log(f"Evaluating component(s): {target_components}")

    valid_bands: list[float] = []
    all_prompts: list[str] = []
    for _, row in df.iterrows():
        try:
            band = float(row["band"])
        except ValueError:
            continue
        valid_bands.append(band)
        for comp in target_components:
            messages = [{"role": "user", "content": _format_input(row["question"], row["essay"], comp)}]
            all_prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            )

    all_generated: list[str] = []
    logged_sample = False
    with torch.no_grad():
        for i in range(0, len(all_prompts), _EVAL_BATCH_SIZE):
            batch = all_prompts[i : i + _EVAL_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(model.device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=2048)
            input_len = enc["input_ids"].shape[1]
            for o in out:
                raw = tokenizer.decode(o[input_len:], skip_special_tokens=False).strip()
                all_generated.append(raw)
                if not logged_sample:
                    print(f"\n[eval sample — full generation with thinking]\n{raw}\n{'─'*60}", flush=True)
                    logged_sample = True
            _log(f"  eval batch {i // _EVAL_BATCH_SIZE + 1}/{-(-len(all_prompts) // _EVAL_BATCH_SIZE)}")

    y_true: list[float] = []
    y_pred: list[float] = []
    n_skipped = 0
    for i, band in enumerate(valid_bands):
        component_preds: list[float] = []
        for j in range(n_components):
            raw = all_generated[i * n_components + j]
            try:
                import re as _re
                raw_clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL)
                raw_clean = _re.sub(r"<\|.*?\|>", "", raw_clean).strip()
                pred = max(1.0, min(9.0, round(float(raw_clean) * 2) / 2))
                component_preds.append(pred)
            except (ValueError, TypeError):
                _log(f"  unparseable: row {i} comp {j} → {raw!r}")
        if not component_preds:
            n_skipped += 1
            continue
        y_true.append(band)
        y_pred.append(round(float(np.mean(component_preds)) * 2) / 2)

    _log(f"Evaluation complete. {len(y_true)} predictions ({n_skipped} rows skipped).")
    return y_true, y_pred


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def _build_dataset(df, tokenizer):
    from datasets import Dataset

    has_feedback = "feedback_text" in df.columns

    def _to_chat(row):
        score_str = str(int(_BANDS[row["label"]]))
        if has_feedback and row.get("feedback_text"):
            clean = re.sub(r"<score>[^<]*</score>\s*", "", row["feedback_text"], flags=re.IGNORECASE)
            clean = re.sub(r"<confidence>[^<]*</confidence>\s*", "", clean, flags=re.IGNORECASE).strip()
            assistant_content = f"<think>\n{clean}\n</think>\n{score_str}"
        else:
            assistant_content = score_str
        messages = [
            {"role": "user", "content": _format_input(row["question"], row["essay"], row["component"])},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    return Dataset.from_dict({
        "text": [_to_chat(row) for _, row in df.iterrows()],
        "label": df["label"].tolist(),
    })


def _format_input(question: str, essay: str, component: str) -> str:
    component_name = COMPONENT_HEADINGS.get(component, component)
    return (
        f"You are an IELTS examiner. Score the essay below for the '{component_name}' component "
        f"and explain your reasoning in detail.\n\n"
        f"Question: {question}\n\nEssay: {essay}"
    )
