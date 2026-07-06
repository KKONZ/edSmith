"""Scorer training and evaluation — Qwen3 + LoRA + ordinal soft-label CE.

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
# Components trained as separate models in multi-component mode.
# task_response is excluded: its score correlates directly with the overall band.
_SCORER_COMPONENTS = ["coherence", "lexical", "grammar"]


def _log(msg: str) -> None:
    print(f"[edsmith-train] {msg}", flush=True, file=sys.stderr)


def _default_config() -> dict:
    return {
        "model_name": "unsloth/Qwen3-4B-Thinking-2507",
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
    enable_thinking: bool = True,
) -> tuple[list[float], list[float]]:
    """Evaluate Scorer on val or test split.

    Auto-detects single vs multi-component from the models directory structure.
    Multi-component: loads each component model, averages predictions for band.

    enable_thinking=True  — inject <think>\\n prefix; model generates feedback then score.
    enable_thinking=False — inject <think>\\n</think>\\n prefix; model outputs score digit only.

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
        return _evaluate(str(models_dir), data_path, enable_thinking=enable_thinking)

    if len(component_dirs) == 1:
        comp, path = next(iter(component_dirs.items()))
        # Honour no_think saved in the model dir's edsmith_config.json.
        edsmith_cfg = json.loads((Path(path) / "edsmith_config.json").read_text()) if (Path(path) / "edsmith_config.json").exists() else {}
        effective_thinking = enable_thinking and not edsmith_cfg.get("no_think", False)
        return _evaluate(path, data_path, component=comp, enable_thinking=effective_thinking)

    # Multi-component: evaluate each model, average predictions per essay
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    _base = 4  # matches _base_band in _evaluate
    valid_rows = [row for _, row in df.iterrows() if _parse_band_val(row.get("band")) is not None]
    y_true = [float(max(_base, int(_parse_band_val(row["band"])))) for row in valid_rows]

    per_component_preds: dict[str, list[float]] = {}
    for comp, path in component_dirs.items():
        _, comp_preds = _evaluate(path, data_path, component=comp, enable_thinking=enable_thinking)
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
        s = str(val).strip()
        if s.startswith("<"):
            s = s.lstrip("<").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------
# Custom loss: weighted CE for thinking tokens + ordinal soft-label CE for score
# ------------------------------------------------------------------

def _ordinal_soft_ce(logits_at_pos, class_tok_ids: list[int], true_class: int):
    """Adjacent-band ordinal credit using full-vocabulary log probabilities.

    Gaussian weights centred on true_class (sigma = (K-1)/3) with the correct
    class zeroed out, so this term only nudges adjacent bands upward. Uses the
    full vocabulary partition — same as score_only_ce — so there is no
    restricted-softmax gradient conflict with CE on the correct class.
    """
    import torch
    K = len(class_tok_ids)
    sigma = (K - 1) / 3
    idx = torch.arange(K, device=logits_at_pos.device, dtype=logits_at_pos.dtype)
    adj_weights = torch.exp(-0.5 * ((idx - true_class) / sigma) ** 2)
    adj_weights[true_class] = 0.0
    adj_weights = adj_weights / adj_weights.sum().clamp(min=1e-7)
    lse = torch.logsumexp(logits_at_pos, dim=0)
    log_p_bands = logits_at_pos[class_tok_ids] - lse  # [K], full-vocab log probs
    return -(adj_weights * log_p_bands).sum()


class _ScoringTrainer:
    """Mixin — provides compute_loss with thinking down-weighting + ordinal soft-label CE.

    Usage: subclass SFTTrainer with this mixin, or swap in after importing.
    """

    _think_weight: float
    _score_weight: float
    _score_only_ce: bool
    _think_id: int
    _end_id: int
    _int_band_tok_ids: list[int]          # token IDs for "4".."9"
    _band_seq_to_int_class: dict          # tuple(tok_ids) → int class 0–5

    def _init_scoring(self, tokenizer, think_weight: float, score_weight: float, score_only_ce: bool = False) -> None:
        import torch.nn.functional as F  # noqa: F401 — trigger import check
        self._think_weight = think_weight
        self._score_weight = score_weight
        self._score_only_ce = score_only_ce

        think_ids = tokenizer.encode("<think>", add_special_tokens=False)
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        assert len(think_ids) == 1 and len(end_ids) == 1, (
            "<think> and </think> must each be a single token in this tokenizer"
        )
        self._think_id = think_ids[0]
        self._end_id = end_ids[0]

        self._int_band_tok_ids = [
            tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(4, 10)
        ]
        # Map integer score strings "4"-"9" → ordinal class index 0–5.
        # Sub-4 bands are collapsed to "4" in _to_chat; 4.0/4.5 both → "4" → class 0.
        self._band_seq_to_int_class = {
            tuple(tokenizer.encode(str(i), add_special_tokens=False)): i - 4
            for i in range(4, 10)
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
        score_mask = torch.zeros(B, L, device=labels.device)  # 1.0 at score token positions only
        ordinal_logit_list: list = []   # logits[b, i-1] at each score position
        ordinal_target_list: list[int] = []
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
                            score_mask[b, i] = 1.0
                            if i > 0:
                                ordinal_logit_list.append(logits[b, i - 1])
                                ordinal_target_list.append(int_class)
                                n_score_found += 1
                            found_score = True
                            break
                i += 1

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_think = think_mask[:, 1:].contiguous()
        shift_score = score_mask[:, 1:].contiguous()

        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, L - 1)

        if self._score_only_ce:
            # CE focused on score token only — strong direct classification signal.
            # think_weight optionally adds CE on the reasoning chain (for thinking mode).
            n_score = shift_score.sum().clamp(min=1)
            ce_loss = (ce_per_token * shift_score).sum() / n_score
            if self._think_weight > 0:
                labeled_mask = (shift_labels != -100).float()
                think_labeled = shift_think * labeled_mask
                n_think = think_labeled.sum().clamp(min=1)
                ce_loss = ce_loss + self._think_weight * (ce_per_token * think_labeled).sum() / n_think
        else:
            # CE — separate normalisation for think vs non-think tokens so that
            # changing feedback length doesn't silently rescale the score-token gradient.
            # think_weight scales the think-CE term; non-think tokens always get weight 1.0.
            labeled_mask = (shift_labels != -100).float()
            think_labeled = shift_think * labeled_mask
            nothink_labeled = (1.0 - shift_think) * labeled_mask
            n_think = think_labeled.sum().clamp(min=1)
            n_nothink = nothink_labeled.sum().clamp(min=1)
            ce_loss = (
                self._think_weight * (ce_per_token * think_labeled).sum() / n_think
                + (ce_per_token * nothink_labeled).sum() / n_nothink
            )

        # Ordinal soft-label CE — regulariser on score token positions.
        # Gaussian soft targets centred on the true band give adjacent bands partial
        # credit, encoding ordinal structure with bounded, well-calibrated gradients.
        # score_weight=0 disables it entirely.
        ord_loss = logits.new_zeros(())
        if ordinal_logit_list and self._score_weight > 0:
            ord_losses = torch.stack([
                _ordinal_soft_ce(l, self._int_band_tok_ids, t)
                for l, t in zip(ordinal_logit_list, ordinal_target_list)
            ])
            ord_loss = ord_losses.mean()

        total = ce_loss + self._score_weight * ord_loss

        step = getattr(self, "state", None)
        step_n = step.global_step if step is not None else -1
        if step_n % 20 == 0:
            mode = "score_only_ce" if self._score_only_ce else f"think_w={self._think_weight}"
            print(
                f"[loss@{step_n}] CE={ce_loss.item():.4f} ({mode})  "
                f"ord={ord_loss.item():.4f} (×{self._score_weight})  "
                f"total={total.item():.4f}  "
                f"labeled={n_labeled_total} think={n_think_labeled} scores_found={n_score_found}",
                flush=True,
            )

        return (total, outputs) if return_outputs else total

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to ensure our compute_loss drives backprop, not Unsloth's CE."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps


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
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")
    _log("Model loaded. Applying LoRA …")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    _log(f"LoRA applied. hidden_size={model.config.hidden_size}")

    dataset = _build_dataset(df, tokenizer, max_len=cfg.get("max_seq_length", 2048), no_think=cfg.get("no_think", False))
    _log(f"Dataset ready ({len(dataset)} items). Building trainer …")

    think_weight = cfg.get("think_weight", 0.0)
    score_weight = cfg.get("score_weight", 1.0)
    score_only_ce = cfg.get("score_only_ce", False)
    if cfg.get("no_think") and think_weight != 0.0:
        _log(f"WARNING: think_weight={think_weight} ignored because no_think=True")
        think_weight = 0.0
    _log(f"Loss weights — think: {think_weight}  score: {score_weight}  score_only_ce: {score_only_ce}")

    from transformers import DataCollatorForSeq2Seq

    class _Trainer(_ScoringTrainer, SFTTrainer):
        pass

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100,
    )
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
        data_collator=data_collator,
    )
    trainer._init_scoring(tokenizer, think_weight=think_weight, score_weight=score_weight, score_only_ce=score_only_ce)
    _log("Starting training …")
    trainer.train()
    _log("Training complete. Saving model …")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "edsmith_config.json").write_text(json.dumps({"component": cfg.get("component"), "no_think": cfg.get("no_think", False)}))
    _log(f"Saved to {out}")
    return str(out)


# ------------------------------------------------------------------
# Evaluation implementation
# ------------------------------------------------------------------

def _evaluate(
    model_path: str,
    eval_data_path: str,
    component: str | None = None,
    enable_thinking: bool = True,
) -> tuple[list[float], list[float]]:
    from unsloth import FastLanguageModel
    from transformers import StoppingCriteria, StoppingCriteriaList
    import pandas as pd
    import torch

    _log(f"Evaluating model at {model_path}  enable_thinking={enable_thinking} …")
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
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")
    tokenizer.padding_side = "left"

    target_components = [component] if component else list(COMPONENT_HEADINGS.keys())
    _log(f"Evaluating component(s): {target_components}")

    _base_band = 4
    _num_classes = 6
    band_tok_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(_base_band, _base_band + _num_classes)]
    band_tok_set = set(band_tok_ids)
    end_think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # qwen3-thinking template's generation prompt already ends with <think>\n.
    # For fast eval (enable_thinking=False) we immediately close the think block
    # so the model outputs a score digit only.
    close_prefix = "" if enable_thinking else "</think>\n"

    valid_bands: list[float] = []
    all_prompts: list[str] = []
    for _, row in df.iterrows():
        band = _parse_band_val(row.get("band"))
        if band is None:
            continue
        valid_bands.append(band)
        for comp in target_components:
            messages = [{"role": "user", "content": _format_input(row["question"], row["essay"], comp)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + close_prefix
            all_prompts.append(prompt)

    n_components = len(target_components)
    all_pred_bands: list[float] = []
    _GEN_BATCH_SIZE = 8

    class _StopOnScore(StoppingCriteria):
        """Stop each sequence as soon as a band token appears after </think>."""
        def __init__(self, end_think_id: int, band_tok_ids: list[int], input_len: int):
            self._end_think_id = end_think_id
            self._band_set = set(band_tok_ids)
            self._input_len = input_len
            self._seen_end_think: dict[int, int] = {}

        def __call__(self, input_ids, scores, **kwargs):
            done = []
            for i in range(input_ids.shape[0]):
                new_ids = input_ids[i, self._input_len:].tolist()
                if i not in self._seen_end_think and self._end_think_id in new_ids:
                    self._seen_end_think[i] = new_ids.index(self._end_think_id)
                if i in self._seen_end_think:
                    after = new_ids[self._seen_end_think[i] + 1:]
                    done.append(any(t in self._band_set for t in after))
                else:
                    done.append(False)
            return all(done)

    max_new_tokens = 4096 if enable_thinking else 10

    with torch.no_grad():
        for i in range(0, len(all_prompts), _GEN_BATCH_SIZE):
            batch = all_prompts[i : i + _GEN_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(model.device)
            input_len = enc["input_ids"].shape[1]

            gen_kwargs: dict = dict(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            if enable_thinking:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    _StopOnScore(end_think_id, band_tok_ids, input_len)
                ])

            out = model.generate(**gen_kwargs)

            for b in range(len(batch)):
                new_ids = out[b, input_len:].tolist()
                if enable_thinking:
                    pred_band = _parse_score_from_ids(new_ids, end_think_id, band_tok_ids, _base_band)
                else:
                    # Score is the first band token in the (very short) output
                    pred_band = next(
                        (float(band_tok_ids.index(t) + _base_band) for t in new_ids if t in band_tok_set),
                        float(_base_band),
                    )
                all_pred_bands.append(pred_band)
                if i == 0 and b == 0:
                    sample_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                    _log(f"[eval sample] {repr(sample_text[:300])}  → pred_band={pred_band}")
            _log(f"  eval batch {i // _GEN_BATCH_SIZE + 1}/{-(-len(all_prompts) // _GEN_BATCH_SIZE)}")

    y_true: list[float] = []
    y_pred: list[float] = []
    for i, band in enumerate(valid_bands):
        comp_preds = [all_pred_bands[i * n_components + j] for j in range(n_components)]
        y_true.append(float(max(_base_band, int(band))))
        y_pred.append(round(sum(comp_preds) / len(comp_preds) * 2) / 2)

    _log(f"Evaluation complete. {len(y_true)} predictions.")
    return y_true, y_pred


def _parse_score_from_ids(new_ids: list[int], end_think_id: int, band_tok_ids: list[int], default: int) -> float:
    """Find the first band token (4-9) after </think> in generated token IDs."""
    if end_think_id not in new_ids:
        return float(default)  # cut off before </think> — don't scan thinking noise
    search_ids = new_ids[new_ids.index(end_think_id) + 1:]
    for tok_id in search_ids:
        if tok_id in band_tok_ids:
            return float(band_tok_ids.index(tok_id) + 4)
    return float(default)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def _build_dataset(df, tokenizer, max_len: int = 2048, no_think: bool = False):
    from datasets import Dataset

    has_feedback = "feedback_text" in df.columns

    def _band_to_score_str(idx: int) -> str:
        return str(max(4, int(_BANDS[idx])))

    unique_labels = sorted(df["label"].unique())
    seen_score_strs: set[str] = set()
    mapping_lines = []
    for idx in unique_labels:
        s = _band_to_score_str(idx)
        mapping_lines.append(f"  label_idx={idx}  band={_BANDS[idx]}  → score_str='{s}'" + (" (collapsed)" if _BANDS[idx] < 4 else ""))
        seen_score_strs.add(s)
    _log(f"Label → score token mapping ({len(seen_score_strs)} unique classes):\n" + "\n".join(mapping_lines))

    # Spot-check: confirm score tokens are single tokens in this tokenizer.
    for band_int in range(4, 10):
        toks = tokenizer.encode(str(band_int), add_special_tokens=False)
        if len(toks) != 1:
            _log(f"WARNING: score token '{band_int}' encodes to {len(toks)} tokens: {toks}")
    _log("Score token single-token check done.")

    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []

    for _, row in df.iterrows():
        score_str = _band_to_score_str(row["label"])

        # Instruction prefix — used to find the response boundary for label masking.
        # qwen3-thinking template adds <think>\n as the generation prompt, so n_instruction
        # covers everything up to (and including) the opening <think>\n token.
        instruction_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": _format_input(row["question"], row["essay"], row["component"])}],
            tokenize=False, add_generation_prompt=True,
        )
        n_instruction = len(tokenizer.encode(instruction_text, add_special_tokens=False))

        # Full sequence (instruction + assistant response).
        # qwen3-thinking template preserves <think>...</think> blocks in assistant messages,
        # unlike the default template which strips them when enable_thinking=False.
        if no_think:
            # Score-only ablation: immediately close the think block so training
            # mirrors the enable_thinking=False eval path (no reasoning chain).
            assistant_content = f"</think>\n{score_str}"
        elif has_feedback and row.get("feedback_text"):
            clean = re.sub(r"<score>[^<]*</score>\s*", "", row["feedback_text"], flags=re.IGNORECASE)
            clean = re.sub(r"<confidence>[^<]*</confidence>\s*", "", clean, flags=re.IGNORECASE).strip()
            assistant_content = f"<think>\n{clean}\n</think>\n{score_str}"
        else:
            assistant_content = score_str
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": _format_input(row["question"], row["essay"], row["component"])},
                {"role": "assistant", "content": assistant_content},
            ],
            tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)[:max_len]

        # Mask instruction tokens with -100 so compute_loss only sees response positions.
        cut = min(n_instruction, len(full_ids))
        labels = [-100] * cut + full_ids[cut:]

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    sample_text = tokenizer.decode(all_input_ids[0])
    print(f"\n[train sample — full chat string]\n{sample_text}\n{'─'*60}", flush=True)

    return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})


def _format_input(question: str, essay: str, component: str) -> str:
    component_name = COMPONENT_HEADINGS.get(component, component)
    return (
        f"You are an IELTS examiner. Score the essay below for the '{component_name}' component "
        f"and explain your reasoning in detail.\n\n"
        f"Question: {question}\n\nEssay: {essay}"
    )
