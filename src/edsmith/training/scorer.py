"""Scorer training and evaluation — Qwen3 + LoRA + CORN loss.

GPU-only. Install [training] extras before importing:
    pip install -e ".[training]"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from edsmith.data.parser import COMPONENT_HEADINGS

_BANDS: list[float] = [b / 2 for b in range(2, 19)]  # 1.0 … 9.0  (17 values)
_BAND_TO_IDX: dict[float, int] = {b: i for i, b in enumerate(_BANDS)}
_EVAL_BATCH_SIZE = 8


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

def train(session_id: str, iteration: int, drive_path: Path, cfg: dict | None = None) -> str:
    """Fine-tune Scorer on Phase 1 feedback for this iteration.

    Reads:  {drive_path}/sessions/{session_id}/feedback_iter{n}.parquet
            {drive_path}/sessions/{session_id}/scorer_config.json  (written by init_session)
    Writes: {drive_path}/sessions/{session_id}/models/iter{n}/
    Returns the model path.
    """
    session_dir = drive_path / "sessions" / session_id
    feedback_path = session_dir / f"feedback_iter{iteration}.parquet"
    output_dir = str(session_dir / "models" / f"iter{iteration}")

    if cfg is None:
        scorer_cfg_path = session_dir / "scorer_config.json"
        if scorer_cfg_path.exists():
            loaded = json.loads(scorer_cfg_path.read_text())
            cfg = {**_default_config(), **{k: v for k, v in loaded.items() if v is not None}}
        else:
            cfg = _default_config()

    return _train(str(feedback_path), cfg, output_dir)


def evaluate(
    session_id: str,
    iteration: int,
    split: str,
    drive_path: Path,
) -> tuple[list[float], list[float]]:
    """Evaluate Scorer on val or test split.

    Reads:  {drive_path}/sessions/{session_id}/models/iter{n}/
            {drive_path}/sessions/{session_id}/data/{split}.parquet
    Returns (y_true, y_pred).
    """
    session_dir = drive_path / "sessions" / session_id
    model_path = str(session_dir / "models" / f"iter{iteration}")
    data_path = str(session_dir / "data" / f"{split}.parquet")
    return _evaluate(model_path, data_path)


# ------------------------------------------------------------------
# Training implementation
# ------------------------------------------------------------------

def _train(feedback_path: str, cfg: dict, output_dir: str) -> str:
    from unsloth import FastLanguageModel  # must be first — patches transformers/torch
    import pandas as pd
    import torch
    from trl import SFTTrainer, SFTConfig

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

    trainer = SFTTrainer(
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
        compute_loss_func=cfg.get("compute_loss_func"),
    )
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

def _evaluate(model_path: str, eval_data_path: str) -> tuple[list[float], list[float]]:
    from unsloth import FastLanguageModel  # must be first
    import numpy as np
    import pandas as pd
    import torch

    _log(f"Evaluating model at {model_path} …")
    edsmith_cfg_path = Path(model_path) / "edsmith_config.json"
    component: str | None = None
    if edsmith_cfg_path.exists():
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
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

    all_generated: list[str] = []
    with torch.no_grad():
        for i in range(0, len(all_prompts), _EVAL_BATCH_SIZE):
            batch = all_prompts[i : i + _EVAL_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(model.device)
            out = model.generate(**enc, do_sample=False)
            input_len = enc["input_ids"].shape[1]
            for o in out:
                all_generated.append(tokenizer.decode(o[input_len:], skip_special_tokens=True).strip())
            _log(f"  eval batch {i // _EVAL_BATCH_SIZE + 1}/{-(-len(all_prompts) // _EVAL_BATCH_SIZE)}")

    y_true: list[float] = []
    y_pred: list[float] = []
    n_skipped = 0
    for i, band in enumerate(valid_bands):
        component_preds: list[float] = []
        for j in range(n_components):
            raw = all_generated[i * n_components + j]
            try:
                pred = max(1.0, min(9.0, round(float(raw) * 2) / 2))
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

    def _to_chat(row):
        messages = [
            {"role": "user", "content": _format_input(row["question"], row["essay"], row["component"])},
            {"role": "assistant", "content": str(_BANDS[row["label"]])},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return Dataset.from_dict({
        "text": [_to_chat(row) for _, row in df.iterrows()],
        "label": df["label"].tolist(),
    })


def _format_input(question: str, essay: str, component: str) -> str:
    component_name = COMPONENT_HEADINGS.get(component, component)
    return f"Component: {component_name}\n\nQuestion: {question}\n\nEssay: {essay}"
