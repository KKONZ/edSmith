"""edsmith MCP training server — runs inside a Colab GPU session.

Based on the colab-mcp pattern (https://github.com/googlecolab/colab-mcp).
Uses fastmcp 2.x with streamable-http transport to avoid HTTP/2 issues
with Cloudflare tunnels.

Setup in Colab (two cells):

--- Cell 1: install & tunnel ---
    !pip install -q "edsmith[training] @ git+https://github.com/kkonz/edSmith.git" fastmcp
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
         -O cloudflared && chmod +x cloudflared

    import subprocess, re

    proc = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in proc.stderr:
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
        if m:
            print(f"\\nMCP URL: {m.group()}/mcp\\n")
            break

--- Cell 2: start server (blocking) ---
    import unsloth  # must be first — patches transformers before any other import
    import threading
    from edsmith.mcp.server import mcp

    t = threading.Thread(target=mcp.run, kwargs={"transport": "http", "port": 8000}, daemon=True)
    t.start()
    t.join()

Then on your local machine:
    edsmith run-session --config session.yaml --mcp-url https://<tunnel-id>.trycloudflare.com/mcp
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import FastMCP

from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.metrics import compute_all

mcp = FastMCP("edsmith-training")

print("[edsmith-server] Server module loaded", flush=True, file=sys.stderr)

# IELTS component band values — ordinal classes for CORN
_BANDS: list[float] = [b / 2 for b in range(2, 19)]  # 1.0 … 9.0  (17 values)
_NUM_CLASSES: int = len(_BANDS)
_BAND_TO_IDX: dict[float, int] = {b: i for i, b in enumerate(_BANDS)}


# ------------------------------------------------------------------
# Tool: train_scorer
# ------------------------------------------------------------------

@mcp.tool()
async def train_scorer(
    feedback_data: str,
    scorer_config: str,
    output_dir: str,
) -> str:
    """Fine-tune the Scorer (Qwen3 + LoRA + CORN loss) on Phase 1 feedback.

    Args:
        feedback_data: Base64-encoded parquet bytes with columns
            [question, essay, band, component, feedback_text, score].
        scorer_config: JSON string of ScorerConfig fields.
        output_dir: Directory to save the trained model.

    Returns:
        JSON string {"model_path": str}.
    """
    import base64, os, tempfile
    print("[edsmith-server] train_scorer called", flush=True, file=sys.stderr)
    raw = base64.b64decode(feedback_data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(raw)
        tmp_path = f.name
    print(f"[edsmith-server] feedback parquet written to {tmp_path}", flush=True, file=sys.stderr)
    try:
        cfg = json.loads(scorer_config)
        loop = asyncio.get_event_loop()
        model_path = await loop.run_in_executor(None, _train, tmp_path, cfg, output_dir)
        return json.dumps({"model_path": model_path})
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Tool: evaluate_scorer
# ------------------------------------------------------------------

@mcp.tool()
async def evaluate_scorer(
    model_path: str,
    eval_data: str,
    component: str | None = None,
) -> str:
    """Run the trained Scorer on an evaluation set and return predictions.

    Args:
        model_path: Path to the saved model directory (on Colab / Drive).
        eval_data: Base64-encoded parquet bytes with columns [question, essay, band].

    Returns:
        JSON string {"y_true": [...], "y_pred": [...]}.
        Predictions are overall band scores (mean of four component predictions).
    """
    import base64, os, tempfile
    raw = base64.b64decode(eval_data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(raw)
        tmp_path = f.name
    try:
        loop = asyncio.get_event_loop()
        y_true, y_pred = await loop.run_in_executor(None, _evaluate, model_path, tmp_path, component)
        return json.dumps({"y_true": y_true, "y_pred": y_pred})
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Tool: compute_metrics
# ------------------------------------------------------------------

@mcp.tool()
def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """Compute accuracy, adjacent_accuracy, QWK, and SMD."""
    return compute_all(y_true, y_pred, bands=_BANDS)


# ------------------------------------------------------------------
# Training implementation
# ------------------------------------------------------------------

def _log(msg: str) -> None:
    import sys
    print(f"[edsmith-train] {msg}", flush=True, file=sys.stderr)


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
        _log(f"Filtered to component '{cfg['component']}'")
    df["label"] = df["score"].map(_BAND_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    _log(f"Training rows: {len(df)}  label range: {df['label'].min()}–{df['label'].max()}")

    _log(f"Loading model {cfg['model_name']} (4bit) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
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
            max_seq_length=cfg["max_seq_length"],
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
    _log(f"Saved to {out}")

    return str(out)


# ------------------------------------------------------------------
# Evaluation implementation
# ------------------------------------------------------------------

_EVAL_BATCH_SIZE = 8


def _evaluate(model_path: str, eval_data_path: str, component: str | None = None) -> tuple[list[float], list[float]]:
    from unsloth import FastLanguageModel  # must be first
    import numpy as np
    import pandas as pd
    import torch

    _log(f"Evaluating model at {model_path} …")
    df = pd.read_parquet(eval_data_path)
    _log(f"Eval rows: {len(df)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.max_new_tokens = 8
    tokenizer.padding_side = "left"

    target_components = [component] if component else list(COMPONENT_HEADINGS.keys())
    n_components = len(target_components)
    _log(f"Evaluating component(s): {target_components}")

    # Build all prompts upfront and filter rows with unparseable bands
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
            all_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Batched inference
    all_generated: list[str] = []
    with torch.no_grad():
        for i in range(0, len(all_prompts), _EVAL_BATCH_SIZE):
            batch = all_prompts[i : i + _EVAL_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=4096, padding=True).to(model.device)
            out = model.generate(**enc, do_sample=False)
            input_len = enc["input_ids"].shape[1]
            for o in out:
                all_generated.append(tokenizer.decode(o[input_len:], skip_special_tokens=True).strip())
            _log(f"  eval batch {i // _EVAL_BATCH_SIZE + 1}/{-(-len(all_prompts) // _EVAL_BATCH_SIZE)}")

    # Aggregate component predictions per row
    y_true: list[float] = []
    y_pred: list[float] = []
    for i, band in enumerate(valid_bands):
        component_preds: list[float] = []
        for j in range(n_components):
            try:
                pred = max(1.0, min(9.0, round(float(all_generated[i * n_components + j]) * 2) / 2))
            except ValueError:
                pred = 5.0
            component_preds.append(pred)
        y_true.append(band)
        y_pred.append(round(float(np.mean(component_preds)) * 2) / 2)

    _log(f"Evaluation complete. {len(y_true)} predictions.")
    return y_true, y_pred


# ------------------------------------------------------------------
# Dataset
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
        "label": df["label"].tolist(),  # ordinal class index — used by CORN loss via compute_loss_func
    })


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_input(question: str, essay: str, component: str) -> str:
    component_name = COMPONENT_HEADINGS.get(component, component)
    return f"Component: {component_name}\n\nQuestion: {question}\n\nEssay: {essay}"


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
