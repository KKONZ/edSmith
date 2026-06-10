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
    import nest_asyncio
    nest_asyncio.apply()
    from edsmith.mcp.server import mcp
    mcp.run(transport="http", port=8000)

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
        y_true, y_pred = await loop.run_in_executor(None, _evaluate, model_path, tmp_path)
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
    from unsloth import FastModel  # must be first — patches transformers/torch
    import numpy as np
    import pandas as pd
    import torch
    from coral_pytorch.losses import corn_loss
    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

    _log(f"Python {sys.version}")
    _log(f"Config: {cfg}")
    _log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _log(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    _log("Loading feedback data …")
    df = pd.read_parquet(feedback_path)
    df = df.dropna(subset=["score"])
    df["label"] = df["score"].map(_BAND_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    _log(f"Training rows: {len(df)}  label range: {df['label'].min()}–{df['label'].max()}")

    _log(f"Loading model {cfg['model_name']} (4bit={cfg['load_in_4bit']}) …")
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    _log("Model loaded. Applying LoRA …")
    model = FastModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    _log(f"LoRA applied. hidden_size={model.config.hidden_size}")

    corn_head = torch.nn.Linear(model.config.hidden_size, _NUM_CLASSES - 1).to(model.device)
    _log(f"CORN head on device: {model.device}")

    class _CORNTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = corn_head(last_hidden)
            loss = corn_loss(logits, labels, num_classes=_NUM_CLASSES)
            return (loss, outputs) if return_outputs else loss

    _log(f"Tokenizing {len(df)} examples (max_length={cfg['max_seq_length']}) …")
    dataset = _ScorerDataset(df, tokenizer, cfg["max_seq_length"])
    _log(f"Dataset ready ({len(dataset)} items). Building trainer …")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = _CORNTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    _log("Starting training …")
    trainer.train()
    _log("Training complete. Saving model …")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    torch.save(corn_head.state_dict(), str(out / "corn_head.pt"))
    _log(f"Saved to {out}")

    return str(out)


# ------------------------------------------------------------------
# Evaluation implementation
# ------------------------------------------------------------------

def _evaluate(model_path: str, eval_data_path: str) -> tuple[list[float], list[float]]:
    from unsloth import FastModel  # must be first
    import numpy as np
    import pandas as pd
    import torch
    from coral_pytorch.dataset import corn_label_from_logits

    _log(f"Evaluating model at {model_path} …")
    df = pd.read_parquet(eval_data_path)
    _log(f"Eval rows: {len(df)}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    hidden_size = model.config.hidden_size
    corn_head = torch.nn.Linear(hidden_size, _NUM_CLASSES - 1).to(model.device)
    corn_head.load_state_dict(
        torch.load(str(Path(model_path) / "corn_head.pt"), map_location=model.device)
    )

    model.eval()
    corn_head.eval()

    y_true: list[float] = []
    y_pred: list[float] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            component_preds: list[float] = []
            for component in COMPONENT_HEADINGS:
                text = _format_input(row["question"], row["essay"], component)
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(model.device)

                out = model(**enc, output_hidden_states=True)
                last_hidden = out.hidden_states[-1][:, -1, :]
                logits = corn_head(last_hidden)
                idx = int(corn_label_from_logits(logits).item())
                component_preds.append(_BANDS[min(idx, len(_BANDS) - 1)])

            pred_band = round(float(np.mean(component_preds)) * 2) / 2
            y_true.append(float(row["band"]))
            y_pred.append(pred_band)

    _log(f"Evaluation complete. {len(y_true)} predictions.")
    return y_true, y_pred


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class _ScorerDataset:
    def __init__(self, df, tokenizer, max_length: int) -> None:
        self._encodings = tokenizer(
            [
                _format_input(row["question"], row["essay"], row["component"])
                for _, row in df.iterrows()
            ],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self._labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
            "labels": self._labels[idx],
        }


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
