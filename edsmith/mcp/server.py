"""edsmith MCP training server — runs inside a Colab GPU session.

Setup in Colab (two cells):

--- Cell 1: install & tunnel ---
    !pip install -q edsmith mcp coral-pytorch unsloth
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\
         -O cloudflared && chmod +x cloudflared

    import re, subprocess, threading

    def _start_tunnel():
        proc = subprocess.Popen(
            ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
            stderr=subprocess.PIPE,
        )
        for line in proc.stderr:
            m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line.decode())
            if m:
                print(f"\\nMCP URL: {m.group()}/sse")
                break

    threading.Thread(target=_start_tunnel, daemon=True).start()

--- Cell 2: start server (blocking) ---
    import subprocess
    subprocess.run(["python", "-m", "edsmith.mcp.server"])

Then on your local machine:
    edsmith run-session --config session.yaml --mcp-url https://<tunnel-id>.trycloudflare.com/sse
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mcp.server.fastmcp import FastMCP

from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.metrics import compute_all

mcp = FastMCP("edsmith-training", json_response=True)

# IELTS component band values — ordinal classes for CORN
_BANDS: list[float] = [b / 2 for b in range(2, 19)]  # 1.0 … 9.0  (17 values)
_NUM_CLASSES: int = len(_BANDS)
_BAND_TO_IDX: dict[float, int] = {b: i for i, b in enumerate(_BANDS)}


# ------------------------------------------------------------------
# Tool: train_scorer
# ------------------------------------------------------------------

@mcp.tool()
async def train_scorer(
    feedback_path: str,
    scorer_config: str,  # JSON-encoded ScorerConfig.model_dump()
    output_dir: str,
) -> str:
    """Fine-tune the Scorer (Qwen3 + LoRA + CORN loss) on Phase 1 feedback.

    Args:
        feedback_path: Path to parquet file with columns
            [question, essay, band, component, feedback_text, score].
        scorer_config: JSON string of ScorerConfig fields.
        output_dir: Directory to save the trained model.

    Returns:
        JSON string {"model_path": str}.
    """
    cfg = json.loads(scorer_config)
    loop = asyncio.get_event_loop()
    model_path = await loop.run_in_executor(
        None, _train, feedback_path, cfg, output_dir
    )
    return json.dumps({"model_path": model_path})


# ------------------------------------------------------------------
# Tool: evaluate_scorer
# ------------------------------------------------------------------

@mcp.tool()
async def evaluate_scorer(
    model_path: str,
    eval_data_path: str,
) -> str:
    """Run the trained Scorer on an evaluation set and return predictions.

    Args:
        model_path: Path to the saved model directory.
        eval_data_path: Path to parquet file with columns [question, essay, band].

    Returns:
        JSON string {"y_true": [...], "y_pred": [...]}.
        Predictions are overall band scores (mean of four component predictions).
    """
    loop = asyncio.get_event_loop()
    y_true, y_pred = await loop.run_in_executor(
        None, _evaluate, model_path, eval_data_path
    )
    return json.dumps({"y_true": y_true, "y_pred": y_pred})


# ------------------------------------------------------------------
# Tool: compute_metrics
# ------------------------------------------------------------------

@mcp.tool()
def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """Compute accuracy, adjacent_accuracy, QWK, and SMD.

    Args:
        y_true: Ground-truth band scores.
        y_pred: Predicted band scores.

    Returns:
        Dict with keys: accuracy, adjacent_accuracy, qwk, smd.
    """
    return compute_all(y_true, y_pred, bands=_BANDS)


# ------------------------------------------------------------------
# Training implementation
# ------------------------------------------------------------------

def _train(feedback_path: str, cfg: dict, output_dir: str) -> str:
    from coral_pytorch.losses import corn_loss
    from transformers import TrainingArguments, Trainer, default_data_collator
    from unsloth import FastModel

    df = pd.read_parquet(feedback_path)
    df = df.dropna(subset=["score"])

    df["label"] = df["score"].map(_BAND_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    model = FastModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
    )

    # CORN classification head attached to the model so the Trainer can see it
    model.corn_head = torch.nn.Linear(model.config.hidden_size, _NUM_CLASSES - 1).to(model.device)

    class _CORNTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = model.corn_head(last_hidden)
            loss = corn_loss(logits, labels, num_classes=_NUM_CLASSES)
            return (loss, outputs) if return_outputs else loss

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
        train_dataset=_ScorerDataset(df, tokenizer, cfg["max_seq_length"]),
        data_collator=default_data_collator,
    )
    trainer.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Save classifier head separately (not part of the PEFT model)
    torch.save(classifier.state_dict(), str(out / "classifier.pt"))

    return str(out)


# ------------------------------------------------------------------
# Evaluation implementation
# ------------------------------------------------------------------

def _evaluate(model_path: str, eval_data_path: str) -> tuple[list[float], list[float]]:
    from unsloth import FastModel
    from coral_pytorch.dataset import corn_label_from_logits

    df = pd.read_parquet(eval_data_path)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    hidden_size = model.config.hidden_size
    classifier = torch.nn.Linear(hidden_size, _NUM_CLASSES - 1).to(model.device)
    classifier.load_state_dict(
        torch.load(str(Path(model_path) / "classifier.pt"), map_location=model.device)
    )

    model.eval()
    classifier.eval()

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
                logits = classifier(last_hidden)
                idx = int(corn_label_from_logits(logits).item())
                component_preds.append(_BANDS[min(idx, len(_BANDS) - 1)])

            # Overall band = mean of four component predictions, rounded to 0.5
            pred_band = round(float(np.mean(component_preds)) * 2) / 2
            y_true.append(float(row["band"]))
            y_pred.append(pred_band)

    return y_true, y_pred


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class _ScorerDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self._encodings = tokenizer(
            [
                _format_input(row["question"], row["essay"], row["component"])
                for _, row in df.iterrows()
            ],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self._labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
    mcp.run(transport="sse")
