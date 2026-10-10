"""Lazy, serialized inference with one resident checkpoint and no text logging."""

import csv
import gc
import os
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.utils.explain import ExplanationOptions, FlairdExplainer

PREFIX = "MahmoodAnaam/flaird-modernbert-large-"
MODELS = {
    f"{fusion.title()} · {task} · {encoder}": PREFIX
    + fusion
    + "-"
    + task
    + ("-frozen" if encoder == "frozen" else "")
    for fusion in ("attention", "concatenation")
    for task in ("multitask", "single-task")
    for encoder in ("trainable", "frozen")
}
DEFAULT_MODEL = os.getenv("MODEL_ID", PREFIX + "attention-multitask")
MAX_CHARACTERS = 30_000
POLICIES = {
    "Training normalization": "training",
    "TIRA normalization": "tira",
    "Raw text": "none",
}


def load_examples(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("examples.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"text", "attack", "model", "label", "generator_label"}
    if not rows or any(
        not required <= row.keys() or not row["text"].strip() for row in rows
    ):
        raise ValueError(
            "examples.csv must contain nonempty text and provenance columns"
        )
    return rows


class ModelRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.model_id = self.model = self.tokenizer = None

    def _load(self, model_id):
        if model_id not in MODELS.values():
            raise ValueError("Select one of the eight FLAIRD checkpoints")
        if self.model_id == model_id:
            return
        self.model_id = self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_id, trust_remote_code=True
            )
            .float()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=getattr(model.config, "_commit_hash", None),
            trust_remote_code=True,
        )
        device = os.getenv(
            "FLAIRD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        if device not in ("cpu", "cuda"):
            raise ValueError("FLAIRD_DEVICE must be cpu or cuda")
        model.to(device)
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id

    def explain(self, text, model_id, threshold=0.5, policy="training"):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter English text or choose an example before analyzing")
        if len(text) > MAX_CHARACTERS:
            raise ValueError(f"Text is limited to {MAX_CHARACTERS:,} characters")
        if policy not in POLICIES.values():
            raise ValueError("Unknown normalization policy")
        options = ExplanationOptions(threshold=float(threshold))
        with self.lock:
            self._load(model_id)
            return (
                FlairdExplainer(self.model, options)
                .explain_text(
                    text, self.tokenizer, max_length=512, preprocessing=policy
                )
                .to_dict()
            )


runtime = ModelRuntime()
