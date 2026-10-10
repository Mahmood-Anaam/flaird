"""Opt-in validation with trained Hub checkpoints and the 20 demo examples.

Run from the repository root with PYTHONPATH=src. This is a functional smoke
check of explanations, not a new benchmark evaluation or a training job.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.data.data_collator import preprocess_text
from flaird.utils.explain import FlairdExplainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    models = args.models or [
        json.loads(p.read_text())["hub_model_id"]
        for p in sorted((root / "configs").glob("flaird-modernbert-large-*.json"))
    ]
    examples_path = root / "demo" / "examples.csv"
    examples = list(csv.DictReader(examples_path.open(encoding="utf-8")))
    torch.set_num_threads(args.threads)
    report = {
        "purpose": "Functional explanation validation; not a benchmark or generator-accuracy evaluation",
        "loader": "AutoModelForSequenceClassification and AutoTokenizer with trust_remote_code=True",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "pystylometry",
                "numpy",
                "huggingface_hub",
            )
        },
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "models": [],
    }
    for model_id in models:
        print(f"Validating {model_id}", flush=True)
        # Exercise the published custom code and trained weights through AutoModel.
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_id, trust_remote_code=True
            )
            .float()
            .eval()
        )
        revision = getattr(model.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        explainer = FlairdExplainer(model)
        rows = []
        for index, example in enumerate(examples):
            result = explainer.explain_text(example["text"], tokenizer)
            normalized = preprocess_text(example["text"])
            batch = tokenizer(
                normalized, return_tensors="pt", truncation=True, max_length=512
            )
            batch["forensic_features"] = model.extract_forensic_features([normalized])
            with torch.no_grad():
                direct = model(**batch, output_fusion_states=True)
            error = abs(float(direct.logits[0, 0]) - result.prediction["machine_logit"])
            if error > 1e-5:
                raise AssertionError(
                    f"Forward mismatch: {model_id}, example {index}, error {error}"
                )
            if not result.diagnostics["completeness_passed"]:
                raise AssertionError(
                    f"Completeness failed: {model_id}, example {index}, {result.diagnostics}"
                )
            json.dumps(result.to_dict(), allow_nan=False)
            assert all(p.grad is None for p in model.parameters())
            rows.append(
                {
                    "example_index": index,
                    "example_label": example["label"],
                    "attack": example["attack"],
                    "machine_probability": result.prediction["machine_probability"],
                    "forward_logit_error": error,
                    "completeness_residual": result.diagnostics[
                        "completeness_residual"
                    ],
                    "completeness_tolerance": result.diagnostics[
                        "completeness_tolerance"
                    ],
                    "integration_steps_used": result.diagnostics[
                        "integration_steps_used"
                    ],
                    "semantic_truncated": result.metadata["semantic_truncated"],
                    "generator_distribution_available": result.prediction[
                        "generator_probabilities"
                    ]
                    is not None,
                    "attention_available": result.groups[0]["attention"] is not None,
                }
            )
        report["models"].append(
            {"model_id": model_id, "revision": revision, "examples": rows}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Passed {len(rows)} examples; revision {revision}", flush=True)
        del model, explainer, tokenizer, batch, direct
        gc.collect()


if __name__ == "__main__":
    main()
