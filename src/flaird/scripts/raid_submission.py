import json
import os
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from flaird.data.data_collator import DataCollator

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

DEFAULT_MODELS = [
    "yusr9/flaird-modernbert-large-attention-multitask",
    "yusr9/flaird-modernbert-large-concatenation-multitask",
    "yusr9/flaird-modernbert-large-attention-single-task",
    "yusr9/flaird-modernbert-large-concatenation-single-task",
    "yusr9/flaird-modernbert-large-attention-multitask-frozen",
    "yusr9/flaird-modernbert-large-concatenation-multitask-frozen",
    "yusr9/flaird-modernbert-large-attention-single-task-frozen",
    "yusr9/flaird-modernbert-large-concatenation-single-task-frozen",
]


def load_raid_dataset(data_path: str | None = None, split: str = "test") -> pd.DataFrame:
    """
    Load RAID test dataset either from a local CSV file or via Hugging Face datasets.
    """
    if data_path and os.path.exists(data_path):
        click.echo(f"Loading RAID dataset from local file: {data_path}")
        df = pd.read_csv(data_path)
    else:
        click.echo(f"Loading RAID dataset from Hugging Face hub (split: {split})...")
        from datasets import load_dataset

        ds = load_dataset("liamdugan/raid", split=split)
        df = ds.to_pandas()

    if "id" not in df.columns:
        df["id"] = df.index
    return df


def run_batch_inference(
    test_df: pd.DataFrame,
    model_path: str,
    device: str | torch.device = "auto",
    batch_size: int = 16,
) -> list[float]:
    """
    Run batched inference on RAID test DataFrame using FLAIRD model.
    """
    target_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    click.echo(f"Loading model & tokenizer from: {model_path} onto {target_device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    model.to(target_device)
    model.eval()

    test_ds = Dataset.from_pandas(test_df)
    collator = DataCollator(tokenizer, include_labels=False)
    dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    scores = []
    for batch in tqdm(dataloader, desc=f"Evaluating {model_path.split('/')[-1]}"):
        batch = {k: v.to(target_device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        logits = outputs.logits.reshape(-1)
        probs = torch.sigmoid(logits).cpu().tolist()
        scores.extend(probs)

    return scores


def create_metadata(
    detector_name: str,
    hf_link: str,
    website: str = "https://github.com/Mahmood-Anaam/flaird",
    paper_link: str = "https://github.com/Mahmood-Anaam/flaird",
    contact_info: str = "mahmood.anaam@example.com",
) -> dict:
    """Create metadata dictionary following RAID template format."""
    return {
        "date_released": "2025-03-01",
        "detector_name": detector_name,
        "website": website,
        "paper_link": paper_link,
        "huggingface_link": hf_link,
        "github_link": "https://github.com/Mahmood-Anaam/flaird",
        "contact_info": contact_info,
    }


@click.command()
@click.option(
    "--model",
    "model_path",
    default="yusr9/flaird-modernbert-large-attention-multitask",
    help="Hugging Face model repository name or local model path.",
)
@click.option(
    "--all-models",
    is_flag=True,
    help="If set, run predictions for all 8 trained FLAIRD models.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to local RAID test dataset CSV (e.g., test.csv).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="raid_submissions",
    help="Base output directory where RAID submissions will be stored.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Inference batch size.",
)
def main(model_path, all_models, data_path, output_dir, batch_size):
    """
    RAID Leaderboard Submission Generator for FLAIRD models.
    Generates predictions.json and metadata.json according to RAID leaderboard requirements.
    """
    models_to_run = DEFAULT_MODELS if all_models else [model_path]
    test_df = load_raid_dataset(data_path=data_path, split="test")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    for model_id in models_to_run:
        detector_name = model_id.split("/")[-1]
        detector_dir = base_out / detector_name
        detector_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n--- Processing RAID Submission for: {detector_name} ---")
        scores = run_batch_inference(
            test_df=test_df,
            model_path=model_id,
            batch_size=batch_size,
        )

        # Structure predictions format: list of scores or dict mapping id -> score
        # RAID accepts list of predictions or dict matching test dataset order
        predictions_file = detector_dir / "predictions.json"
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(scores, f)

        metadata = create_metadata(
            detector_name=detector_name,
            hf_link=f"https://huggingface.co/{model_id}",
        )
        metadata_file = detector_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        click.echo(f"Successfully generated:")
        click.echo(f"  - {predictions_file}")
        click.echo(f"  - {metadata_file}")


if __name__ == "__main__":
    main()
