"""Generate RAID leaderboard submissions for FLAIRD detectors."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

import click
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

DEFAULT_USERNAME = os.getenv("HF_USERNAME", "MahmoodAnaam")
DEFAULT_MODEL = f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask"
DEFAULT_MODELS = [
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-multitask",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-single-task",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-single-task",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-multitask-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-multitask-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-attention-single-task-frozen",
    f"{DEFAULT_USERNAME}/flaird-modernbert-large-concatenation-single-task-frozen",
]
SEED = 42


def create_metadata(
    detector_name: str,
    model_id: str,
    contact_info: str,
) -> dict[str, str]:
    """Return metadata in the format expected by the RAID leaderboard."""
    return {
        "date_released": date.today().isoformat(),
        "detector_name": detector_name,
        "huggingface_link": f"https://huggingface.co/{model_id}",
        "contact_info": contact_info,
    }


def _scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert binary classifier logits to P(machine-generated)."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1))
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise RuntimeError(
        f"Expected one or two output logits, got shape {tuple(logits.shape)}"
    )


def prepare_loader(
    dataset: Dataset,
    tokenizer: Any,
    feature_column: str | None,
    apply_text_preprocessing: bool,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """Tokenize once, then reuse the inexpensive tensor batches for every model.

    This matters for ``--all-models``: repeatedly running Python collation,
    regex preprocessing and tokenization used to make the GPU wait for the CPU
    eight times. RAID feature configurations already contain feature vectors.
    """
    if "generation" not in dataset.column_names:
        raise click.ClickException(
            "The RAID dataset must contain a 'generation' column."
        )
    if feature_column and feature_column not in dataset.column_names:
        raise click.ClickException(
            f"Feature column '{feature_column}' is not present in the dataset. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )
    if not feature_column:
        raise click.ClickException(
            "RAID inference requires precomputed forensic features. Pass --feature-column "
            "(for example, forensic_features) or use the *_features dataset configuration."
        )

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = batch["generation"]
        if apply_text_preprocessing:
            from flaird.data.data_collator import preprocess_text

            texts = [preprocess_text(str(text)) for text in texts]
        return tokenizer(texts, truncation=True, max_length=max_length)

    prepared = dataset.map(tokenize, batched=True, desc="Tokenizing RAID inputs")
    columns = ["input_ids", "attention_mask", feature_column]
    if "token_type_ids" in prepared.column_names:
        columns.append("token_type_ids")
    prepared = prepared.with_format("torch", columns=columns)
    return torch.utils.data.DataLoader(
        prepared,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def predict(
    loader: torch.utils.data.DataLoader,
    model_id: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> list[float]:
    """Run GPU inference for a prepared RAID loader and return scores in order."""
    click.echo(f"Loading {model_id} on {device} ...")
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    scores: list[float] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.inference_mode(), autocast:
        for batch in tqdm(
            loader, desc=f"Predicting {model_id.split('/')[-1]}", unit="batch"
        ):
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            batch_scores = _scores_from_logits(model(**batch).logits)
            scores.extend(batch_scores.float().cpu().tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def write_submission(
    output_dir: Path,
    model_id: str,
    ids: list[Any],
    scores: list[float],
    contact_info: str,
) -> None:
    """Write the RAID-required predictions.json and metadata.json files."""
    detector_name = model_id.rsplit("/", 1)[-1]
    detector_dir = output_dir / detector_name
    detector_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({"id": ids, "score": scores})
    predictions_file = detector_dir / "predictions.json"
    predictions_df.to_json(predictions_file, orient="records", lines=True)

    metadata = create_metadata(detector_name, model_id, contact_info)
    metadata_file = detector_dir / "metadata.json"
    with (metadata_file).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )
    click.echo(f"Wrote {predictions_file}")
    click.echo(f"Wrote {metadata_file}")


@click.command()
@click.option("--model", "model_id", default=DEFAULT_MODEL, show_default=True)
@click.option("--all-models", is_flag=True, help="Run all eight trained FLAIRD models.")
@click.option("--dataset-path", default="liamdugan/raid", show_default=True)
@click.option("--dataset-name", default="raid_test", show_default=True)
@click.option("--feature-column", default=None)
@click.option("--apply-text-preprocessing", is_flag=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="raid_submissions",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option(
    "--max-length", type=click.IntRange(min=1), default=512, show_default=True
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="DataLoader workers; 0 is fastest for pretokenized Arrow data in most notebooks.",
)
@click.option(
    "--contact-info", default="Email Address: yusrpro9@gmail.com", show_default=True
)
@click.option(
    "--inference-dtype",
    type=click.Choice(["auto", "bfloat16", "float16", "float32"]),
    default="auto",
    show_default=True,
)
def main(
    model_id: str,
    all_models: bool,
    dataset_path: str,
    dataset_name: str,
    feature_column: str | None,
    apply_text_preprocessing: bool,
    output_dir: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
    contact_info: str,
    inference_dtype: str,
) -> None:
    """Generate one RAID submission directory per selected FLAIRD model."""
    set_seed(SEED)
    dataset = load_dataset(dataset_path, name=dataset_name, split="test")
    ids = (
        list(dataset["id"])
        if "id" in dataset.column_names
        else list(range(len(dataset)))
    )
    models = DEFAULT_MODELS if all_models else [model_id]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loaded {len(dataset)} rows from {dataset_path}/{dataset_name}")
    # All official FLAIRD variants use ModernBERT-large and the same tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(models[0], trust_remote_code=True)
    loader = prepare_loader(
        dataset,
        tokenizer,
        feature_column,
        apply_text_preprocessing,
        batch_size,
        max_length,
        num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if inference_dtype == "auto":
        autocast_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if device.type == "cuda" else None)
        )
    else:
        autocast_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": None,
        }[inference_dtype]
    if autocast_dtype is not None and device.type != "cuda":
        raise click.ClickException("bfloat16 and float16 inference require CUDA.")
    for current_model in models:
        scores = predict(loader, current_model, device, autocast_dtype)
        if len(scores) != len(dataset):
            raise RuntimeError(
                f"Model returned {len(scores)} scores for {len(dataset)} dataset rows."
            )
        write_submission(output, current_model, ids, scores, contact_info)

    click.echo(f"Completed {len(models)} detector submission(s) in {output}")


if __name__ == "__main__":
    main()
