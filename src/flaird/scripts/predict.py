"""Run FLAIRD predictions and optional local explanations from the command line."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.data.data_collator import preprocess_text
from flaird.utils.explain import FlairdExplainer


@click.command()
@click.option(
    "--model",
    "model_id",
    default="MahmoodAnaam/flaird-modernbert-large-attention-multitask",
    show_default=True,
)
@click.option("--text", default=None, help="Text to analyse.")
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSONL file with a text column.",
)
@click.option("--text-column", default="text", show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write JSON/JSONL results here.",
)
@click.option(
    "--max-length", type=click.IntRange(min=1), default=512, show_default=True
)
@click.option("--top-k", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--explain/--no-explain", default=True, show_default=True)
def main(model_id, text, input_file, text_column, output, max_length, top_k, explain):
    """Classify one text (--text) or all JSONL records (--input-file)."""
    if (text is None) == (input_file is None):
        raise click.UsageError("Provide exactly one of --text or --input-file.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    explainer = FlairdExplainer(model, tokenizer, max_length=max_length)

    records = (
        [{text_column: text}]
        if text is not None
        else [
            json.loads(line)
            for line in Path(input_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    results = []
    for record in records:
        if text_column not in record:
            raise click.ClickException(f"Missing '{text_column}' in an input record.")
        prepared = preprocess_text(str(record[text_column]))
        if explain:
            result = explainer.explain(prepared, top_k=top_k).to_dict()
        else:
            inputs = tokenizer(
                prepared, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            inputs["forensic_features"] = model.extract_forensic_features([prepared])
            with torch.inference_mode():
                logits = model(**inputs).logits
            result = {
                "machine_probability": float(
                    torch.sigmoid(logits.reshape(-1)[0]).item()
                )
            }
        if "id" in record:
            result["id"] = record["id"]
        results.append(result)

    rendered = json.dumps(
        results[0] if text is not None else results, ensure_ascii=False, indent=2
    )
    if output:
        path = Path(output)
        if text is None:
            path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in results
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"Wrote {path}")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    main()
