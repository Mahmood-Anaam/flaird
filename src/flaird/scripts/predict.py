import json
from pathlib import Path

import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.utils import explain_text


@click.command()
@click.option(
    "--model-id", required=True, help="Local checkpoint path or Hugging Face model ID."
)
@click.option("--text", help="Text to analyze.")
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="UTF-8 text file to analyze.",
)
@click.option(
    "--max-length",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
    help="Maximum tokenized sequence length.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of forensic features to include in the ranked explanation.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Inference device.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the JSON report to this path instead of stdout.",
)
def main(model_id, text, input_file, max_length, top_k, device, output):
    """Generate a FLAIRD prediction and forensic explanation for one text."""
    if (text is None) == (input_file is None):
        raise click.UsageError("Provide exactly one of --text or --input-file.")

    if input_file is not None:
        text = input_file.read_text(encoding="utf-8")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    try:
        report = explain_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
            top_k=top_k,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if output is None:
        click.echo(serialized)
    else:
        output.write_text(serialized + "\n", encoding="utf-8")
        click.echo(f"Explanation written to {output}")


if __name__ == "__main__":
    main()
