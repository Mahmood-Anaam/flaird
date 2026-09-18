import json
import sys

import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.utils.explain import FlairdExplainer


@click.command()
@click.argument("model_path")
@click.option("--text", required=True, help="English text to analyze.")
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Optional path to save JSON report to.",
)
@click.option(
    "--device",
    default="auto",
    help="Device to run inference on (cpu, cuda, or auto).",
)
def main(model_path, text, output_file, device):
    """
    FLAIRD Forensic Inference and Explanation CLI.

    Analyzes an English text and produces detailed forensic classification and explanations.
    MODEL_PATH: HuggingFace hub repository name or local directory path.
    """
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        model.to(target_device)
        model.eval()
    except Exception as e:
        click.echo(f"Error loading model from {model_path}: {e}", err=True)
        sys.exit(1)

    explainer = FlairdExplainer(model=model, tokenizer=tokenizer)
    report = explainer.explain(text=text)

    json_report = json.dumps(report, indent=2)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_report)
        click.echo(f"Report saved to {output_file}")
    else:
        click.echo(json_report)


if __name__ == "__main__":
    main()
