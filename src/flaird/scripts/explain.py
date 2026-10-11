"""Export a reproducible explanation without starting the Gradio application."""

import json
from pathlib import Path

import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.utils.explain import ExplanationOptions, FlairdExplainer


@click.command()
@click.option(
    "--model",
    "model_id",
    required=True,
    help="FLAIRD Hub ID or local checkpoint directory.",
)
@click.option(
    "--revision",
    default=None,
    help="Pin the model and tokenizer to a Hub commit or tag.",
)
@click.option(
    "--text-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--baseline-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON array of 35 raw features in FEATURE_NAMES order.",
)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--max-length", type=click.IntRange(min=2), default=512, show_default=True
)
@click.option(
    "--preprocessing",
    type=click.Choice(["training", "tira", "none"]),
    default="training",
    show_default=True,
)
@click.option(
    "--integration-steps", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option(
    "--max-integration-steps",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
)
@click.option(
    "--integration-batch-size", type=click.IntRange(min=1), default=8, show_default=True
)
def main(
    model_id,
    revision,
    text_file,
    output,
    baseline_file,
    device,
    max_length,
    preprocessing,
    integration_steps,
    integration_batch_size,
    max_integration_steps,
):
    """Explain English text using the installed FLAIRD architecture and weights."""
    try:
        text = text_file.read_text(encoding="utf-8")
        baseline = (
            json.loads(baseline_file.read_text(encoding="utf-8"))
            if baseline_file
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        # Use the checkpoint's registered custom architecture, as in trained-model inference.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        model = model.float().to(torch.device(device)).eval()
        options = ExplanationOptions(
            integration_steps=integration_steps,
            integration_batch_size=integration_batch_size,
            max_integration_steps=max_integration_steps,
        )
        result = FlairdExplainer(model, options).explain_text(
            text,
            tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            baseline_features=baseline,
        )
        result.metadata.update(
            {"requested_model": model_id, "requested_revision": revision}
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved explanation to {output}")
    if not result.diagnostics["completeness_passed"]:
        click.echo(
            "Warning: completeness tolerance exceeded; inspect diagnostics before interpreting ranks.",
            err=True,
        )


if __name__ == "__main__":
    main()
