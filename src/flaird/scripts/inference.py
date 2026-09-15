import json

import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.data.data_collator import DataCollator


@click.command()
@click.argument("model_path")
@click.option("--text", required=True, help="English text to analyze.")
def main(model_path, text):
    """
    Inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    ).eval()
    batch = DataCollator(tokenizer, include_labels=False)([{"text": text}])
    with torch.inference_mode():
        output = model(**batch)
    probability = torch.sigmoid(output.logits).item()
    report = {
        "probability": probability,
        "feature_attention": output.model_outputs.fusion_outputs.feature_attention,
    }

    click.echo(json.dumps(report.__dict__, indent=2))
