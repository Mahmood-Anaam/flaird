"""Gradio Space for interactive FLAIRD detection and evidence inspection."""

import os

import gradio as gr
import pandas as pd
import spaces
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.data.data_collator import preprocess_text
from flaird.utils.explain import FlairdExplainer

MODEL_ID = os.environ.get(
    "MODEL_ID", "MahmoodAnaam/flaird-modernbert-large-attention-multitask"
)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
MODEL = (
    AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    .to(DEVICE)
    .eval()
)
EXPLAINER = FlairdExplainer(MODEL, TOKENIZER, max_length=MAX_LENGTH)


def chart_frame(values: dict[str, float], column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"indicator": list(values), column: list(values.values())}
    ).sort_values(column)


@spaces.GPU
def analyse(text: str):
    """Return a decision, the fusion evidence, and concise local explanations."""
    explanation = EXPLAINER.explain(preprocess_text(text), top_k=12)
    result = {
        "Machine-generated": round(explanation.machine_probability, 4),
        "Human-written": round(1 - explanation.machine_probability, 4),
    }
    generator = (
        pd.DataFrame(
            {
                "generator": list(explanation.generator_probabilities),
                "probability": list(explanation.generator_probabilities.values()),
            }
        ).sort_values("probability", ascending=False)
        if explanation.generator_probabilities
        else pd.DataFrame({"generator": [], "probability": []})
    )
    attention = chart_frame(explanation.feature_attention or {}, "attention")
    contributions = chart_frame(explanation.feature_contributions, "contribution")
    tokens = pd.DataFrame(
        explanation.token_contributions
        or [{"token": "No token evidence available", "score": 0.0}]
    )
    status = f"**Decision:** {explanation.predicted_label}  \\n**Confidence:** {explanation.confidence:.1%}"
    if explanation.fusion_gate is not None:
        status += f"  \\n**Semantic fusion gate:** {explanation.fusion_gate:.1%} (higher values favour Transformer semantic evidence)."
    status += "  \n*Evidence scores are local gradient-based sensitivities, not causal proof. Use this tool as decision support.*"
    return result, status, generator, attention, contributions, tokens


with gr.Blocks(
    theme=gr.themes.Soft(), title="FLAIRD | Forensic AI-text detection"
) as demo:
    gr.Markdown(
        "# FLAIRD\n"
        "**Forensic Linguistic and AI-Integrated Robust Detector** — combines ModernBERT semantic signals with 35 interpretable linguistic indicators."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Text to analyse",
                lines=16,
                placeholder="Paste English text here…",
            )
            run = gr.Button("Analyse text", variant="primary")
        with gr.Column(scale=2):
            decision = gr.Label(label="Binary prediction", num_top_classes=2)
            summary = gr.Markdown()
            gr.Markdown("### Likely generator family (auxiliary task)")
            generator = gr.Dataframe(
                headers=["generator", "probability"], interactive=False
            )
    with gr.Row():
        attention = gr.BarPlot(
            x="attention",
            y="indicator",
            orientation="horizontal",
            title="Cross-attention over forensic feature groups",
        )
        contributions = gr.BarPlot(
            x="contribution",
            y="indicator",
            orientation="horizontal",
            title="Forensic group contribution to this prediction",
        )
    tokens = gr.Dataframe(
        headers=["token", "score"], label="Most influential tokens", interactive=False
    )
    run.click(
        analyse, text, [decision, summary, generator, attention, contributions, tokens]
    )

if __name__ == "__main__":
    demo.launch()
