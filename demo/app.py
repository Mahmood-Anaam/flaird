import csv
import os
import re
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import pandas as pd


DEFAULT_MODEL_ID = "MahmoodAnaam/flaird-modernbert-large-attention-multitask"
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
EXAMPLES_PATH = Path(__file__).with_name("examples.csv")

EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3})?[\s*\.-]?\(?\d{1,4}\)?[\s*\.-]?\d{2,4}[\s*\.-]?\d{2,6}"
)


def load_examples() -> list[dict[str, str]]:
    with EXAMPLES_PATH.open(encoding="utf-8", newline="") as examples_file:
        return [
            {key: value or "" for key, value in row.items()}
            for row in csv.DictReader(examples_file)
            if row.get("text", "").strip()
        ]


EXAMPLES = load_examples()
EXAMPLE_CHOICES = [
    (
        f"{index + 1:02d} · {example['label'].title()} · "
        f"{example['model'] or 'Unknown source'} · {example['attack'] or 'Original'}",
        str(index),
    )
    for index, example in enumerate(EXAMPLES)
]


@lru_cache(maxsize=1)
def load_model(model_id: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval(), device


def preprocess(text: str) -> str:
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = USER_MENTION_PATTERN.sub("[USER]", text)
    return PHONE_PATTERN.sub(" [PHONE]", text).replace("  [PHONE]", " [PHONE]").strip()


def select_example(example_index: str | None):
    if example_index is None:
        return "", "Choose an example to view its metadata."
    example = EXAMPLES[int(example_index)]
    metadata = (
        f"**Reference label:** {example['label'] or 'Unavailable'} | "
        f"**Generator:** {example['generator_label'] or 'Unavailable'} | "
        f"**Transformation:** {example['attack'] or 'Unspecified'}"
    )
    return example["text"], metadata


def analyze(text: str, model_id: str, top_k: int):
    if not text or not text.strip():
        raise gr.Error("Enter text or choose an example before analyzing.")
    if not model_id or not model_id.strip():
        raise gr.Error("Enter a model ID or set the MODEL_ID environment variable.")

    clean_text = preprocess(text)
    try:
        from flaird.utils import explain_text

        tokenizer, model, device = load_model(model_id.strip())
        report = explain_text(
            model=model,
            tokenizer=tokenizer,
            text=clean_text,
            max_length=MAX_LENGTH,
            top_k=top_k,
        )
    except Exception as exc:
        raise gr.Error(f"Text analysis failed: {exc}") from exc

    machine_probability = report["machine_probability"]
    human_probability = report["human_probability"]
    label = (
        "Likely machine-generated"
        if report["predicted_label"] == "machine"
        else "Likely human-written"
    )
    summary = f"""
### {label}
**Machine probability:** {machine_probability:.1%} · **Human probability:** {human_probability:.1%}

The model score is not proof of authorship. The signals below are analytical aids, not causal evidence.
"""

    class_plot = pd.DataFrame(
        {
            "Class": ["Human", "Machine-generated"],
            "Probability": [human_probability, machine_probability],
        }
    )
    feature_rows = report["feature_attributions"]
    feature_plot = pd.DataFrame(
        {
            "Feature": [row["feature"] for row in reversed(feature_rows)],
            "Signed attribution": [
                row["attribution"] for row in reversed(feature_rows)
            ],
            "Direction": [row["direction"] for row in reversed(feature_rows)],
        }
    )
    feature_table = pd.DataFrame(
        [
            {
                "Feature": row["feature"],
                "Extracted value": row["value"],
                "Attribution": row["attribution"],
                "Direction": row["direction"],
            }
            for row in feature_rows
        ]
    )

    group_rows = report["feature_group_attention"] or []
    group_plot = pd.DataFrame(
        {
            "Feature group": [row["group"] for row in reversed(group_rows)],
            "Attention weight": [row["attention"] for row in reversed(group_rows)],
        },
        columns=["Feature group", "Attention weight"],
    )
    generator_rows = (report["generator_probabilities"] or [])[:5]
    generator_plot = pd.DataFrame(
        {
            "Generator": [row["generator"] for row in reversed(generator_rows)],
            "Probability": [row["probability"] for row in reversed(generator_rows)],
        },
        columns=["Generator", "Probability"],
    )

    fusion_description = (
        f"Fusion: `{report['fusion_type']}` | "
        f"Tokens: `{report['token_count']}` | Device: `{device}`"
    )
    if report["possibly_truncated"]:
        fusion_description += (
            " | Input reached the token limit and may have been truncated."
        )
    if report["fusion_gate"] is not None:
        fusion_description += f" | Mean fusion gate: `{report['fusion_gate']:.3f}`"

    return (
        summary,
        class_plot,
        feature_plot,
        feature_table,
        gr.update(value=group_plot, visible=bool(group_rows)),
        gr.update(value=generator_plot, visible=bool(generator_rows)),
        fusion_description,
    )


CSS = """
:root {
  --ink: #172a24;
  --muted: #63736c;
  --line: #dce6df;
  --paper: #f4f7f2;
  --accent: #087e63;
  --lime: #d9f36a;
}
body, .gradio-container { background: var(--paper) !important; color: var(--ink); }
.gradio-container { max-width: 1320px !important; }
.hero { padding: 30px 0 20px; border-bottom: 1px solid var(--line); }
.hero h1 { font-size: 34px !important; line-height: 1.15 !important; color: var(--ink); }
.hero p { color: var(--muted); max-width: 760px; line-height: 1.8; }
.section-title { margin: 22px 0 8px; color: var(--ink); }
.primary-button { background: var(--accent) !important; color: white !important; border: 0 !important; }
.result-panel { border-inline-start: 4px solid var(--accent); padding-inline-start: 16px; }
#text-input textarea { direction: ltr; text-align: left; }
footer { display: none !important; }
@media (max-width: 700px) {
  .hero { padding-top: 16px; }
  .hero h1 { font-size: 28px !important; }
}
"""


with gr.Blocks(title="FLAIRD | Machine-Generated Text Analysis") as demo:
    gr.Markdown(
        """
        # FLAIRD | Forensic Text Analysis
        Analyze English text with a hybrid of contextual representations and forensic-linguistic features.
        Inspect the signals associated with the model's prediction.
        **This is a research decision-support tool, not a definitive authorship judgment.**
        """,
        elem_classes="hero",
    )

    with gr.Row():
        with gr.Column(scale=7, min_width=320):
            gr.Markdown("### Text to analyze", elem_classes="section-title")
            example_picker = gr.Dropdown(
                choices=EXAMPLE_CHOICES,
                value=None,
                label="Demo examples",
                info="Human and machine-written examples with varied transformations.",
            )
            example_metadata = gr.Markdown("Choose an example to view its metadata.")
            text_input = gr.Textbox(
                label="English text",
                placeholder="Paste text here or choose a demo example...",
                lines=14,
                max_lines=24,
                elem_id="text-input",
            )
            example_picker.change(
                select_example,
                inputs=example_picker,
                outputs=[text_input, example_metadata],
            )
            with gr.Row():
                model_input = gr.Textbox(value=MODEL_ID, label="Model ID", scale=4)
                top_k_input = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=8,
                    step=1,
                    label="Features to show",
                    scale=2,
                )
            analyze_button = gr.Button(
                "Analyze text",
                variant="primary",
                elem_classes="primary-button",
            )
        with gr.Column(scale=5, min_width=300):
            gr.Markdown("### Prediction", elem_classes="section-title")
            summary_output = gr.Markdown(elem_classes="result-panel")
            class_plot = gr.BarPlot(
                label="Class probabilities",
                x="Class",
                y="Probability",
                y_lim=[0, 1],
                height=240,
            )
            status_output = gr.Markdown()

    gr.Markdown("## Forensic-linguistic evidence", elem_classes="section-title")
    with gr.Row():
        attribution_plot = gr.BarPlot(
            label="Largest local feature contributions",
            x="Signed attribution",
            y="Feature",
            color="Direction",
            color_map={"machine": "#c45839", "human": "#087e63", "neutral": "#89958e"},
            height=390,
        )
        attention_plot = gr.BarPlot(
            label="Attention across feature groups",
            x="Attention weight",
            y="Feature group",
            height=390,
            visible=False,
        )

    feature_table = gr.Dataframe(
        label="Top feature details", interactive=False, wrap=True
    )
    generator_plot = gr.BarPlot(
        label="Generator-class probabilities (top 5)",
        x="Probability",
        y="Generator",
        y_lim=[0, 1],
        height=300,
        visible=False,
    )
    gr.Markdown(
        """
        **Reading the charts:** A positive attribution means the feature locally increased the machine logit;
        a negative value means it decreased it. Attention weights describe allocation within the fusion layer,
        not causality. Long inputs may be truncated at the token limit.
        """
    )

    analyze_button.click(
        analyze,
        inputs=[text_input, model_input, top_k_input],
        outputs=[
            summary_output,
            class_plot,
            attribution_plot,
            feature_table,
            attention_plot,
            generator_plot,
            status_output,
        ],
        api_name="analyze",
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CSS, share=True)
