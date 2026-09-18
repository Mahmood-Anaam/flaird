import os
import re
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.modeling.features import FEATURE_NAMES
from flaird.utils.explain import FlairdExplainer

# Available trained FLAIRD models on HuggingFace Hub
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "yusr9/flaird-modernbert-large-attention-multitask")

AVAILABLE_MODELS = {
    "FLAIRD ModernBERT Attention Multi-Task (Recommended)": "yusr9/flaird-modernbert-large-attention-multitask",
    "FLAIRD ModernBERT Concatenation Multi-Task": "yusr9/flaird-modernbert-large-concatenation-multitask",
    "FLAIRD ModernBERT Attention Single-Task": "yusr9/flaird-modernbert-large-attention-single-task",
    "FLAIRD ModernBERT Concatenation Single-Task": "yusr9/flaird-modernbert-large-concatenation-single-task",
    "FLAIRD ModernBERT Attention Multi-Task (Frozen Encoder)": "yusr9/flaird-modernbert-large-attention-multitask-frozen",
    "FLAIRD ModernBERT Concatenation Multi-Task (Frozen Encoder)": "yusr9/flaird-modernbert-large-concatenation-multitask-frozen",
    "FLAIRD ModernBERT Attention Single-Task (Frozen Encoder)": "yusr9/flaird-modernbert-large-attention-single-task-frozen",
    "FLAIRD ModernBERT Concatenation Single-Task (Frozen Encoder)": "yusr9/flaird-modernbert-large-concatenation-single-task-frozen",
}

# Global cache for loaded models and tokenizers
MODEL_CACHE = {}


def load_model_and_tokenizer(model_id_or_name: str):
    target_id = AVAILABLE_MODELS.get(model_id_or_name, model_id_or_name)
    if target_id in MODEL_CACHE:
        return MODEL_CACHE[target_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(target_id, trust_remote_code=True)
        .eval()
        .to(device)
    )
    MODEL_CACHE[target_id] = (model, tokenizer, device)
    return model, tokenizer, device


def preprocess(text: str) -> str:
    if not text:
        return ""
    EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}"
    )
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
    text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.strip()


def analyze_text(text: str, model_choice: str):
    if not text or len(text.strip()) == 0:
        return (
            "Please enter valid English text to analyze.",
            None,
            None,
            None,
            None,
        )

    clean_text = preprocess(text)
    model, tokenizer, device = load_model_and_tokenizer(model_choice)

    explainer = FlairdExplainer(model=model, tokenizer=tokenizer)
    report = explainer.explain(clean_text)

    # 1. Verdict & Probabilities
    verdict = report["verdict"].upper()
    prob_machine = report["machine_probability"]
    prob_human = report["human_probability"]

    verdict_md = f"""
    ### Detection Verdict: **{verdict}**
    - **Machine Generation Probability:** `{prob_machine:.2%}`
    - **Human Authorship Probability:** `{prob_human:.2%}`

    > **Narrative Summary:** {report['narrative_summary']}
    """

    # 2. Plot: Feature Group Importance (Bar / Radar Chart)
    fg_imp = report["feature_group_importance"]
    fg_df = pd.DataFrame(
        [
            {"Feature Group": k.replace("_", " ").title(), "Importance / Weight": v}
            for k, v in fg_imp.items()
        ]
    )
    fg_df = fg_df.sort_values(by="Importance / Weight", ascending=True)

    fig_importance = px.bar(
        fg_df,
        x="Importance / Weight",
        y="Feature Group",
        orientation="h",
        title="Forensic Feature Group Importance / Attention Weight",
        color="Importance / Weight",
        color_continuous_scale="Viridis",
    )
    fig_importance.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)

    # 3. Plot: Generator Breakdown (if available)
    fig_generator = None
    gen_analysis = report["generator_analysis"]
    if gen_analysis:
        gen_probs = gen_analysis["generator_probabilities"]
        gen_df = pd.DataFrame(
            [{"Generator": k, "Probability": v} for k, v in gen_probs.items()]
        ).sort_values(by="Probability", ascending=False)

        fig_generator = px.bar(
            gen_df,
            x="Generator",
            y="Probability",
            title=f"Multi-Task Generator Attribution (Top: {gen_analysis['top_generator']})",
            color="Probability",
            color_continuous_scale="Plasma",
        )
        fig_generator.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)

    # 4. Detailed Raw Features Breakdown Table
    raw_feats = report["raw_features"]
    raw_df = pd.DataFrame(
        [{"Forensic Metric": k, "Extracted Value": round(v, 4)} for k, v in raw_feats.items()]
    )

    return (
        verdict_md,
        fig_importance,
        fig_generator,
        raw_df,
        report,
    )


# Example Texts
EXAMPLE_1 = """Duke Ellington, a titan of jazz, revolutionized the genre through his innovative compositions, showcasing a remarkable ability to integrate voice and instrumental music. Among the notable figures who contributed to this artistic symphony was Ivie Anderson, whose scat singing mirrored the improvisational prowess of musicians like Nanton. In "Ring Dem Bells," Ellington ingeniously interweaves scat singing as a dialogue with saxophones, creating a dynamic call and response that underscores his vision of music as a fluid conversation between voices and instruments. Ellington consistently emphasized the voice as an instrument of equal importance to traditional brass and woodwinds."""

EXAMPLE_2 = """I was walking down 5th avenue when it suddenly started raining cats and dogs! I quickly grabbed my old umbrella, but the wind was blowing so hard it flipped inside out within seconds. Ended up taking shelter in a small cozy coffee shop nearby, ordered a hot cappuccino, and listened to the gentle pitter-patter of raindrops against the window pane. It turned out to be the highlight of my morning."""


with gr.Blocks(title="FLAIRD Interactive Forensics & Disinformation Detector") as demo:
    gr.Markdown(
        """
        # 🔬 FLAIRD: Forensic Linguistics and AI for Disinformation Detection
        ### Multi-Task Hybrid Architecture combining Stylometric Forensics & ModernBERT Transformer Representations
        Select a trained FLAIRD model, enter text to analyze, and receive instant forensic verdict and linguistic breakdown.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="FLAIRD ModernBERT Attention Multi-Task (Recommended)",
                label="Select Trained FLAIRD Model",
            )
            input_text = gr.Textbox(
                lines=10,
                placeholder="Enter English text here to analyze for machine generation...",
                label="Input Text for Forensic Analysis",
                value=EXAMPLE_1,
            )
            analyze_btn = gr.Button("🔍 Run Forensic Analysis", variant="primary")

            gr.Examples(
                examples=[[EXAMPLE_1, "FLAIRD ModernBERT Attention Multi-Task (Recommended)"], [EXAMPLE_2, "FLAIRD ModernBERT Attention Multi-Task (Recommended)"]],
                inputs=[input_text, model_selector],
            )

        with gr.Column(scale=1):
            verdict_output = gr.Markdown(label="Classification Verdict & Probabilities")
            importance_plot = gr.Plot(label="Feature Group Importance")
            generator_plot = gr.Plot(label="Multi-Task Generator Identification")

    with gr.Accordion("📋 Raw Forensic Feature Values & Structured JSON Output", open=False):
        with gr.Row():
            features_table = gr.Dataframe(label="35 Extracted Forensic Metrics")
            raw_json_output = gr.JSON(label="Structured Diagnostic JSON Report")

    analyze_btn.click(
        fn=analyze_text,
        inputs=[input_text, model_selector],
        outputs=[
            verdict_output,
            importance_plot,
            generator_plot,
            features_table,
            raw_json_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()
