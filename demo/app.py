import os
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = os.environ.get(
    "MODEL_ID", "MahmoodAnaam/flaird-modernbert-large-attention-multitask"
)
max_length = int(os.environ.get("MAX_LENGTH", 512))

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = (
    AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    .eval()
    .to(device)
)


def preprocess(text: str) -> str:
    EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}"
    )
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
    text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.strip()


def predict(text: str) -> str:
    text = preprocess(text)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs["forensic_features"] = model.extract_forensic_features(
        [text], return_tensors=True
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_fusion_states=True)

    logits = outputs.logits.cpu()[0]
    score = torch.sigmoid(logits).numpy()

    return f"machine score: {score}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Text(),
    outputs=gr.Text(),
)
demo.launch()
