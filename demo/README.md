---
title: Flaird
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: FLAIRD:Forensic Linguistic and AI-Integrated Robust Detector
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# FLAIRD Demo

An interactive interface for analyzing English text with FLAIRD. It presents
machine and human class probabilities, the strongest local forensic-feature
attributions, feature-group attention, and generator-class probabilities when
the selected model supports them.

## Run Locally

Python 3.12 and the project environment are required. From the repository root:

```bash
pip install -e .
python demo/app.py
```

Open `http://127.0.0.1:7860`. The default model is
`MahmoodAnaam/flaird-modernbert-large-attention-multitask`. Change it in the
Model ID field or with the `MODEL_ID` environment variable. `MAX_LENGTH` sets
the maximum token count and defaults to 512.

The UI and examples load without downloading a checkpoint. The tokenizer and
model are loaded only on the first analysis request, using `AutoTokenizer` and
`AutoModelForSequenceClassification` with `trust_remote_code=True`. The first
analysis therefore needs internet access unless the model files are cached
locally. The process caches the loaded model for subsequent requests.

## Hugging Face Spaces

To create a standalone Space, publish the contents of this directory at the
Space repository root so `app.py`, `README.md`, `requirements.txt`, and
`examples.csv` are at the root. The metadata at the top of this README selects
the Gradio SDK and entry point. `requirements.txt` installs the project and its
dependencies. Set `MODEL_ID` in the Space settings to select a published
checkpoint; set `MAX_LENGTH` if needed. The first analysis downloads model
weights, so choose Space hardware with enough memory for the checkpoint.

## Use the Interface

1. Choose an example or paste English text into the input field.
2. Review the example's reference metadata; it is descriptive and is not sent
   to the model.
3. Select **Analyze text** to view the prediction and charts.
4. Change the number of displayed features to show a shorter or longer ranking.

The app reads examples directly from `examples.csv` and supports multiline CSV
cells. The file contains 20 examples (19 machine-generated and one human) from
multiple generators and transformation types. It is an imbalanced demo
collection, not an evaluation set.

## Reading the Charts and Limitations

- Class probabilities show the classifier output; they do not prove authorship.
- A positive attribution means a feature locally increased the machine logit;
  a negative attribution means it decreased it. The UI uses gradient-times-input.
- Attention weights show how the fusion layer allocates attention across feature
  groups; they are not causal importance scores. They are unavailable for some
  concatenation models.
- Generator probabilities may be unavailable for single-task models.
- Inputs may be truncated at the token limit. The UI warns when the sequence
  reaches `MAX_LENGTH`.

Do not use a FLAIRD result as an automated decision or definitive evidence of
authorship. The system is intended for research support and human-reviewed
analysis.

## Technical Documentation

- [Model architecture and experiment design](../docs/model-architecture.md)
- [Dataset preparation and reproducibility notes](../docs/dataset-preparation.md)
- [PAN26 and RAID evaluation results](../docs/evaluation-results.md)
- [Explanation method and limitations](../docs/explanation_module.md)
