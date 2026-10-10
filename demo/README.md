---
title: FLAIRD
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Detect machine-generated text and explore forensic explanations
---

# FLAIRD detection and explanation workbench

An English-text research interface for all eight trained FLAIRD checkpoints. Paste text or select one of the 20 original CSV examples; inspect origin scores, signed forensic feature effects, group sensitivity, available attention diagnostics, and the auxiliary generator distribution. Download a structured explanation report without the original text.

## Run from the GitHub repository

Use Python 3.12 and install a PyTorch build appropriate for your CPU or CUDA host, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
python demo/app.py
```

Open `http://localhost:7860`. No checkpoint is downloaded merely by opening the app. The first analysis loads the selected model; switching checkpoints evicts the previous one. CPU analysis works, but a GPU makes large encoders more responsive. No paid hardware is provisioned automatically.

## Deploy the demo folder to a Hugging Face Space

Create/select a **Gradio Space** and upload the contents of this folder to its repository root, preserving these files:

```text
README.md             # Space metadata, Python and Gradio versions
app.py                # entry point, Blocks and events
runtime.py            # lazy AutoModel loading and serialized analysis
presentation.py       # charts and summaries
style.css             # responsive visual design
requirements.txt      # Python package and Plotly dependencies
examples.csv          # unmodified demonstration texts and annotations
```

`requirements.txt` pins the explanation package to the phase-one commit so the deployment works while its PR is awaiting merge. The Space builds dependencies automatically. After both PRs merge, update the pin deliberately to a reviewed immutable commit; using an unmerged `main` would omit the explanation implementation.

This app supports ordinary **CPU or dedicated CUDA GPU** Spaces. ZeroGPU is not configured: its allocation decorator and device lifecycle need a separate integration. Select a host with enough RAM for ModernBERT-large in float32, including loading overhead. Only one model is resident; the global inference queue allows one active analysis and eight waiting events. Test host memory and cold-start latency before public use. This phase provides a deployable folder; it does not claim that a live Space has been published.

## Environment settings

Set variables in the shell or in **Space Settings → Variables and secrets**. `.env.example` is a reference; `.env` is not automatically loaded.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `MahmoodAnaam/flaird-modernbert-large-attention-multitask` | Initial checkpoint; must be one of the eight built-in IDs |
| `FLAIRD_DEVICE` | CUDA when available, otherwise CPU | Override with `cpu` or `cuda` |
| `FLAIRD_CPU_THREADS` | `4` | Positive CPU thread count |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Listen address |
| `GRADIO_SERVER_PORT` | `7860` | Listen port |
| `HF_TOKEN` | unset | Optional Hub authentication; configure as a secret |

The public checkpoints need no token. Both model and tokenizer load through Transformers Auto classes with `trust_remote_code=True`; the tokenizer uses the model's resolved Hub revision. Each report records that revision. Do not put access tokens in this repository.

## Read the results

- **Decision:** machine when its probability is strictly greater than the selected threshold. Scores are uncalibrated model outputs.
- **Feature effects:** largest 15 signed integrated-gradient effects; the full table preserves all 35 feature values and ranks. Positive values move the machine logit upward relative to the reference; negative values move it downward.
- **Feature groups:** sums of feature effects and separate zero-reference group replacement tests. Replacement effects are percentage-point changes and are not additive.
- **Attention:** actual head-average attention over seven forensic groups, available only for attention fusion. It is not token saliency or causal importance.
- **Generator family:** independent auxiliary scores, including `human`; available only for multitask models and not verified generator identity.
- **Quality & report:** completeness residual/tolerance, zero reference, semantic gate diagnostic, preprocessing, precision, truncation, revision and JSON export.

Inputs are limited to 30,000 characters. Semantics use at most 512 tokens; forensic features cover the full feature text. The default normalization matches training. Advanced alternatives reproduce TIRA's raw forensic branch or leave both branches raw. Changing inputs/settings clears stale reports and export controls. Choosing an example shows its **source annotations separately**; these annotations never enter inference.

Feature effects hold semantic evidence fixed and depend on a zero-feature numerical reference, which may be outside the data distribution. Failed numerical checks display **Needs review**. Origin scores cannot establish factual falsity, author intent, or coordinated campaigns.

## Tests and methodology

See [Gradio implementation and operations](../docs/GRADIO_INTERFACE.md), [explanation mathematics](../docs/EXPLANATION_MODULE.md), and [acceptance evidence](../docs/validation/gradio-checks.json) in the GitHub repository. Links to sibling docs apply to the full GitHub checkout rather than a Space containing only this folder.

```bash
python -m pip install pytest plotly==6.6.0
python -m pytest tests/test_demo.py tests/test_explain.py -q
# Opt-in live browser/real-weight checks (from the full repo):
python -m pip install playwright==1.51.0
python -m playwright install chromium
PYTHONPATH=.:src python tests/validate_demo.py --output docs/validation/gradio-checks.json
```
