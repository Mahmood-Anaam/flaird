# Installation, operation and reproducibility

## Environment and dependencies

The package declares Python ≥3.10. Validation for the explanation/UI/documentation phases used Python 3.12, Transformers 5.16.1, `pystylometry` 1.4.3, Gradio 6.27.0 and Plotly 6.6.0. Install PyTorch separately for the target CPU/CUDA environment; it is imported by the implementation but not declared in the package's dependency list. A compatible GPU/PyTorch stack is required for the original BF16/fused-optimizer training request.

```bash
git clone https://github.com/Mahmood-Anaam/flaird.git
cd flaird
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU inference; choose the appropriate official PyTorch distribution for CUDA.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[demo,test]'
```

The training entry point additionally imports `torchsummary`; install `torchsummary==1.5.1` and `accelerate` for training-command imports. A successful `--help` does not prove compatibility of a complete training run. The editable package command can install tracking dependencies even when only local inference is needed; no tracking service is contacted merely by the example below.

For authenticated Hub data access use `hf auth login` or provide a token through the environment/secret mechanism. Never put a token in a committed configuration, URL or example. Public model loading itself does not require a token. `trust_remote_code=True` is required by the registered FLAIRD architecture; for repeatable execution pin the verified model and tokenizer to the same revision.

## Minimal trained-model inference

This example reproduces the **training-style** preprocessing policy: normalize before both tokenization and forensic extraction. It loads the trained hybrid checkpoint through the requested Transformers auto classes and supplies both inputs.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flaird.data.data_collator import preprocess_text
from flaird.modeling.features import ForensicFeatureExtractor

model_id = "MahmoodAnaam/flaird-modernbert-large-attention-multitask"
# For a controlled run, also pass revision="<verified Hub commit>" to both calls.
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, trust_remote_code=True
).float().eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
text = preprocess_text("A sample English document for origin analysis.")
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
inputs["forensic_features"] = torch.tensor(
    [ForensicFeatureExtractor()(text)], dtype=torch.float32
)
with torch.no_grad():
    output = model(**inputs)
    print("Machine score:", output.logits.sigmoid().item())
    if output.generator_logits is not None:
        print("Family distribution:", output.generator_logits.softmax(-1).tolist())
```

For CUDA move both model and all input tensors to the same device. Input features are raw measurements; signed-log scaling happens inside the model. Single-task checkpoints return no generator logits. A machine score is not a truthfulness score or proof of authorship.

## Explanation export and Gradio

Create an English UTF-8 file, then run:

```bash
flaird-explain --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \
  --text-file sample.txt --output outputs/explanation.json \
  --preprocessing training --device cpu
python demo/app.py
```

Optional explanation flags include `--revision`, `--baseline-file` (35 raw values in canonical order), `--max-length`, `--integration-steps`, `--max-integration-steps` and `--integration-batch-size`. [Explanation documentation](EXPLANATION_MODULE.md) defines the report fields, logit attribution, interventions and convergence/completeness checks. Preserve those checks when interpreting feature ranks.

The Gradio UI serves at port 7860 by default, loads a selected checkpoint lazily on analysis, and includes the 20 examples in `demo/examples.csv`. It offers all eight variants, threshold controls, training/TIRA/raw policies, group and feature charts, auxiliary probabilities when available, and JSON export. The default attention/multi-task/trainable checkpoint supports attention diagnostics and has the saved RAID result; concatenation/multi-task/trainable has the strongest saved PAN26 Mean.

See [Gradio documentation](GRADIO_INTERFACE.md) and [demo deployment guide](../demo/README.md) for environment variables and Hugging Face Space layout. The Space-ready directory is implemented; publication to a live Space is a separate deployment action, not evidence supplied by this repository.

## Preprocessing policy matrix

| Path | Semantic text | Feature text | Stored vectors |
| --- | --- | --- | --- |
| Prepared training/validation | Already normalized | Already normalized | Read from `forensic_features` |
| Explanation/UI `training` | Normalized | Normalized | Computed for selected text |
| Explanation/UI `tira` | Normalized | Original | Computed for selected text |
| Explanation/UI `none` | Original | Original | Computed for selected text |
| Default TIRA collator fallback | Normalized | Original | None in ordinary input JSONL |
| Default RAID CLI | Original | Original | None unless selected explicitly |
| Prepared RAID configuration | Already normalized `generation` | Prepared normalized text | Selected by `--feature-column` |

All completed experiments use 512 semantic tokens. The feature branch sees whole text under its selected policy. A preprocessing change, feature package upgrade or longer token limit creates a new inference condition. Record it separately rather than asserting equivalence to a historical leaderboard result.

## Local PAN-style evaluation

Given `answers.jsonl` records `{"id": ..., "label": continuous_machine_score}` and matching truth records with binary labels:

```bash
mkdir -p outputs/evaluation
flaird-evaluator answers.jsonl truth.jsonl outputs/evaluation \
  --outfile-name evaluation.json
```

Validate unique IDs and coverage first. The local scorer defaults absent answers to 0.5; summary JSON alone cannot reveal whether a historical hosted run had missing predictions. Metric definitions and tie handling are in [PAN26 evaluation](PAN26_EVALUATION.md).

For local TIRA inference, inspect the supported options with `flaird-tira-test --help`; input is text-bearing JSONL and the output is a directory containing `predictions.jsonl`. The manual GitHub TIRA workflow provisions an offline model cache and runs a configurable smoke test. It requires a TIRA-authorized environment and is not launched by these instructions. Preserve the exact hosted run ID, software/container digest and checkpoint/data revision when submitting.

## RAID shard generation

The following documents the notebook's prepared-data path, not a re-run of the saved leaderboard release. It requires access to the named prepared test configuration. Use a **new** output directory for every model/data/policy manifest.

```bash
for shard_index in 0 1 2 3 4; do
  flaird-raid-test \
    --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \
    --dataset-path MahmoodAnaam/flaird-raid-pan26 \
    --dataset-name raid_test_features \
    --feature-column forensic_features \
    --output-dir outputs/raid-prepared-new-run \
    --batch-size 16 --num-shards 5 --shard-index "$shard_index" \
    --contact-info "Email Address: eng.mahmood.anaam@gmail.com"
done
```

Alternatively select `liamdugan/raid`, `raid_test` and omit `--feature-column` for raw-default extraction. `--apply-text-preprocessing` normalizes semantic text; fallback vectors still use original text through the collator. Neither example certifies the historical submission policy. The current CLI lacks revision flags for all benchmark inputs; record resolved Hub/dataset commits externally, or use a separately versioned local snapshot workflow for a future fully pinned submission.

Shards use zero-based CLI indices and one-based filenames under `shards_5/`. Once all are present the script writes merged `predictions.json` inside the model output directory. Validate unique IDs, complete expected coverage and finite [0,1] scores before a future upload. Existing shard files can be reused based on paths alone, so do not mix runs. No leaderboard submission was performed during this documentation phase.

## Training command and path adaptation

Training has already finished; this command is an operational reference only:

```bash
python -m pip install torchsummary==1.5.1 accelerate
# Copy a selected JSON and replace Colab cache/output paths for your environment.
flaird-train /absolute/path/to/local-experiment.json
```

The eight authoritative [configurations](../configs/) must remain unchanged when describing their reported results. A local adaptation should be a separately named copy and record every change. The `warmup_steps` float and omitted best-model reload in one configuration need historical execution evidence before claiming their exact runtime effects. The unimplemented `predict.py` stub does not register a `flaird-predict` command.

## Verification and artifact ledger

```bash
python docs/tools/research_tables.py --check
python -m pytest -q
```

`research_tables.py` reads the committed eight configs, all 32 TIRA rows and 10,881 RAID records. It checks TP/FN recall identities, regenerates marked Markdown tables and compares source hashes/evidence JSON. Running without `--check` updates those sections after intentional changes. It does not contact the Hub, train, or submit a benchmark.

[Phase-one checkpoint validation](validation/) and [explanation documentation](EXPLANATION_MODULE.md) record real-model smoke tests over all eight checkpoints and all 20 examples. Their tested Hub revisions identify explanation validation, not necessarily the historical training or benchmark revisions. Keep that distinction in audit reports.

A reproducible future run should retain: environment lock; code/tree SHA; model and tokenizer revision; dataset source/configuration/revision; split/source-group manifest; feature version/order; text policy; maximum length; threshold/calibration source; precision/device settings; unique-ID coverage; trainer state and selected checkpoint; predictions and evaluator version. Missing fields in historical artifacts are explicitly unknown, not reconstructed by assumption.
