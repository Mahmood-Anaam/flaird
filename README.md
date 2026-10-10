# FLAIRD

**Forensic Linguistic and AI-Integrated Robust Detector** combines ModernBERT-large with 35 forensic-linguistic measurements to detect English human-versus-machine text. Multi-task variants also predict one of ten generator families or human. The repository includes eight completed training configurations, saved PAN/TIRA and RAID results, a tested explanation module and a professional Gradio application prepared for Hugging Face Spaces.

The thesis title is *Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns*. The implemented classifier supplies text-origin evidence; determining factuality, deceptive intent or campaign coordination requires additional system components.

[Technical documentation](docs/README.md) · [Academic proposal](docs/THESIS_PROPOSAL_ACADEMIC.md) · [Gradio guide](demo/README.md) · [Checkpoints on Hugging Face](https://huggingface.co/MahmoodAnaam)

## Main results

On the saved **PAN26 test** evaluation, trainable attention/multi-task reaches **Mean 0.836 (83.6%)**, and trainable concatenation/multi-task reaches **Mean 0.894 (89.4%)**, the strongest of these eight variants. Mean averages five benchmark metrics; it is **not classification accuracy**. Brier in this table is the complement of Brier loss.

<!-- research:pan:start -->
| Variant | Roc-Auc | Brier | C@1 | F1 | F05U | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Attention / multi-task / frozen | 0.835 | 0.740 | 0.651 | 0.713 | 0.828 | 0.753 |
| Attention / multi-task / trainable | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | 0.836 |
| Attention / single-task / frozen | 0.823 | 0.731 | 0.658 | 0.715 | 0.841 | 0.754 |
| Attention / single-task / trainable | 0.815 | 0.697 | 0.650 | 0.713 | 0.827 | 0.741 |
| Concatenation / multi-task / frozen | 0.774 | 0.744 | 0.655 | 0.719 | 0.829 | 0.744 |
| Concatenation / multi-task / trainable | 0.959 | 0.868 | 0.833 | 0.878 | 0.933 | 0.894 |
| Concatenation / single-task / frozen | 0.773 | 0.704 | 0.625 | 0.681 | 0.814 | 0.719 |
| Concatenation / single-task / trainable | 0.830 | 0.722 | 0.671 | 0.729 | 0.848 | 0.760 |
<!-- research:pan:end -->

The one saved **RAID leaderboard** result is associated with attention/multi-task/trainable (release 2026-09-30). It reports machine recall **99.7678% at 5% FPR**, **99.0006% at 1% FPR**, and AUROC **0.998726**, across all saved attack conditions. Domain-specific thresholds are fitted to human reference scores. This recall is not the PAN composite or overall binary accuracy; no saved RAID result for the other seven variants is present.

Read [PAN/TIRA analysis](docs/PAN26_EVALUATION.md) and [RAID analysis](docs/RAID_EVALUATION.md) for all source IDs, cross-dataset results, domain/generator/attack breakdowns, equations and limitations. Original JSON files remain unchanged. Benchmark checkpoint revisions and full trainer logs were not retained, so current checkpoint validation cannot certify exact historical benchmark reproduction.

## Published models and experiment design

Eight configurations cross attention/concatenation, single-/multi-task and frozen/trainable encoder. All request one epoch, seed 42, 512 semantic tokens, learning rate 2e-5 and BF16 training. Trainable per-device batch is 128; frozen batch is 256. Multi-task loss combines weighted binary BCE with 0.2 × weighted eleven-class cross-entropy.

<!-- research:models:start -->
| Variant | PAN26 Mean | Hugging Face checkpoint |
| --- | --- | --- |
| Attention / multi-task / frozen | 0.753 | [flaird-modernbert-large-attention-multitask-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-multitask-frozen) |
| Attention / multi-task / trainable | 0.836 | [flaird-modernbert-large-attention-multitask](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-multitask) |
| Attention / single-task / frozen | 0.754 | [flaird-modernbert-large-attention-single-task-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-single-task-frozen) |
| Attention / single-task / trainable | 0.741 | [flaird-modernbert-large-attention-single-task](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-single-task) |
| Concatenation / multi-task / frozen | 0.744 | [flaird-modernbert-large-concatenation-multitask-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-multitask-frozen) |
| Concatenation / multi-task / trainable | 0.894 | [flaird-modernbert-large-concatenation-multitask](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-multitask) |
| Concatenation / single-task / frozen | 0.719 | [flaird-modernbert-large-concatenation-single-task-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-single-task-frozen) |
| Concatenation / single-task / trainable | 0.760 | [flaird-modernbert-large-concatenation-single-task](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-single-task) |
<!-- research:models:end -->

These models are loaded using the Transformers auto classes with `trust_remote_code=True`. For controlled use pin the same verified revision for model and tokenizer. [Training documentation](docs/TRAINING_EXPERIMENTS.md) records literal settings, including `warmup_steps=0.06` rather than an assumed warmup ratio, and one omitted best-model reload setting. The development JSON is not a ninth benchmark run.

## Architecture and output

Two evidence branches run in parallel. ModernBERT-large produces semantic token states with masked mean pooling. The forensic branch measures the entire selected text, applies a signed-log transform, and projects seven groups into 128-dimensional states. Attention fusion uses eight heads and a learned coordinate-wise gate; concatenation fusion joins semantic pooling with the average group state. A shared representation feeds the prediction heads.

| Interface | Meaning |
| --- | --- |
| `input_ids`, `attention_mask` | Tokenized text, at most 512 tokens in completed experiments |
| `forensic_features` | B × 35 raw finite values in canonical order; mandatory |
| `logits` | B × 1; sigmoid is machine score, human score is its complement |
| `generator_logits` | B × 11 in multi-task models; absent in single-task models |
| Requested fusion diagnostics | Seven group attention weights and mean gate for attention models; absent in concatenation models |

Feature groups cover lexical diversity, function words, surface composition, structural organization, punctuation, entropy and repetition. These are measurable proxies, not dependency parsing or discourse-coherence analysis. Family probabilities are independent of the binary head and include human; they are not verified generator identities. Weighted training does not establish population-calibrated probabilities. [Full architecture](docs/MODEL_ARCHITECTURE.md) defines all 35 values, equations and tensor shapes.

## Installation and local interface

Python 3.12 is the validated development environment; the package declares Python ≥3.10. Install PyTorch for your hardware separately, then the editable package:

```bash
git clone https://github.com/Mahmood-Anaam/flaird.git
cd flaird
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[demo,test]'
python demo/app.py
```

The application uses port 7860 by default. Checkpoints load lazily when analysis starts. Select a model, enter English text or choose one of the twenty `demo/examples.csv` examples, then inspect origin scores, forensic plots, group interventions and available family/attention diagnostics. Threshold and preprocessing controls are explicit, and JSON explanation export is available.

The default attention/multi-task/trainable model provides attention diagnostics and the recorded RAID result. Select concatenation/multi-task/trainable for the highest saved PAN26 Mean. CPU inference is supported; first model downloads and integrated-gradient analysis can take longer than classification alone. The Space-ready deployment files are in `demo`; no live Space URL has been published as part of these artifacts. See [deployment instructions](demo/README.md).

## Trained-model inference

A plain tokenizer-only pipeline omits the forensic input. Use both auto loaders and supply the ordered feature vector:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flaird.data.data_collator import preprocess_text
from flaird.modeling.features import ForensicFeatureExtractor

model_id = "MahmoodAnaam/flaird-modernbert-large-attention-multitask"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, trust_remote_code=True
).float().eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
text = preprocess_text("A sample English document for origin analysis.")
batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
batch["forensic_features"] = torch.tensor(
    [ForensicFeatureExtractor()(text)], dtype=torch.float32
)
with torch.no_grad():
    output = model(**batch)
print(output.logits.sigmoid().item())
```

This normalizes text before both tokenization and extraction, matching prepared training-style inputs. TIRA's default fallback instead uses normalized semantic text and original forensic text; raw RAID defaults normalize neither. [Reproducibility](docs/REPRODUCIBILITY.md) documents these policies, CUDA use, authentication and revision pinning.

## Explanation and command-line tools

The explanation module computes integrated gradients on transformed forensic features while holding semantic evidence fixed. It reports baseline-relative **logit** contributions, group interventions, numerical completeness and attention/gate diagnostics where available. Attention is routing information, and zero-baseline interventions are numerical comparisons that can be outside the training distribution. They are not causal proof of authorship or token-level saliency.

```bash
flaird-explain --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \
  --text-file sample.txt --output outputs/explanation.json --preprocessing training
python docs/tools/research_tables.py --check
python -m pytest -q
```

`sample.txt` must be an existing UTF-8 English text file. Registered tools are `flaird-train`, `flaird-evaluator`, `flaird-tira-test`, `flaird-raid-test` and `flaird-explain`. The `predict.py` file is a stub; no `flaird-predict` command is provided. Training imports additionally require `torchsummary` and `accelerate`; see the [command guide](docs/REPRODUCIBILITY.md) before adapting the original Colab paths. Training has already completed and these documentation commands do not retrain or submit a leaderboard evaluation.

The [explanation document](docs/EXPLANATION_MODULE.md) and [Gradio document](docs/GRADIO_INTERFACE.md) provide API details and phase-specific validation. Real-checkpoint explanation validation covers all eight variants and all twenty examples. Its pinned model revisions identify that validation, not the unrecorded historical benchmark revisions.

## Dataset and repository structure

The preparation notebook combines PAN source splits and a binary-label-stratified RAID row split, resulting in **5,077,945 training** and **565,171 validation** rows. Feature extraction runs after identifier normalization. Source IDs for related variants are discarded before the RAID split; group-disjointness and cross-source deduplication are not demonstrated. [Dataset documentation](docs/DATASET_PREPARATION.md) reconstructs the notebook, aliases, imbalance, weights and required future audits.

| Path | Responsibility |
| --- | --- |
| `src/flaird/modeling/` | Model configuration, ordered features, fusion and pretrained model classes |
| `src/flaird/data/` | Dataset loading, dynamic padding, preprocessing and feature/label collation |
| `src/flaird/utils/explain.py` | Forensic attribution, interventions and explanation reports |
| `src/flaird/scripts/` | Training, benchmark adapters, evaluator and explanation CLI |
| `src/flaird/utils/` | Arguments and supporting utilities |
| `configs/` | Eight completed experiment JSON files plus a development configuration |
| `notebooks/` | Dataset preparation and training workflows |
| `demo/` | Gradio application, runtime, rendering, twenty examples and Space configuration |
| `tests/` | Model/explanation/UI checks |
| `docs/` | Academic/technical documentation and original evaluation artifacts |
| `docs/tools/` | Offline table regeneration and evidence verification |
| `docs/validation/` | Validation evidence and source-hash research ledger |
| `.github/workflows/` | Manual TIRA submission/smoke-test workflow |

## Limitations and research direction

The current evidence is English-focused, single-seed and closed-set for generator families. There are no retained family attribution scores, encoder-only/forensic-only baselines or source-disjoint guarantees. RAID recall uses benchmark-tuned domain thresholds and cannot be treated as universal operational accuracy. Very short text may trigger feature fallbacks, and long text has different branch context lengths.

Priorities are immutable data/model manifests, source/prompt overlap audits, matched-batch repeated runs, uncertainty intervals, calibration, unseen-generator evaluation, explanation stability studies and multilingual validation. Campaign-level analysis additionally needs provenance, fact verification and temporal/network evidence. The [academic proposal](docs/THESIS_PROPOSAL_ACADEMIC.md) specifies research questions, completed findings and future work.

## References and license

- Warner et al., [ModernBERT](https://arxiv.org/abs/2412.13663), 2024.
- Dugan et al., [RAID](https://arxiv.org/abs/2405.07940), 2024.
- [PAN26 generated-content analysis](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html).
- Sundararajan et al., [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html), 2017.

Package metadata declares Apache License 2.0. Upstream datasets, base models and dependencies have their own terms. Author: Mahmood Anaam — `eng.mahmood.anaam@gmail.com`.
