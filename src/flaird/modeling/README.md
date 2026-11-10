---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: answerdotai/ModernBERT-large
datasets:
- MahmoodAnaam/flaird-raid-pan26
tags:
- flaird
- ai-generated-text-detection
- stylometry
- explainability
- multitask-learning
metrics:
- roc_auc
- f1
---

# FLAIRD ModernBERT-large: attention multitask

**Checkpoint:** [MahmoodAnaam/flaird-modernbert-large-attention-multitask](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-multitask)

FLAIRD combines a fine-tuned ModernBERT-large encoder with 35 forensic-linguistic measurements, grouped into seven learned feature tokens. Gated multihead cross-attention integrates the pooled semantic state with these tokens. A shared representation supports one machine-positive binary logit and an auxiliary 11-class generator-family head.

This card describes the attention-multitask checkpoint. The [collection](https://huggingface.co/collections/MahmoodAnaam/flaird) includes seven comparator variants. Their scores must not be attributed to this checkpoint. [Live research demo](https://huggingface.co/spaces/MahmoodAnaam/flaird).

## Intended use and boundaries

The model supports English document-level AI-text-origin research and human-reviewed forensic analysis. It does not establish factual falsehood, deceptive intent, exact model authorship or coordinated campaign activity. Human writing may score as machine and generated writing may score as human. Consequential decisions require independent evidence.

The binary sigmoid is a model score affected by class-weighted training, not a calibrated probability of authorship. The auxiliary head estimates one of its known dataset families, including human, and has no unknown class. English-only evidence does not establish multilingual validity. Very short documents, unseen generators, new attacks and distribution shifts can reduce reliability.

## Architecture

| Component | Configuration |
|---|---|
| Encoder | `answerdotai/ModernBERT-large`, fine-tuned |
| Semantic pooling | Masked mean; released pipeline uses at most 512 tokens |
| Forensic stream | 35 raw measurements → sign-log transform → seven group projections |
| Group token width | 128 |
| Fusion | Eight-head cross-attention with one pooled semantic query, vector gate and product interaction |
| Shared head | Linear → GELU → LayerNorm |
| Binary head | Dropout → linear, one machine-positive logit |
| Auxiliary head | Dropout → linear, 11 labels |
| Dropout defaults | 0.1 for feature/fusion/classifier |

Groups cover lexical diversity, function-word style, surface composition, structural organization, punctuation/rhetoric, entropy, and repetition/burstiness. They do not include parser-based syntax, readability indices or discourse coherence measurements.

## Training data and procedure

Committed preparation outputs record 5,077,945 training and 565,171 validation rows combining PAN26 and RAID-derived partitions. RAID's labeled source was split 90/10 by row, stratified by binary label with seed 42. Source/near-duplicate isolation was not audited. The separate 672,000-row RAID held-out input was reserved for submission inference.

Texts undergo email/mention/phone-pattern substitution; 35 ordered features are materialized from prepared text. Machine rows comprise approximately 96.98% of training. Prepared dataset access may require authorization, and upstream data terms remain applicable.

| Training parameter | Value |
|---|---|
| Epochs | 1 |
| Per-device train/eval batch | 128 / 128 |
| Learning rate / optimizer | 0.00002 / AdamW fused |
| Scheduler | Cosine |
| Warmup | `warmup_steps=0.06`, treated as a fraction by Transformers 5.16.1 |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Seed / precision | 42 / bf16 |
| Binary positive weight | 0.0311727005543984 |
| Auxiliary objective weight | 0.2 |
| Checkpoint selection | Minimum validation loss; best model reloaded |

The objective combines weighted binary cross-entropy and weighted generator cross-entropy. Human examples also participate in the generator task. Class weights and complete settings are specified in the [source training guide](https://github.com/Mahmood-Anaam/flaird/blob/develop/docs/TRAINING_CONFIGURATION.md). The package pins Transformers 5.16.1, PyTorch 2.8.0 and datasets 4.0.0; exact historical environment provenance is not fully recorded in the training notebook.

## Evaluation

The following values are transcribed from this checkpoint's committed TIRA result artifact. `Brier` is **1 − Brier loss**, so higher is better. Mean averages ROC-AUC, Brier complement, c@1, F1 and F0.5u; it is not accuracy. F1 is machine-positive at a strict threshold > 0.5.

| Archived dataset | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PAN25 test | 0.991 | 0.940 | 0.933 | 0.949 | 0.977 | 0.958 |
| PAN26 test | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | 0.836 |
| Eloquent test | 0.899 | 0.676 | 0.644 | 0.771 | 0.892 | 0.776 |
| PAN26 smoke training set | 1.000 | 0.967 | 0.971 | 0.968 | 0.987 | 0.978 |

The archived RAID aggregate records AUROC 0.998726, machine TPR 0.997678 at 5% human FPR and 0.990006 at 1% FPR. These are controlled-FPR detection rates, not ordinary accuracy at threshold 0.5. The hardest aggregated attack conditions at 1% FPR include paraphrase and zero-width spaces (approximately 0.961 and 0.960 TPR).

Attention multitask exceeds the concatenation counterpart on PAN25 Mean but trails it on PAN26 (0.836 versus 0.894). More newly initialized fusion layers may need a different training budget, but this remains an untested explanation rather than a proven cause. No multi-seed confidence intervals or generator-head test metrics are present in the archive. Full dataset identifiers, timestamps, all variants and discussion are in the [evaluation report](https://github.com/Mahmood-Anaam/flaird/blob/develop/docs/EVALUATION_RESULTS.md).

## Usage

Install the FLAIRD package from the source repository, then supply both token tensors **and the ordered forensic feature tensor**. A generic text-only Transformers pipeline does not prepare the required forensic input.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flaird.data.data_collator import preprocess_text

model_id = 'MahmoodAnaam/flaird-modernbert-large-attention-multitask'
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
text = preprocess_text('A sufficiently long English document to analyze ...')
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
features = model.extract_forensic_features([text], return_tensors=True)
with torch.inference_mode():
    output = model(**inputs, forensic_features=features)
    machine_score = torch.sigmoid(output.logits).item()
    family_scores = torch.softmax(output.generator_logits, dim=-1)[0]
print(machine_score)
```

The example runs on CPU; if moving the model to GPU, move token tensors to the same device. Loading custom Hub code requires `trust_remote_code=True`; use reviewed checkpoint revisions for controlled deployments.

For full explanations:

```bash
flaird-predict --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \
  --input document.txt --output explanation.json --report
```

## Explanation and interpretation

The explainer holds pooled semantics fixed and masks raw forensic measurements relative to zero. Exact group Shapley values allocate the forensic logit difference across seven groups; exact Owen values allocate each group across its measurements. Default computation uses 78,592 distinct masks and one encoder pass. A sampled option reports approximate feature values and standard errors.

Positive binary contributions support machine relative to the reference; negative contributions support human. They are logit values, not percentage points. Zero raw input does not necessarily produce zero learned feature tokens, and masked combinations can be linguistically unrealistic. Attention weights and mean gate are routing diagnostics, not causal percentages. No word-level attribution is implemented.

## Reproducibility, license and contact

Repository: [Mahmood-Anaam/flaird](https://github.com/Mahmood-Anaam/flaird). Author: Mahmood Anaam. Contact: `eng.mahmood.anaam@gmail.com`.

Software license: Apache-2.0. Follow the base model and dataset terms separately. The model card's results are repository snapshots; no live leaderboard position or external deployment update is claimed by this document.
