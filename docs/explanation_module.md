# FLAIRD Explanation Module

## Purpose and Scope

The explanation module turns one FLAIRD sequence-classification result into a
structured, local evidence report. It is designed for the decision-support
workflow described in the thesis proposal: show the binary machine-text score,
the auxiliary generator distribution when that head exists, the forensic
features most influential to the machine logit, and the feature-group allocation
used by attention fusion.

The implementation is `src/flaird/utils/explain.py`. Its public entry point is
`flaird.utils.explain_text`. This module does not load checkpoints, contact the
Hugging Face Hub, or render a user interface. The caller supplies an already
loaded sequence-classification model and tokenizer. This keeps model lifecycle
and presentation concerns outside the explanation logic.

## Inputs and Outputs

`explain_text(model, tokenizer, text, max_length=512, top_k=10)` accepts one
non-empty string. It tokenizes with truncation, extracts the same 35 forensic
features used by the FLAIRD model, moves tensors to the model's device, and
requests `output_fusion_states=True` for this inference call. The model's prior
training/evaluation mode is restored even if explanation fails.

The returned dictionary contains:

| Key | Meaning |
| --- | --- |
| `machine_probability` | Probability assigned to the machine class. |
| `human_probability` | Complement of the machine probability. |
| `predicted_label` | `machine` at probability $\geq 0.5$, otherwise `human`. |
| `generator_probabilities` | Generator-class probabilities, sorted descending, or `None` when the auxiliary head is absent. |
| `feature_attributions` | The top `top_k` forensic features, ranked by absolute attribution. |
| `all_feature_attributions` | All 35 features in the same ranked order. |
| `feature_group_attention` | Attention-fusion weights sorted descending, or `None` for fusion variants without attention. |
| `fusion_gate` | Scalar mean gate value when available; otherwise `None`. |
| `fusion_type` | Configured fusion type, such as `attention` or `concatenation`. |
| `token_count` | Number of non-padding tokens, including special tokens when an attention mask is provided. |
| `possibly_truncated` | True when the returned token sequence reaches `max_length`; this is a warning signal, not proof of truncation. |

Each feature attribution has its feature name, extracted value, signed
attribution, absolute importance, and a direction. `machine` means the local
feature contribution increases the machine logit; `human` means it decreases
that logit; `neutral` means the computed contribution is zero.

## Explanation Calculations

### Binary decision score

The trained FLAIRD classifiers use one logit for the binary decision. For logit
$z_m$, the machine probability is

$$
P(\text{machine}\mid x)=\sigma(z_m)=\frac{1}{1+e^{-z_m}}.
$$

The human probability is its complement. The default decision threshold of 0.5
is a display convention; it is not a calibrated or policy-optimized threshold.
For a model that emits two class logits, the implementation uses softmax and
selects the class named `machine` in `config.id2label` (falling back to class 1
if that label is unavailable).

### Forensic-feature attribution

For each raw extracted feature $x_i$, the module differentiates the machine
logit with respect to the input feature and computes signed gradient-times-input:

$$
a_i=x_i\frac{\partial z_m}{\partial x_i}.
$$

Features are sorted by $|a_i|$. The sign indicates the local direction of the
contribution around this particular input. The forensic feature encoder applies
its configured transform and learned group projections inside the model; the
gradient is taken through those operations, while $x_i$ is the original scalar
feature extracted from the text. It is therefore a local sensitivity summary,
not a counterfactual or a causal claim. Correlated features, nonlinear
interactions, feature scaling, and the reference value of zero can affect this
approximation. A zero-valued feature may receive zero gradient-times-input even
when its derivative is nonzero.

### Feature-group attention and fusion gate

Attention fusion pools the semantic token states into one query, attends over
the seven projected forensic feature-group states, and returns the normalized
group attention weights. The weights describe how the attention operation
allocates its feature context for this example. The implementation reports the
group names and their weights from the model configuration, not fabricated
per-token or per-feature attention.

The fusion layer combines semantic state $s$ and forensic context $c$ using a
vector gate $g$:

$$
m=c+g\odot(s-c).
$$

The exposed scalar is the mean of the gate dimensions. Values nearer one imply
more semantic state in this interpolation; values nearer zero imply more
forensic context. This is a summary of the learned interpolation, not a measure
of the final classifier's reliance on either stream.

For concatenation fusion, feature-group attention and the gate do not exist and
are returned as `None`. Generator probabilities are likewise `None` for
single-task checkpoints. These are valid architecture variants in the completed
experiment set, not inference errors.

## Model Loading and Example

The intended model-loading interface is Transformers' auto classes with remote
FLAIRD code enabled. This is an example for deployment or interactive use; it
is deliberately not executed during local tests because downloading trained
weights can be large and slow.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flaird.utils import explain_text

model_id = "<your-published-flaird-model-id>"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

report = explain_text(
    model=model,
    tokenizer=tokenizer,
    text="Text to analyze.",
    max_length=512,
    top_k=10,
)
```

The caller should apply any presentation-layer preprocessing consistently
before both feature extraction and tokenization. If personally identifying
information is redacted by an application, pass the redacted text to this
function so the displayed evidence describes exactly the analyzed input.

## Reliability and Validation

Input validation rejects blank text, non-positive sequence limits, and
non-positive `top_k`. The module supports one-logit and two-logit binary heads,
attention and concatenation fusion, and optional generator heads. Inference uses
evaluation mode, but gradients remain enabled for the forensic-feature
attribution; the model's original top-level training flag is restored in a
`finally` block.

`tests/test_explain.py` uses a small deterministic stub model and tokenizer. It
checks prediction and attribution values, attention ordering, generator
probabilities, mode restoration, the concatenation/single-task case, and blank
input rejection. These tests do not instantiate a FLAIRD checkpoint or access
the network. They validate the explanation contract and control flow, not the
quality or calibration of explanations from a trained checkpoint.

## Limitations and Responsible Use

1. A detector score is not proof of authorship. Human editing, paraphrasing,
   domain shift, and unseen generators can change predictions.
2. Gradient-times-input and attention weights are model diagnostics. Neither
   establishes that a linguistic feature caused the classification.
3. The auxiliary generator head predicts among its configured training labels;
   it must not be interpreted as reliable identification of an arbitrary
   real-world model.
4. The fixed 0.5 display threshold is not a substitute for threshold selection
   on an application-specific validation set or for probability calibration.
5. `possibly_truncated` is intentionally conservative: an input whose token
   count equals the configured limit may or may not have been truncated.
6. The model is English-focused. Results should not be generalized to other
   languages without dedicated training and evaluation.

The interface should present the score together with uncertainty and the
evidence limitations above. It should not make consequential decisions
automatically.

## Future Work

- Validate score calibration and select operating thresholds against the
  intended use case, including false-positive and false-negative costs.
- Compare gradient-times-input with integrated gradients and controlled
  feature-ablation tests on held-out data.
- Measure explanation stability under paraphrase, small edits, domain changes,
  and different generator families.
- Add uncertainty estimates and out-of-distribution warnings.
- Add token-level explanations only after selecting and validating a suitable
  method; feature-group attention is not a substitute for token attribution.
- Evaluate whether explanation quality aligns with expert forensic-linguistic
  judgments.