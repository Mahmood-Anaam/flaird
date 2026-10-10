# PAN26 / TIRA evaluation: results and interpretation

## Evidence and metric identity

The source is eight [TIRA result JSON files](tira-evaluation/), each containing four dataset rows. The target challenge row is exactly `pan26-generative-ai-detection-20260507-test`. Results below preserve the saved three-decimal values. The repository does not include private test labels, prediction files, sample counts or benchmark checkpoint revisions; those cannot be reconstructed from summary JSON alone.

**83.6% and 89.4% are PAN26 composite Mean scores, not classification accuracy.** Their variants are trainable attention/multi-task and trainable concatenation/multi-task, respectively. Ranking or placement against other participants is not recorded by these files and is not claimed.

## Full PAN26 results

All five component metrics and Mean are higher-is-better. `Brier` denotes the **complement** of Brier loss.

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

<!-- research:runs:start -->
| Configuration | PAN26 TIRA run |
| --- | --- |
| flaird-modernbert-large-attention-multitask-frozen | 2026-09-20-09-10-36 |
| flaird-modernbert-large-attention-multitask | 2026-09-20-11-24-22 |
| flaird-modernbert-large-attention-single-task-frozen | 2026-09-20-11-40-00 |
| flaird-modernbert-large-attention-single-task | 2026-09-20-11-16-58 |
| flaird-modernbert-large-concatenation-multitask-frozen | 2026-09-20-11-53-54 |
| flaird-modernbert-large-concatenation-multitask | 2026-09-20-11-20-59 |
| flaird-modernbert-large-concatenation-single-task-frozen | 2026-09-20-11-36-09 |
| flaird-modernbert-large-concatenation-single-task | 2026-09-20-11-06-49 |
<!-- research:runs:end -->

## Metrics and thresholds

The local [evaluator](../src/flaird/scripts/evaluator.py) defines compatible named metrics. Saved TIRA summary values are external outputs; matching metric names does not prove the hosted evaluator used this exact source revision.

For true labels $y_i\in\{0,1\}$ and machine scores $p_i$, ROC-AUC measures ranking across thresholds. A useful probabilistic interpretation is the probability that a randomly selected machine text receives a higher score than a randomly selected human text, with half credit for ties. It does not describe performance at a specific deployment threshold.

The reported Brier component is

$$Brier_{complement}=1-\frac1N\sum_i\big(clip(p_i,0,1)-y_i\big)^2.$$

Higher is better; raw Brier loss is $1-Brier_{complement}$. The two headline variants therefore have saved raw losses 0.208 and 0.132. Brier combines discrimination and probability error; these values alone do not establish calibrated reliability under a new population.

With $n_c$ correct answered cases and $n_u$ unanswered cases,

$$c@1=\frac1N\left(n_c+\frac{n_u n_c}{N}\right).$$

The local evaluator treats **exactly** $p=0.5$ as unanswered for c@1 and F0.5u; it does not define a tolerance band. Scores greater than 0.5 are machine predictions. In local F1 and confusion-matrix calculations, exactly 0.5 becomes a human decision, so the handling of ties differs across metrics.

$$F1=\frac{2TP}{2TP+FP+FN},\qquad F_{0.5u}=\frac{1.25TP}{1.25TP+0.25(FN+n_u)+FP}.$$

For F0.5u, FN counts machine labels with $p<0.5$, while $n_u$ counts all unanswered cases. This emphasizes precision and penalizes nonresponse. Undefined local ROC-AUC or F metrics return `None`; the local Mean substitutes zero for undefined components.

$$Mean=\frac{ROC\text{-}AUC+Brier_{complement}+c@1+F1+F_{0.5u}}5.$$

The local evaluator averages unrounded components, then rounds output values to three decimals. Re-averaging the five already-rounded displayed components can differ slightly from the saved Mean and is not evidence of an error.

## Cross-dataset Mean scores

The four rows in each source file are distinct evaluations. The smoke dataset is a training smoke test, not the primary challenge result; do not replace PAN26 test scores with smoke scores.

<!-- research:cross:start -->
| Variant | PAN25 | PAN26 | Eloquent | Smoke |
| --- | --- | --- | --- | --- |
| Attention / multi-task / frozen | 0.878 | 0.753 | 0.736 | 0.876 |
| Attention / multi-task / trainable | 0.958 | 0.836 | 0.776 | 0.978 |
| Attention / single-task / frozen | 0.861 | 0.754 | 0.717 | 0.872 |
| Attention / single-task / trainable | 0.953 | 0.741 | 0.743 | 0.954 |
| Concatenation / multi-task / frozen | 0.798 | 0.744 | 0.723 | 0.674 |
| Concatenation / multi-task / trainable | 0.951 | 0.894 | 0.774 | 0.978 |
| Concatenation / single-task / frozen | 0.847 | 0.719 | 0.718 | 0.840 |
| Concatenation / single-task / trainable | 0.952 | 0.760 | 0.767 | 0.953 |
<!-- research:cross:end -->

Exact identifiers are:

- PAN25: `pan25-generative-ai-detection-20260508-test`.
- PAN26: `pan26-generative-ai-detection-20260507-test`.
- Eloquent: `eloquent-20260617-test`.
- Smoke: `pan26-generative-ai-detection-smoke-test-20260330-training`.

## Findings and justified comparisons

Concatenation/multi-task/trainable is strongest on PAN26 among these eight variants (0.894). Its advantage over attention/multi-task/trainable is 0.058, or 5.8 Mean percentage points. Attention/multi-task/trainable is slightly stronger on PAN25 (0.958 versus 0.951) and Eloquent (0.776 versus 0.774); both have 0.978 on the smoke dataset. These small differences are descriptive rounded results, without significance testing.

Trainable versus frozen differences are +0.083 for attention multi-task and +0.150 for concatenation multi-task. Single-task differences are −0.013 for attention and +0.041 for concatenation. Encoder training is therefore not uniformly beneficial in every saved comparison. Freezing also changes configured per-device batch size, which is a confound.

Multi-task versus single-task differences with trainable encoders are +0.095 for attention and +0.134 for concatenation; frozen differences are −0.001 and +0.025. The result is consistent with an interaction between auxiliary supervision and backbone adaptation, not a guarantee that an auxiliary head improves every detector.

The weaker Eloquent scores across all variants show that high scores on one benchmark do not eliminate distribution shift. Without per-example labels, class composition or error traces, the repository cannot identify a specific linguistic mechanism for each error. Explanation outputs for selected examples must not substitute for dataset-level causal analysis.

## Inference policy and reproducibility

The TIRA CLI uses `AutoModelForSequenceClassification` / `AutoTokenizer` with remote code, the collator's default maximum length of 512, and emits `predictions.jsonl` with `id` and `label` (continuous machine score). Semantic text receives identifier normalization by default, while fallback forensic extraction reads original text. Training preparation normalizes text before features are extracted. This policy difference is documented rather than silently retroactively corrected; the saved evaluation files do not record a per-run preprocessing manifest.

The manual [GitHub TIRA workflow](../.github/workflows/upload-software-to-tira.yml) supports selecting the eight models and defaults to a smoke-test dry run. It does not itself demonstrate a fresh PAN26 private-test evaluation. Its container uses an offline Hub cache prepared beforehand. [Reproducibility](REPRODUCIBILITY.md) lists local inference/evaluator commands and the information required to reproduce a specific hosted submission.

The local evaluator substitutes 0.5 for missing IDs and rejects prediction IDs outside the truth set. Input records are stored in dictionaries, so duplicate IDs overwrite earlier entries. Operators should validate uniqueness and complete coverage before scoring; incomplete coverage should be reported, not concealed as ordinary model abstention. The helper `optimize_pred_scores` exists but is not called by the scoring CLI. Test labels must not be used to tune deployment thresholds.

## Limits and next analysis

The supplied results are one completed configuration per design cell, not multiple independent seed replicates. No confidence intervals, statistical tests, latency distributions, per-domain confusion matrices or held-out generator analyses are present. Aggregated results cannot establish disinformation detection, open-world attribution or robust multilingual performance.

Future reporting should retain checkpoint and dataset revisions, preprocessing manifests, predictions where policy permits, per-domain metrics, human false-positive rates and confidence intervals. Compare calibration under a validation-only protocol and keep public smoke checks separate from held-out challenge scores.

## References

- [Official PAN26 generated-content analysis task](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html).
- Peñas and Rodrigo, [A Simple Measure to Assess Nonresponse](https://aclanthology.org/P11-1142/), 2011.
- [Saved TIRA artifacts](tira-evaluation/) and [local evaluator source](../src/flaird/scripts/evaluator.py).
