# FLAIRD Evaluation Results

## Evaluation Scope

This report summarizes the committed result artifacts in `docs/tira-evaluation/` and `docs/raid-evaluation/`. It does not rerun evaluation or load model checkpoints. The PAN26 challenge test results are from the dataset named `pan26-generative-ai-detection-20260507-test`; the RAID results come from the leaderboard artifact released on 2026-09-30.

These are benchmark observations, not guarantees of performance on arbitrary text. The configurations share one seed and one reported training run each; no confidence intervals or repeated-seed analysis are present in the checked-in artifacts.

## PAN26 TIRA Results

The table reports the PAN26 test row for each of the eight submitted checkpoints. The `Mean` column is the arithmetic mean of ROC-AUC, the repository's complemented Brier score, C@1, F1, and F0.5u.

| Configuration / model | Fusion | Generator task | Encoder | ROC-AUC | Brier* | C@1 | F1 | F0.5u | Mean |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [attention-multitask](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-multitask) | Attention | Multi-task | Trainable | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | **0.836** |
| [attention-multitask-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-multitask-frozen) | Attention | Multi-task | Frozen | 0.835 | 0.740 | 0.651 | 0.713 | 0.828 | 0.753 |
| [attention-single-task](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-single-task) | Attention | Binary only | Trainable | 0.815 | 0.697 | 0.650 | 0.713 | 0.827 | 0.741 |
| [attention-single-task-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-attention-single-task-frozen) | Attention | Binary only | Frozen | 0.823 | 0.731 | 0.658 | 0.715 | 0.841 | 0.754 |
| [concatenation-multitask](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-multitask) | Concatenation | Multi-task | Trainable | 0.959 | 0.868 | 0.833 | 0.878 | 0.933 | **0.894** |
| [concatenation-multitask-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-multitask-frozen) | Concatenation | Multi-task | Frozen | 0.774 | 0.744 | 0.655 | 0.719 | 0.829 | 0.744 |
| [concatenation-single-task](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-single-task) | Concatenation | Binary only | Trainable | 0.830 | 0.722 | 0.671 | 0.729 | 0.848 | 0.760 |
| [concatenation-single-task-frozen](https://huggingface.co/MahmoodAnaam/flaird-modernbert-large-concatenation-single-task-frozen) | Concatenation | Binary only | Frozen | 0.773 | 0.704 | 0.625 | 0.681 | 0.814 | 0.719 |

Raw values and run identifiers are preserved in the eight JSON files under [`tira-evaluation/`](tira-evaluation/). The two requested headline means are 0.836 for trainable attention multi-task and 0.894 for trainable concatenation multi-task.

### Interpreting the Comparisons

- Among these eight saved runs, trainable concatenation multi-task has the highest PAN26 mean (0.894), followed by trainable attention multi-task (0.836). Their difference is 0.058 mean-score points.
- Within the trainable attention pair, the multi-task run exceeds the single-task run by 0.095 mean points. Within trainable concatenation, the difference is 0.134.
- Freezing the encoder lowers the multi-task mean by 0.083 for attention and 0.150 for concatenation in these runs. The single-task frozen results are slightly higher than their trainable counterparts for these configurations, so the effect is not uniform.
- These contrasts are descriptive ablations, not causal or statistically significant findings. A single run per cell, shared seed, possible dataset overlap, and no confidence intervals limit stronger conclusions.

### Metric Definitions and Direction

The evaluator in `src/flaird/scripts/evaluator.py` computes:

- **ROC-AUC:** threshold-independent ranking discrimination.
- **Brier*:** $1-\operatorname{BrierLoss}$, clipped to $[0,1]$ before the loss is computed. Higher is better in this repository; this is the complement of the usual loss, not raw Brier loss.
- **C@1:** a measure that rewards correct predictions while allowing abstention at exactly 0.5.
- **F1:** positive-class F1 after a fixed probability threshold of 0.5.
- **F0.5u:** F0.5-style score where predictions exactly at 0.5 count as unanswered cases.
- **Mean:** the equal-weight average of the five preceding scores, rounded to three decimals in the result artifact.

$$
\operatorname{Mean}=\frac{\operatorname{ROC\!\text{-}\!AUC}+\operatorname{Brier}^{*}+C@1+F1+F_{0.5u}}{5}.
$$

The challenge score is therefore not plain accuracy. The fixed threshold and lack of calibration analysis should be considered when using probabilities operationally.

## RAID Leaderboard Result

The RAID artifact is for `flaird-modernbert-large-attention-multitask`, released 2026-09-30. Its result file has 10,881 grouped score rows plus two top-level aggregates. The aggregate values are:

| RAID aggregate | AUROC | Artifact `accuracy` at `0.05` | TP / FN at `0.05` | Artifact `accuracy` at `0.01` | TP / FN at `0.01` |
| --- | ---: | ---: | ---: | ---: | ---: |
| All recorded conditions | **0.998726** | 0.997678 | 651,284 / 1,516 | 0.990006 | 646,276 / 6,524 |
| No adversarial attack | 0.999646 | 0.999559 | 54,376 / 24 | 0.997610 | 54,270 / 130 |

For the all-condition aggregate, TP+FN is 652,800; for no-adversarial it is 54,400. In both cases, the JSON's `accuracy` value equals TP/(TP+FN), which is positive-class recall from the included counts. The artifact keys it by `0.05` and `0.01`, but this repository does not contain the leaderboard's full metric specification. Do not relabel those fields as overall accuracy or infer their operating-point semantics without consulting the RAID leaderboard definition.

The aggregate AUROC is exceptionally high, including for the all-attack aggregate. This is evidence on the particular RAID leaderboard set and its recorded conditions, not proof of robustness to every unseen generator, domain, attack, or collection process. The no-adversarial score is a subset result and must not be presented as the overall result.

The raw RAID JSON is automatically generated and includes a warning not to edit it manually: [RAID results](raid-evaluation/flaird-modernbert-large-attention-multitask/results.json). The adjacent metadata currently leaves `huggingface_link` null even though the training configuration names the checkpoint; the model link above is derived from that configured Hub ID, not from the RAID metadata.

## Limitations and Follow-up Evaluation

1. Re-run all eight models on the exact same PAN26 and RAID revisions and publish dataset revisions, split counts, and run IDs alongside scores.
2. Use multiple random seeds and bootstrap confidence intervals to test whether the observed architecture and auxiliary-task differences are stable.
3. Audit exact and near duplicates, prompt/source overlap, generator overlap, and dataset leakage before attributing performance to generalization.
4. Report class-wise precision/recall, confusion matrices, calibration plots, and threshold-dependent operating characteristics in addition to the aggregate mean.
5. Keep RAID's attack-wise, domain-wise, and generator-wise results disaggregated; report macro as well as example-weighted summaries.
6. Assess calibration and choose deployment thresholds on validation data only. Do not tune from leaderboard test feedback.
7. Test paraphrase, unseen-generator, cross-domain, and text-length shifts, and quantify explanation stability separately from classification quality.
