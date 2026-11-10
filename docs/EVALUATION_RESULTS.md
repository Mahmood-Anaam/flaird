# Evaluation protocol, archived results and discussion

## 1. Evidence and scope

All eight main experiments are complete according to project status. The numeric evidence available in this repository is eight [TIRA JSON files](tira-evaluation/), each containing four dataset records, and one [RAID result file](raid-evaluation/flaird-modernbert-large-attention-multitask/results.json). This document transcribes those artifacts without rerunning training, recalculating unavailable predictions, or inventing missing results.

The [training notebook](../notebooks/flaird_training.ipynb) contains no saved outputs. There are no archived generator confusion matrices, seed distributions, confidence intervals, or complete training logs for all eight runs. Thus the following comparisons are descriptive single-run comparisons of released checkpoints. Dataset names and run timestamps are retained below for traceability.

## 2. Metrics and decision conventions

[evaluator.py](../src/flaird/scripts/evaluator.py) computes the PAN-style metrics below. Positive means machine-generated. Scores are sigmoid outputs. A strict score **> 0.5** is positive for thresholded evaluation; exactly 0.5 counts as unanswered in abstention-aware metrics. The dashboard's 0.35–0.65 inconclusive band is a separate display convention and is not used in these archived metric formulas.

| Metric | Interpretation | Direction / implementation caveat |
|---|---|---|
| ROC-AUC | Ranking of positive versus negative examples across thresholds | Higher is better; not a calibrated operating point |
| `Brier` | **1 minus** mean squared error between score and binary label | Higher is better; this repository does not report raw Brier loss under this name |
| c@1 | Rewards correct decisions and exact-score abstentions | Only scores exactly 0.5 are unanswered |
| F1 | Machine-positive harmonic mean of precision/recall | At strict > 0.5; not macro-F1 |
| F0.5u | Precision-emphasizing F score with unanswered cases in denominator | Exact abstentions contribute to its penalty |
| Mean | Arithmetic mean of the preceding five metrics | A composite, not accuracy; missing values become zero in local implementation |
| Confusion | Rows true human/machine, columns predicted human/machine | Local evaluator can return it, but archived TIRA summary files omit it |

For $n$ examples, $n_c$ correct answered decisions and $n_u$ unanswered cases:

$$
c@1=\frac{n_c+(n_u n_c/n)}{n},\qquad
B^*=1-\frac1n\sum_i(s_i-y_i)^2,
$$

$$
F_{0.5u}=\frac{1.25\,TP}{1.25\,TP+0.25(FN+n_u)+FP}.
$$

The local evaluator rounds metric outputs to three decimals after computing its composite from unrounded component values. A rounded AUC of 1.000 is not proof of perfect discrimination. A high complemented Brier score is also not proof of calibration, especially under class imbalance and weighted training.

## 3. TIRA overview: all eight experiments

Abbreviations: A = attention, C = concatenation, M = multitask, S = single-task, F = frozen encoder, T = fine-tuned encoder. Each value is the archived `Mean`.

<!-- TIRA_SUMMARY_START -->
| Experiment | PAN25 | PAN26 | Eloquent | Smoke |
|---|---:|---:|---:|---:|
| attention-multitask-frozen | 0.878 | 0.753 | 0.736 | 0.876 |
| attention-multitask | 0.958 | 0.836 | 0.776 | 0.978 |
| attention-single-task-frozen | 0.861 | 0.754 | 0.717 | 0.872 |
| attention-single-task | 0.953 | 0.741 | 0.743 | 0.954 |
| concatenation-multitask-frozen | 0.798 | 0.744 | 0.723 | 0.674 |
| concatenation-multitask | 0.951 | 0.894 | 0.774 | 0.978 |
| concatenation-single-task-frozen | 0.847 | 0.719 | 0.718 | 0.840 |
| concatenation-single-task | 0.952 | 0.760 | 0.767 | 0.953 |
<!-- TIRA_SUMMARY_END -->

## 4. Complete TIRA metric tables

### pan25-generative-ai-detection-20260508-test

| Experiment | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean | Run (archive timestamp) |
|---|---:|---:|---:|---:|---:|---:|---|
| [attention-multitask-frozen](tira-evaluation/flaird-modernbert-large-attention-multitask-frozen.json) | 0.935 | 0.854 | 0.822 | 0.858 | 0.922 | 0.878 | 2026-09-20-12-05-44 |
| [attention-multitask](tira-evaluation/flaird-modernbert-large-attention-multitask.json) | 0.991 | 0.940 | 0.933 | 0.949 | 0.977 | 0.958 | 2026-09-20-11-34-18 |
| [attention-single-task-frozen](tira-evaluation/flaird-modernbert-large-attention-single-task-frozen.json) | 0.930 | 0.835 | 0.798 | 0.837 | 0.906 | 0.861 | 2026-09-20-11-49-58 |
| [attention-single-task](tira-evaluation/flaird-modernbert-large-attention-single-task.json) | 0.988 | 0.935 | 0.925 | 0.943 | 0.972 | 0.953 | 2026-09-20-11-18-01 |
| [concatenation-multitask-frozen](tira-evaluation/flaird-modernbert-large-concatenation-multitask-frozen.json) | 0.866 | 0.787 | 0.719 | 0.761 | 0.856 | 0.798 | 2026-09-20-11-57-28 |
| [concatenation-multitask](tira-evaluation/flaird-modernbert-large-concatenation-multitask.json) | 0.988 | 0.931 | 0.922 | 0.941 | 0.974 | 0.951 | 2026-09-20-11-30-57 |
| [concatenation-single-task-frozen](tira-evaluation/flaird-modernbert-large-concatenation-single-task-frozen.json) | 0.913 | 0.824 | 0.781 | 0.820 | 0.896 | 0.847 | 2026-09-20-11-46-11 |
| [concatenation-single-task](tira-evaluation/flaird-modernbert-large-concatenation-single-task.json) | 0.986 | 0.933 | 0.924 | 0.942 | 0.973 | 0.952 | 2026-09-20-11-12-57 |

### pan26-generative-ai-detection-20260507-test

| Experiment | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean | Run (archive timestamp) |
|---|---:|---:|---:|---:|---:|---:|---|
| [attention-multitask-frozen](tira-evaluation/flaird-modernbert-large-attention-multitask-frozen.json) | 0.835 | 0.740 | 0.651 | 0.713 | 0.828 | 0.753 | 2026-09-20-09-10-36 |
| [attention-multitask](tira-evaluation/flaird-modernbert-large-attention-multitask.json) | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | 0.836 | 2026-09-20-11-24-22 |
| [attention-single-task-frozen](tira-evaluation/flaird-modernbert-large-attention-single-task-frozen.json) | 0.823 | 0.731 | 0.658 | 0.715 | 0.841 | 0.754 | 2026-09-20-11-40-00 |
| [attention-single-task](tira-evaluation/flaird-modernbert-large-attention-single-task.json) | 0.815 | 0.697 | 0.650 | 0.713 | 0.827 | 0.741 | 2026-09-20-11-16-58 |
| [concatenation-multitask-frozen](tira-evaluation/flaird-modernbert-large-concatenation-multitask-frozen.json) | 0.774 | 0.744 | 0.655 | 0.719 | 0.829 | 0.744 | 2026-09-20-11-53-54 |
| [concatenation-multitask](tira-evaluation/flaird-modernbert-large-concatenation-multitask.json) | 0.959 | 0.868 | 0.833 | 0.878 | 0.933 | 0.894 | 2026-09-20-11-20-59 |
| [concatenation-single-task-frozen](tira-evaluation/flaird-modernbert-large-concatenation-single-task-frozen.json) | 0.773 | 0.704 | 0.625 | 0.681 | 0.814 | 0.719 | 2026-09-20-11-36-09 |
| [concatenation-single-task](tira-evaluation/flaird-modernbert-large-concatenation-single-task.json) | 0.830 | 0.722 | 0.671 | 0.729 | 0.848 | 0.760 | 2026-09-20-11-06-49 |

### eloquent-20260617-test

| Experiment | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean | Run (archive timestamp) |
|---|---:|---:|---:|---:|---:|---:|---|
| [attention-multitask-frozen](tira-evaluation/flaird-modernbert-large-attention-multitask-frozen.json) | 0.794 | 0.653 | 0.611 | 0.744 | 0.877 | 0.736 | 2026-09-20-12-07-13 |
| [attention-multitask](tira-evaluation/flaird-modernbert-large-attention-multitask.json) | 0.899 | 0.676 | 0.644 | 0.771 | 0.892 | 0.776 | 2026-09-20-11-32-04 |
| [attention-single-task-frozen](tira-evaluation/flaird-modernbert-large-attention-single-task-frozen.json) | 0.813 | 0.621 | 0.577 | 0.715 | 0.861 | 0.717 | 2026-09-20-11-47-23 |
| [attention-single-task](tira-evaluation/flaird-modernbert-large-attention-single-task.json) | 0.764 | 0.663 | 0.637 | 0.768 | 0.883 | 0.743 | 2026-09-20-11-19-09 |
| [concatenation-multitask-frozen](tira-evaluation/flaird-modernbert-large-concatenation-multitask-frozen.json) | 0.856 | 0.637 | 0.565 | 0.705 | 0.854 | 0.723 | 2026-09-20-11-58-34 |
| [concatenation-multitask](tira-evaluation/flaird-modernbert-large-concatenation-multitask.json) | 0.852 | 0.684 | 0.656 | 0.780 | 0.897 | 0.774 | 2026-09-20-11-20-18 |
| [concatenation-single-task-frozen](tira-evaluation/flaird-modernbert-large-concatenation-single-task-frozen.json) | 0.815 | 0.627 | 0.575 | 0.713 | 0.859 | 0.718 | 2026-09-20-11-35-53 |
| [concatenation-single-task](tira-evaluation/flaird-modernbert-large-concatenation-single-task.json) | 0.804 | 0.687 | 0.663 | 0.789 | 0.894 | 0.767 | 2026-09-20-11-07-58 |

### pan26-generative-ai-detection-smoke-test-20260330-training

| Experiment | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean | Run (archive timestamp) |
|---|---:|---:|---:|---:|---:|---:|---|
| [attention-multitask-frozen](tira-evaluation/flaird-modernbert-large-attention-multitask-frozen.json) | 0.934 | 0.864 | 0.853 | 0.815 | 0.917 | 0.876 | 2026-09-20-03-02-44 |
| [attention-multitask](tira-evaluation/flaird-modernbert-large-attention-multitask.json) | 1.000 | 0.967 | 0.971 | 0.968 | 0.987 | 0.978 | 2026-09-20-10-56-35 |
| [attention-single-task-frozen](tira-evaluation/flaird-modernbert-large-attention-single-task-frozen.json) | 0.927 | 0.849 | 0.853 | 0.815 | 0.917 | 0.872 | 2026-09-20-10-54-32 |
| [attention-single-task](tira-evaluation/flaird-modernbert-large-attention-single-task.json) | 0.979 | 0.944 | 0.941 | 0.933 | 0.972 | 0.954 | 2026-09-20-10-58-52 |
| [concatenation-multitask-frozen](tira-evaluation/flaird-modernbert-large-concatenation-multitask-frozen.json) | 0.924 | 0.774 | 0.647 | 0.400 | 0.625 | 0.674 | 2026-09-20-10-54-10 |
| [concatenation-multitask](tira-evaluation/flaird-modernbert-large-concatenation-multitask.json) | 1.000 | 0.966 | 0.971 | 0.968 | 0.987 | 0.978 | 2026-09-20-10-58-20 |
| [concatenation-single-task-frozen](tira-evaluation/flaird-modernbert-large-concatenation-single-task-frozen.json) | 0.889 | 0.828 | 0.824 | 0.769 | 0.893 | 0.840 | 2026-09-20-11-34-44 |
| [concatenation-single-task](tira-evaluation/flaird-modernbert-large-concatenation-single-task.json) | 0.965 | 0.955 | 0.941 | 0.933 | 0.972 | 0.953 | 2026-09-20-11-00-49 |

## 5. Matched fusion differences

The table reports attention Mean minus concatenation Mean, computed from archived three-decimal values. Positive favors attention.

| Supervision / encoder | PAN25 | PAN26 | Eloquent | Smoke |
|---|---:|---:|---:|---:|
| multitask | +0.007 | -0.058 | +0.002 | +0.000 |
| multitask-frozen | +0.080 | +0.009 | +0.013 | +0.202 |
| single-task | +0.001 | -0.019 | -0.024 | +0.001 |
| single-task-frozen | +0.014 | +0.035 | -0.001 | +0.032 |

## 6. RAID: attention multitask checkpoint

Only this checkpoint has a RAID result artifact in the reviewed repository. The file contains 10,881 overlapping breakdown records; they must not be averaged as if they were independent examples. The rows below select fully aggregated domain/model/decoding/repetition conditions for each attack.

| Attack / aggregate | Machine examples | AUROC | TPR at 5% FPR | TPR at 1% FPR |
|---|---:|---:|---:|---:|
| none | 54,400 | 0.999645 | 0.999559 | 0.997610 |
| whitespace | 54,400 | 0.999655 | 0.999283 | 0.997537 |
| upper_lower | 54,400 | 0.999639 | 0.999375 | 0.997426 |
| synonym | 54,400 | 0.999341 | 0.998548 | 0.995404 |
| perplexity_misspelling | 54,400 | 0.999610 | 0.999393 | 0.997390 |
| paraphrase | 54,400 | 0.994284 | 0.992335 | 0.960754 |
| number | 54,400 | 0.999633 | 0.999559 | 0.997537 |
| insert_paragraphs | 54,400 | 0.999321 | 0.998897 | 0.995790 |
| homoglyph | 54,400 | 0.998414 | 0.995680 | 0.985276 |
| article_deletion | 54,400 | 0.999686 | 0.999449 | 0.997721 |
| alternative_spelling | 54,400 | 0.999630 | 0.999504 | 0.997482 |
| zero_width_space | 54,400 | 0.995848 | 0.990551 | 0.960147 |
| all | 652,800 | 0.998726 | 0.997678 | 0.990006 |

RAID's nested field is named `accuracy`, but the reported quantity is `tp/(tp+fn)`: **machine detection rate / TPR at a controlled human false-positive rate**, not overall binary accuracy. The file supplies per-domain thresholds and attained FPR values for eight domains. There is no universal threshold of 0.5 in this controlled-FPR table.

The aggregate has 652,800 machine cases. At 5% FPR it records TP 651,284 and FN 1,516 (TPR 0.997678); at 1% FPR TP 646,276 and FN 6,524 (TPR 0.990006). The notebook's held-out input has 672,000 rows, so the machine count is not the total input size. Metadata records release date 2026-09-30. These are stored benchmark outputs, not a claim about a live leaderboard rank or comparative results for the other seven models.

Paraphrasing and zero-width spaces are the most difficult aggregated attack conditions at 1% FPR, with TPR approximately 0.960754 and 0.960147, compared with 0.997610 for unmodified text. This supports reporting attack-level performance even when the overall aggregate is high. It does not establish robustness against arbitrary future attacks or unseen generators.

## 7. Interpretation of fusion and multitask results

### 7.1 Proposed attention versus concatenation

For the two fully fine-tuned multitask models, attention leads on PAN25: Mean 0.958 versus 0.951, F1 0.949 versus 0.941, and AUC 0.991 versus 0.988. On Eloquent it has slightly higher composite Mean (0.776 versus 0.774) and markedly higher AUC (0.899 versus 0.852), while concatenation has better Brier complement (0.684 versus 0.676), c@1 (0.656 versus 0.644), and F1 (0.780 versus 0.771). The two methods therefore differ in ranking and threshold behavior; no single “winner” should hide this distinction.

On PAN26, concatenation multitask has a clear descriptive advantage: Mean 0.894 versus 0.836, F1 0.878 versus 0.809, AUC 0.959 versus 0.932, and Brier complement 0.868 versus 0.792. It would be incorrect to dismiss this as a rounding effect. Smoke Mean ties at 0.978, with Brier complement differing by 0.001. The smoke set is a deployment/integration check and must not be treated as a fourth independent broad generalization study with equal evidential weight.

### 7.2 Optimization-budget explanation: plausible, not established

The attention design introduces Q/K/V and output projections, several normalization layers, an $H$-dimensional gate, and an additional output map over gated and multiplicative evidence. These parameters begin from random initialization while the encoder is pretrained. The concatenation alternative also trains FeatureEncoder, a fusion projection and prediction heads from scratch, but its fusion is shallower and has fewer learned transformations. At encoder width 1,024 and feature width 128, the implemented attention fusion has 6,566,144 parameters versus 1,182,720 for concatenation, a difference of 5,383,424 ([module counts](ARCHITECTURE.md)). Consequently attention may require more optimization steps or a different learning-rate allocation to align its new modules with the pretrained encoder.

All main runs use one epoch and one nominal learning rate. Such a common recipe controls some factors but does not ensure equal convergence across architectures. The hypothesis that attention is under-optimized is compatible with the PAN26 deficit, and with attention's positive results elsewhere: frozen multitask attention exceeds concatenation on every archived composite (PAN25 +0.080, PAN26 +0.009, Eloquent +0.013, smoke +0.202); frozen single-task attention leads on PAN25 and PAN26, although Eloquent is lower by 0.001. Fully fine-tuned single-task attention leads PAN25 slightly and trails on PAN26/Eloquent. This pattern suggests conditional value from attention, rather than universal inferiority or guaranteed superiority.

However, these comparisons **do not prove** that longer training will reverse the PAN26 ordering. Alternative explanations include negative transfer, overfitting to dominant training sources, checkpoint selection by joint loss, class-weight-induced score shifts, sensitivity to initialization, or mismatches between training and evaluation distributions/preprocessing. The archived file [historical model-card summary](TRAINING_HISTORY.md) shows a historical attention-multitask validation loss falling from about 0.0801 at epoch 0.1 to 0.0231 at epoch 1.0. That is consistent with continued learning within one run, but lacks matched concatenation curves and does not establish remaining generalization gains beyond epoch one.

A defensible thesis statement is: “The lower PAN26 performance of gated attention may reflect a larger optimization requirement for newly initialized fusion modules under a fixed one-epoch schedule. Its superiority in several other matched comparisons supports investigating this possibility, but controlled longer-budget and multi-seed studies are required to distinguish it from competing explanations.”

A future test should vary update budget and fusion-specific learning rate, retain identical data and evaluation protocol, use multiple seeds, and compare validation and held-out learning curves. This is a proposed experiment, not an additional completed run in this release.

### 7.3 Auxiliary supervision and frozen encoders

Fine-tuned multitask supervision improves PAN26 Mean over single-task by +0.095 for attention (0.836 versus 0.741) and +0.134 for concatenation (0.894 versus 0.760). PAN25 improvements are +0.005 for attention and -0.001 for concatenation. On Eloquent both fine-tuned multitask variants improve their respective single-task composites. These observations support the auxiliary objective in some conditions, with a small PAN25 counterexample.

Frozen concatenation multitask falls below frozen concatenation single-task on PAN25 and smoke, while improving PAN26 and Eloquent. This demonstrates that auxiliary supervision is not uniformly beneficial. The gradient interaction and weighted class taxonomy require analysis rather than an assumption that an extra head always helps.

Fine-tuning generally improves results over corresponding frozen variants, especially PAN25 and smoke. Yet attention single-task PAN26 has Mean 0.741 when fine-tuned versus 0.754 frozen, illustrating that representation adaptation can fail to transfer. Frozen runs also use batch 256 versus 128 and one configuration omits best-model restoration; freezing comparisons therefore include training-protocol differences. They are not a perfectly isolated estimate of encoder freezing alone.

## 8. Limits on conclusions

The archive lacks raw prediction scores and test labels for reconstructing uncertainty or conducting paired tests. Differences near 0.001 should not be described as statistically significant. Dataset sizes for the archived TIRA test records are not included, so confidence intervals cannot be reconstructed from the summaries. Internal training validation also has an unaudited row-split/source-overlap risk documented in [data preparation](DATASET_PREPARATION.md).

The eight variants do not include a matched semantic-only or forensic-only detector. Thus fusion comparisons cannot establish that explicit features outperform ModernBERT alone. No archived generator-specific performance supports an exact-family authorship claim. Calibration, baseline sensitivity of explanations, and independent human evaluation of explanations remain separate research questions. Finally, generated-text detection does not measure factual falsehood, malicious intent or campaign coordination.

## 9. Reproduction paths

- `flaird-evaluator ANSWERS.jsonl TRUTH.jsonl OUTPUT_DIRECTORY` evaluates local PAN-format files, matching IDs and filling missing predictions with 0.5.
- `flaird-tira-test --input-directory INPUT --output-directory OUTPUT --model MODEL_ID` creates one sigmoid score per text through the TIRA collator path.
- `flaird-raid-test --model MODEL_ID --shard-index 0 --num-shards 5 ...` generates a contiguous RAID shard; all shards must be available before merging into `predictions.json`. Dataset/feature/preprocessing flags must match the intended inference route.

The [training guide](TRAINING_CONFIGURATION.md) and [demo guide](DEMO.md) distinguish checkpoint generation, benchmark submissions and deployed explanations. Primary benchmark reference: Dugan et al. (2024), [RAID](https://aclanthology.org/2024.acl-long.674/).
