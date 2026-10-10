# Phase one: repository review and explanation readiness

> Historical phase-one snapshot. The interface and comprehensive research documentation were completed in later phases; see [the current documentation index](README.md).

## Delivery boundary

This phase completes the explanation backend, its runnable export command, functional validation, and detailed explanation documentation. It assumes the eight configured experiments have already finished training and evaluation. It does not initiate another training run or replace benchmark results.

See [EXPLANATION_MODULE.md](EXPLANATION_MODULE.md) for implementation rationale, equations, reference semantics, input/output contracts, and research limitations. The next authorized phase will build the professional Gradio interface inside `demo`; the subsequent documentation phase will expand the dataset/evaluation analysis, thesis proposal, and project README.

## Reviewed sources

The review covered the tracked repository structure, the thesis proposal, every Python file under `src/flaird/modeling`, `data`, `scripts`, and `utils`, the trainer, all experiment JSON files, both notebook cell sources and their saved text/table outputs, demo code/dependencies/examples, project packaging, Docker/TIRA workflow, and the stored TIRA/RAID evaluation JSON. The bundled RADAR repository export is background reference material rather than evidence of the FLAIRD implementation or its results.

Several documentation files are currently empty or skeletal: the main README, model/src README files, and many sections of `THESIS_PROPOSAL_ACADEMIC.md`. Their missing content is an outstanding documentation task, not evidence that proposed modules exist.

## Completed experiment design

The eight full experiments form a $2\times2\times2$ design: attention versus concatenation fusion; binary-only versus binary plus generator-family training; frozen versus trainable ModernBERT-large. `flaird-dev-multitask.json` is a separate development configuration, not a ninth full experiment.

| Configuration suffix | Fusion | Generator head | Encoder | Batch size | PAN26 Mean |
|---|---|---|---|---:|---:|
| attention-multitask | attention | 11 classes | trainable | 128 | 0.836 |
| attention-multitask-frozen | attention | 11 classes | frozen | 256 | 0.753 |
| attention-single-task | attention | absent | trainable | 128 | 0.741 |
| attention-single-task-frozen | attention | absent | frozen | 256 | 0.754 |
| concatenation-multitask | concatenation | 11 classes | trainable | 128 | 0.894 |
| concatenation-multitask-frozen | concatenation | 11 classes | frozen | 256 | 0.744 |
| concatenation-single-task | concatenation | absent | trainable | 128 | 0.760 |
| concatenation-single-task-frozen | concatenation | absent | frozen | 256 | 0.719 |

Names are prefixed with `flaird-modernbert-large-`; their configured Hub namespace is `MahmoodAnaam`. The PAN26 values come specifically from `pan26-generative-ai-detection-20260507-test`, not PAN25, Eloquent, or the smoke-test dataset.

Common settings include one epoch, token length 512, seed 42, learning rate $2\times10^{-5}$, weight decay 0.01, cosine scheduling, `adamw_torch_fused`, maximum gradient norm 1.0, bf16, and stored forensic features. Evaluation and saves are configured at `0.10` step fractions, with a limit of two saved checkpoints. All full configs use `loss` as the best-model metric; `load_best_model_at_end` is explicitly true in seven configs but omitted in `attention-single-task-frozen`. This difference should be retained when documenting checkpoint selection.

The JSON files contain the literal setting `warmup_steps: 0.06`. This is **not** the same configuration key as `warmup_ratio: 0.06`; the preparation of later documentation must not silently rename it or claim an intended 6% schedule as a verified training fact. Actual trainer state/logs are needed to establish the executed warmup behavior.

The weighted binary objective is

$$\mathcal{L}_{\mathrm{binary}}=-\frac1N\sum_n\left[w_+y_n\log\sigma(z_n)+(1-y_n)\log(1-\sigma(z_n))\right],$$

with configured $w_+=0.0311727005543984$. Multitask variants add $0.2\mathcal{L}_{\mathrm{generator}}$, using weighted eleven-class cross-entropy. Single-task variants have zero generator-loss weight and no auxiliary head. The low positive weight counters the machine-majority training distribution; it does not make sigmoid outputs calibrated probabilities.

## Dataset preparation as actually implemented

The preparation notebook combines PAN26 and RAID. Its saved outputs show:

| Component | Training rows | Validation rows |
|---|---:|---:|
| PAN26 | 23,707 | 3,589 |
| RAID | 5,054,238 | 561,582 |
| Combined FLAIRD | 5,077,945 | 565,171 |

RAID's original training split has 5,615,820 rows. The notebook performs a row-level, binary-label-stratified 90/10 split with seed 42. Its separate 672,000-row RAID test split is reserved for leaderboard predictions. PAN26's supplied train/validation split is retained.

The notebook renames `generation` to `text` and `domain` to `genre`, derives `human=0` and `machine=1`, attaches dataset provenance, normalizes email/mention/phone patterns, maps model names to ten machine families plus human, and adds the same ordered 35-feature extractor used by the Python model. Features are extracted from the **normalized** text. The feature datasets are published as `flaird_train_features` and `flaird_validation_features`; normalized RAID test text and features are prepared as `raid_test_features`.

The saved machine-family counts total 4,924,437 training rows, leaving 153,508 human rows. This is approximately 96.98% machine and 3.02% human; $153508/4924437\approx0.0311727$, consistent with the configured binary positive weight.

The notebook heading describes preparation as leakage-safe and balanced, but its executed code does not demonstrate source-group splitting, deduplication, or balanced resampling. In fact, source-link columns are dropped before the RAID row split. Related source/attack variants may therefore cross its internal train/validation boundary; that is a methodological risk requiring an audit, not a claim that held-out leaderboard labels were used. Later academic documentation should distinguish completed operations from proposed safeguards.

The saved outputs establish the operations and counts shown in the notebook. They do not alone prove the precise revision of every dataset currently on the Hub or successful execution of every publication cell. No dataset was regenerated during this phase.

## Evaluation evidence and implications

PAN26 Mean is the arithmetic mean of ROC-AUC, complemented Brier score, c@1, F1, and F0.5u in the saved evaluator outputs. It is a composite metric, not classification accuracy. The two results described as approximately 83 and 89 are precisely 83.6% and 89.4% on that composite scale.

| Trainable multitask variant | ROC-AUC | Brier complement | c@1 | F1 | F0.5u | Mean |
|---|---:|---:|---:|---:|---:|---:|
| attention | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | 0.836 |
| concatenation | 0.959 | 0.868 | 0.833 | 0.878 | 0.933 | 0.894 |

The observed improvement supports further study of the multitask variants under this training/evaluation setting. Eight single completed runs do not establish statistical significance, universal superiority of a fusion method, or the causal benefit of auxiliary supervision. Repeated seeds and controlled comparisons would be needed for those claims.

Only `docs/raid-evaluation/flaird-modernbert-large-attention-multitask` currently contains RAID results. Its saved `score_agg` is:

| RAID slice | Machine recall at 5% FPR | Machine recall at 1% FPR | AUROC |
|---|---:|---:|---:|
| All attacks | 0.9976776961 | 0.9900061275 | 0.9987255193 |
| No adversarial attack | 0.9995588235 | 0.9976102941 | 0.9996453508 |

The JSON's field named `accuracy` records $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ on machine examples at the specified FPR. For all attacks, the denominators are 652,800 machine examples: TP/FN are 651,284/1,516 at 5% FPR and 646,276/6,524 at 1% FPR. There are domain-specific thresholds, not a shared 0.5 cutoff.

Among the aggregate attack slices, paraphrase and zero-width space are weaker at 1% FPR, with recalls 0.9607536765 and 0.9601470588 respectively. The slices and `all` aggregates overlap; summing all 10,881 score rows would double-count observations. These existing artifacts support a later detailed domain/generator/attack analysis, not RAID results for all eight experiments.

The RAID metadata inside the saved result contains no populated model link, so the containing directory provides the current association to the attention multitask variant. The result hashes and release date should be retained in later reporting. No benchmark predictions or published scores were altered.

## Demo readiness

`demo/examples.csv` contains 20 examples with `text`, `attack`, `model`, `label`, and `generator_label`. It includes all eleven output labels, one human example, adversarial transformations such as whitespace, paraphrase, and zero-width spaces, and missing attack values for some PAN-style examples. The UI should display supplied provenance separately from predictions, treat blank attack metadata as unspecified, and never feed ground-truth labels into the explainer.

The present demo returns only a machine score, loads the model on startup, and does not yet implement example selection, structured explanations, or charts. Phase one's result schema supplies the numerical data required for that work. Token attention should not be fabricated: the implemented fusion attention is over feature groups.

## Validation performed

The offline tests and the trained-checkpoint report are reproducible through the commands in [EXPLANATION_MODULE.md](EXPLANATION_MODULE.md). The JSON under `docs/validation` records model revisions, installed versions, per-example forward differences, completeness checks, available heads/diagnostics, and truncation. These are functional checks of the explanation layer, distinct from TIRA and RAID benchmark evaluations.

The trained-model loader uses the explicitly requested `AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)` and `AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)` pathway. Integration remains compatible with their published custom-code modules without modifying trained weights.

## Trained-checkpoint validation results

All 25 offline tests passed. All eight published checkpoints passed all 20 examples (160 trained-weight checks) using CPU float32 and the requested AutoModel/AutoTokenizer loading path. The maximum absolute difference between explanation and independent full-forward logits was zero for every checkpoint. Every attribution satisfied the configured completeness criterion.

| Checkpoint suffix | Examples passed | Maximum integration nodes | Maximum absolute completeness residual |
|---|---:|---:|---:|
| attention-multitask-frozen | 20/20 | 256 | 0.066911 |
| attention-multitask | 20/20 | 256 | 0.021678 |
| attention-single-task-frozen | 20/20 | 512 | 0.069989 |
| attention-single-task | 20/20 | 512 | 0.027060 |
| concatenation-multitask-frozen | 20/20 | 256 | 0.021152 |
| concatenation-multitask | 20/20 | 256 | 0.005286 |
| concatenation-single-task-frozen | 20/20 | 256 | 0.016174 |
| concatenation-single-task | 20/20 | 256 | 0.003935 |

Residuals are in logit units and are evaluated against each example's absolute-plus-relative tolerance; they are not errors in predicted probability. Individual rank stability and baseline sensitivity remain future research checks. Immutable checkpoint revisions, package versions, and per-example details are recorded in [explanation-checkpoints.json](validation/explanation-checkpoints.json).

## Remaining phases

1. After instruction to continue: build and validate the organized Gradio UI, example chooser, probability/generator summaries, signed feature plots, group attention, replacement effects, and numerical-quality indicators in `demo`.
2. After the following instruction: write detailed dataset, architecture, training, and evaluation documentation, complete the thesis proposal and technical README, and present precise result tables and verified model links.
3. Treat deployment to a Hugging Face Space as a concrete follow-on task once the interface and dependencies have been validated. No Space was created or published during phase one.
