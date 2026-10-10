# Eight completed training experiments

The user reports that all eight experiments have finished training and evaluation. Their committed JSON configurations specify the intended protocol, and saved TIRA artifacts establish reported downstream results. Full trainer states, loss curves, hardware logs and best-checkpoint manifests are not committed here. This document records configurations without inventing those missing execution details or launching a new training run.

## Experimental design

The design crosses fusion (attention/concatenation), objective (binary single-task/binary plus family multi-task), and encoder adaptation (frozen/trainable): $2\times2\times2=8$ variants. Frozen configurations use twice the per-device batch size. Accordingly, comparisons involving freezing also vary batch size and are not a perfectly isolated causal test of encoder adaptation.

<!-- research:experiments:start -->
| Configuration | Fusion | Auxiliary head | Encoder | Train/eval batch per device | Best-model reload explicitly set |
| --- | --- | --- | --- | --- | --- |
| [flaird-modernbert-large-attention-multitask-frozen](../configs/flaird-modernbert-large-attention-multitask-frozen.json) | attention | 11-class | frozen | 256 | true |
| [flaird-modernbert-large-attention-multitask](../configs/flaird-modernbert-large-attention-multitask.json) | attention | 11-class | trainable | 128 | true |
| [flaird-modernbert-large-attention-single-task-frozen](../configs/flaird-modernbert-large-attention-single-task-frozen.json) | attention | none | frozen | 256 | omitted |
| [flaird-modernbert-large-attention-single-task](../configs/flaird-modernbert-large-attention-single-task.json) | attention | none | trainable | 128 | true |
| [flaird-modernbert-large-concatenation-multitask-frozen](../configs/flaird-modernbert-large-concatenation-multitask-frozen.json) | concatenation | 11-class | frozen | 256 | true |
| [flaird-modernbert-large-concatenation-multitask](../configs/flaird-modernbert-large-concatenation-multitask.json) | concatenation | 11-class | trainable | 128 | true |
| [flaird-modernbert-large-concatenation-single-task-frozen](../configs/flaird-modernbert-large-concatenation-single-task-frozen.json) | concatenation | none | frozen | 256 | true |
| [flaird-modernbert-large-concatenation-single-task](../configs/flaird-modernbert-large-concatenation-single-task.json) | concatenation | none | trainable | 128 | true |
<!-- research:experiments:end -->

`configs/flaird-dev-multitask.json` is a separate development configuration, not a ninth completed benchmark experiment. All eight published checkpoint links and PAN26 scores appear in [the root README](../README.md).

## Common configuration

| Setting | Committed value | Meaning |
| --- | --- | --- |
| Encoder/tokenizer | `answerdotai/ModernBERT-large` | Pretrained semantic backbone |
| `model_name_or_path` | `null` | Build FLAIRD from pretrained encoder, not resume a trained FLAIRD model |
| Feature dimensions | 35 values; seven 128-dimensional groups | Configuration defaults in model code |
| `max_seq_length` | 512 | Semantic branch truncation limit |
| Training data | `MahmoodAnaam/flaird-raid-pan26`, `flaird_train_features` | Precomputed normalized text/features |
| Validation data | Same dataset, `flaird_validation_features` | Prepared validation configuration |
| Training shuffle / seed | `true` / 42 | Same seed across eight configurations |
| Epochs | 1 | Requested pass count |
| Learning rate | 0.00002 | Shared learning rate; no separate encoder/head rates specified |
| Weight decay | 0.01 | Optimizer regularization |
| Optimizer | `adamw_torch_fused` | Fused AdamW request |
| Scheduler | `cosine` | Decaying learning-rate schedule |
| `warmup_steps` | **0.06** | Literal field; not `warmup_ratio` |
| Gradient norm limit | 1.0 | Clipping |
| Precision | `bf16=true` | Training request; depends on supported hardware |
| Gradient checkpointing | false | No activation checkpointing requested |
| Evaluation/save strategy | `steps`; `eval_steps=save_steps=0.10` | Fractional step intervals interpreted by installed Transformers |
| Checkpoint retention | 2 | `save_total_limit` |
| Best-model metric | `loss`, lower is better | Combined loss in multi-task runs |
| Logging | Every 200 steps; first step enabled | W&B and TensorBoard reporters |
| DataLoader | Four workers; pinned memory | Throughput settings |
| `remove_unused_columns` | false | Preserve forensic inputs |
| Label names | `labels`, `generator_labels` | Collator supplies both even when family head is absent |
| Hub | `push_to_hub=true`, `hub_strategy="checkpoint"` | Configured publishing behavior |
| Resume | `null` | No checkpoint resume specified |

The literal `warmup_steps=0.06` must not be described as a verified 6% warmup. Its runtime interpretation requires the historical Transformers version/trainer state; no separate `warmup_ratio=0.06` is present. Changing this field is a new training protocol, not a documentation correction to historical runs.

`load_best_model_at_end=true` is explicit in seven configurations and omitted in attention/single-task/frozen. Document that omission rather than assuming all runs restored a best checkpoint. The saved benchmark artifacts do not identify whether their evaluated checkpoint was final or best, or its immutable Hub revision.

Configs contain Google Drive cache/output paths. Copy a configuration and change paths before local execution. Effective batch is $B_{device}\times N_{devices}\times accumulation$; the actual device count and executed accumulation are not evidenced by saved logs. One seed does not supply confidence intervals or seed-robust significance.

## Loss equations and rationale

Let $y_n\in\{0,1\}$ be human/machine, $\ell_n$ the binary logit and $p_n=\sigma(\ell_n)$. Binary loss uses `BCEWithLogitsLoss` with the machine positive weight $w_+=0.0311727005543984$:

$$\mathcal L_{binary}=-\frac1B\sum_n\left[w_+y_n\log p_n+(1-y_n)\log(1-p_n)\right].$$

Machine texts dominate the prepared source. Choosing $w_+=N_{human}/N_{machine}$ balances total class loss mass under comparable per-example losses. It is deliberately below one. It does not balance each minibatch or remove source/family imbalance.

For auxiliary family labels $c_n$, weights $a_c$ and family softmax $q_{n,c}$, PyTorch's weighted mean cross-entropy is

$$\mathcal L_{family}=\frac{-\sum_n a_{c_n}\log q_{n,c_n}}{\sum_n a_{c_n}}.$$

The denominator is the sum of selected target weights, not simply batch size. Multi-task runs use $\mathcal L=\mathcal L_{binary}+0.2\mathcal L_{family}$. Single-task runs disable the auxiliary classifier and set its loss coefficient to zero. The family weights and their inverse-square-root frequency reconstruction are documented in [Dataset preparation](DATASET_PREPARATION.md).

Family supervision is intended to encourage shared representations of generation style. It remains closed-set: eleven configured labels are not proof of identifying an arbitrary new generator. The minority families have very few examples, and no generator-attribution confusion matrices or per-family validation metrics are retained in this repository.

Weighted BCE changes the population optimum. If an unweighted conditional machine probability is η, minimizing the idealized weighted risk yields

$$p^*=\frac{w_+\eta}{1-\eta+w_+\eta}.$$

This analytical fact explains why the returned sigmoid should not be presented as a certified population posterior. Actual calibration additionally depends on optimization and distribution shift. Temperature scaling, domain-specific calibration and operational threshold selection require held-out data and separate reporting.

## Training implementation and execution record

`flaird-train` parses the JSON through `HfArgumentParser`, initializes the tokenizer, constructs the hybrid model from the pretrained encoder, optionally sets encoder parameters `requires_grad=False`, loads prepared dataset configurations, and trains through `FlairdTrainer`. Freezing parameters does not itself guarantee that a module runs in evaluation mode throughout training.

The training collator uses `apply_text_preprocessing=False` because prepared text and features were already normalized. It dynamically pads/truncates tokens, reads the stored vector, and constructs binary and family labels. Runtime extraction is a fallback if a stored vector is absent. Preserve `feature_column="forensic_features"` and validate the schema before using a different data source.

[The training notebook](../notebooks/flaird_training.ipynb) illustrates Google Drive mounting, secrets, GPU checks and command execution. Its Colab and Kaggle path examples require environment adaptation; they are not evidence of a particular accelerator, duration, cost or full execution history. `train.py` imports `torchsummary`, which is not declared in `pyproject.toml`; the training setup therefore includes it explicitly in [Reproducibility](REPRODUCIBILITY.md).

## Reading the completed comparisons

On PAN26, trainable multi-task concatenation reaches Mean 0.894 and trainable multi-task attention reaches 0.836. Compared with the corresponding trainable single-task variants, gains are 0.134 and 0.095, respectively. With frozen encoders, multi-task minus single-task is +0.025 for concatenation and −0.001 for attention.

These interactions support a descriptive conclusion that auxiliary supervision is most beneficial in the two trainable comparisons in this record. They do not establish universal superiority of multi-task training. Attention offers routing diagnostics but does not outperform concatenation on every dataset. [PAN26 evaluation](PAN26_EVALUATION.md) includes complete metrics, cross-dataset results and interpretation limits.

## Future controlled experiments

Retain immutable training environments, dataset revisions, model commit SHAs, trainer states and full loss/metric logs. Repeat multiple seeds with paired document-level bootstrap intervals. Match effective batch across frozen/trainable runs. Evaluate encoder-only and forensic-only baselines, feature ablations, alternative loss weights, correctly specified warmup schedules and validation-only calibration. New experiments should be versioned separately from the eight completed configurations.
