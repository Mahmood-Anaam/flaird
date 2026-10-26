# Training configuration and objective

## 1. Completed experiment design

The project owner confirms that all eight main experiments have completed training and evaluation. This document records their committed configurations rather than rerunning expensive experiments or altering the final checkpoints. The [training notebook](../notebooks/flaird_training.ipynb) is an orchestration guide with **no saved cell outputs**. Its clone cell targets `main`, whereas this release reviews `develop`; the exact code revision used for each historical run is not recorded in that notebook. The [result archive](EVALUATION_RESULTS.md) is the source for benchmark measurements.

The eight runs form a factorial comparison of attention/concatenation fusion, frozen/fine-tuned encoder, and binary/multitask supervision. The `-frozen` suffix refers only to encoder gradients; feature projections, fusion and task heads remain trainable. Every newly constructed variant first loads `answerdotai/ModernBERT-large` encoder weights. Both fusion variants train their forensic projections and prediction heads from scratch.

## 2. Shared run settings

| Setting | Committed value | Meaning |
|---|---|---|
| Encoder / tokenizer | `answerdotai/ModernBERT-large` | Pretrained contextual encoder |
| Initial FLAIRD checkpoint | `model_name_or_path: null` | Construct hybrid model, then copy pretrained encoder weights |
| Input limit | 512 tokens | Longer token sequences are truncated |
| Train dataset | `MahmoodAnaam/flaird-raid-pan26`, `flaird_train_features` | Prepared training texts and stored feature vectors |
| Validation dataset | Same repository, `flaird_validation_features` | Internal validation |
| Dataset shuffle | true | Uses default shuffle seed 42 |
| Epochs / learning rate | 1 / 0.00002 | Same nominal schedule across main experiments |
| Optimizer | `adamw_torch_fused` | AdamW; framework defaults supply unspecified beta/epsilon values |
| Weight decay / clipping | 0.01 / maximum gradient norm 1.0 | Regularization and gradient control |
| Scheduler / warmup | cosine / `warmup_steps: 0.06` | Fractional warmup in the pinned Transformers version; see below |
| Evaluation / saving | Every `0.1` of total training steps | Fractional step intervals, rounded by Trainer |
| Checkpoint retention | 2 | Limits retained local checkpoints |
| Logging | First step and every 200 steps | W&B and TensorBoard reporting |
| Random seed | 42 | Does not establish multi-seed uncertainty |
| Precision | `bf16: true` | Requires a compatible runtime |
| Data loader | 4 workers, pinned memory | Loading configuration |
| Gradient checkpointing | false | Model also declares it unsupported |
| Unused columns | Not removed | Collator receives text/features/labels |
| Labels | `labels`, `generator_labels` | Two label tensors supplied by collator |
| Publication | `push_to_hub: true`, strategy `checkpoint` | Checkpoint publishing during training |
| Resume | null | No configured resume checkpoint |

The code uses the prepared feature column and `apply_text_preprocessing=False`, avoiding repeated preprocessing of already normalized training text. Padding is dynamic within batches.

### 2.1 Fractional warmup is version-specific

In the checked **Transformers 5.16.1** implementation, `TrainingArguments.get_warmup_steps(T)` returns `ceil(T * warmup_steps)` when the value is below one. Thus 0.06 means approximately 6% of optimizer steps, not 0.06 literal steps. For a 39,672-step schedule this is 2,381 warmup steps. Older Transformers APIs may instead require an integer `warmup_steps` and a separate ratio field. Reproduction must use the configured version rather than silently translating fields.

With 5,077,945 rows, one device, accumulation 1, no dropped final batch, and batch 128, the nominal count is `ceil(5077945 / 128) = 39,672` updates. Batch 256 gives 19,836. These counts are **derived**, not recovered training logs. Multiple devices, accumulation, limits or Trainer behavior can change them. Equal epochs do not imply equal updates when frozen runs use twice the batch size.

## 3. Weighted binary and auxiliary losses

For machine-positive label $y_i\in\{0,1\}$ and binary logit $z_i$, the binary objective is

$$
\mathcal{L}_{\mathrm{bin}}=-\frac1B\sum_{i=1}^{B}
\left[w_+y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right],
\qquad w_+=0.0311727005543984.
$$

Machine texts are the majority class, so their positive term is downweighted. This is not a class-balanced sampler and does not change corpus composition. It changes the score's relation to the corpus posterior. Under an idealized weighted-risk optimum, $s=w_+p/(1-p+w_+p)$ for unweighted posterior $p$; this is an explanatory derivation, not a calibration correction applied by the software.

For generator label $c_i$ and class weights $a_c$, PyTorch's weighted mean cross-entropy is

$$
\mathcal{L}_{\mathrm{gen}}=
-\frac{\sum_{i=1}^{B}a_{c_i}\log p_{i,c_i}}{\sum_{i=1}^{B}a_{c_i}}.
$$

This denominator matters: the generator term is not simply a batch mean of weighted losses. The model adds it only when binary labels, generator labels, an active generator head and positive generator loss weight are present:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{bin}}+\lambda\mathcal{L}_{\mathrm{gen}},
\qquad\lambda=0.2\text{ for multitask},\quad\lambda=0\text{ for binary-only}.
$$

Human examples participate in generator classification as class 10. All multitask configurations store the following weights; the repository does not retain their derivation procedure.

| ID | Family | Cross-entropy weight |
|---:|---|---:|
| 0 | GPT | 0.04221510 |
| 1 | Meta-LLaMA | 0.06678689 |
| 2 | MPT | 0.04729609 |
| 3 | Cohere | 0.06691101 |
| 4 | Mistral | 0.04726863 |
| 5 | Gemini | 0.97986811 |
| 6 | DeepSeek | 1.69372451 |
| 7 | Falcon | 1.71478915 |
| 8 | Bison | 3.12307358 |
| 9 | Qwen | 3.08830738 |
| 10 | human | 0.12975964 |

## 4. Run-by-run configuration matrix

The names below omit the common `flaird-modernbert-large-` prefix. Each configuration's `output_dir`, `run_name`, and `hub_model_id` follow its full experiment name. The absolute `/content/drive/MyDrive/flaird/` paths are Colab-specific and must be adapted for another filesystem.

<!-- CONFIG_TABLE_START -->
| Configuration | Frozen encoder | Generator head / labels | Train / eval batch per device | Auxiliary weight | Best model selection |
|---|---|---|---|---:|---|
| [attention-multitask-frozen](../configs/flaird-modernbert-large-attention-multitask-frozen.json) | True | True / 11 | 256 / 256 | 0.2 | loss (minimize) |
| [attention-multitask](../configs/flaird-modernbert-large-attention-multitask.json) | False | True / 11 | 128 / 128 | 0.2 | loss (minimize) |
| [attention-single-task-frozen](../configs/flaird-modernbert-large-attention-single-task-frozen.json) | True | False / 0 | 256 / 256 | 0.0 | No explicit best-model reload |
| [attention-single-task](../configs/flaird-modernbert-large-attention-single-task.json) | False | False / 0 | 128 / 128 | 0.0 | loss (minimize) |
| [concatenation-multitask-frozen](../configs/flaird-modernbert-large-concatenation-multitask-frozen.json) | True | True / 11 | 256 / 256 | 0.2 | loss (minimize) |
| [concatenation-multitask](../configs/flaird-modernbert-large-concatenation-multitask.json) | False | True / 11 | 128 / 128 | 0.2 | loss (minimize) |
| [concatenation-single-task-frozen](../configs/flaird-modernbert-large-concatenation-single-task-frozen.json) | True | False / 0 | 256 / 256 | 0.0 | loss (minimize) |
| [concatenation-single-task](../configs/flaird-modernbert-large-concatenation-single-task.json) | False | False / 0 | 128 / 128 | 0.0 | loss (minimize) |
<!-- CONFIG_TABLE_END -->

Seven main configurations explicitly reload the checkpoint with minimum evaluation loss. `attention-single-task-frozen.json` omits `load_best_model_at_end`, so it inherits the framework default rather than enabling best-checkpoint restoration. This is an experimental difference, not corrected retroactively. Selection on combined multitask loss also differs conceptually from selection on binary detection quality. The primary binary metric summary should therefore be interpreted alongside the checkpoint-selection rule.

Single-task configurations retain the stored generator class weight list but disable the generator head and set the generator loss weight to zero, so those class weights do not affect their objective.

## 5. Development configuration

`flaird-dev-multitask.json` uses frozen attention fusion, 1,024 training rows, 512 evaluation rows, batch 64, learning rate 0.0001, one epoch, evaluation/saving every half of total steps, logging every step, TensorBoard reporting, and best-model selection by maximum `mean`. It is a smoke/development run and is excluded from the eight-run benchmark tables.

## 6. Training entry point and reproducibility

```bash
python -m pip install -e .
python -m pip install torchsummary
flaird-train configs/flaird-modernbert-large-attention-multitask.json
```

`train.py` imports `torchsummary`, which is not declared in the baseline package dependencies. It also expects reporting integrations and a suitable GPU for bf16. These commands describe the source entry point; they do not imply that this documentation release retrained the models. Private dataset or Hub publication access requires the user's existing authorized credentials. Do not hardcode tokens in notebooks or Git URLs.

For an independent reproduction, retain the resolved packages, CUDA/GPU, number of devices, dataset revisions, initial model revision, exact config, optimizer/scheduler state and Trainer logs. [historical model-card summary](TRAINING_HISTORY.md) contains a historical attention-multitask training table ending at 39,672 steps, but is an older generated artifact and lacks checkpoint/revision provenance. It can support a historical trajectory discussion, not replace complete logs for all eight runs.

Architecture defaults not overridden by JSON include seven 128-dimensional feature tokens, eight fusion heads, mean semantic pooling, GELU activations and 0.1 feature/fusion/classifier dropout. See [ARCHITECTURE.md](ARCHITECTURE.md). A comparative test of longer training should give attention and concatenation matched update budgets, multiple seeds and validation-based stopping; none of those additional runs is claimed as complete.
