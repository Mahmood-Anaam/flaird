---
library_name: transformers
tags:
- generated_from_trainer
metrics:
- f1
model-index:
- name: flaird-modernbert-large-attention-multitask
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flaird-modernbert-large-attention-multitask

This model is a fine-tuned version of [](https://huggingface.co/) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0231
- Roc-auc: 1.0
- Brier: 0.995
- C@1: 0.994
- F1: 0.997
- F05u: 0.999
- Mean: 0.997

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 128
- eval_batch_size: 128
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.06
- num_epochs: 1

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Roc-auc | Brier | C@1   | F1    | F05u  | Mean  |
|:-------------:|:------:|:-----:|:---------------:|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0.0903        | 0.1000 | 3968  | 0.0801          | 0.997   | 0.984 | 0.979 | 0.989 | 0.995 | 0.989 |
| 0.0603        | 0.2000 | 7936  | 0.0585          | 0.999   | 0.985 | 0.979 | 0.989 | 0.995 | 0.99  |
| 0.0516        | 0.3001 | 11904 | 0.0460          | 0.999   | 0.992 | 0.989 | 0.994 | 0.997 | 0.994 |
| 0.0433        | 0.4001 | 15872 | 0.0385          | 0.999   | 0.989 | 0.985 | 0.992 | 0.997 | 0.992 |
| 0.0369        | 0.5001 | 19840 | 0.0335          | 1.0     | 0.994 | 0.991 | 0.995 | 0.998 | 0.996 |
| 0.0301        | 0.6001 | 23808 | 0.0291          | 1.0     | 0.993 | 0.991 | 0.995 | 0.998 | 0.995 |
| 0.0294        | 0.7001 | 27776 | 0.0267          | 1.0     | 0.996 | 0.994 | 0.997 | 0.999 | 0.997 |
| 0.0254        | 0.8002 | 31744 | 0.0244          | 1.0     | 0.994 | 0.993 | 0.996 | 0.998 | 0.996 |
| 0.0239        | 0.9002 | 35712 | 0.0233          | 1.0     | 0.995 | 0.994 | 0.997 | 0.999 | 0.997 |
| 0.0228        | 1.0    | 39672 | 0.0231          | 1.0     | 0.995 | 0.994 | 0.997 | 0.999 | 0.997 |


### Framework versions

- Transformers 5.16.1
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.23.1
