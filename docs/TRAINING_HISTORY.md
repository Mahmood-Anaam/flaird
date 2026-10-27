# Historical attention-multitask training summary

This table is preserved from `src/flaird/modeling/README.md` at baseline commit `76ade95cfb71f0ab4959a6021d04096523918a36`. It identifies itself as attention-multitask training output. It is a historical model-card summary, not a newly executed experiment or a substitute for raw Trainer logs. Dataset/model revision, hardware, exact runtime and the corresponding checkpoint hash were not recorded alongside it.

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

The original card reports final validation loss 0.0231 and composite Mean 0.997. This is internal validation, not the external TIRA PAN26 Mean of 0.836. These values address different evaluation populations. Continued loss reduction inside the recorded epoch is compatible with ongoing learning, but does not demonstrate that additional epochs would improve external generalization.

The original card's framework section lists Transformers 5.16.1, PyTorch 2.11.0+cu128, datasets 4.0.0 and tokenizers 0.23.1. The package specification pins PyTorch 2.8.0. This difference is preserved as a provenance gap rather than silently reconciled. No corresponding training-history table is archived for the other seven variants.
