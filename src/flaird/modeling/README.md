# FLAIRD modeling package

FLAIRD fuses ModernBERT semantic states with seven learned groups of 35 deterministic forensic features. Attention and concatenation variants share a binary machine-origin classifier and optionally an eleven-class family classifier.

| File | Responsibility |
| --- | --- |
| `configuration_flaird.py` | Model settings, canonical groups and family mapping |
| `features.py` | Ordered finite scalar extraction |
| `modeling_fusion.py` | Masked pooling, attention/gate and concatenation |
| `modeling_flaird.py` | Feature projections, pretrained encoder, shared head, logits and losses |
| `__init__.py` | Exports and Transformers registration |

The trained-model contract requires `input_ids`, `attention_mask` and `forensic_features` (B × 35). Load checkpoints with `AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)` and the corresponding `AutoTokenizer`. The sole binary logit is the machine score after sigmoid; family logits are optional and independent.

See [full architecture and feature definitions](../../../docs/MODEL_ARCHITECTURE.md), [training losses](../../../docs/TRAINING_EXPERIMENTS.md) and [executable usage](../../../docs/REPRODUCIBILITY.md). Feature order and preprocessing policy are checkpoint interfaces and must be preserved.
