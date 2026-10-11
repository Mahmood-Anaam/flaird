# FLAIRD Dataset Preparation

## Scope and Sources

The preparation workflow is in `notebooks/flaird_dataset_preparation.ipynb`. It combines English text from PAN 2026 and RAID (Dugan et al., 2024) for binary machine-text detection and optional generator-family classification.

| Source | Role in the prepared corpus | Facts recorded in the notebook |
| --- | --- | --- |
| PAN 2026 | Official train and validation portions | 23,707 training examples and 3,589 validation examples; columns are `id`, `text`, `model`, `label`, and `genre`. |
| RAID | Its training split is divided into training and validation; its `raid_test` split is held out for leaderboard prediction | The notebook describes RAID as approximately 5.6 million rows from 11 LLMs, 11 domains, and 12 attack types. `raid_test` is retained separately and is not used to make the train/validation corpus. |

The PAN 2026 challenge test set is not constructed by this notebook; PAN26 evaluation results are recorded separately in [evaluation-results.md](evaluation-results.md).

## Transformations in the Notebook

1. Load the PAN 2026 train and validation splits. Cast labels to a two-class `ClassLabel`, where `0` is human and `1` is machine, and add a source split marker.
2. Load RAID's `raid` training split and the separate `raid_test` split. For RAID training rows, keep the identifier, generation, model, domain, and attack fields; rename `generation` to `text` and `domain` to `genre`; then assign `0` to model name `human` and `1` to other model names.
3. Split RAID training data 90/10 with stratification on the binary label and seed 42. Mark the resulting subsets as `raid-train` and `raid-validation`.
4. Concatenate PAN training with RAID training, and PAN validation with RAID validation. This aims to preserve the official PAN validation set while adding a stratified RAID validation subset.
5. Normalize email addresses, user mentions, and phone-like strings to `[EMAIL]`, `[USER]`, and `[PHONE]`. Map dataset model names to 11 generator classes (ten machine families plus human).
6. Extract 35 forensic-linguistic features for the normalized text. Keep source identifiers, text, genre, attack, original model, model family, binary label, and generator label in the prepared records.
7. Push the prepared train, validation, and RAID-test datasets to the configured Hugging Face dataset repository. Training configs consume the feature-enriched train and validation configurations.

## Feature Inventory

The ordered feature vector contains:

| Measurement family | Features |
| --- | --- |
| Lexical diversity | `ttr`, `root_ttr`, `log_ttr`, `mtld`, `yule_k`, `yule_i`, `hapax_ratio`, `dis_hapax_ratio`, `sichel_s`, `honore_r` |
| Function-word style | `determiner_ratio`, `preposition_ratio`, `conjunction_ratio`, `pronoun_ratio`, `auxiliary_ratio`, `function_word_ratio`, `function_word_diversity` |
| Surface composition | `avg_word_length`, `vowel_consonant_ratio`, `digit_ratio`, `uppercase_ratio`, `whitespace_ratio` |
| Structure | `avg_sentence_length_chars`, `sentence_length_cv`, `paragraph_length_cv`, `newline_density` |
| Punctuation and rhetoric | `punctuation_density`, `punctuation_variety`, `question_density`, `exclamation_density`, `quote_density` |
| Entropy | `character_bigram_entropy`, `word_bigram_entropy` |
| Repetition and burstiness | `word_burstiness`, `repeated_bigram_ratio` |

The implementation and exact order are authoritative in `src/flaird/modeling/features.py`. `src/flaird/modeling/configuration_flaird.py` groups these 35 values into seven feature tokens for attention fusion. Feature extraction is deterministic; values that are non-finite or unavailable for very short inputs are normalized to zero.

## Label Semantics

The binary label is `0 = human`, `1 = machine`. Generator labels use a fixed mapping: `GPT`, `Meta-LLaMA`, `MPT`, `Cohere`, `Mistral`, `Gemini`, `DeepSeek`, `Falcon`, `Bison`, `Qwen`, and `human`. Multiple model identifiers map to a family, so auxiliary classification is family-level rather than exact model identification.

The notebook's narrative EDA describes the merged data as strongly machine-heavy (approximately 97% machine) and notes that GPT, Mistral, and MPT are frequent families. These are notebook narrative findings, not freshly recomputed statistics in this documentation; exact counts should be regenerated from the pushed dataset before publication.

## Reproducibility and Leakage Audit

The notebook source available in this repository reports no executed cells in the current editor metadata. Its markdown describes intended findings, and its code records the transformation recipe, but an independent run is needed to establish current counts and successful publication.

A schema inconsistency in the checked-in code should be resolved before treating the notebook as a clean-room reproducible script: the documented PAN schema does not include `attack`, while the selected RAID training columns do; the notebook concatenates PAN and RAID datasets before aligning that field. Hugging Face `concatenate_datasets` requires compatible columns/features. Add an explicit missing-value `attack` column to the PAN splits (or drop that field consistently) and verify feature order/types before concatenating.

The code uses the official PAN validation split and a seeded, label-stratified RAID split. The notebook does not show a cross-source duplicate or near-duplicate audit, a group split by prompt/source, or a generator-held-out training split. Therefore, the phrase “leakage-safe” in the notebook heading should be treated as an intention rather than a demonstrated guarantee. Shared prompts, duplicate texts, and source/template overlap should be audited before making strong generalization claims.

The RAID test set is distinct in the notebook and is reserved for submission generation. Preserve that separation; do not use its examples, labels, or leaderboard feedback to select checkpoints or tune thresholds.

## Suggested Reproduction Checks

- Verify row counts and binary/generator label distributions after every source transformation and after concatenation.
- Confirm no null or unmapped `model` values remain before generator-family conversion.
- Align schemas and feature types explicitly before calling `concatenate_datasets`.
- Check exact and near duplicates across train, validation, and sources; record the policy and counts.
- Confirm no test records enter feature fitting, checkpoint selection, threshold selection, or hyperparameter tuning.
- Recompute feature vectors from the normalized text and verify a sample against the stored 35-value vectors.
- Record the dataset revision, preparation code revision, and final split counts with each experiment.
