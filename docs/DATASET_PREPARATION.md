# Dataset preparation and provenance

This document reconstructs the executed preparation recorded in [the preparation notebook](../notebooks/flaird_dataset_preparation.ipynb). Counts below are saved notebook outputs, not counts obtained by downloading the latest upstream dataset. The training and validation configurations consume the resulting Hub configurations. Historical Hub revisions, upstream file hashes and complete preprocessing logs were not retained in this repository; the notebook is therefore an implementation record with partial provenance rather than a fully immutable dataset release manifest.

## Purpose and source inventory

PAN26 contributes task-specific examples; RAID contributes scale, generator variation, domains and adversarial modifications. Combining them aims to teach an origin detector that uses both contextual representations and forensic style. These are authorship labels, not factuality or malicious-intent labels.

| Source and configuration | Saved split | Rows | Role |
| --- | --- | ---: | --- |
| `${HF_USERNAME}/pan26` | train | 23,707 | PAN training input |
| `${HF_USERNAME}/pan26` | validation | 3,589 | PAN validation input |
| `liamdugan/raid`, `raid` | train | 5,615,820 | RAID labeled development source |
| `liamdugan/raid`, `raid_test` | test | 672,000 | Separate, unlabeled leaderboard input |

`HF_USERNAME` is read from a notebook secret. The PAN source identifier is parameterized; the later experiment configurations explicitly identify `MahmoodAnaam/flaird-raid-pan26` as the prepared dataset. The RAID source has evolved since preparation, so current upstream totals must not replace these saved counts. RAID test initially exposes `id` and `generation`, without labels available to the notebook. It is not concatenated into training.

## Executed sequence

1. **Inspect sources (cells 4–26).** Load the provided PAN train/validation splits and RAID development/test configurations; inspect features, sample rows and lengths. Source selection is by configuration, not a locally reconstructed PAN random split.
2. **Reduce RAID columns (cell 27).** Keep `id`, `generation`, `model`, `domain`, `attack`; rename `generation` to `text` and `domain` to `genre`. The notebook discards `source_id`, `adv_source_id`, decoding settings and repetition penalty at this stage.
3. **Create binary labels and split RAID (cells 28–33).** Map `human` to 0 and every other model to 1, cast to a two-class `ClassLabel`, and use `train_test_split(test_size=0.10, stratify_by_column="label", seed=42)`.
4. **Attach provenance and combine (cells 34–39).** Set `dataset` to `pan26-train`, `pan26-validation`, `raid-train` or `raid-validation`, then concatenate corresponding source splits.
5. **Normalize and label generator families (cells 40–44).** Replace contact identifiers, produce binary/family labels and cast family IDs. Mapping runs in batches of 2,000 with four processes.
6. **Explore the combined training set (cells 45–62).** Inspect family frequencies and text-length histograms; derive descriptive plots. `text_len` is an exploratory dataframe column. Restricting a histogram to 5,000 characters affects the plot, not the stored training text.
7. **Precompute forensic vectors (cell 64).** Apply the 35-value extractor to normalized combined train/validation text, with batches of 2,000 and four processes.
8. **Prepare the separate leaderboard configuration (cell 65).** Normalize RAID test `generation` and compute its vectors. This cell has no saved execution output. It documents the intended preparation operation, without proving the resulting upload or its revision.
9. **Publish named configurations (cells 67–68).** The notebook contains `push_to_hub` calls. They have no saved output; a heading describes private storage, but the calls themselves do not set `private=True`. Access settings and upload success require Hub-side evidence.

## Split sizes and source imbalance

| Prepared split | RAID rows | PAN rows | Total |
| --- | ---: | ---: | ---: |
| Training | 5,054,238 | 23,707 | 5,077,945 |
| Validation | 561,582 | 3,589 | 565,171 |

Training is approximately 99.53% RAID and 0.47% PAN by row count. Concatenation supplies no explicit source-balancing sampler or source-specific loss. Consequently, task-specific PAN behavior cannot be presumed to receive equal influence merely because two sources are present.

Saved training labels contain **4,924,437 machine texts** and **153,508 human texts** (approximately 96.98% / 3.02%). Label stratification maintains binary prevalence in the RAID row split; it does not explicitly balance generator families, genres or attacks.

## Final schema and class mapping

| Field | Meaning |
| --- | --- |
| `id` | Original source ID; not globally re-keyed in the notebook |
| `dataset` | Four-way source/split provenance string |
| `text` | Normalized text used for prepared training/validation |
| `genre` | Source genre/domain |
| `attack` | Source attack description, where present |
| `model` | Original model label |
| `model_family` | Canonical family string |
| `label` | Binary origin: human=0, machine=1 |
| `generator_label` | Integer family class in the order below |
| `forensic_features` | Ordered list of 35 finite scalar values |

| Family ID | Family | Training rows | Configured auxiliary weight |
| --- | --- | ---: | ---: |
| 0 | GPT | 1,450,356 | 0.04221510 |
| 1 | Meta-LLaMA | 579,466 | 0.06678689 |
| 2 | MPT | 1,155,473 | 0.04729609 |
| 3 | Cohere | 577,318 | 0.06691101 |
| 4 | Mistral | 1,156,816 | 0.04726863 |
| 5 | Gemini | 2,692 | 0.97986811 |
| 6 | DeepSeek | 901 | 1.69372451 |
| 7 | Falcon | 879 | 1.71478915 |
| 8 | Bison | 265 | 3.12307358 |
| 9 | Qwen | 271 | 3.08830738 |
| 10 | human | 153,508 | 0.12975964 |

The human class is intentionally included in the auxiliary task. Family probabilities are predictions over all eleven labels, not probabilities conditioned on a machine decision.

### Generator aliases recorded in cell 41

| Family | Accepted source strings |
| --- | --- |
| GPT | `gpt3`, `gpt2`, `chatgpt`, `gpt4`, `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`, `o3-mini`, `gpt-4.5-preview`, `gpt-4-turbo-paraphrase`, `gpt-4-turbo` |
| Meta-LLaMA | `llama-chat`, `llama-2-7b-chat`, `llama-2-70b-chat`, `llama-3.3-70b-instruct`, `llama-3.1-8b-instruct` |
| Mistral | `mistral`, `mistral-chat`, `ministral-8b-instruct-2410`, `mistral-7b-instruct-v0.2`, `mixtral-8x7b-instruct-v0.1` |
| Gemini | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-pro`, `gemini-pro-paraphrase` |
| DeepSeek | `deepseek-r1-distill-qwen-32b` |
| Falcon | `falcon3-10b-instruct` |
| Bison | `text-bison-002` |
| Qwen | `qwen1.5-72b-chat-8bit` |
| MPT | `mpt`, `mpt-chat` |
| Cohere | `cohere`, `cohere-chat` |
| human | `human` |

DeepSeek's distilled Qwen model is assigned to DeepSeek by the notebook's explicit alias policy. There is no unknown-family bucket. The generator-label function lowercases before lookup and fails on unrecognized strings; `model_family` uses a direct lookup of the original string. Future mixed-case or new model labels need explicit validation rather than silent assumptions.

## Normalization and feature computation

`preprocess_text` replaces emails with `[EMAIL]`, account mentions with `[USER]`, and matches of a broad phone regex with `[PHONE]`, then strips boundary whitespace. Its purpose is to reduce dependence on identifiers and align prepared inputs. It does not generally lowercase, remove punctuation or collapse all whitespace.

The phone expression can match ordinary numeric sequences as well as phone numbers. This may alter years, counts and adversarial number modifications; the transformation is therefore part of the model's evidence policy, not a guarantee of perfect anonymization. Feature extraction in the preparation notebook occurs **after** normalization. Semantic tokens are later truncated to 512; feature extraction uses the entire normalized text. This creates an intentional difference in visible context between branches for long documents.

`pystylometry` functions are called without overriding their defaults. In the inspected 1.4.3 implementation, Yule/hapax and bigram entropy functions aggregate chunk-level statistics using a default 1,000-word chunk size. A future package upgrade can change tokenization or aggregation even if the 35 names remain unchanged. Preserve a version lock and reference fixtures for a new dataset release. Full feature definitions and zero fallbacks are in [Model architecture](MODEL_ARCHITECTURE.md).

## Weight rationale

The binary weight in every completed configuration is exactly consistent with the saved counts:

$$w_+=\frac{N_{human}}{N_{machine}}=\frac{153508}{4924437}=0.0311727005543984.$$

Machine is the positive class and the majority, so its positive loss is downweighted. The auxiliary weights are mathematically consistent, within approximately 6 × 10⁻⁸, with normalized inverse square-root frequencies:

$$a_c=\frac{n_c^{-1/2}}{\frac1{11}\sum_{j=0}^{10}n_j^{-1/2}}.$$

This is a reconstruction from the counts and literal JSON weights; the notebook does not contain an executed derivation of these constants. Square-root weighting increases influence for rare families less aggressively than full inverse frequency. It does not create more independent examples, remove domain confounding or establish reliable attribution for the smallest families.

## Leakage and reproducibility boundaries

The RAID split is **row-level** and occurs after source-group identifiers are dropped. Clean texts and related adversarial variants may share an underlying source across train/validation; the notebook does not provide a group-disjoint guarantee. It contains no explicit cross-source deduplication, prompt-group split, near-duplicate audit or held-out-family training protocol. These are future audit tasks, not claims of observed leakage in every row.

PAN validation remains the supplied split. Its independence from RAID sources and from leaderboard prompts is not verified in this repository. Strong leaderboard performance must be read alongside these provenance limitations. No test labels were used by the visible preparation code, but saved artifacts alone cannot certify all upstream overlap properties.

For the next release, retain source/variant/prompt IDs, commit dataset revisions and hashes, audit exact and near duplicates, publish group-disjoint splits, and report class/genre/family/attack counts per split. Compare identifier normalization with raw text under a controlled protocol. Recompute features only for a new version; do not silently relabel the historical benchmarks as results of that version.

## References

- [Preparation notebook](../notebooks/flaird_dataset_preparation.ipynb).
- [RAID dataset and tooling](https://github.com/liamdugan/raid).
- Dugan et al., [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://arxiv.org/abs/2405.07940), 2024. This corrects the RAID arXiv identifier cited in the notebook.
- [Prepared dataset identifier](https://huggingface.co/datasets/MahmoodAnaam/flaird-raid-pan26); access and revision must be resolved in the operator's Hub account.
