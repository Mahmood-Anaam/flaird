# Notebook execution and output audit

Cell indices are zero-based positions in notebook JSON. All code-cell execution counters are null. Saved outputs can therefore reflect prior or out-of-order execution. Each output block was inspected, including the embedded PNG figures; absent outputs are marked explicitly. No notebook was rerun during this release.

The preparation record contains no saved Python exception outputs. Progress widgets that show 0% are retained display snapshots, not independently verified job completion signals. The final dataset representations are stronger evidence for recorded row counts.

See [dataset preparation](DATASET_PREPARATION.md), [training](TRAINING_CONFIGURATION.md) and [historical model-card training summary](TRAINING_HISTORY.md) for interpretation.

## flaird_dataset_preparation.ipynb

- SHA-256 of reviewed baseline notebook: `798139dc3f7c682bb59764d1b311b2962d8ac2c2a709266d4c1892ce9424b9ed`.
- 71 cells; 46 code cells; 37 saved output blocks.

| Cell | Type | Source / operation | Retained output evidence |
|---:|---|---|---|
| 0 | markdown | Colab launcher badge | No output (narrative) |
| 1 | markdown | # FLAIRD dataset preparation | No output (narrative) |
| 2 | markdown | ## Configuration | No output (narrative) |
| 3 | code | import os | No saved output; execution/result not demonstrated here |
| 4 | markdown | ## Imports | No output (narrative) |
| 5 | code | !pip install pystylometry[all] --q | Text output: Preparing metadata (setup.py) ... done |
| 6 | code | import os | No saved output; execution/result not demonstrated here |
| 7 | markdown | ## PAN-2026 Dataset | No output (narrative) |
| 8 | markdown | Load the official PAN-2026 training and validation splits from the HuggingFace Hub. | No output (narrative) |
| 9 | code | pan26_train = load_dataset(f"{HF_USERNAME}/pan26",split="train") | No saved output; execution/result not demonstrated here |
| 10 | code | pan26_train | Dataset representation; rows 23707 |
| 11 | code | pan26_train.column_names | Column-name/schema list |
| 12 | markdown | **Training split** — 23,707 samples across 5 columns: id, text, model, label, genre. | No output (narrative) |
| 13 | code | pan26_validation | Dataset representation; rows 3589 |
| 14 | code | pan26_validation.column_names | Column-name/schema list |
| 15 | markdown | **Validation split** — 3,589 samples with the same schema. | No output (narrative) |
| 16 | markdown | Cast the label column to ClassLabel. | No output (narrative) |
| 17 | code | class_labels = ClassLabel(names=["human", "machine"], num_classes=2) | No saved output; execution/result not demonstrated here |
| 18 | code | pan26_train = pan26_train.add_column(name="dataset", column=["pan26-train"] * len(pan26_train)) | No saved output; execution/result not demonstrated here |
| 19 | code | pan26_train , pan26_validation | Dataset representation; rows 23707, 3589 |
| 20 | markdown | ## RAID Dataset | No output (narrative) |
| 21 | markdown | The RAID benchmark ([Dugan et al., 2024](https://arxiv.org/abs/2406.07441)) contains 5.6 M rows genera… | No output (narrative) |
| 22 | markdown | raid_test is held out and never enters preparation. It has no labels and is retained solely for RAID l… | No output (narrative) |
| 23 | code | raid_train = load_dataset("liamdugan/raid","raid",split="train") | No saved output; execution/result not demonstrated here |
| 24 | code | raid_train, raid_test | Dataset representation; rows 5615820, 672000 |
| 25 | code | raid_train.column_names, raid_test.column_names | Text output: (['id', |
| 26 | markdown | Retain only the columns required for training; rename generation → text and domain → genre to align wi… | No output (narrative) |
| 27 | code | raid_train = raid_train.select_columns(["id", "generation", "model", "domain", "attack"]) | Column-name/schema list |
| 28 | markdown | Add integer label column: 0 = human, 1 = machine, mirroring the PAN-2026 convention. | No output (narrative) |
| 29 | code | labels = [0 if model.lower() == "human" else 1 for model in raid_train["model"]] | No saved output; execution/result not demonstrated here |
| 30 | code | raid_train | Dataset representation; rows 5615820 |
| 31 | markdown | Cast the label column to ClassLabel. | No output (narrative) |
| 32 | code | class_labels = ClassLabel(names=["human", "machine"], num_classes=2) | Saved progress snapshot (1 line(s)) |
| 33 | code | raid_split = raid_train.train_test_split( | Dataset representation; rows 5054238, 561582 |
| 34 | code | raid_train = raid_split['train'] | No saved output; execution/result not demonstrated here |
| 35 | code | raid_train, raid_validation | Dataset representation; rows 5054238, 561582 |
| 36 | code | raid_train = raid_train.add_column(name="dataset", column=["raid-train"] * len(raid_train)) | Saved progress snapshot (1 line(s)); Saved progress snapshot (1 line(s)) |
| 37 | code | raid_train, raid_validation | Dataset representation; rows 5054238, 561582 |
| 38 | markdown | ## Flaird Dataset | No output (narrative) |
| 39 | code | import datasets | Dataset representation; rows 5077945, 565171 |
| 40 | code | """Deterministic forensic-linguistic features used by FLAIRD.""" | No saved output; execution/result not demonstrated here |
| 41 | code | import re | No saved output; execution/result not demonstrated here |
| 42 | code | flaird_validation = flaird_validation.map(normalize_batch, batched=True, batch_size=2000, num_proc=4) | Saved progress snapshot (1 line(s)); Dataset representation; rows 565171 |
| 43 | code | flaird_train = flaird_train.map(normalize_batch, batched=True, batch_size=2000, num_proc=4) | Saved progress snapshot (1 line(s)); Dataset representation; rows 5077945 |
| 44 | code | class_generator_labels = ClassLabel(names=list(flaird_model_set.keys()), num_classes=len(flaird_model_… | Saved progress snapshot (1 line(s)); Saved progress snapshot (1 line(s)) |
| 45 | code | flaird_train, flaird_validation | No saved output; execution/result not demonstrated here |
| 46 | code | flaird_train = flaird_train.select_columns([ | Dataset representation; rows 5077945, 565171 |
| 47 | markdown | ## Exploratory Data Analysis | No output (narrative) |
| 48 | code | flaird_train_df = flaird_train.to_pandas() | No saved output; execution/result not demonstrated here |
| 49 | code | flaird_train_df.head() | DataFrame preview; includes text_len before its visible assignment |
| 50 | code | flaird_train_df.info() | DataFrame schema, row count and memory summary |
| 51 | code | flaird_validation_df.head() | DataFrame preview; includes text_len before its visible assignment |
| 52 | code | flaird_validation_df.info() | DataFrame schema, row count and memory summary |
| 53 | markdown | **Label distribution** | No output (narrative) |
| 54 | code | fig, axes = plt.subplots(1, 2, figsize=(12, 4)) | Embedded EDA PNG inspected; reproduced in dataset guide |
| 55 | markdown | **Genre distribution** | No output (narrative) |
| 56 | code | fig, axes = plt.subplots(1, 2, figsize=(14, 5)) | Embedded EDA PNG inspected; reproduced in dataset guide |
| 57 | markdown | **AI model by family distribution (training)** | No output (narrative) |
| 58 | code | model_counts = flaird_train_df[flaird_train_df['label'] == 1]['model_family'].value_counts() | Embedded EDA PNG inspected; reproduced in dataset guide |
| 59 | code | model_counts | Exact machine-family counts; transcribed in dataset guide |
| 60 | markdown | **Text length distribution** | No output (narrative) |
| 61 | code | for df, name in [(flaird_train_df, 'Train'), (flaird_validation_df, 'Validation')]: | Embedded EDA PNG inspected; reproduced in dataset guide |
| 62 | markdown | ## Features | No output (narrative) |
| 63 | code | flaird_train_features = flaird_train.map(add_features, batched=True, batch_size=2000, num_proc=4) | Saved progress snapshot (1 line(s)); Dataset representation; rows 5077945 |
| 64 | code | flaird_validation_features = flaird_validation.map(add_features, batched=True, batch_size=2000, num_pr… | Saved progress snapshot (1 line(s)); Dataset representation; rows 565171 |
| 65 | code | feature_extractor = ForensicFeatureExtractor() | No saved output; execution/result not demonstrated here |
| 66 | markdown | ## Push to hub | No output (narrative) |
| 67 | code | flaird_train.push_to_hub(f"{HF_USERNAME}/flaird-raid-pan26", config_name="flaird_train", split="train") | No saved output; execution/result not demonstrated here |
| 68 | code | flaird_train_features.push_to_hub(f"{HF_USERNAME}/flaird-raid-pan26", config_name="flaird_train_featur… | No saved output; execution/result not demonstrated here |
| 69 | markdown | ## Summary | No output (narrative) |
| 70 | markdown | This notebook details the preparation of the FLAIRD dataset, which combines the PAN-2026 and RAID data… | No output (narrative) |

## flaird_training.ipynb

- SHA-256 of reviewed baseline notebook: `10ce8bf75032a9adfda3b477b6e1660a24b6e120ac0a8b128f5b54c9f1bcf4a3`.
- 37 cells; 24 code cells; 0 saved output blocks.

| Cell | Type | Source / operation | Retained output evidence |
|---:|---|---|---|
| 0 | markdown | Colab launcher badge | No output (narrative) |
| 1 | markdown | # FLAIRD training | No output (narrative) |
| 2 | markdown | ## Install flaird | No output (narrative) |
| 3 | code | BRANCH = "main" | No saved output; execution/result not demonstrated here |
| 4 | code | !pip install -e '/content/flaird' --q | No saved output; execution/result not demonstrated here |
| 5 | code | import IPython | No saved output; execution/result not demonstrated here |
| 6 | markdown | ## Configuration | No output (narrative) |
| 7 | code | from google.colab import drive | No saved output; execution/result not demonstrated here |
| 8 | markdown | **Configure GitHub** | No output (narrative) |
| 9 | code | import os | No saved output; execution/result not demonstrated here |
| 10 | markdown | **Configure Hugging Face** | No output (narrative) |
| 11 | code | import os | No saved output; execution/result not demonstrated here |
| 12 | markdown | **Configure Weights & Biases** | No output (narrative) |
| 13 | code | import os | No saved output; execution/result not demonstrated here |
| 14 | markdown | ## Verify the accelerator | No output (narrative) |
| 15 | code | import torch | No saved output; execution/result not demonstrated here |
| 16 | markdown | ## Select an experiment | No output (narrative) |
| 17 | code | CONFIG_FILE = "flaird-modernbert-large-attention-multitask-frozen.json" | No saved output; execution/result not demonstrated here |
| 18 | markdown | ## Train | No output (narrative) |
| 19 | code | !flaird-train /content/flaird/configs/{CONFIG_FILE} | No saved output; execution/result not demonstrated here |
| 20 | markdown | ## RAID Submission | No output (narrative) |
| 21 | code | import os | No saved output; execution/result not demonstrated here |
| 22 | code | import os | No saved output; execution/result not demonstrated here |
| 23 | code | # shard-index => 0,1,2,3,4 | No saved output; execution/result not demonstrated here |
| 24 | code | import os | No saved output; execution/result not demonstrated here |
| 25 | code | !git config --global user.email "eng.mahmood.anaam@gmail.com" | No saved output; execution/result not demonstrated here |
| 26 | code | from pathlib import Path | No saved output; execution/result not demonstrated here |
| 27 | code | !cd /content/raid && git status | No saved output; execution/result not demonstrated here |
| 28 | code | !cd /content/raid && git add . | No saved output; execution/result not demonstrated here |
| 29 | code | !cd /content/raid && git commit -m "Add submissions" | No saved output; execution/result not demonstrated here |
| 30 | code | !cd /content/raid && git status | No saved output; execution/result not demonstrated here |
| 31 | code | !cd /content/raid && git push | No saved output; execution/result not demonstrated here |
| 32 | code | !cd /content/raid && git status | No saved output; execution/result not demonstrated here |
| 33 | markdown | ## TIRA Submission | No output (narrative) |
| 34 | code | # run workfow github action | No saved output; execution/result not demonstrated here |
| 35 | markdown | ## Disconnect & Delete Runtime | No output (narrative) |
| 36 | code | from google.colab import runtime | No saved output; execution/result not demonstrated here |

## Interpretation safeguards

- Preparation cells 54, 56, 58 and 61 preserve actual plotted figures; exact family counts are separately saved in cell 59.
- Cells 63–64 preserve the combined feature dataset row counts and schema.
- Preparation cell 65 and publication cells 67–68 contain no saved success output.
- All training-notebook cells lack saved output, including training, accelerator inspection, RAID inference and Git publication commands. Completion status comes from project context and the separate metric archive.
- The preparation notebook does not implement the leakage audit or resampling implied by its introductory wording. The documentation records actual operations.
- Private credentials are referenced through Colab secret lookups; no secret values are reproduced in this audit.
