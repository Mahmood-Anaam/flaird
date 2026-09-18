# FLAIRD Submission & Evaluation Guide

This guide provides comprehensive instructions for running inference, launching the Hugging Face Space demo, executing evaluation/submissions on **TIRA (PAN CLEF 2026 Shared Task)**, and generating submissions for the **RAID Leaderboard** across all 8 trained FLAIRD models.

---

## 1. Overview of Trained FLAIRD Models

FLAIRD (*Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns*) integrates 35 stylometric forensic metrics with ModernBERT transformer representations across two fusion strategies (Attention vs Concatenation) and two task configurations (Single-Task vs Multi-Task). All 8 trained models are hosted on Hugging Face Hub under the repository namespace `yusr9/`:

| # | Model Identifier | Fusion | Task Type | Encoder Status |
|---|------------------|--------|-----------|----------------|
| 1 | `yusr9/flaird-modernbert-large-attention-multitask` | Attention | Multi-Task | Fine-tuned |
| 2 | `yusr9/flaird-modernbert-large-concatenation-multitask` | Concatenation | Multi-Task | Fine-tuned |
| 3 | `yusr9/flaird-modernbert-large-attention-single-task` | Attention | Single-Task | Fine-tuned |
| 4 | `yusr9/flaird-modernbert-large-concatenation-single-task` | Concatenation | Single-Task | Fine-tuned |
| 5 | `yusr9/flaird-modernbert-large-attention-multitask-frozen` | Attention | Multi-Task | Frozen |
| 6 | `yusr9/flaird-modernbert-large-concatenation-multitask-frozen` | Concatenation | Multi-Task | Frozen |
| 7 | `yusr9/flaird-modernbert-large-attention-single-task-frozen` | Attention | Single-Task | Frozen |
| 8 | `yusr9/flaird-modernbert-large-concatenation-single-task-frozen` | Concatenation | Single-Task | Frozen |

> **Note:** All models use `AutoTokenizer` and `AutoModelForSequenceClassification` with `trust_remote_code=True`.

---

## 2. Local Forensic Inference CLI

You can perform forensic text classification and generate detailed diagnostic explanations using the CLI command `flaird-inference`:

```bash
flaird-inference yusr9/flaird-modernbert-large-attention-multitask \
  --text "Duke Ellington, a titan of jazz, revolutionized the genre through his innovative compositions..." \
  --output-file explanation.json
```

---

## 3. Hugging Face Space Interactive Demo (`demo/`)

The Hugging Face Space interface provides interactive visualizations (Plotly bar charts for feature importance and generator probabilities).

To launch locally:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

---

## 4. TIRA Platform Submission (PAN 2026 Shared Task)

FLAIRD uses Docker containers and GitHub Actions for continuous evaluation on TIRA.

### 4.1 Running Predictions Locally with TIRA Script

```bash
flaird-tira-submission \
  --input-directory /path/to/tira_input \
  --output-directory /path/to/tira_output \
  --model yusr9/flaird-modernbert-large-attention-multitask \
  --batch-size 16
```

### 4.2 TIRA Submission via GitHub Actions Workflow

1. Navigate to your repository on GitHub -> **Actions** -> **Upload Software to TIRA**.
2. Click **Run workflow**.
3. (Optional) Provide the desired model in the `model` input parameter (e.g., `yusr9/flaird-modernbert-large-concatenation-multitask`).
4. The workflow will automatically build the Docker image, run verification against the TIRA smoke-test dataset, and push the software submission to TIRA.

---

## 5. RAID Leaderboard Submission (`src/flaird/scripts/raid_submission.py`)

The RAID leaderboard evaluates machine text detectors on unseen generators, domains, and adversarial attacks.

### 5.1 Evaluating a Single Model

To evaluate a single model on a local RAID test CSV file:

```bash
flaird-raid-submission \
  --model yusr9/flaird-modernbert-large-attention-multitask \
  --data-path /path/to/raid_test.csv \
  --output-dir raid_submissions
```

### 5.2 Batch Submissions for All 8 Trained Models

To generate submission folders (`predictions.json` and `metadata.json`) for **all 8 trained models**:

```bash
flaird-raid-submission \
  --all-models \
  --data-path /path/to/raid_test.csv \
  --output-dir raid_submissions \
  --batch-size 16
```

### 5.3 Submitting to RAID GitHub Repository

1. Fork the official [RAID repository](https://github.com/liamdugan/raid).
2. Copy the output folder `raid_submissions/<detector-name>` into `leaderboard/submissions/<detector-name>` in your fork.
3. Verify that `predictions.json` and `metadata.json` are present.
4. Create a Pull Request to the RAID repository. The automated bot will evaluate your submission and compute performance metrics across all splits.
