# Phase-three documentation validation

Validation date: 2026-10-10. This phase documents existing experiments; it did not change weights, configurations, notebooks or original benchmark JSON, retrain a model, or send a leaderboard submission.

## Checks performed

| Check | Result |
| --- | --- |
| Existing model/explanation/UI tests | `35 passed in 13.12s` |
| Research table/evidence check | Eight configs, 32 TIRA rows and 10,881 RAID filtered records verified |
| RAID recall arithmetic | Every saved filtered record checked against TP/(TP+FN) |
| Feature contract | All 35 feature names/order match notebook source, package source and architecture document |
| Dataset sizes/family counts | Read from saved notebook outputs; training total 5,077,945, validation 565,171 |
| Binary weight reconstruction | Exact equality to 153,508 / 4,924,437 |
| Auxiliary weight reconstruction | Maximum absolute difference 5.7246645202013724e-08 |
| Markdown checks | Local links resolve; fenced blocks balanced |
| Research utility quality | Ruff check and formatting passed |
| Registered command imports | All five commands return success for `--help` |
| README Python inference example | Executed successfully with the real attention/multi-task checkpoint |
| Documented explanation CLI | Produced a valid JSON report for the first CSV example; completeness passed |
| Local evaluator command | Two known toy answers produced expected metrics; this is an operational smoke check, not a benchmark result |
| Git whitespace check | Passed before committing |

Training-command import initially revealed the undeclared `torchsummary` dependency. Installing `torchsummary==1.5.1` and `accelerate` made `flaird-train --help` succeed. This prerequisite is documented, without changing the completed training protocol or asserting that a new full training run was tested.

## Real-checkpoint command smoke check

The explanation CLI loaded `MahmoodAnaam/flaird-modernbert-large-attention-multitask`, resolved revision `ab52d15b7af2a2793c607bd853585d630c8237af`, used float32 CPU, training-style preprocessing and 512-token maximum. The first `demo/examples.csv` row used 296 tokens. Integration increased from 64 to 128 to 256 steps; completeness residual was −0.005576641361455081 within tolerance 0.034792614936828614. This is numerical explanation validation, not a new external evaluation or proof that this revision generated the historical leaderboard scores.

## Environment

Python 3.12.14; PyTorch 2.14.1+cpu; Transformers 5.16.1; pystylometry 1.4.3; pytest 9.1.1; Gradio 6.27.0; Plotly 6.6.0; Ruff 0.17.0; accelerate 1.15.0; torchsummary 1.5.1.

## Rechecking

```bash
python docs/tools/research_tables.py --check
python -m pytest -q
ruff check docs/tools/research_tables.py
ruff format --check docs/tools/research_tables.py
git diff --check
```

The generator reads committed inputs and compares marked tables and [research-evidence.json](research-evidence.json). It preserves source hashes and does not access a network. Benchmarks remain limited by missing historical model/data revisions, trainer logs, predictions and independent source-disjoint audits, as described in the evaluation reports.
