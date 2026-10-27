# Release review: interface and academic documentation

Review date: 2026-10-26. Base branch: `develop`. Reviewed baseline commit: `76ade95cfb71f0ab4959a6021d04096523918a36`.

## 1. Scope and changes

| Area | Files | Result |
|---|---|---|
| Progress display | `demo/app.py`, `demo/ui/controller.py` | Remove injected progress object/callback and explicitly hide progress for Analyze, Example and Clear |
| Visual system | `demo/ui/theme.py`, `demo/ui/style.css`, `demo/ui/components.py`, `demo/ui/plots.py` | New masthead/hero, readable KPI tiles, neutral panels, consistent spacing, responsive cards and navigation; horizontal evidence charts and hidden-tab resizing |
| Interpretation labels | `demo/ui/components.py`, `demo/ui/controller.py`, `demo/app.py` | Clarify zero-feature reference, gate and magnitude summaries; sampled versus exact feature explanation |
| Feature descriptions | `src/flaird/utils/explain.py` | Correct punctuation-density denominator and uppercase/whitespace descriptions; no change to feature values |
| Explanation theory | `docs/EXPLAINABILITY_MODULE.md` | Formal masking game, Shapley/Owen derivations, examples, exact/sampled distinction and limits |
| Feature specification | `docs/FORENSIC_FEATURES.md` | All 35 features, version-specific denominators/chunking, examples and failure semantics |
| Architecture | `docs/ARCHITECTURE.md` | Dimensions, modules, rationales, FeatureEncoder/attention relationship and comparator behavior |
| Data and notebooks | `docs/DATASET_PREPARATION.md`, `docs/NOTEBOOK_EXECUTION_AUDIT.md`, `docs/assets/` | Recorded cell evidence, counts, four original EDA figures and an initial dashboard preview, split/provenance limitations |
| Training | `docs/TRAINING_CONFIGURATION.md`, `docs/TRAINING_HISTORY.md` | All eight configurations, weighted losses, version-specific warmup and preserved historical card table |
| Evaluation | `docs/EVALUATION_RESULTS.md` | All 32 TIRA records, fusion differences, RAID attack slices and qualified discussion |
| Academic proposal | `docs/THESIS_PROPOSAL_ACADEMIC.md` | Complete questions, hypotheses, methodology, observations, validity and references |
| Project entry points | Root `README.md`, `docs/README.md`, `demo/README.md`, `docs/DEMO.md` | Concise project navigation, model/Space badges and publication instructions |
| Model card | `src/flaird/modeling/README.md` | Checkpoint-specific metadata, intended use, data, settings, actual metrics, usage and limitations |
| Regression verification | `tests/test_release.py`, `tests/test_docs.py` | Offline numerical/UI/provenance checks and Markdown structure/link checks |

The extractor calculations, model architecture, experiment JSONs, notebook contents, archived metric JSONs and deployment workflow are preserved. The work does not retrain models, change dataset partitions, regenerate benchmark results, merge the PR or dispatch deployment.

## 2. Research corrections and interpretation decisions

- Removed references to a missing verification script/CLI and unsupported historical test-count and latency claims.
- Distinguished zero raw forensic input from a literal absence of the learned forensic branch; learned biases can produce nonzero feature tokens.
- Distinguished exact group consistency from finite-sample Owen estimation, whose within-group sums can deviate from exact group values.
- Corrected punctuation density to punctuation per 100 words and uppercase ratio to the library's counted-letter denominator; documented library chunk summaries.
- Documented that feature extraction failure shares one library `try` block, while direct features can remain nonzero.
- Replaced unsupported “leakage-safe/balanced” characterizations with the actual row-stratified split and loss-weighted training.
- Explained that TIRA `Brier` is a complement, RAID `accuracy` is machine TPR at a target FPR, and UI inconclusive bands are separate from evaluator abstention.
- Preserved the historical training-card table with provenance instead of treating the output-free training notebook as a full experiment log.
- Discussed attention's additional optimization burden as a plausible hypothesis, alongside counterevidence and alternative explanations. Concatenation also has newly trained layers.
- Separated text-origin detection from factuality, intent, exact authorship and campaign coordination.

## 3. Verification record

| Check | Result / scope |
|---|---|
| Offline regression suite | 16 tests passed: eight model/fusion/pooling/head combinations, independent toy-game Owen enumeration, feature definitions, UI events, archived result tables and Markdown checks |
| Numerical explanation accounting | Tiny random two-layer ModernBERT; both fusions, both poolings, with/without generator; one encoder call and 78,592 masks; forward error < 1e-5 and exact allocation errors < 1e-6 |
| UI construction and event configuration | Gradio 6.27.0; 73 components; all three events hide progress and have no progress tracker injection |
| Browser visual/interaction review | Chromium 153 via Playwright; desktop 1440 px and phone 390 px; all four tabs exercised, including mobile overflow navigation |
| Busy state | Zero visible application progress bars during an Analyze request |
| Responsive figures | Hidden-tab/viewport transitions checked; all visible Plotly SVG widths match their 328 px phone containers; document width stays 390 px |
| Markdown rendering | 15 Markdown documents rendered with Markdown-it, KaTeX 0.19.1 and Mermaid 12.2.0; 129 mathematical expressions and four diagrams; no math errors or missing repository images |
| Evidence provenance | All 32 archived TIRA records and 13 fully aggregated RAID attack/overall rows checked against displayed tables; four notebook EDA figures preserved |
| Source checks | Python compilation and `git diff --check` passed; local documentation links/fences checked |

Browser result screens use explicitly labeled synthetic weights and fixed features, not predictions from the released checkpoints. This permits UI/shape verification without reinterpreting benchmark results. The relevant QA stack used PyTorch 2.8.0+cpu, Transformers 5.16.1, Gradio 6.27.0, Plotly 7.1.0 and pystylometry 1.4.3; the environment is not a complete recreation of the historical GPU training runtime. One third-party `pronouncing`/`pkg_resources` deprecation warning was observed.

Reproduce the offline suite after installing the package, `pytest` and `spaces`:

```bash
GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 PYTHONPATH=src:demo python -m pytest tests -q
python -m compileall -q src demo tests
git diff --check
```

## 4. Remaining evidence and operational boundaries

This release checks local UI behavior and numerical accounting using tiny random models. It does not constitute a new evaluation of released ModernBERT-large checkpoint quality or a GPU/ZeroGPU end-to-end deployment run. Detection results remain the archived project artifacts. No new real-checkpoint explanation latency, calibration or human-faithfulness measurements are claimed.

No multi-seed uncertainty, matched longer-training experiment, source-overlap audit or per-family evaluation is supplied by the reviewed artifacts. Notebook publication calls lack saved success confirmations. These gaps are explicitly documented; they are not silently filled with assumptions.

The deployed Space is unchanged until the PR is merged and the existing manual synchronization workflow is dispatched with the intended target/ref/privacy. `demo/requirements.txt` imports the mutable `develop` package; retain the resolved commit/version for a publication record. The PR is the reviewable delivery of these changes, not an assertion that external hosting has already updated.
