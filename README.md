# FLAIRD

**Forensic Linguistic and AI-Integrated Robust Detector** — an English AI-generated text detector combining ModernBERT, 35 forensic-linguistic features and hierarchical explanations.

[![Hugging Face models](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Model_collection-FFD21E?style=flat)](https://huggingface.co/collections/MahmoodAnaam/flaird)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Live_demo-FFD21E?style=flat)](https://huggingface.co/spaces/MahmoodAnaam/flaird)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Model and explanation

- **Hybrid representation:** ModernBERT-large semantics plus seven groups of stylistic measurements.
- **Two fusion strategies:** gated cross-attention and a learned concatenation comparator.
- **Eight completed experiments:** attention/concatenation × single/multitask × frozen/fine-tuned encoder.
- **Inspectable predictions:** input diagnostics, binary score, optional generator-family distribution, group Shapley and feature Owen contributions.

FLAIRD estimates text origin. It does not verify factuality, deceptive intent or campaign coordination. Scores are affected by training class weights and are not calibrated authorship probabilities.

## Quick start

Python 3.10+; use a compatible GPU runtime for the committed bf16 training configurations.

```bash
git clone -b develop https://github.com/Mahmood-Anaam/flaird.git
cd flaird
python -m pip install -e .
flaird-predict --text "A sufficiently long English document to analyze ..." --report
```

```bash
# Exact structured attribution (default)
flaird-predict --input document.txt --output explanation.json --report

# Faster feature-attribution estimate
flaird-predict --input document.txt --owen-method permutation --owen-permutations 512
```

The default checkpoint is `MahmoodAnaam/flaird-modernbert-large-attention-multitask`. Exact explanation evaluates 78,592 lightweight feature/fusion masks after one encoder pass; CPU latency depends on hardware. Model loading uses `trust_remote_code=True`.

## Dashboard

```bash
python -m pip install spaces
python demo/app.py
```

The dashboard compares the released attention and concatenation multitask checkpoints. It presents the decision, baseline-relative path decomposition, ranked features and attention diagnostics. The semantic encoder reads at most 512 tokens; forensic extraction uses the supplied text, capped at 30,000 characters in the demo. See [deployment and interface guide](docs/DEMO.md).

## Results

Archived TIRA composite Mean for the fine-tuned multitask variants:

| Fusion | PAN25 | PAN26 | Eloquent |
|---|---:|---:|---:|
| Attention | 0.958 | 0.836 | 0.776 |
| Concatenation | 0.951 | 0.894 | 0.774 |

Performance depends on the benchmark. [Complete results and discussion](docs/EVALUATION_RESULTS.md) cover all eight runs and the available RAID artifact, including metric definitions and limitations. Internal validation is row-stratified; source-level overlap has not been audited.

## Documentation

| Guide | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Components, dimensions, FeatureEncoder, fusion, task heads |
| [Forensic features](docs/FORENSIC_FEATURES.md) | All 35 definitions, examples and numerical conventions |
| [Dataset preparation](docs/DATASET_PREPARATION.md) | Sources, counts, labels, EDA and inference alignment |
| [Notebook audit](docs/NOTEBOOK_EXECUTION_AUDIT.md) | Every cell and its retained output evidence |
| [Training](docs/TRAINING_CONFIGURATION.md) | Eight configurations, weighted objectives and reproduction |
| [Evaluation](docs/EVALUATION_RESULTS.md) | Full benchmark tables and critical interpretation |
| [Explainability](docs/EXPLAINABILITY_MODULE.md) | Shapley/Owen theory, examples, API and limitations |
| [Thesis proposal](docs/THESIS_PROPOSAL_ACADEMIC.md) | Academic motivation, questions, methodology and findings |
| [Model card](src/flaird/modeling/README.md) | Attention-multitask checkpoint specification |
| [Release review](docs/RELEASE_REVIEW.md) | Changes, verification and remaining evidence gaps |

Training starts with `flaird-train configs/EXPERIMENT.json`; see the [training guide](docs/TRAINING_CONFIGURATION.md) for Colab paths, dataset access and the additional `torchsummary` import. Package entry points also support PAN evaluation, TIRA inference and sharded RAID submissions. The Space sync workflow is manually dispatched; merging code does not automatically deploy it.

## License

Software: [Apache License 2.0](LICENSE). Dataset access and pretrained-model terms are governed by their respective sources.
