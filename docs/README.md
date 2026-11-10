# FLAIRD technical and research documentation

This documentation describes the implemented release and archived experiments. It distinguishes observed results, implementation properties and research hypotheses.

| Reading order | Document | Main question |
|---:|---|---|
| 1 | [Thesis proposal](THESIS_PROPOSAL_ACADEMIC.md) | What research problem is operationalized and what is actually supported? |
| 2 | [System architecture](ARCHITECTURE.md) | How do semantic and forensic representations interact? |
| 3 | [Feature specification](FORENSIC_FEATURES.md) | What does each of the 35 ordered measurements mean? |
| 4 | [Data preparation](DATASET_PREPARATION.md) | Where do the data and labels come from? |
| 5 | [Notebook audit](NOTEBOOK_EXECUTION_AUDIT.md) | What output evidence is retained for each cell? |
| 6 | [Training configuration](TRAINING_CONFIGURATION.md) | Which objective and settings define the eight runs? |
| 7 | [Historical training summary](TRAINING_HISTORY.md) | What limited learning-curve evidence was preserved from the prior card? |
| 8 | [Evaluation and discussion](EVALUATION_RESULTS.md) | Which architectural comparisons are supported by archived results? |
| 9 | [Explanation specification](EXPLAINABILITY_MODULE.md) | What are Shapley/Owen values and what exactly do they explain? |
| 10 | [Dashboard and deployment](DEMO.md) | How are evidence and deployment presented? |
| 11 | [Release review](RELEASE_REVIEW.md) | What changed and what was verified? |

## Markdown rendering conventions

Documents use GitHub-compatible `$...$` inline mathematics and `$$` display blocks. Mermaid diagrams use fenced `mermaid` blocks with short quoted labels and top-down flow. Tables use standard Markdown, and the four EDA figures are repository-relative PNG assets extracted from the notebook.

GitHub renders these formats natively. A plain-text editor or Markdown viewer without math/Mermaid support will show source syntax; enable those capabilities when exporting to thesis authoring tools. The release review records the renderers used to check mathematical syntax and diagram rendering locally. That check does not imply every third-party Markdown application has identical rendering support.
