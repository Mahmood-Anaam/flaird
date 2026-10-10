# FLAIRD technical and research documentation

The implementation and eight training/evaluation configurations are complete. This documentation describes their saved evidence, the implemented explanation module and Gradio interface. It separates the origin-detection model from broader future disinformation-analysis objectives.

| Document | Contents |
| --- | --- |
| [Dataset preparation](DATASET_PREPARATION.md) | Notebook reconstruction, provenance, split sizes, generator aliases, schema, normalization, imbalance and leakage boundaries |
| [Model architecture](MODEL_ARCHITECTURE.md) | Tensor contract, 35 ordered features, seven groups, semantic encoder, both fusion variants and output interpretation |
| [Training experiments](TRAINING_EXPERIMENTS.md) | Eight-run design, exact common settings, loss equations, weights, configuration differences and missing execution evidence |
| [PAN26 evaluation](PAN26_EVALUATION.md) | All eight challenge results, cross-dataset scores, TIRA run IDs, metric equations and comparative analysis |
| [RAID evaluation](RAID_EVALUATION.md) | One saved leaderboard release, recall/FPR semantics, aggregates, domains, attacks, generators and thresholds |
| [Explanation module](EXPLANATION_MODULE.md) | Integrated gradients, group interventions, diagnostics, limitations and validation |
| [Gradio interface](GRADIO_INTERFACE.md) | UI behavior, examples, plots, model selection and deployment configuration |
| [Reproducibility](REPRODUCIBILITY.md) | Installation, trained-model loading, commands, policy matrix and artifact manifests |
| [Academic thesis proposal](THESIS_PROPOSAL_ACADEMIC.md) | Research problem, objectives, questions, hypotheses, methods, findings and future work |
| [Documentation validation](validation/PHASE_THREE_VALIDATION.md) | Checks of tables, links, command imports and real-checkpoint examples |
| [Phase-one review](PHASE_ONE_REVIEW.md) | Historical repository analysis before the interface and documentation phases |

## Reading order and evidence

Start with architecture, dataset preparation and training; then read the two evaluation reports. PAN Mean and RAID recall are different metrics. The headline PAN values are 0.836 and 0.894; the saved RAID record belongs only to attention/multi-task/trainable. Neither benchmark establishes factuality or campaign coordination.

The original TIRA/RAID JSON files and notebooks remain unchanged. [Research evidence](validation/research-evidence.json) contains generated summaries and hashes of their source bytes. [Table generation/checking](tools/research_tables.py) prevents transcription drift in marked documentation tables. Missing training logs and benchmark checkpoint revisions are described explicitly in the reports.

The professional Gradio application is prepared in `demo`; a live Space URL is not available in these artifacts. Explanation and UI implementation documents retain their own phase-specific validation evidence.
