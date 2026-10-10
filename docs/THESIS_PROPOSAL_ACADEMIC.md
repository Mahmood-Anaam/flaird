# Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns

## Research proposal and implementation status

**System:** FLAIRD — Forensic Linguistic and AI-Integrated Robust Detector.

**Author:** Mahmood Anaam.

**Scope of this revision:** implemented English machine-origin detection, completed eight-configuration training/evaluation record, explanation module, Gradio interface and research documentation.

The thesis title motivates a broader application to disinformation analysis. The implemented model identifies textual origin and, optionally, a closed-set generator family. It does **not** determine whether a statement is false, whether an author intended deception, or whether multiple accounts form a coordinated campaign. Origin evidence can support investigation, but campaign-level conclusions require provenance, temporal/network evidence and factual verification. This distinction governs the research questions and claims below.

## 1. Abstract

Generative language models make fluent synthetic text inexpensive to produce. Origin detection can assist forensic investigation, but detector performance may change across genres, generators, adversarial edits and decision thresholds. This research develops FLAIRD, a hybrid classifier integrating ModernBERT-large contextual representations with 35 deterministic lexical, character, structural and repetition measurements. Seven learned feature-group tokens are fused with semantic evidence through either gated cross-attention or concatenation. A binary origin objective is optionally accompanied by eleven-class generator-family supervision.

Eight completed configurations cross fusion, auxiliary supervision and encoder freezing. The prepared corpus combines PAN26 source splits with a label-stratified RAID row split, producing 5,077,945 training and 565,171 validation examples. Saved PAN26 TIRA results give composite Mean 0.836 for trainable attention/multi-task and 0.894 for trainable concatenation/multi-task. One saved RAID leaderboard release associated with attention/multi-task reports all-condition machine recall 99.7678% at 5% domain-tuned FPR and 99.0006% at 1% FPR, with AUROC approximately 0.998726. These are distinct evaluation protocols, not interchangeable accuracy estimates.

The implemented explanation module reports forensic integrated gradients, group baseline interventions, attention/gate diagnostics and numerical completeness checks. A Gradio interface exposes predictions, uncertainty-relevant controls, examples and visual explanations. Remaining research concerns include group-disjoint data audits, repeat-seed uncertainty, calibration, unseen-generator testing and campaign-level integration. Missing immutable benchmark checkpoint revisions and full trainer logs limit exact historical reproduction.

## 2. Background and research problem

Text generation quality reduces the reliability of simple intuitions about fluency, vocabulary or punctuation. A contextual encoder can learn rich patterns but may exploit source shortcuts; forensic scalar features are legible to investigators yet can be length-sensitive, correlated or modified by editing. A hybrid representation offers a testable way to combine the two evidence types.

RAID evaluates detector variation across domains, generators, decoding settings, repetition penalties and adversarial modifications. PAN26 offers a separate challenge context. Strong performance on one source does not establish universal robustness. Operational false accusations against human writers also make calibration and human false-positive rates central to evaluation.

The concrete problem is to produce a reproducible origin classifier whose architecture, inputs, losses, benchmark metrics and explanations can be inspected. The broader forensic objective is decision support: an analyst should understand which observable signals influence a score and which claims the system cannot substantiate.

## 3. Aim, objectives and contributions

The aim is to investigate whether combining contextual semantics and forensic measurements improves machine-origin detection and enables useful, bounded explanations.

| Objective | Implemented method | Evidence and remaining work |
| --- | --- | --- |
| Construct a hybrid detector | ModernBERT-large plus seven groups of 35 measurements | Model code and trained checkpoint validation |
| Compare fusion strategies | Attention and concatenation | Eight configurations and TIRA results; no encoder-only baseline yet |
| Investigate auxiliary family supervision | Binary objective plus weighted eleven-class objective | Multi-/single-task comparisons; family attribution metrics remain unavailable |
| Examine backbone adaptation | Frozen/trainable encoders | Completed variants; batch-size confounding and single-seed limitation |
| Support forensic explanation | Feature integrated gradients, group interventions, routing diagnostics | Completeness and real-checkpoint validation; user-study evidence remains future work |
| Assess robustness | Saved PAN/TIRA and one RAID release | Protocol-specific results; unseen generators and source-disjoint audits remain necessary |
| Provide accessible operation | Gradio examples, charts and JSON export | Space-ready demo; live deployment is a separate action |
| Document reproducibility | Notebook reconstruction, exact configs, artifact hashes and commands | Historical environment/checkpoint gaps identified explicitly |

Contributions are the implemented hybrid architecture, organized forensic feature interface, factorial experiment record, bounded explanation toolkit and documented operational system. This proposal does not claim priority over all hybrid detectors, a causal explanation of authorship, or completion of network-level disinformation detection.

## 4. Research questions and hypotheses

**RQ1:** How do attention and concatenation compare on held-out origin-detection benchmarks?

**H1:** Semantic-conditioned feature attention may improve some distributions, but its added flexibility need not dominate simpler concatenation. The completed PAN26 result favors concatenation/multi-task/trainable; attention/multi-task is slightly stronger on saved PAN25/Eloquent Mean. Thus a universal attention-superiority hypothesis is not supported.

**RQ2:** Does auxiliary generator supervision improve the binary objective?

**H2:** Family supervision can regularize the shared representation, especially with encoder adaptation. Completed trainable comparisons show Mean gains of 0.095 and 0.134; frozen attention changes by −0.001. Results support a conditional benefit, pending repeats and attribution metrics.

**RQ3:** Is adapting the encoder preferable to freezing it?

**H3:** Adaptation can improve task alignment but may interact with objective and optimization. Saved comparisons are mixed for single-task attention and favorable for both multi-task variants. Changed per-device batch size prevents attributing all differences solely to freezing.

**RQ4:** Can explanations faithfully summarize the implemented model's forensic dependence?

**H4:** Integrated gradients and controlled feature interventions can characterize local model behavior when numerical completeness and intervention limits are exposed. Existing checks establish numerical consistency, not human interpretability, causal authorship evidence or robustness of every explanation under perturbation.

**RQ5:** How stable are performance and explanations under domain and adversarial changes?

**H5:** Domain thresholds, editing and generator composition affect operational performance. Saved RAID slices identify weaker paraphrase/zero-width-space and Cohere conditions; cross-benchmark differences motivate further tests. Independence from training source variants is not established.

## 5. System architecture

The system has four implemented layers: prepared dataset and feature extraction; hybrid origin/family modeling; explanation; and a Gradio analyst interface. [Model architecture](MODEL_ARCHITECTURE.md) gives the complete tensor graph and ordered feature table.

For text t, tokenizer input produces contextual states $H\in\mathbb R^{L\times1024}$, and deterministic extraction produces $x\in\mathbb R^{35}$. Features use signed-log compression and seven group-specific projections to states $F\in\mathbb R^{7\times128}$. Masked mean pooling gives semantic state s.

Attention fusion uses eight heads with s as a single query and F as keys/values, then constructs gated mixed and multiplicative states. Concatenation fusion joins s with the mean feature state. Both emit a 1,024-dimensional shared representation and one binary logit; multi-task variants additionally emit eleven family logits.

$$p_{machine}=\sigma(\ell),\qquad p_{human}=1-p_{machine},\qquad q=softmax(v).$$

The semantic branch sees at most 512 tokens in the completed experiments. The forensic branch measures whole text under the selected preprocessing policy. Input features are surface/stylometric proxies; syntactic parsing, discourse coherence and token saliency are not implemented capabilities.

## 6. Dataset methodology

[Dataset preparation](DATASET_PREPARATION.md) reconstructs notebook operations and saved outputs. PAN contributes 23,707 training and 3,589 validation rows. RAID labeled development has 5,615,820 rows and is split by binary label at 90%/10%, seed 42. Combining sources yields 5,077,945 / 565,171 rows. The 672,000-row unlabeled RAID leaderboard set remains separate.

Preparation replaces email, mention and broad phone-pattern matches, maps generators to ten machine families plus human, and computes 35 features on normalized text. Stored fields retain origin and source labels. Training has 4,924,437 machine examples and 153,508 human examples; RAID represents about 99.53% of its rows.

The split occurs at row level after source/variant IDs are discarded. There is no demonstrated exact/near-duplicate or prompt-group audit, so related source variants may cross training/validation. The work must not claim guaranteed group-disjoint generalization. Future releases should preserve source IDs, audit overlaps and publish immutable split manifests before further training.

## 7. Training methodology and loss design

Eight experiments implement $2\times2\times2$ combinations of fusion, auxiliary task and freezing. They request one epoch, seed 42, 512 tokens, learning rate $2\times10^{-5}$, weight decay 0.01, cosine scheduling, fused AdamW, BF16 and norm clipping 1. Trainable per-device batch is 128, frozen batch 256. [Training experiments](TRAINING_EXPERIMENTS.md) lists every common setting and explicit configuration difference.

The binary loss is weighted BCE:

$$\mathcal L_b=-\frac1B\sum_n\left[w_+y_n\log p_n+(1-y_n)\log(1-p_n)\right],\qquad w_+=153508/4924437.$$

The auxiliary objective uses weighted cross-entropy with normalized inverse-square-root family weights. Multi-task loss is $\mathcal L_b+0.2\mathcal L_g$. This aims to reduce domination by common generator families while sharing evidence for binary detection. It does not establish open-set generator attribution.

The literal config field `warmup_steps=0.06` is not a documented `warmup_ratio=0.06`. Seven configs explicitly reload the best loss checkpoint; attention/single-task/frozen omits that setting. Historical trainer states are needed to determine executed warmup and checkpoint selection. Configuration intent and saved benchmark evidence are kept distinct.

## 8. Explanation methodology

For a fixed semantic state and transformed forensic vector z, integrated gradients against baseline $z_0$ estimate

$$IG_i=(z_i-z_{0,i})\int_0^1\frac{\partial\ell(z_0+\alpha(z-z_0);s)}{\partial z_i}\,d\alpha.$$

Completeness checks compare $\sum_i IG_i$ with $\ell(z;s)-\ell(z_0;s)$. The implementation uses numerical integration and adaptive resolution with reported tolerances. Positive values raise the machine logit relative to the numerical baseline; negative values lower it. They are not probabilities or direct factual evidence of authorship.

Group interventions replace a selected group's coordinates with baseline values and report output changes. They can be off-manifold and do not simulate a naturally edited sentence. Attention/gate diagnostics expose fusion routing but are not causal attribution. Concatenation variants have no such routing diagnostics and still support forensic attribution/interventions.

The explanation module has been tested offline and against all eight real checkpoints across the 20 demo examples. Existing evidence reports direct-forward agreement and completeness. Research on stability, expert usefulness and baseline sensitivity remains future work. [The explanation document](EXPLANATION_MODULE.md) contains exact APIs, formulas and validation artifacts.

## 9. Evaluation methodology and completed findings

PAN reports ROC-AUC, Brier complement, c@1, F1 and F0.5u, with Mean averaging those five components. The headline 0.836 / 0.894 scores are this composite. RAID reports machine recall at human FPR operating points with domain-specific thresholds, plus unthresholded AUROC. They should be discussed separately.

<!-- research:pan:start -->
| Variant | Roc-Auc | Brier | C@1 | F1 | F05U | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Attention / multi-task / frozen | 0.835 | 0.740 | 0.651 | 0.713 | 0.828 | 0.753 |
| Attention / multi-task / trainable | 0.932 | 0.792 | 0.754 | 0.809 | 0.895 | 0.836 |
| Attention / single-task / frozen | 0.823 | 0.731 | 0.658 | 0.715 | 0.841 | 0.754 |
| Attention / single-task / trainable | 0.815 | 0.697 | 0.650 | 0.713 | 0.827 | 0.741 |
| Concatenation / multi-task / frozen | 0.774 | 0.744 | 0.655 | 0.719 | 0.829 | 0.744 |
| Concatenation / multi-task / trainable | 0.959 | 0.868 | 0.833 | 0.878 | 0.933 | 0.894 |
| Concatenation / single-task / frozen | 0.773 | 0.704 | 0.625 | 0.681 | 0.814 | 0.719 |
| Concatenation / single-task / trainable | 0.830 | 0.722 | 0.671 | 0.729 | 0.848 | 0.760 |
<!-- research:pan:end -->

The one saved RAID release gives all-condition recall 99.7678% at 5% FPR and 99.0006% at 1% FPR, AUROC approximately 0.998726. Unmodified recall is 99.9559% / 99.7610%. Paraphrase and zero-width-space yield approximately 96.08% / 96.01% recall at 1% FPR. These are descriptive benchmark slices; their overlapping variants are not independent replicates.

[PAN26 evaluation](PAN26_EVALUATION.md) and [RAID evaluation](RAID_EVALUATION.md) include source IDs, full tables and limitations. No confidence intervals, seed replicates, family-head confusion matrices or benchmark checkpoint revisions are supplied. The findings establish performance under the recorded protocols, not universal detection accuracy or coordinated-disinformation recognition.

## 10. Interface, deployment and forensic use

The Gradio system provides text entry, twenty selectable examples, all eight checkpoints, threshold and preprocessing controls, origin/family views, integrated-gradient and group-intervention plots, attention diagnostics where applicable and downloadable JSON. It clears obsolete results when input changes and handles absent auxiliary/routing outputs explicitly.

The default attention/multi-task model exposes the richest implemented diagnostics and has the saved RAID result; users can select concatenation/multi-task for the strongest saved PAN26 Mean. A Space-ready directory is present. Live deployment, hardware selection and operational availability remain a distinct deployment phase.

Scores and explanations should support review of provenance hypotheses. A human writer's unusual style, translation, editing or language background can affect predictions. No automated punitive decision or truthfulness conclusion follows from a single model score. System documentation explains these model limits as part of the output's meaning rather than presenting origin detection as complete campaign analysis.

## 11. Validity, reproducibility and limitations

Internal validity is constrained by source dominance, class imbalance, row-level variant splitting, changed batch sizes and a single seed. External validity is constrained by English-focused features, closed-set family labels, historical dataset snapshots and unverified source overlap. Construct validity requires separating origin, factuality, intent and coordination. Explanation completeness supports local numerical accounting but not causal interpretation.

Historical artifact validity is constrained by missing execution logs and model/data commit IDs for benchmark submissions. The repository now records hashes of supplied source files and generates results tables from them; this preserves current evidence without retroactively filling unknowns. Training/TIRA/raw processing differences are documented in [Reproducibility](REPRODUCIBILITY.md).

## 12. Future work and proposed evaluation plan

1. Preserve source/prompt/variant provenance and publish exact/near-duplicate audits with group-disjoint splits.
2. Repeat seed-controlled experiments and match effective batches; bootstrap at the source-document level to account for correlated variants.
3. Add encoder-only, forensic-only and group-removal baselines to measure contributions beyond the current factorial design.
4. Evaluate generator attribution with confusion matrices, macro-F1 and held-out/open-set generator protocols.
5. Calibrate only on held-out validation distributions and report human FPR, precision and recall at deployment-relevant thresholds.
6. Stress-test preprocessing, Unicode edits, paraphrase, short text and long-text truncation; evaluate explanation sensitivity to baselines and perturbations.
7. Extend features/language coverage through separately trained, validated multilingual versions rather than assuming English lists transfer unchanged.
8. Conduct expert usability studies measuring whether explanations improve analyst decisions and calibration of confidence.
9. Integrate factual verification, source provenance and temporal/network analysis for a separately scoped campaign-analysis system, with independent labels for each construct.
10. Publish a reproducible release bundle with code/model/data revisions, environment lock, evaluator version and permitted prediction artifacts.

## 13. Conclusion

FLAIRD provides an implemented hybrid origin detector with eight completed configuration comparisons, bounded forensic explanations and a usable interface. Saved results favor trainable multi-task concatenation on PAN26, while attention offers routing diagnostics and the recorded RAID release. The next research contribution depends on stronger provenance, controlled uncertainty analysis and broader evaluation, and campaign-level claims require additional independently evaluated system components.

## References

- Dugan et al. (2024), [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://arxiv.org/abs/2405.07940).
- Warner et al. (2024), [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663).
- Sundararajan, Taly and Yan (2017), [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html).
- Peñas and Rodrigo (2011), [A Simple Measure to Assess Nonresponse](https://aclanthology.org/P11-1142/).
- [PAN26 official task description](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html).
- [pystylometry implementation](https://github.com/craigtrim/pystylometry).
- [Committed configurations](../configs/), [TIRA artifacts](tira-evaluation/) and [RAID artifacts](raid-evaluation/).
