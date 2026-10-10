# RAID leaderboard evaluation

## Saved evidence and scope

The repository contains one saved RAID evaluation, under [attention/multi-task](raid-evaluation/flaird-modernbert-large-attention-multitask/). `metadata.json` identifies detector FLAIRD, release date **2026-09-30** and contact `eng.mahmood.anaam@gmail.com`. The corresponding `results.json` includes aggregate scores, 10,881 overlapping filtered records, per-domain thresholds and achieved false-positive rates.

The folder associates this result with the trainable attention/multi-task experiment. The saved result's model, website, paper and GitHub link fields are null, and no checkpoint revision is recorded. Association to a particular current Hub commit cannot be certified from these artifacts. The other seven variants have no saved RAID leaderboard result in this repository; their PAN scores must not be represented as RAID measurements.

The original files are preserved unchanged. [The evidence ledger](validation/research-evidence.json) records their SHA-256 file hashes alongside the source configuration and notebook hashes. `_submission_hash` and `_results_hash` are leaderboard fields; they are not Git or Hub model revision identifiers.

## Metric semantics

Upstream [RAID evaluation code](https://github.com/liamdugan/raid/blob/main/raid/evaluate.py) separates machine texts from human reference texts. It finds a threshold for each domain using human, unmodified examples to approach the requested FPR. Machine examples are then thresholded using the matching domain threshold with `score >= threshold`.

The field named `accuracy` in a machine slice has only positive ground-truth labels:

$$accuracy_{RAID}=TP/(TP+FN)=TPR=Recall_{machine}.$$

It is **recall at a specified human false-positive operating point**, not conventional overall binary accuracy. The saved artifacts have target FPRs 5% and 1%, achieved in each of the eight domains. They do not provide TN counts in each machine slice, so ordinary binary accuracy cannot be read directly from those records.

AUROC uses unthresholded machine scores and human reference scores, with human references filtered only by the selected domain. For the `all` domain, these are pooled across domains; AUROC is not computed from the domain-thresholded decisions. The inspected current upstream source corroborates the metric interpretation but does not identify the historical evaluator commit that generated this release.

## Aggregate result

In this table `all` includes every attack condition, while `none` is the unmodified-text subset.

<!-- research:raid-summary:start -->
| attack | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| all | 99.7678% | 99.0006% | 0.998726 | 646,276 / 6,524 |
| none | 99.9559% | 99.7610% | 0.999645 | 54,270 / 130 |
<!-- research:raid-summary:end -->

For all conditions, 5% FPR gives TP=651,284 and FN=1,516; 1% FPR gives TP=646,276 and FN=6,524, both over **652,800 machine rows**. The unmodified subset contains 54,400 machine rows. These are row/variant counts, not independent source-document counts. The notebook's test total is 672,000, so subtracting machine rows suggests 19,200 human rows; this is an inference from two artifacts, not a directly saved human-count field.

No claim of leaderboard placement is made: the saved files contain this detector's metrics, not a dated ranking of competitors.

## Domain operating thresholds

<!-- research:thresholds:start -->
| Domain | Threshold @ 5% | Threshold @ 1% | Achieved FPR @ 5% / 1% |
| --- | --- | --- | --- |
| abstracts | 0.000578372 | 0.032225101 | 0.05 / 0.01 |
| books | 0.295424560 | 0.889174560 | 0.05 / 0.01 |
| news | 0.000208983 | 0.127864013 | 0.05 / 0.01 |
| poetry | 0.113907973 | 0.520157973 | 0.05 / 0.01 |
| recipes | 0.000097159 | 0.000394705 | 0.05 / 0.01 |
| reddit | 0.106041914 | 0.777916914 | 0.05 / 0.01 |
| reviews | 0.090868961 | 0.918993961 | 0.05 / 0.01 |
| wiki | 0.006418087 | 0.508371212 | 0.05 / 0.01 |
<!-- research:thresholds:end -->

Thresholds differ substantially by domain. At 1% FPR, the saved books threshold is about 0.8892, reviews about 0.9190 and recipes about 0.000395. A universal 0.5 demo threshold is therefore not the same protocol as the leaderboard's domain-tuned operating points. These thresholds use leaderboard human reference distributions; copying them into an unknown deployment population does not guarantee the same FPR.

## Single-axis breakdowns

Each table changes one axis while setting the other four axes to `all`. Records across these tables overlap. Summing them together, or averaging all 10,881 records without regard to their overlapping filters, would produce misleading totals.

### Domain

<!-- research:raid-domain:start -->
| domain | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| abstracts | 99.9988% | 99.8750% | 0.997637 | 81,498 / 102 |
| books | 99.7708% | 99.1054% | 0.998168 | 80,870 / 730 |
| news | 99.9975% | 99.7096% | 0.999530 | 81,363 / 237 |
| poetry | 99.6176% | 99.1900% | 0.999140 | 80,939 / 661 |
| recipes | 100.0000% | 100.0000% | 0.999908 | 81,600 / 0 |
| reddit | 99.4314% | 98.1397% | 0.998497 | 80,082 / 1,518 |
| reviews | 99.3860% | 96.9902% | 0.997953 | 79,144 / 2,456 |
| wiki | 99.9400% | 98.9951% | 0.999311 | 80,780 / 820 |
<!-- research:raid-domain:end -->

Reviews and Reddit have the lowest domain recall at 1% FPR (approximately 96.99% and 98.14%). Recipes achieves 100% on this saved slice. This does not establish zero failure probability outside the measured rows. Domain-dependent human score distributions and the selected thresholds affect these recall differences.

### Generator

<!-- research:raid-model:start -->
| model | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| llama-chat | 99.9961% | 99.9180% | 0.999793 | 76,737 / 63 |
| mpt | 99.8464% | 99.0443% | 0.998746 | 76,066 / 734 |
| mpt-chat | 99.9753% | 99.7227% | 0.999555 | 76,587 / 213 |
| gpt2 | 99.9349% | 99.3958% | 0.999195 | 76,336 / 464 |
| mistral | 99.6341% | 98.5586% | 0.998096 | 75,693 / 1,107 |
| mistral-chat | 99.9779% | 99.7513% | 0.999563 | 76,609 / 191 |
| gpt3 | 99.9062% | 99.3490% | 0.998815 | 38,150 / 250 |
| cohere | 98.3411% | 94.4245% | 0.993776 | 36,259 / 2,141 |
| chatgpt | 99.9401% | 99.7865% | 0.999665 | 38,318 / 82 |
| gpt4 | 99.9635% | 99.7083% | 0.999572 | 38,288 / 112 |
| cohere-chat | 99.1719% | 96.9609% | 0.996612 | 37,233 / 1,167 |
<!-- research:raid-model:end -->

Cohere is the weakest listed generator slice at 1% FPR (approximately 94.42%), followed by Cohere-chat (96.96%). These are **binary detection** slices for known source generators, not the family classifier's attribution accuracy. Family-head evaluation would require separate family predictions and labels.

### Attack

<!-- research:raid-attack:start -->
| attack | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| none | 99.9559% | 99.7610% | 0.999645 | 54,270 / 130 |
| whitespace | 99.9283% | 99.7537% | 0.999655 | 54,266 / 134 |
| upper_lower | 99.9375% | 99.7426% | 0.999639 | 54,260 / 140 |
| synonym | 99.8548% | 99.5404% | 0.999341 | 54,150 / 250 |
| perplexity_misspelling | 99.9393% | 99.7390% | 0.999610 | 54,258 / 142 |
| paraphrase | 99.2335% | 96.0754% | 0.994284 | 52,265 / 2,135 |
| number | 99.9559% | 99.7537% | 0.999633 | 54,266 / 134 |
| insert_paragraphs | 99.8897% | 99.5790% | 0.999321 | 54,171 / 229 |
| homoglyph | 99.5680% | 98.5276% | 0.998414 | 53,599 / 801 |
| article_deletion | 99.9449% | 99.7721% | 0.999686 | 54,276 / 124 |
| alternative_spelling | 99.9504% | 99.7482% | 0.999630 | 54,263 / 137 |
| zero_width_space | 99.0551% | 96.0147% | 0.995848 | 52,232 / 2,168 |
<!-- research:raid-attack:end -->

The stored result covers twelve conditions: unmodified plus eleven adversarial conditions. Paraphrase and zero-width-space have the lowest 1%-FPR recalls (approximately 96.08% and 96.01%). These shifts identify priorities for controlled stress tests. They do not reveal whether failures arose from tokenization, feature extraction, semantics or the operating threshold without per-example analysis and ablations.

### Decoding and repetition penalty

<!-- research:raid-decoding:start -->
| decoding | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| greedy | 99.8655% | 99.4605% | 0.999262 | 324,639 / 1,761 |
| sampling | 99.6700% | 98.5407% | 0.998189 | 321,637 / 4,763 |
<!-- research:raid-decoding:end -->

<!-- research:raid-repetition_penalty:start -->
| repetition_penalty | Recall @ 5% FPR | Recall @ 1% FPR | AUROC | TP / FN @ 1% |
| --- | --- | --- | --- | --- |
| no | 99.6541% | 98.5845% | 0.998261 | 416,421 / 5,979 |
| yes | 99.9761% | 99.7635% | 0.999578 | 229,855 / 545 |
<!-- research:raid-repetition_penalty:end -->

Sampling has lower 1%-FPR recall than greedy decoding (98.54% versus 99.46%). Repetition-penalty-enabled rows have higher recall than disabled rows (99.76% versus 98.58%). These subsets can differ in generator composition and other settings; the aggregates are not controlled causal effects of decoding or repetition penalty.

## Submission pipeline and processing policy

`flaird-raid-test` loads a trained checkpoint with remote code, reads a selected dataset configuration, computes continuous scores, writes resumable JSONL shards and merges them into a final JSON array. Defaults are `liamdugan/raid`, `raid_test`, maximum length 512, batch 16, no identifier normalization, and five shards. A zero-based `--shard-index` is required. A separate prepared configuration can supply normalized text and stored forensic vectors, as illustrated in [Reproducibility](REPRODUCIBILITY.md).

Resume is file-based: an existing shard path may be reused without verifying its checkpoint revision, dataset revision or preprocessing manifest. The current merger does not establish complete unique-ID coverage by itself. For a future submission, use a fresh output directory per pinned model/data/policy, verify unique IDs and expected row count, and preserve a manifest before uploading. The CLI's historical default contact differs from the saved metadata; commands explicitly override it with the user's requested address.

The preparation notebook includes normalization/feature computation for leaderboard test text, whereas raw-default CLI inference does not normalize. Saved result metadata does not fully identify which path produced the release. This uncertainty prevents a claim of exact benchmark reproduction merely by running today's defaults.

## Comparison with PAN26

RAID all-condition AUROC is about 0.998726, while the corresponding attention/multi-task PAN26 AUROC is 0.932. Their operating-point and aggregate objectives differ: RAID recall is evaluated under domain-tuned FPR, whereas PAN's Mean combines ranking, probability error and fixed-threshold metrics. Comparing 99% RAID recall numerically to 83.6% PAN Mean as though they were the same accuracy is invalid.

RAID training material dominates the prepared training set. Row-level source splitting, possible related-variant overlap and missing immutable dataset revisions constrain claims of out-of-distribution generalization. A high saved leaderboard result is useful evidence under that benchmark protocol; independence from all training source material is not established by the repository's summaries.

## Future evaluation

Retain benchmark revisions, exact predictions and preprocessing manifests; audit source/prompt overlaps; evaluate group-disjoint splits and unseen generators; calibrate on separate validation human texts; report human false positives and uncertainty at domain and attack level. Apply the same RAID protocol to the remaining seven checkpoints before claiming that the best PAN variant also leads on RAID. No additional submission or new benchmark run was performed for this documentation phase.

## References

- Dugan et al., [RAID paper](https://arxiv.org/abs/2405.07940), 2024.
- [RAID repository and leaderboard instructions](https://github.com/liamdugan/raid).
- [Saved result](raid-evaluation/flaird-modernbert-large-attention-multitask/results.json), [saved metadata](raid-evaluation/flaird-modernbert-large-attention-multitask/metadata.json).
