# FLAIRD Explanation Module

Implementation and theory of `src/flaird/utils/explain.py` (library), `src/flaird/scripts/predict.py` (CLI `flaird-predict`) and `src/flaird/scripts/verify_explain.py` (verification, `flaird-verify-explain`).

The module explains *why* FLAIRD produced a prediction, in a form a forensic reader can inspect. It is a decision-support tool: it shows evidence and its limits, it does not replace human judgement.

---

## 1. What it explains

FLAIRD fuses two streams: a ModernBERT **semantic** state and 35 **forensic-linguistic** features organized in 7 groups (`FeatureEncoder` → 7 group tokens). The explanation mirrors that hierarchy:

| Layer | Question | Method |
|-------|----------|--------|
| 0. Input reliability | Can this text be analyzed reliably? | rule-based diagnostics |
| 1. Decision | What did the model decide? | machine score, verdict band, generator distribution, head consistency |
| 2. Path balance | How much came from meaning vs. style? | `v(∅)` decomposition, fusion gate reading |
| 3. Feature groups | Which style dimensions pushed the decision? | exact Shapley values over 7 groups |
| 4. Features | Which individual indicators? | exact Owen values over 35 features |
| Cross-checks | Is the explanation internally consistent? | additivity errors, attention vs. attribution agreement |

Everything is computed from the input text and the checkpoint. **No dataset statistics, reference files or caches are read.** There is one entry point and no analysis modes.

```
text ─► preprocess ─► tokenizer ─► ModernBERT (ONE forward pass) ─► pooled semantic vector H
          │                                                              │
          └─► 35 forensic features x ───────────────┐                    │
                                                    ▼                    ▼
                       value function v(mask) = head(fusion(H, FeatureEncoder(x ⊙ mask + b ⊙ ¬mask)))
                                                    │
                       ┌────────────────────────────┼──────────────────────────┐
                       ▼                            ▼                          ▼
              group Shapley (128 masks)    feature Owen (78,592 masks)   v(1), v(0) → path balance
```

## 2. Method

### 2.1 The value function

Let `H` be the pooled semantic vector (`pool_semantic_states`, computed once), `x ∈ R^35` the raw forensic features, `m ∈ {0,1}^35` a mask and `b` the baseline value (default `0`). Define

```
v_binary(m) = classifier( head( fusion( H, FeatureEncoder( m·x + (1−m)·b ) ) ) )              # machine logit
v_gen(m; c) = z_c − logsumexp(z_{j≠c}),   z = generator_classifier( head( fusion( ... ) ) )   # class log-odds
```

* `sigmoid(v_binary(1))` is the machine score (`id2label = {0: "machine"}`).
* Both read-outs share one head evaluation per mask, so explaining the generator head costs almost nothing extra.
* **Pooling trick.** Both fusion layers only use the *pooled* semantic vector. `H` is therefore passed as a `[B,1,hidden]` tensor with no attention mask; for `cls` and `mean` pooling the fusion layer then sees exactly `H`. The encoder is never re-run during attribution, and no `[B,L,hidden]` tensor is materialized. Test T1 proves `v(1)` equals the real forward logit (error ≈ 3·10⁻⁷).

### 2.2 Group Shapley values (Layer 3)

With `n = 7` groups, `N = {1..n}`, and `w_m(s) = s!(m−s−1)!/m!`:

```
φ_k = Σ_{S ⊆ N∖{k}} w_n(|S|) · [ v(S ∪ {k}) − v(S) ]            Σ_k φ_k = v(1) − v(0)
```

Exact (128 coalitions). `φ_k > 0` pushes toward *machine*, `< 0` toward *human*.

### 2.3 Owen values (Layer 4)

The Owen value is the Shapley value restricted to orderings in which the features of each group stay contiguous. For feature `i` in group `G_k` (size `g`):

```
φ_i = Σ_{S ⊆ N∖{k}} w_n(|S|) · Σ_{T ⊆ G_k∖{i}} w_g(|T|) · [ v(U(S) ∪ T ∪ {i}) − v(U(S) ∪ T) ]
```

`U(S)` is the union of the features of the groups in `S`. Two properties hold exactly and are tested:

* **Group consistency:** `Σ_{i∈G_k} φ_i = φ_k` — feature values always add up to their group's value, so the two layers cannot disagree.
* **Efficiency:** `Σ_i φ_i = v(1) − v(0)`.

*Why Owen and not plain Shapley or gradients?* Plain Shapley over 35 features needs 2³⁵ coalitions and ignores the group structure the model itself uses. Gradient methods (e.g. integrated gradients) need a path and a completeness check and do not align with the group tokens. Owen values use one idea ("what happens to the decision if this evidence is removed?") at both levels, with exact additivity.

**Exact computation.** The needed masks are *(union of whole groups) ∪ (subset of one group)*. For the default groups this is `2⁷ + 2⁶·Σ_k(2^{g_k} − 2) = 78,592` distinct masks (79,360 before de-duplication). They are enumerated as `uint64` bitmasks, de-duplicated, evaluated in chunks of 4,096, and accumulated in `float64` with vectorized weight tables.

**Fallback.** `ExplainConfig(owen_method="permutation")` uses antithetic group-contiguous permutations (default 512, fixed seed) and reports per-feature standard errors. Efficiency still holds exactly (each permutation telescopes). The method used is always recorded in `attribution.method`; the module never switches silently. If the exact method would exceed `max_exact_evaluations` it raises an error that points to the fallback.

### 2.4 Path balance (Layer 2)

```
semantic_logit = v(0)              decision with a neutral forensic stream
forensic_delta = v(1) − v(0) = Σ_k φ_k
final_logit    = v(1)              semantic_logit + forensic_delta = final_logit
```

This is a **semantic-anchored** decomposition: the forensic stream is neutralized while the semantic stream is kept. The encoder cannot be cheaply neutralized, so the opposite decomposition is not offered. For attention fusion the module also reports the mean gate value `ḡ` (near 1 means the fused state is dominated by the semantic stream) and, per group, the head-averaged fusion attention `α_k`. Attention is a *reading distribution*, not a causal effect; the cross-check `attention_vs_phi_spearman` compares it with `|φ_k|`.

### 2.5 Baseline

The default baseline is `b = 0` for every raw feature: *no forensic information*. `FeatureEncoder` applies `sign(x)·log1p(|x|)`, so a zero raw vector is a zero transformed vector, the same state the extractor produces when feature extraction fails. Limitations:

* All-zero features is not a realistic text (no real text has zero lexical diversity), so the model may behave unusually at `v(0)`, and contribution magnitudes depend on this reference. Attribution always answers "relative to no forensic information".
* `baseline_value` is configurable; the choice should be checked on real checkpoints (see §7).

### 2.6 Layer 1 and Layer 0

* **Decision:** `machine_score = sigmoid(logit)`; `prediction_at_0_5` uses the PAN threshold; the verdict band is "Likely machine-generated" (score ≥ 0.65), "Likely human-written" (≤ 0.35) or "Inconclusive". The score is prior-adjusted (training used `pos_weight ≈ 0.031`), **not** a calibrated probability.
* **Generator** (multitask checkpoints): softmax over 11 classes, top-3, entropy, `p_human`, and a consistency check `|(1 − p_human) − machine_score| ≤ 0.25`. It is a model-family signature estimate, not authorship attribution.
* **Input diagnostics:** word/sentence/token counts, truncation and coverage (the encoder reads at most `max_length` tokens), zero-width characters, mixed-alphabet words (homoglyphs), unusual whitespace, PII placeholders, and all-zero features. Reliability is **low** if features are all zero or the text has fewer than 30 words; **medium** if coverage < 0.5, zero-width characters or homoglyphs exist, or more than 5 % of whitespace runs are unusual; otherwise **high**.

### 2.7 Preprocessing note

Training and the dataset notebook compute features on **preprocessed** text. `DataCollator` (used by `tira_test.py`) computes features from the raw text. The explainer follows training (preprocessed text for both streams). The collator was not changed, so published TIRA numbers remain reproducible; the small discrepancy is documented here.

## 3. Usage

```python
from flaird.utils.explain import FlairdExplainer, ExplainConfig, explain_to_markdown

explainer = FlairdExplainer.from_pretrained(
    "MahmoodAnaam/flaird-modernbert-large-attention-multitask", ExplainConfig()
)  # AutoModelForSequenceClassification + AutoTokenizer, trust_remote_code=True
result = explainer.explain("Some text to analyze ...")
print(result.decision.verdict, result.decision.machine_score)
print(result.balance.semantic_logit, result.balance.forensic_delta)
for feature in result.top_features(5):
    print(feature.name, feature.phi, feature.direction)
print(explain_to_markdown(result))
payload = result.to_dict()  # JSON-safe
what_if = explainer.counterfactual_logit("...", ["lexical_diversity"])  # group removed
```

CLI:

```bash
flaird-predict --model MahmoodAnaam/flaird-modernbert-large-attention-multitask --text "..."
flaird-predict --input essay.txt --output result.json --report
flaird-predict --input texts.jsonl --output results.jsonl        # field "text" per line
flaird-predict --text "..." --generator-target human             # attribute another class
flaird-predict --text "..." --owen-method permutation            # sampled fallback
```

Capabilities are derived from the checkpoint: concatenation fusion has no attention/gate, single-task checkpoints have no generator head; those fields are `None` with a note in `meta.notes`.

## 4. Output structure

`FlairdExplanation`: `meta`, `input` (diagnostics), `decision`, `generator`, `balance`, `groups` (7), `features` (35, ranked by `abs_rank`), `generator_attribution`, `checks`, `attribution`. All values are JSON-safe (floats rounded to 6 decimals, non-finite values become `null`).

Reading guide: a positive `phi` pushes toward machine, negative toward human, in logit units relative to the baseline. `checks.*_error` fields should be ≤ 10⁻³ (observed ≈ 10⁻⁷ or smaller).

## 5. Performance (measured)

Measured in the development sandbox (**1 CPU core**, random weights, hidden size 1024, feature token dim 128, 8 fusion heads, tiny 2-layer encoder). The real ModernBERT-large forward pass was **not** measured (no Hugging Face access in the sandbox); only the attribution stage is representative, because it does not touch the encoder.

| Fusion | Method | Evaluations | Attribution time | Note |
|--------|--------|-------------|------------------|------|
| attention | exact | 78,592 | 14.9 s | |
| attention | permutation (512) | 8,588 | 1.8 s | max feature difference vs. exact: 0.027 (4 % of max \|φ\|) |
| concatenation | exact | 78,592 | 3.4 s | |
| concatenation | permutation (512) | 8,588 | 0.42 s | max difference 0.0019 (1 % of max \|φ\|) |

A GPU or more CPU cores reduce these times. A possible future optimization is to precompute the query-dependent half of the attention/gate linear layers once per text.

## 6. Tests

`pytest` (35 tests in total, all offline, using a tiny random model and a locally built tokenizer; ~20 s). The 29 tests below cover the explanation module, the CLI and the verification script; 6 further tests (`tests/test_demo.py`) cover the dashboard logic ([DEMO.md](DEMO.md)):

| ID | What is verified |
|----|------------------|
| T1, T2 | `v(1)` equals the real forward logit and `v(0)` equals a real forward with zero features (both fusions, both poolings, with/without generator head); error ≤ 10⁻⁴ |
| T3, T4, T5 | group efficiency, Owen efficiency, Owen–Shapley group consistency ≤ 10⁻³ |
| T6 | exact Owen equals brute-force enumeration of all group-contiguous permutations on a toy game with cross- and within-group interactions (≤ 10⁻⁹) |
| T7 | null players get 0, symmetric players get equal values |
| T8 | concatenation and single-task capabilities degrade gracefully |
| T9 | generator target selection and error message |
| T10 | diagnostics: zero-width, homoglyph, whitespace, short text, truncation, all-zero features |
| T11 | JSON round trip, finite, rounded |
| T12 | determinism (also for the permutation method with a fixed seed) |
| T13 | distinct evaluation count equals the analytic 78,592 |
| T14 | CLI on text, `.txt`, `.jsonl`, `--output`, and error exits |
| T15 | exactly one encoder call per explanation |
| T16 | no dataset/reference-file reads |
| extra | permutation estimator converges on the toy game; `counterfactual_logit`; verification script |

## 7. Verification with real checkpoints (to be run on Colab)

```bash
python -m flaird.scripts.verify_explain \
    --model MahmoodAnaam/flaird-modernbert-large-attention-multitask \
    --examples demo/examples.csv --output verify_attention.json
```

For each of the 20 examples it prints the score, label, generator top-1 vs. truth, the maximum additivity error, attribution time and a **faithfulness test**: removing the 2 top-attributed groups must move the logit in the direction the attribution predicts, and by more than removing 2 random groups. The summary reports the fractions of rows passing (`errors_ok`, `direction_ok`, `top_beats_random`), label agreement and mean attribution time. Use it also to check whether `v(0)` (the semantic-only logit) is sensible for the chosen baseline. **These real-checkpoint results are not yet available**; the script was only run against the tiny random model.

## 8. Limitations

* Attribution is relative to a baseline of "no forensic information"; contribution sizes depend on that reference.
* The path-balance decomposition is semantic-anchored; interactions between the streams are attributed to the forensic side.
* Owen values explain the *model*, not ground truth: a feature pointing toward machine is evidence for the model, not proof.
* The module explains forensic features and fusion; it does not attribute individual words or sentences.
* The machine score is prior-adjusted and not a calibrated probability.
* Exact attribution on a slow CPU with attention fusion takes ~15 s; use the permutation fallback if needed.
* Assumes the checkpoint's remote code matches this repository's module structure (`model.model.fusion`, `model.head`, `model.classifier`, `model.generator_classifier`); the verification script detects a mismatch through `value_function_error`.

## 9. Future work

Comparison against human/machine reference populations of the features; text-level attribution (sentences, tokens); encoder-side counterfactuals for a two-sided path decomposition; faster fusion evaluation by factorizing the query-dependent computation; calibration of the machine score; multilingual support.
