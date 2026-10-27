# Forensic-linguistic feature specification

## 1. Purpose and implementation contract

[ForensicFeatureExtractor](../src/flaird/modeling/features.py) returns 35 ordered floating-point values. These are observable stylistic measurements, not independent tests of human authorship. The classifier learns their joint relevance. A high or low value has no universal machine/human direction; an Owen attribution describes its effect for one checkpoint, document and reference input.

The seven groups in `DEFAULT_FEATURE_GROUPS` are the learned feature tokens and the explanation coalitions. Feature order must remain compatible with a checkpoint. Reordering values while retaining the same dimensions silently changes their meaning.

The tables below distinguish library definitions from direct FLAIRD calculations. Library formulas were checked against **pystylometry 1.4.3**. The package specification allows `>=1.4.3,<2`; the dataset notebook installs an unpinned version. Therefore exact historical dependency provenance is not recoverable from that notebook alone. Reproduction should record the resolved version and compare stored vectors before regenerating them.

## 2. Notation and tokenization

For a token sequence, $N$ denotes tokens, $V$ distinct types, $V_r$ types occurring exactly $r$ times, and $M_2=\sum_r r^2V_r$. For a list $a$:

$$
\operatorname{CV}(a)=\frac{\operatorname{std}_{\mathrm{population}}(a)}{\operatorname{mean}(a)}.
$$

FLAIRD uses zero when a list is empty or its mean is zero. Its direct word counts use lowercased matches of `\b[\w'-]+\b`; sentence segments use nonempty pieces after `[.!?]+`; paragraphs use nonempty pieces after `\n\s*\n`. CV sentence/paragraph lengths use whitespace splitting **inside** those segments. These conventions differ from the library's tokenizers and character-level sentence handling. Do not reuse a single $N$ across incompatible implementations when checking an example.

**Library chunking matters.** TTR's main values are global in 1.4.3. MTLD, Yule, hapax, character metrics, and entropy expose summaries over the library's default 1,000-word chunks. Many selected fields are means of chunk measurements, rather than the formula applied once to all tokens. The formulas below define the underlying measurement; the returned library attribute defines aggregation. Every part of the input is available to extraction, but this does not imply a single global statistic for each field.

## 3. Lexical diversity: indices 0–9

The purpose is to represent vocabulary variety and concentration with complementary length sensitivities. A repetitive technical document and a repetitive generated document can have similar values; domain and length are confounders.

| Index | Feature / library attribute | Definition | Illustrative interpretation |
|---:|---|---|---|
| 0 | `ttr` / `ttr` | $V/N$ | Five distinct types in ten tokens yield 0.5. Longer documents often have smaller TTR even for the same writer. |
| 1 | `root_ttr` / `root_ttr` | $V/\sqrt{N}$ | For $N=100,V=40$, value 4. It reduces, but does not eliminate, length dependence. |
| 2 | `log_ttr` / `log_ttr` | $\log V/\log N$; 0 for $N\leq1$ | The same example gives approximately 0.801. Also called Herdan's C. |
| 3 | `mtld` / `mtld_average` | Average forward/backward MTLD; default TTR threshold 0.72 | Longer segments before diversity falls below the threshold suggest sustained lexical variety. It is not a readability score. |
| 4 | `yule_k` / `yule_k` | $10^4(M_2-N)/N^2$ | Higher concentration of repeated types raises K. |
| 5 | `yule_i` / `yule_i` | $V^2/(M_2-N)$ | In this library, larger values indicate less concentrated vocabulary. Undefined values become zero downstream; I is not simply $1/K$. |
| 6 | `hapax_ratio` / `hapax_ratio` | $V_1/N$ | Types used once divided by token count, **not** vocabulary size. |
| 7 | `dis_hapax_ratio` / `dis_hapax_ratio` | $V_2/N$ | Types used twice divided by tokens; their occurrences are not counted twice in the numerator. |
| 8 | `sichel_s` / `sichel_s` | $V_2/V$ | Vocabulary share represented by twice-used types. This has a different denominator from feature 7. |
| 9 | `honore_r` / `honore_r` | $100\log N/(1-V_1/V)$ | Many once-used types increase R. An infinite result is converted to zero, so zero may indicate an undefined case. |

For `cat cat dog bird bird`, $N=5,V=3,V_1=1,V_2=2,M_2=9$. In one chunk, TTR is 0.6, hapax ratio 0.2, dis-hapax ratio 0.4, Sichel's S $2/3$, Yule's K 1600, and Yule's I 2.25. This demonstrates why repeated types and repeated occurrences must not be confused.

## 4. Function-word style: indices 10–16

The purpose is to capture habits in grammatical connecting words with fixed English lexicons. These are dictionary-based counts, **not contextual POS tags** or a syntactic parser. A word can occur in multiple category lists. The total union counts each matching token once, so category ratios need not sum to the total ratio.

| Index | Feature | Definition | Example and analytical purpose |
|---:|---|---|---|
| 10 | `determiner_ratio` | Determiner matches / library word count | “the”, “a”, “this”: article and reference habits. |
| 11 | `preposition_ratio` | Preposition matches / words | “in”, “of”, “on”: relational phrase frequency. |
| 12 | `conjunction_ratio` | Conjunction matches / words | “and”, “but”, “or”: explicit linking of clauses or items. |
| 13 | `pronoun_ratio` | Pronoun matches / words | “I”, “they”, “it”: personal versus impersonal reference patterns. |
| 14 | `auxiliary_ratio` | Auxiliary matches / words | “is”, “have”, “will”: surface tense/modal habits; ambiguous words are not disambiguated. |
| 15 | `function_word_ratio` | `total_function_word_ratio`: union matches / words | Includes library categories such as particles beyond the five separately exposed ratios. |
| 16 | `function_word_diversity` | Distinct matching function words / all matching function-word tokens | If six function-word tokens contain three types, diversity is 0.5. It measures within-function-word variety, not share of total vocabulary. |

For a category with two matches in ten library word tokens, its ratio is 0.2. Replacing two “and” occurrences by “but” changes diversity and potentially rhetorical style without necessarily changing the conjunction ratio. This motivates using both counts and diversity.

## 5. Surface composition: indices 17, 21–24

These features encode orthography and formatting. Their relevance can arise from writing convention, preprocessing artifacts or adversarial manipulation rather than generation itself.

| Index | Feature | Definition in pystylometry 1.4.3 | Example / caveat |
|---:|---|---|---|
| 17 | `avg_word_length` | Mean alphanumeric character count of whitespace-delimited words containing alphanumeric characters | “cat, dogs” gives lengths 3 and 4, mean 3.5; comma is excluded. |
| 21 | `vowel_consonant_ratio` | English vowel count / English consonant count | “cat” gives $1/2$; absent consonants may yield nonfinite values, subsequently zeroed. |
| 22 | `digit_ratio` | Digits / total characters | “a1 b” has one digit in four characters, 0.25, before chunk handling. Phone-like strings may already be replaced. |
| 23 | `uppercase_ratio` | Uppercase alphabetic count / counted English letters | “Ab” gives 0.5. The denominator is letters, **not all characters**. Non-English input can violate intuitive ratio bounds. |
| 24 | `whitespace_ratio` | Whitespace / total characters in library chunks | Two spaces in a ten-character chunk give 0.2. Library chunk reconstruction can alter original whitespace; this is not a lossless whitespace audit. |

## 6. Structural organization: indices 18, 27, 28, 34

The purpose is to summarize variation in document structure without requiring a parser. Uniform or variable length is a descriptive observation, not an authorship rule.

| Index | Feature | Definition | Worked interpretation |
|---:|---|---|---|
| 18 | `avg_sentence_length_chars` | Library mean sentence length in characters, including its delimiter handling, aggregated over chunks | Longer clauses can increase it, but abbreviations and repeated punctuation affect segmentation. |
| 27 | `sentence_length_cv` | Population CV of whitespace-word counts in FLAIRD sentence segments | Sentence lengths 2 and 6 yield mean 4, standard deviation 2, CV 0.5. |
| 28 | `paragraph_length_cv` | Population CV of whitespace-word counts in blank-line-separated paragraphs | One paragraph gives zero variation regardless of its length. |
| 34 | `newline_density` | `text.count("\n") / max(1,N)` using direct regex words | Two newline characters and ten words yield 0.2; a blank line typically contributes two characters. |

Repeated punctuation such as `?!` is split as a run by FLAIRD's sentence regex. The library character routine can use a different segmentation, so features 18 and 27 intentionally do not share a denominator. Neither measures grammatical tree depth or discourse coherence.

## 7. Punctuation and rhetoric: indices 19, 20, 31–33

These indicators capture rhetorical presentation and formatting intensity. Quotes, questions and exclamations can be topic- or genre-dependent.

| Index | Feature | Definition | Example / caveat |
|---:|---|---|---|
| 19 | `punctuation_density` | Library punctuation count / whitespace word count × 100, summarized over chunks | Two recognized punctuation marks in four words yield **50**, not a character fraction. |
| 20 | `punctuation_variety` | Distinct recognized punctuation types, summarized over chunks | A period, comma and question mark produce variety 3 in one chunk. It is not normalized to text length. |
| 31 | `question_density` | Number of `?` / `max(1, sentence segments)` | “Why?? Yes.” has two marks and two segments: 1. Multiple marks can make the ratio exceed one. |
| 32 | `exclamation_density` | Number of `!` / `max(1, sentence segments)` | “Stop!!!” gives 3 for one segment. |
| 33 | `quote_density` | Count of `"`, `“`, `”` / `max(1,N)` | One opening and one closing quote contribute two. Single apostrophes are not counted. |

The original description of punctuation density as punctuation per character was incorrect for the checked library version. This release corrects the documentation and display description while preserving the extractor and trained feature semantics.

## 8. Information entropy: indices 25–26

For empirical bigram frequencies $p_b$, the underlying statistic is Shannon entropy in bits:

$$
H_2=-\sum_{b:p_b>0}p_b\log_2 p_b.
$$

| Index | Feature | Measurement | Example / limitation |
|---:|---|---|---|
| 25 | `character_bigram_entropy` | Mean library chunk entropy of overlapping character bigrams | If two bigram types occur equally often, entropy is 1 bit. Spaces and case can contribute under the library tokenizer/chunk rules. |
| 26 | `word_bigram_entropy` | Mean library chunk entropy of overlapping word bigrams | Three distinct equally frequent pairs yield $\log_2 3\approx1.585$ bits. |

These are empirical distributions of pair **types**. They are not a pretrained language model's likelihood, conditional next-word entropy, perplexity score from ModernBERT, or a semantic coherence measure. High empirical entropy often follows from many unique bigrams and is strongly affected by length. The rationale is to expose local distributional regularity cheaply, not to replace probabilistic language modeling.

## 9. Repetition and burstiness: indices 29–30

| Index | Feature | Direct calculation | Example / limitation |
|---:|---|---|---|
| 29 | `word_burstiness` | Population CV of counts of distinct regex word types | Frequencies `[3,1]` yield CV 0.5. It measures frequency inequality, not the temporal spacing of repeated words. Permuting tokens preserves it. |
| 30 | `repeated_bigram_ratio` | Number of distinct word-pair types with count > 1 / `max(1, number of distinct pair types)` | “a b a b” has pair types `a b` (twice), `b a` (once): ratio $1/2$. This is not repeated occurrences / all occurrences. |

The group targets repetition concentration and repeated local phrases. A text can have high word-frequency inequality and low pair repetition, hence both indicators are retained. The name “burstiness” should be interpreted according to the implemented frequency-CV definition.

## 10. Extraction failure, scaling and explanation

The library calls share one `try` block. If **any** of those calls raises `ValueError` or `ZeroDivisionError`, **all library-backed result objects** in that block become `None`; their attributes yield zero. Direct features (indices 27–34) are still calculated. `_finite` additionally maps invalid scalar conversions, NaN and infinity to zero. The implementation does not record a separate failure flag per feature, and exceptions outside these handled types propagate.

Consequently, zero may mean a genuine absence, an undefined statistic, or a handled library failure. An all-zero vector is flagged by the explainer, but partial library failure can remain undetected. Very short texts warrant caution even when some direct features are nonzero.

The model applies $\operatorname{sign}(x)\log(1+|x|)$, followed by learned group projections. There is no dataset-fitted standardization. See [architecture](ARCHITECTURE.md) for why this matters to attention. An attribution of +0.4 is a **machine-logit contribution relative to the masking baseline**, not “40% machine,” nor a threshold on the raw feature.

```python
from flaird.modeling.features import FEATURE_NAMES, ForensicFeatureExtractor

extractor = ForensicFeatureExtractor()
text = 'One small example. A second sentence repeats one word.'
measurements = extractor.extract_dict(text)
vector = extractor(text)
assert len(vector) == 35
assert list(measurements) == list(FEATURE_NAMES)
```

For training-aligned use, apply `preprocess_text` before extraction. The feature extractor itself intentionally does not perform that substitution. Features are computed over the complete supplied text; in the demo this is at most 30,000 characters, while the encoder independently sees at most 512 tokens.

## 11. Research interpretation and reproducibility

The feature families are motivated by stylometry, but the repository contains no isolated statistical validation that every feature is independently discriminative. Correlation is expected: lexical richness measures share vocabulary counts, and punctuation/structure measures overlap. Owen values respect predefined groups; they do not remove these statistical dependencies or turn masking into realistic text editing.

There are no parser-based syntactic complexity, readability-index, discourse-relation, or semantic coherence features in this release. Such measures remain possible extensions requiring new datasets, checkpoint compatibility decisions and ablations. English lexicons and sentence heuristics do not establish multilingual validity.

Primary implementation reference: Craig Trim, [pystylometry repository](https://github.com/craigtrim/pystylometry), especially `lexical/`, `character/character_metrics.py`, and `ngrams/entropy.py`. Version-specific package: [pystylometry 1.4.3](https://pypi.org/project/pystylometry/1.4.3/). Foundational definitions include Yule (1944), *The Statistical Study of Literary Vocabulary*, and Shannon (1948), *A Mathematical Theory of Communication*. These motivate measurements; FLAIRD's chosen library conventions govern its actual inputs.
