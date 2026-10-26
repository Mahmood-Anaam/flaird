# Forensic-Linguistic Features

FLAIRD's forensic stream consists of **35 deterministic features** computed by `flaird.modeling.features.ForensicFeatureExtractor`. They are fixed, ordered, human-readable measurements of how a text is written. This document defines them, explains why each group is relevant to machine-text detection, and describes the numerical conventions.

---

## 1. Design principles

1. **Interpretable by construction.** Every feature has a plain-language meaning, so an analyst can reason about the evidence.
2. **Deterministic.** The same text always gives the same vector; no learned component is involved.
3. **Content-light.** Features describe style, vocabulary distribution and structure rather than topics.
4. **Computed on the whole preprocessed text.** The feature vector uses the complete text after placeholder substitution, whereas the neural encoder reads at most 512 tokens.
5. **Robust failure mode.** Some library routines reject extremely short inputs; in that case the affected values become `0.0`, and non-finite values are replaced by `0.0`. The explanation module flags the all-zero vector as an unreliable input.

## 2. Notation

For a text, let `N` be the number of word tokens (lower-cased, matched by `\b[\w'-]+\b`), `V` the number of distinct words (types), `V_i` the number of words that occur exactly `i` times, `M_2 = Σ_i i² V_i`, and `CV(a) = std(a) / mean(a)` the coefficient of variation (0 for an empty list or zero mean). Sentences are split at `[.!?]+`; paragraphs at blank lines.

Library-computed measures (`pystylometry`, version ≥ 1.4.3) are marked ⚙; features computed directly in FLAIRD are marked ✎. Library measures follow the standard definitions of the stylometry literature; where a numerical convention matters, the library implementation is authoritative.

## 3. The 35 features by group

The groups are the 7 *feature tokens* of the model (`DEFAULT_FEATURE_GROUPS`). Within each group, the position in the 35-dimensional vector is shown.

### 3.1 Lexical diversity (10 features, indices 0–9)

Question answered: *how rich and varied is the vocabulary?* Language models tend to produce fluent but statistically regular vocabularies; humans show more idiosyncratic vocabulary and more rare words.

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 0 | `ttr` | ⚙ | Type-token ratio `V / N`. |
| 1 | `root_ttr` | ⚙ | Root TTR `V / √N` (corrects the length dependence of TTR). |
| 2 | `log_ttr` | ⚙ | Log TTR `log V / log N` (another length correction). |
| 3 | `mtld` | ⚙ | Measure of Textual Lexical Diversity (average of forward and backward passes): mean length of word runs that keep a running TTR above a threshold. Robust to text length. |
| 4 | `yule_k` | ⚙ | Yule's K `10⁴ (M_2 − N) / N²`; larger values mean a more repetitive vocabulary. |
| 5 | `yule_i` | ⚙ | Yule's I, the inverse-type measure of vocabulary richness (larger means richer). |
| 6 | `hapax_ratio` | ⚙ | Proportion of hapax legomena (words occurring once). |
| 7 | `dis_hapax_ratio` | ⚙ | Proportion of dis legomena (words occurring twice). |
| 8 | `sichel_s` | ⚙ | Sichel's S, based on `V_2 / V`. |
| 9 | `honore_r` | ⚙ | Honoré's R `100 log N / (1 − V_1 / V)`; high values mean many once-used words. |

### 3.2 Function-word style (7 features, indices 10–16)

Question answered: *what are the writer's habits with grammatical "glue" words?* Function words are used largely unconsciously and are a classical authorship signal; models reproduce their distribution well on average but not in their natural variability.

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 10 | `determiner_ratio` | ⚙ | Share of determiners (the, a, this, …). |
| 11 | `preposition_ratio` | ⚙ | Share of prepositions. |
| 12 | `conjunction_ratio` | ⚙ | Share of conjunctions. |
| 13 | `pronoun_ratio` | ⚙ | Share of pronouns. |
| 14 | `auxiliary_ratio` | ⚙ | Share of auxiliary verbs. |
| 15 | `function_word_ratio` | ⚙ | Total share of function words. |
| 16 | `function_word_diversity` | ⚙ | Variety of distinct function words used. |

### 3.3 Surface composition (5 features, indices 17, 21–24)

Question answered: *what does the text look like at the character level?* Encoding artifacts, formatting habits and adversarial character manipulations show up here.

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 17 | `avg_word_length` | ⚙ | Mean word length in characters. |
| 21 | `vowel_consonant_ratio` | ⚙ | Vowels divided by consonants. |
| 22 | `digit_ratio` | ⚙ | Share of digit characters. |
| 23 | `uppercase_ratio` | ⚙ | Share of uppercase characters. |
| 24 | `whitespace_ratio` | ⚙ | Share of whitespace characters (also sensitive to whitespace attacks). |

### 3.4 Structural organization (4 features, indices 18, 27, 28, 34)

Question answered: *how is the text organized into sentences and paragraphs?* Human writing tends to vary sentence length more ("burstiness"); generated text is often more regular.

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 18 | `avg_sentence_length_chars` | ⚙ | Mean sentence length in characters. |
| 27 | `sentence_length_cv` | ✎ | `CV` of sentence lengths measured in words. |
| 28 | `paragraph_length_cv` | ✎ | `CV` of paragraph lengths in words. |
| 34 | `newline_density` | ✎ | Line breaks per word: `count("\n") / max(1, N)`. |

### 3.5 Punctuation and rhetoric (5 features, indices 19, 20, 31–33)

Question answered: *how does the writer punctuate and address the reader?*

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 19 | `punctuation_density` | ⚙ | Punctuation characters per character. |
| 20 | `punctuation_variety` | ⚙ | Number of distinct punctuation marks used. |
| 31 | `question_density` | ✎ | Question marks per sentence: `count("?") / max(1, #sentences)`. |
| 32 | `exclamation_density` | ✎ | Exclamation marks per sentence. |
| 33 | `quote_density` | ✎ | Quotation marks (`"`, `“`, `”`) per word. |

### 3.6 Information entropy (2 features, indices 25, 26)

Question answered: *how predictable are character and word sequences?* Text sampled from a language model is, by construction, high-probability under a language model; bigram entropy provides a cheap proxy for local predictability.

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 25 | `character_bigram_entropy` | ⚙ | Shannon entropy of the character-bigram distribution. |
| 26 | `word_bigram_entropy` | ⚙ | Shannon entropy of the word-bigram distribution. |

### 3.7 Repetition and burstiness (2 features, indices 29, 30)

Question answered: *how are repeated words and phrases distributed?* Human text clusters topical words in bursts; generated text repeats phrases more uniformly (or avoids repetition more aggressively).

| # | Feature | Source | Definition and meaning |
|--:|---------|:------:|------------------------|
| 29 | `word_burstiness` | ✎ | `CV` of the word-frequency counts (unevenness of the frequency distribution). |
| 30 | `repeated_bigram_ratio` | ✎ | Distinct word bigrams that occur more than once, divided by the number of distinct bigrams. |

## 4. Numerical conventions inside the model

* Raw values are fed to the model unnormalized; the `FeatureEncoder` applies `sign(x)·log(1 + |x|)` and one learned projection per group (see [ARCHITECTURE.md](ARCHITECTURE.md), §3.3).
* A raw vector of zeros maps to zero after the logarithm, which is why it is used as the "no forensic information" baseline of the explanation module.
* No normalization statistics (means, variances) are computed from the data, so the extractor and the model have no hidden dependency on the training corpus.

## 5. Why not more (or learned) features?

The set was chosen to cover the classical stylometric families with a small, cheap, library-backed vector that is stable on texts of 100–1000 words. The `FeatureEncoder` is generic: a different list of features and group assignment can be supplied through the model configuration (`forensic_feature_dim`, `feature_group_names`, `feature_group_indices`), but a checkpoint is only valid for the feature definition it was trained with.

## 6. Known limitations

* Feature values depend on text length for some measures (TTR); the length-corrected variants are included for this reason, but very short texts remain unreliable.
* Sentence and paragraph splitting is rule-based and English-oriented.
* Library routines that reject very short inputs produce zeros, which look identical to a genuinely degenerate text; the explanation module therefore reports this case explicitly.
* The thesis proposal anticipates features for syntactic complexity, discourse coherence and readability. The released extractor does **not** contain such features (it has no parser-based or discourse-level measures); the 35 features cover lexical, function-word, character-level, structural, punctuation, entropy and repetition evidence only.
