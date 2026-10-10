# FLAIRD Gradio interface: design, interpretation and operations

## Scope and objective

Phase two turns the trained detector and the verified phase-one explanation engine into a usable research workbench. It implements the interface in `demo`, preserves completed experiment settings and benchmark results, and introduces no training or new benchmark claims. Phase three remains responsible for the comprehensive dataset, experiment, PAN26/RAID and thesis/README documentation.

The interface must make three questions answerable: what did the binary detector decide; which forensic signals changed its logit relative to the selected reference; and how trustworthy is the numerical explanation? It also exposes the trained auxiliary and fusion diagnostics without treating them as explanations of author identity or factual truth.

## Components and dependencies

| File | Responsibility |
|---|---|
| `demo/app.py` | Gradio Blocks, layout, events, session reports, export lifecycle, launch |
| `demo/runtime.py` | Eight-checkpoint registry, input validation, lazy model loading, cache eviction, inference lock |
| `demo/presentation.py` | Plotly adapters, escaped source metadata, decision cards and numerical diagnostics |
| `demo/style.css` | Responsive header, cards, spacing, source annotation styling and mobile rules |
| `demo/examples.csv` | Existing twenty examples, unchanged |
| `demo/requirements.txt` | Immutable phase-one package source, PyTorch and Plotly |
| `pyproject.toml` | Plotly demo/test extras; pytest discovery restricted to actual `tests/` |
| `demo/README.md` | Space frontmatter and deployment/user guide |
| `tests/test_demo.py` | Offline UI and result contracts |
| `tests/validate_demo.py` | Opt-in trained-weight and live-browser acceptance workflow |

Gradio 6.27.0 matches the repository's pinned dependency and the Space frontmatter. Theme and stylesheet are supplied to `Blocks.launch`, consistent with the Gradio 6 API. Plotly 6.6.0 supplies interactive hovers and graph export controls. These visualizations use the explanation result directly: the presentation layer does not recompute gradients, substitute another model, or renormalize feature effects into fictitious importance percentages.

The source dependency is pinned to phase-one commit `41f157fc8edcc7f331c8abc8fb4cb916bbcee1e1`. This is necessary while phase-one PR #3 is open: the existing main branch does not contain `FlairdExplainer`. It also makes standalone Space builds reproducible with respect to the local explanation code. Upstream model IDs resolve a Hub revision at load time; that resolved revision is recorded, and the tokenizer is pinned to the same revision. Future deployments can pin model revisions explicitly when promoting a fixed research release.

## Interface organization

The header gives the system identity and task. The left input panel contains a sample selector, English text, a collapsed settings panel, Analyze and Clear. The right panel starts with the decision, machine score and explanation-quality cards, then origin-score bars and four tabs:

1. **Feature effects:** largest fifteen signed effects plus all thirty-five raw values, baseline values and ranks.
2. **Feature groups:** group-summed effects, replacement sensitivity and a collapsed attention diagnostic.
3. **Generator family:** all eleven available auxiliary classes, sorted by probability; a clear unavailable state for single-task checkpoints.
4. **Quality & report:** completeness, gate, reference/input logit, sum of attributions, preprocessing, precision, device, revision, warnings, full JSON and a download workflow.

The strongest effects appear first so the overview remains readable. The full table preserves the feature order and indices supplied by the scientific module. Human/machine scores remain complements of the same binary prediction. The threshold marker is drawn on the machine bar only. Color encodes effect direction: coral for a positive machine-logit effect, teal for a negative effect. Axis labels and signed numbers provide the same information without requiring color discrimination.

At 390px width, the input and results stack vertically and the three decision cards become a single column. Charts remain interactive; longer content and the full feature table can be inspected in their component containers. The hero's auxiliary badge is hidden to preserve useful space. Custom CSS targets named elements/classes rather than generated Gradio class identifiers.

## Inference and lifecycle

Application import/build reads only the local CSV and constructs components. It does not load a model or launch a server. The guarded `main` sets CPU threads and launches on port 7860 by default. This permits offline component tests and ordinary Space startup.

On analysis, the runtime validates nonempty input, the 30,000-character bound, the normalization policy and threshold. The eight IDs are a fixed registry matching the completed experiment configuration files. Loading uses the user's required API:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, trust_remote_code=True
).float().eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=getattr(model.config, "_commit_hash", None),
    trust_remote_code=True,
)
```

Model parameters remain trained checkpoint values. Float32 is chosen to support attribution accuracy. CPU/CUDA placement is selected by availability or `FLAIRD_DEVICE`. The model's semantic encoder runs once within the existing explainer; semantic states are then reused for integrated gradients and replacement tests. The interface does not wrap the explainer in inference mode, which would prevent gradient-based explanations.

A single lock covers cache switching and the complete analysis, avoiding concurrent evictions or memory allocation across sessions. Repeating the same model selection reuses its resident model/tokenizer; selecting a different model releases references and collects garbage before loading the replacement. An explicit CUDA cache release assists switching on GPU hosts. Failed loads do not populate a half-valid cache. Eight models are never retained simultaneously by this runtime.

The Gradio queue admits one active analysis and at most eight waiting events. A streamed callback first clears the prior result and disables input/model/threshold/policy/sample/analyze/clear controls. It then returns either the complete new result or an error state and restores controls. Input/settings edits clear the report and disable downloads. Selecting an example updates text and source annotations, then clears previous analysis; the user still explicitly requests analysis.

This controls ordinary browser interactions. Programmatic API clients remain responsible for correlating their own requests; the global lock guarantees model integrity, but session/request scheduling is not a benchmark throughput service. Future deployments needing many users should isolate model workers and provide request IDs.

## Mathematical meaning of each view

Let the machine logit be \(z(s,x)\), with fixed semantic representation \(s\), observed forensic features \(x\), and reference \(b=0\). The interface displays \(p_M=\sigma(z)\), \(p_H=1-p_M\). At threshold \(\tau\), the decision is machine iff \(p_M>\tau\), matching the explainer rather than rounding displayed percentages before comparison. Threshold changes affect the decision rule, not the underlying model or attribution target.

For feature \(i\), the plotted value is

\[
 A_i=(x_i-b_i)\int_0^1 \frac{\partial z(s,b+\alpha(x-b))}{\partial x_i}\,d\alpha.
\]

The feature overview selects the fifteen smallest magnitude ranks; it retains each signed value. These are **logit units**, not probability percentages. A group \(G\) displays \(A_G=\sum_{i\in G}A_i\). Neither the semantic encoder nor the entire document is explained by these conditional feature effects.

For a group replacement, \(x^{(G\leftarrow b)}\) keeps the original features outside \(G\) and assigns reference values inside \(G\). The separate sensitivity chart displays

\[
 \Delta p_G=100\left[\sigma(z(s,x))-\sigma(z(s,x^{(G\leftarrow b)}))\right]
\]

in **percentage points**. The factor 100 is a unit conversion, not a normalization across groups. Interactions and nonlinearities mean these replacement deltas need not sum to a total. A positive delta means the actual group's features raise the machine score relative to replacing them with zeros, conditional on fixed semantics.

Attention is the trained attention-fusion output averaged over heads, indexed by the seven forensic groups. Its percentages sum to approximately 100 by attention normalization. This describes routing, not a causal contribution estimate. The mean semantic gate is an internal interpolation diagnostic and is displayed as a scalar, not a percentage of the final prediction. Concatenation variants expose neither diagnostic; no substitute values are invented.

Multitask generator probabilities are the softmax of the independent auxiliary head. The distribution includes `human`, can disagree with the binary head, and is not \(P(\text{generator}\mid\text{machine})\). Single-task variants have no generator head. The charts state these limits explicitly.

## Numerical quality and text scope

The quality card propagates the explainer's actual completeness result:

\[
 r=z(s,x)-z(s,b)-\sum_i A_i,\qquad
 |r|\le \epsilon_a+\epsilon_r|z(s,x)-z(s,b)|.
\]

Defaults are 64 integration points, adaptive doubling up to 1024, absolute tolerance 0.001 and relative tolerance 0.01. The UI does not claim success merely because analysis completed. A failed criterion becomes **Needs review**, and the quality tab recommends withholding reliance on the ranked effects. The report retains convergence details for further inspection. Passing completeness demonstrates numerical consistency, not causal validity or predictive calibration.

The zero-feature reference is a numerical baseline and may be outside the data distribution. The reference logit and attribution sum make this visible. Scientific alternatives such as training-only empirical baselines belong in future methodological work rather than being fabricated in the interface.

Semantic tokens are capped at 512 to match experiment settings. Forensic features cover the full feature text; long input therefore gives the two branches different scope. The result card states tokens used/available and truncation. The 30,000-character input bound protects feature extraction and request duration without silently truncating the forensic text. The normalization selector maps directly to `training`, `tira`, or `none`; the chosen policy is stored in every report.

## Examples and provenance

`examples.csv` contains twenty rows with `text`, `attack`, `model`, `label` and `generator_label`. Selection preserves the original text, including attack-related whitespace and unusual characters, until the user-selected inference preprocessing runs. Source annotations are displayed in a separate provenance card and never passed to the classifier or attribution routine. Blank attacks are rendered as not recorded, rather than asserting an attack-free sample.

The examples are demonstrations, not a balanced test set: there is only one recorded human row. Neither their source annotations nor the UI acceptance runs justify an accuracy estimate. Editing sample text clears its provenance so an edited passage cannot retain an apparently verified label. Metadata is escaped before insertion in HTML.

## Export, failures and deployment

The report is held in Gradio session state. Preparing an export writes a unique temporary JSON file, removes that session's previous source export, and enables its download link. Session cleanup removes the source file after the configured state lifetime/disconnection cleanup; Gradio's cache maintenance removes aged served copies on an hourly cycle. A new analysis/edit invalidates the visible export link. Files never share a deterministic cross-session filename. Exports omit the original text and example annotations, but contain forensic values, predictions, settings, checkpoint revision and warnings; they are research records, not anonymization guarantees.

Input errors become escaped, actionable alerts. Unexpected model/host failures produce a concise generic message and restore controls; server logs retain the exception for maintainers. The application does not log submitted text explicitly. Gradio/host infrastructure policies still apply; applications handling sensitive data should evaluate those policies before deployment.

Deploy by copying `demo/` contents to the root of a Gradio Space. The existing frontmatter specifies Python 3.12, Gradio 6.27.0 and `app.py`. CPU or a dedicated CUDA GPU is supported; no hardware purchase, Space creation or public deployment happens implicitly in this implementation. ZeroGPU needs a separate lifecycle integration and is not advertised as supported. A standard host needs sufficient RAM for float32 ModernBERT-large plus temporary loading overhead; validate actual host usage and cold start before offering a public endpoint. See `demo/README.md` for commands and environment settings.

## Validation and remaining work

The offline contracts validate all twenty exact sample selections and the eight-ID registry, signed plotting values/ranks, probability and percentage-point conversions, absent-head states, failed-quality messaging, escaped provenance, early invalid-input rejection, streamed success/failure restoration, isolated export files/cleanup, and lazy application construction. These supplement the 25 phase-one explanation tests.

The final default `python -m pytest -q` run passed **35 tests**: 25 explanation tests and 10 interface tests. Pytest discovery is restricted to `tests/` because `src/flaird/scripts/tira_test.py` is a production evaluation command containing a function named `test`, not a pytest fixture-based test. Ruff lint/format and Git whitespace checks also passed. Browser validation additionally confirmed checkpoint switching to concatenation single-task frozen, a 65% threshold, both unavailable-head panels, and real inference in a fresh 390px session. The evidence file includes implementation hashes and completion time.

The opt-in acceptance script uses all eight real published checkpoints through the production runtime, checks completeness and chart availability, verifies cache reuse, and exercises the running Gradio page in Chromium. Browser checks include blank-input recovery, CSV sample selection, actual inference, feature/group/attention/generator/quality views, JSON download, stale-result/provenance/export invalidation, Clear and mobile overflow. `docs/validation/gradio-checks.json` records versions, immutable checkpoint revisions and the completed checks. These runs are functional validation, not new PAN26/RAID measurements. Visual screenshots are inspected during implementation; they do not replace numerical checks.

Future work includes target-Space deployment validation, ZeroGPU support if required, fixed model-release revision configuration, production memory/latency measurements, model-worker isolation for higher concurrency, empirically calibrated decision thresholds, training-only reference selection, and user studies of explanation comprehension. Token saliency or causal claims would require separate methods and validation. Phase three will expand the research documentation using the existing training and evaluation evidence.

## Primary references

- [Gradio 6 migration guide](https://www.gradio.app/guides/gradio-6-migration-guide): launch-level theme/CSS and current component API.
- [Gradio queuing](https://www.gradio.app/guides/queuing): concurrency and queue controls.
- [Hugging Face Space configuration](https://huggingface.co/docs/hub/spaces-config-reference): frontmatter and runtime settings.
- [Integrated Gradients](https://arxiv.org/abs/1703.01365): attribution definition and completeness property; FLAIRD's conditional boundary and quadrature implementation are documented separately in `EXPLANATION_MODULE.md`.
