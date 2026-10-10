"""Faithful plots and escaped summaries; no inference or attribution recomputation."""

from html import escape

import plotly.graph_objects as go

TEAL = "#0f766e"
CORAL = "#c2413b"
INK = "#193549"


def label(name):
    return name.replace("_", " ").capitalize()


def style(fig, height=370):
    return fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Arial, sans-serif", "size": 12, "color": INK},
        margin={"l": 12, "r": 24, "t": 25, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"gridcolor": "#e8eef2", "zerolinecolor": "#93a4b2"},
    )


def empty_plot(message):
    fig = style(go.Figure(), 220)
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
    )
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    return fig


def bars(names, values, title, signed=False, height=370):
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[CORAL if x > 0 else TEAL for x in values] if signed else TEAL,
            hovertemplate="%{y}<br>%{x:.5f}<extra></extra>",
        )
    )
    style(fig, height)
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title=title)
    if signed:
        fig.add_vline(x=0, line_width=1, line_color=INK)
    return fig


def charts(report):
    p = report["prediction"]
    probability = bars(
        ["Human", "Machine"],
        [100 * p["human_probability"], 100 * p["machine_probability"]],
        "Model score (%)",
        height=210,
    )
    probability.update_xaxes(range=[0, 100])
    # Threshold belongs only to the machine bar; the human complement is not tested.
    probability.add_shape(
        type="line",
        x0=100 * p["threshold"],
        x1=100 * p["threshold"],
        y0=0.5,
        y1=1.5,
        xref="x",
        yref="y",
        line={"dash": "dot", "color": CORAL},
    )
    top = sorted(report["features"], key=lambda row: row["rank"])[:15]
    features = bars(
        [label(r["name"]) for r in top],
        [r["attribution"] for r in top],
        "Signed effect on machine logit",
        signed=True,
        height=510,
    )
    groups = report["groups"]
    group_effect = bars(
        [label(r["name"]) for r in groups],
        [r["attribution"] for r in groups],
        "Sum of feature effects (logit units)",
        signed=True,
    )
    replacement = bars(
        [label(r["name"]) for r in groups],
        [100 * r["probability_delta"] for r in groups],
        "Original score − group replaced score (percentage points)",
        signed=True,
    )
    attention = (
        bars(
            [label(r["name"]) for r in groups],
            [100 * r["attention"] for r in groups],
            "Head-average group attention (%)",
        )
        if groups[0]["attention"] is not None
        else empty_plot("Group attention is unavailable for concatenation models")
    )
    generators = p["generator_probabilities"]
    if generators is not None:
        ranked = sorted(generators, key=lambda r: -r["probability"])
        generator = bars(
            [r["label"] for r in ranked],
            [100 * r["probability"] for r in ranked],
            "Auxiliary model score (%)",
            height=420,
        )
        generator.update_xaxes(range=[0, 100])
    else:
        generator = empty_plot("Generator head is unavailable for single-task models")
    return probability, features, group_effect, replacement, attention, generator


def summary(report):
    p, d, m = report["prediction"], report["diagnostics"], report["metadata"]
    decision = "Machine-generated" if p["label"] == "machine" else "Human-written"
    quality = "Passed" if d["completeness_passed"] else "Needs review"
    return f"""<div class="result-grid">
      <div class="metric"><span>MODEL DECISION</span><strong>{decision}</strong><small>Threshold {p["threshold"]:.0%}</small></div>
      <div class="metric"><span>MACHINE SCORE</span><strong>{p["machine_probability"]:.2%}</strong><small>Model probability · uncalibrated</small></div>
      <div class="metric"><span>EXPLANATION CHECK</span><strong>{quality}</strong><small>{d["integration_steps_used"]} integration points</small></div>
    </div><p class="scope-note">Semantic input: {m["tokens_used"]} / {m["tokens_before_truncation"]} tokens ·
    {"truncated at 512 tokens" if m["semantic_truncated"] else "within token limit"} · Forensic features: full text</p>"""


def diagnostics(report):
    d, m = report["diagnostics"], report["metadata"]
    gate = (
        "Unavailable for concatenation"
        if d["mean_semantic_gate"] is None
        else f"{d['mean_semantic_gate']:.5f} (internal interpolation diagnostic)"
    )
    warnings = "\n".join(f"- {w}" for w in report["warnings"])
    return f"""### Interpretation and numerical quality
Feature effects hold the semantic representation fixed and use a zero-feature reference.
They explain changes in the machine **logit**, rather than word importance or factual accuracy.

| Check | Value |
|---|---|
| Completeness | {"Passed" if d["completeness_passed"] else "**Needs review — do not rely on the ranked effects**"} |
| Residual / allowed tolerance | {d["completeness_residual"]:.6g} / {d["completeness_tolerance"]:.6g} |
| Reference logit / input logit | {d["baseline_logit"]:.6g} / {report["prediction"]["machine_logit"]:.6g} |
| Sum of feature effects | {d["attribution_sum"]:.6g} |
| Mean semantic gate | {gate} |
| Normalization | {m["preprocessing"]} |
| Precision / device | {m["dtype"]} / {m["device"]} |
| Checkpoint revision | `{m["model_revision"]}` |

{warnings}
"""


def provenance(example, index):
    fields = [
        ("Source model", example["model"]),
        ("Recorded label", example["label"]),
        ("Recorded generator", example["generator_label"]),
        ("Attack", example["attack"] or "Not recorded"),
    ]
    return (
        f'<div class="provenance"><b>Example {index + 1:02d} · source metadata</b><p>'
        + " · ".join(f"{name}: {escape(value)}" for name, value in fields)
        + "</p><small>These are dataset annotations, not the system’s prediction. Editing the text clears this provenance.</small></div>"
    )
