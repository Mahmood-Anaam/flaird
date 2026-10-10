"""Offline UI contracts: correct signs/units, absent heads, provenance and failure reset."""

import copy
import json
from pathlib import Path

import gradio as gr
import pytest

from demo import app, presentation
from demo.runtime import MAX_CHARACTERS, MODELS, ModelRuntime, load_examples


@pytest.fixture
def report():
    # UI contract data with intentionally negative and positive effects.
    return {
        "prediction": {
            "machine_probability": 0.7,
            "human_probability": 0.3,
            "threshold": 0.5,
            "label": "machine",
            "machine_logit": 1.5,
            "generator_probabilities": [
                {"label": "human", "probability": 0.25},
                {"label": "GPT", "probability": 0.75},
            ],
        },
        "features": [
            {
                "name": f"feature_{i}",
                "group": "one",
                "value": i,
                "baseline": 0,
                "attribution": (-1) ** i * i / 10,
                "rank": 35 - i,
            }
            for i in range(35)
        ],
        "groups": [
            {
                "name": "one",
                "attribution": -0.2,
                "probability_delta": -0.1,
                "attention": 1.0,
            }
        ],
        "diagnostics": {
            "completeness_passed": True,
            "completeness_residual": 0.0001,
            "completeness_tolerance": 0.001,
            "baseline_logit": 0.2,
            "attribution_sum": 1.3,
            "integration_steps_used": 64,
            "mean_semantic_gate": 0.5,
        },
        "metadata": {
            "tokens_used": 512,
            "tokens_before_truncation": 800,
            "semantic_truncated": True,
            "preprocessing": "training",
            "dtype": "torch.float32",
            "device": "cpu",
            "model_revision": "abc",
        },
        "warnings": ["Conditional effects only"],
    }


def test_examples_and_models_preserve_all_variants():
    examples = load_examples()
    assert len(examples) == 20
    assert len(MODELS) == len(set(MODELS.values())) == 8
    root = Path(__file__).parents[1]
    ids = {
        json.loads(p.read_text())["hub_model_id"]
        for p in (root / "configs").glob("flaird-modernbert-large-*.json")
    }
    assert set(MODELS.values()) == ids
    for i, example in enumerate(examples):
        text, html = app.choose_example(i)
        assert text == example["text"]
        assert "source metadata" in html
        assert "not the system’s prediction" in html
        assert "source metadata" in app.on_text_change(text, i)[0]
        changed = app.on_text_change(text + " edited", i)
        assert changed[0] == "" and changed[11] is None


def test_plot_values_signs_units_and_ranks(report):
    before = copy.deepcopy(report)
    probability, features, group, replacement, attention, generator = (
        presentation.charts(report)
    )
    assert list(probability.data[0].x) == [30, 70]
    top = sorted(report["features"], key=lambda r: r["rank"])[:15]
    assert list(features.data[0].x) == [r["attribution"] for r in top]
    assert list(features.data[0].marker.color) == [
        presentation.CORAL if r["attribution"] > 0 else presentation.TEAL for r in top
    ]
    assert list(group.data[0].x) == [-0.2]
    assert list(replacement.data[0].x) == [-10]
    assert list(attention.data[0].x) == [100]
    assert list(generator.data[0].y) == ["GPT", "human"]
    assert list(generator.data[0].x) == [75, 25]
    assert report == before


def test_optional_heads_and_failed_completeness(report):
    report["groups"][0]["attention"] = None
    report["prediction"]["generator_probabilities"] = None
    report["diagnostics"]["completeness_passed"] = False
    report["diagnostics"]["mean_semantic_gate"] = None
    plots = presentation.charts(report)
    assert not plots[4].data and not plots[5].data
    assert "concatenation" in plots[4].layout.annotations[0].text
    assert "single-task" in plots[5].layout.annotations[0].text
    assert "Needs review" in presentation.summary(report)
    assert "do not rely" in presentation.diagnostics(report)
    assert "truncated at 512" in presentation.summary(report)


def test_provenance_escapes_html():
    html = presentation.provenance(
        {
            "model": '<script>alert("x")</script>',
            "label": "1",
            "generator_label": "GPT",
            "attack": "",
        },
        0,
    )
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize(
    "text,policy",
    [(" ", "training"), ("a" * (MAX_CHARACTERS + 1), "training"), ("text", "invalid")],
)
def test_input_rejected_before_download(text, policy, monkeypatch):
    runtime = ModelRuntime()
    monkeypatch.setattr(
        runtime, "_load", lambda _: pytest.fail("invalid input loaded checkpoint")
    )
    with pytest.raises(ValueError):
        runtime.explain(text, next(iter(MODELS.values())), policy=policy)


def test_callback_success_and_failure_restore_controls(report, monkeypatch):
    monkeypatch.setattr(app.runtime, "explain", lambda *args: report)
    stream = list(app.analyze_text("text", "model", 0.5, "training"))
    assert len(stream) == 2 and all(len(row) == 20 for row in stream)
    assert all(not update["interactive"] for update in stream[0][:7])
    assert all(update["interactive"] for update in stream[-1][:7])
    assert stream[-1][17] == report
    assert stream[-1][18]["interactive"]

    def fail(*args):
        raise ValueError("<invalid>")

    monkeypatch.setattr(app.runtime, "explain", fail)
    failed = list(app.analyze_text("text", "model", 0.5, "training"))[-1]
    assert "&lt;invalid&gt;" in failed[7]
    assert failed[17] is None
    assert not failed[18]["interactive"] and all(v["interactive"] for v in failed[:7])


def test_export_unique_files_cleanup_and_no_text(report):
    _first, path = app.export_report(report, None)
    assert json.loads(Path(path).read_text()) == report
    assert "text" not in json.loads(Path(path).read_text())
    _second, path2 = app.export_report(report, path)
    assert path != path2 and not Path(path).exists()
    app.remove_export(path2)
    assert not Path(path2).exists()
    with pytest.raises(gr.Error):
        app.export_report(None, None)


def test_build_does_not_download_weights(monkeypatch):
    monkeypatch.setattr(
        app.runtime, "_load", lambda _: pytest.fail("import/build downloaded weights")
    )
    blocks = app.build_app()
    assert blocks.enable_queue
    assert blocks._queue.max_size == 8
