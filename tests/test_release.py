"""Offline regression checks for the release's UI and documented numerical contracts.

Run with PYTHONPATH=src:demo python -m pytest tests. No Hub weights are downloaded.
"""

import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import ModernBertConfig

from flaird.modeling import FlairdConfig, FlairdForSequenceClassification
from flaird.modeling.features import FEATURE_NAMES, ForensicFeatureExtractor
from flaird.utils.explain import ExplainConfig, FlairdExplainer, HierarchicalAttributor

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class Tokenizer:
    """Local tensor fixture; feature extraction is isolated in forward equivalence tests."""

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [1, 4, 7, 2]
        if return_tensors:
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        return {"input_ids": ids}


@pytest.mark.parametrize("fusion", ["attention", "concatenation"])
@pytest.mark.parametrize("pooling", ["mean", "cls"])
@pytest.mark.parametrize("generator", [True, False])
def test_cached_game_matches_real_model_and_exact_accounting(
    fusion, pooling, generator
):
    torch.manual_seed(42)
    encoder = ModernBertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=20,
        max_position_embeddings=32,
        global_attn_every_n_layers=3,
        pad_token_id=0,
    )
    encoder._attn_implementation = "eager"
    config = FlairdConfig(
        encoder_config=encoder,
        fusion_type=fusion,
        classifier_pooling=pooling,
        feature_token_dim=8,
        fusion_num_heads=2,
        use_generator_classifier=generator,
        freeze_encoder=False,
    )
    model = FlairdForSequenceClassification(config).eval()
    model.extract_forensic_features = Mock(
        return_value=torch.linspace(0.1, 3.5, 35).reshape(1, -1)
    )
    encoder_spy = Mock(wraps=model.model.encoder.forward)
    model.model.encoder.forward = encoder_spy
    result = FlairdExplainer(
        model, Tokenizer(), ExplainConfig(chunk_size=4096)
    ).explain(" ".join(["evidence"] * 40))
    assert encoder_spy.call_count == 1
    assert result.attribution.num_evaluations == 78592
    assert result.checks.value_function_error < 1e-5
    assert result.checks.efficiency_error < 1e-6
    assert result.checks.group_efficiency_error < 1e-6
    assert result.checks.owen_group_consistency_error < 1e-6
    assert (result.balance.gate is not None) == (fusion == "attention")
    assert (result.generator is not None) == generator
    if generator:
        assert result.checks.generator_value_function_error < 1e-5
        assert result.checks.generator_efficiency_error < 1e-6
    assert len(result.features) == 35
    json.loads(result.to_json())


def test_owen_matches_independent_group_contiguous_order_enumeration():
    groups = [[0, 1], [2], [3]]
    engine = HierarchicalAttributor(groups, 4)

    # Cross-group interaction plus unequal within-group interaction.
    def value(bits):
        x = np.asarray(bits, dtype=float)
        return (
            x @ np.array([1.0, -2.0, 0.5, 0.0])
            + 4 * x[..., 0] * x[..., 2]
            + 3 * x[..., 0] * x[..., 1]
        )

    game = SimpleNamespace(evaluate=lambda bits: value(bits).reshape(-1, 1))
    exact = engine.exact(game)
    samples = []
    for order in itertools.permutations(range(len(groups))):
        for inner in itertools.product(
            *(list(itertools.permutations(groups[k])) for k in order)
        ):
            bits = np.zeros(4)
            phi = np.zeros(4)
            for i in itertools.chain.from_iterable(inner):
                previous = value(bits)
                bits[i] = 1
                phi[i] = value(bits) - previous
            samples.append(phi)
    np.testing.assert_allclose(
        exact.phi_features[:, 0], np.mean(samples, axis=0), atol=1e-12
    )
    # Fourth feature is a null player even in the presence of interactions.
    assert exact.phi_features[3, 0] == 0
    sampled = engine.permutation(game, 512, 13)
    assert (
        abs(sampled.phi_features.sum() - (sampled.v_full - sampled.v_empty).item())
        < 1e-12
    )
    np.testing.assert_allclose(sampled.phi_groups, exact.phi_groups)
    assert sampled.stderr_features is not None


def test_documented_feature_denominators_and_order():
    extractor = ForensicFeatureExtractor()
    values = extractor.extract_dict("cat cat dog bird bird")
    assert list(values) == list(FEATURE_NAMES)
    assert len(extractor("")) == 35
    assert all(math.isfinite(x) for x in extractor(""))
    for key, expected in {
        "ttr": 0.6,
        "hapax_ratio": 0.2,
        "dis_hapax_ratio": 0.4,
        "sichel_s": 2 / 3,
        "yule_k": 1600,
        "yule_i": 2.25,
    }.items():
        assert values[key] == pytest.approx(expected, abs=1e-5)
    values = extractor.extract_dict("Cat, dog bird fish.")
    assert values["punctuation_density"] == pytest.approx(50)
    assert values["uppercase_ratio"] == pytest.approx(1 / 14)
    assert extractor.extract_dict("a b a b")["repeated_bigram_ratio"] == 0.5


def test_ui_events_hide_progress_and_clear_complete_view():
    from app import build_app
    from ui.controller import DashboardController, OUTPUT_FIELDS, empty_view

    controller = DashboardController(examples_path=ROOT / "demo/examples.csv")
    app = build_app(controller)
    events = app.config["dependencies"]
    assert len(events) == 3
    assert all(event["show_progress"] == "hidden" for event in events)
    assert all(not fn.tracks_progress for fn in app.fns.values())
    assert len(empty_view().as_tuple()) == len(OUTPUT_FIELDS)
    clear_fn = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "<lambda>")
    cleared = clear_fn()
    assert cleared[:3] == ("", None, "")
    assert len(cleared) == 3 + len(OUTPUT_FIELDS)
    text, chip = controller.load_example(controller.example_choices()[0])
    assert text and "Ground truth:" in chip
    assert controller.analyze("", next(iter(controller.models)), "Exact (recommended)")[
        1
    ]


def test_documented_tira_tables_match_every_archived_record():
    report = (ROOT / "docs/EVALUATION_RESULTS.md").read_text()
    for path in (ROOT / "docs/tira-evaluation").glob("*.json"):
        name = path.stem.replace("flaird-modernbert-large-", "")
        for row in json.loads(path.read_text()):
            expected = " | ".join(
                f"{row[k]:.3f}"
                for k in ["Roc-Auc", "Brier", "C@1", "F1", "F05U", "Mean"]
            )
            assert (
                f"[{name}](tira-evaluation/{path.name}) | {expected} | {row['Run']}"
                in report
            )


def test_documented_raid_slices_match_archived_values():
    report = (ROOT / "docs/EVALUATION_RESULTS.md").read_text()
    path = (
        ROOT
        / "docs/raid-evaluation/flaird-modernbert-large-attention-multitask/results.json"
    )
    for row in json.loads(path.read_text())["scores"]:
        if all(
            row[k] == "all"
            for k in ["domain", "model", "decoding", "repetition_penalty"]
        ):
            count = row["accuracy"]["0.05"]["tp"] + row["accuracy"]["0.05"]["fn"]
            expected = f"| {row['attack']} | {count:,} | {row['auroc']:.6f} | {row['accuracy']['0.05']['accuracy']:.6f} | {row['accuracy']['0.01']['accuracy']:.6f} |"
            assert expected in report
