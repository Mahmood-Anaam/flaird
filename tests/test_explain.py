"""Offline mathematical and architecture checks; no trained weights are downloaded."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import ModernBertConfig

from flaird.modeling import FlairdConfig, FlairdForSequenceClassification
from flaird.modeling.configuration_flaird import DEFAULT_FEATURE_GROUPS
from flaird.modeling.features import FEATURE_NAMES
from flaird.utils.explain import ExplanationOptions, FlairdExplainer


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(fusion="attention", multitask=True, frozen=False):
    torch.manual_seed(42)
    encoder = ModernBertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        reference_compile=False,
        _attn_implementation="eager",
    )
    return FlairdForSequenceClassification(
        FlairdConfig(
            encoder_config=encoder,
            feature_token_dim=8,
            fusion_num_heads=2,
            fusion_type=fusion,
            use_generator_classifier=multitask,
            freeze_encoder=frozen,
        )
    ).eval()


def inputs():
    return {
        "input_ids": torch.tensor([[1, 4, 7, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "forensic_features": torch.linspace(0.1, 1, 35).unsqueeze(0),
    }


@pytest.mark.parametrize(
    "fusion,multitask,frozen",
    list(
        itertools.product(
            ["attention", "concatenation"],
            [True, False],
            [True, False],
        )
    ),
)
def test_all_eight_variants_match_forward_and_preserve_model(fusion, multitask, frozen):
    model = tiny_model(fusion, multitask, frozen)
    batch = inputs()
    with torch.no_grad():
        original = model(**batch, output_fusion_states=True)
    parameters = [(p.detach().clone(), p.requires_grad) for p in model.parameters()]
    first = next(model.parameters())
    first.grad = torch.full_like(first, 0.125)
    old_gradient = first.grad.clone()
    explainer = FlairdExplainer(model)
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder:
        with torch.no_grad():
            result = explainer.explain_inputs(**batch)
        assert encoder.call_count == 1
    assert result.prediction["machine_logit"] == pytest.approx(
        float(original.logits[0, 0]), abs=1e-7
    )
    assert result.prediction["machine_probability"] == pytest.approx(
        float(original.logits[0, 0].sigmoid())
    )
    assert result.diagnostics["completeness_passed"]
    assert result.diagnostics["completeness_residual"] == pytest.approx(
        0, abs=result.diagnostics["completeness_tolerance"]
    )
    assert sum(row["attribution"] for row in result.groups) == pytest.approx(
        result.diagnostics["attribution_sum"]
    )
    if fusion == "attention":
        assert sum(row["attention"] for row in result.groups) == pytest.approx(
            1, abs=1e-6
        )
        expected = original.fusion_states.fusion_outputs
        assert [row["attention"] for row in result.groups] == pytest.approx(
            expected.feature_attention[0].tolist()
        )
        assert result.diagnostics["mean_semantic_gate"] == pytest.approx(
            float(expected.fusion_gate[0])
        )
    else:
        assert all(row["attention"] is None for row in result.groups)
        assert result.diagnostics["mean_semantic_gate"] is None
    if multitask:
        assert [
            row["probability"] for row in result.prediction["generator_probabilities"]
        ] == pytest.approx(original.generator_logits[0].softmax(-1).tolist())
        assert result.prediction["generator_probabilities"][-1]["label"] == "human"
    else:
        assert result.prediction["generator_probabilities"] is None
    assert [row["name"] for row in result.features] == list(FEATURE_NAMES)
    assert sorted(row["rank"] for row in result.features) == list(range(1, 36))
    json.dumps(result.to_dict(), allow_nan=False)
    assert torch.equal(first.grad, old_gradient)
    for parameter, (value, requires_grad) in zip(model.parameters(), parameters):
        assert torch.equal(parameter, value)
        assert parameter.requires_grad == requires_grad
        if parameter is not first:
            assert parameter.grad is None
    assert not any(module.training for module in model.modules())
    repeat = explainer.explain_inputs(**batch)
    assert repeat.to_dict() == result.to_dict()


class LinearFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, features):
        return features


class LinearFusion(nn.Module):
    def forward(self, semantic, features, mask):
        return SimpleNamespace(
            fused_state=features, feature_attention=None, fusion_gate=None
        )


class LinearEncoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=torch.zeros(1, input_ids.shape[1], 1, dtype=torch.float64)
        )


class LinearOracle(nn.Module):
    """F(x) = bias + w.x provides exact independent IG and replacement answers."""

    def __init__(self):
        super().__init__()
        self.config = FlairdConfig(
            fusion_type="concatenation", use_generator_classifier=False
        )
        self.feature_encoder = LinearFeatureEncoder()
        self.encoder = LinearEncoder()
        self.model = SimpleNamespace(fusion=LinearFusion())
        self.head = nn.Identity()
        self.classifier = nn.Linear(35, 1).double()
        self.generator_classifier = None
        with torch.no_grad():
            self.classifier.weight.copy_(torch.linspace(-1, 1, 35).double())
            self.classifier.bias.fill_(0.3)


def test_linear_oracle_signed_ig_custom_baseline_and_group_replacement():
    model = LinearOracle().eval()
    batch = inputs()
    batch["forensic_features"] = torch.linspace(-0.7, 1.4, 35).double().unsqueeze(0)
    baseline = torch.full((35,), 0.2, dtype=torch.float64)
    result = FlairdExplainer(
        model, ExplanationOptions(integration_steps=4)
    ).explain_inputs(**batch, baseline_features=baseline)
    expected = model.classifier.weight[0].detach() * (
        batch["forensic_features"][0] - baseline
    )
    assert [row["attribution"] for row in result.features] == pytest.approx(
        expected.tolist(), abs=1e-12
    )
    assert result.diagnostics["completeness_residual"] == pytest.approx(0, abs=1e-12)
    for row in result.groups:
        assert row["logit_delta"] == pytest.approx(
            float(expected[row["feature_indices"]].sum()), abs=1e-12
        )
    assert result.metadata["baseline_kind"] == "provided"


def test_identical_baseline_has_zero_effects():
    batch = inputs()
    result = FlairdExplainer(tiny_model()).explain_inputs(
        **batch, baseline_features=batch["forensic_features"]
    )
    assert all(row["attribution"] == 0 for row in result.features)
    assert all(abs(row["logit_delta"]) < 1e-7 for row in result.groups)
    assert result.diagnostics["completeness_passed"]


def test_integration_batch_size_does_not_change_attribution():
    model = tiny_model()
    first = FlairdExplainer(
        model, ExplanationOptions(integration_batch_size=1)
    ).explain_inputs(**inputs())
    second = FlairdExplainer(
        model, ExplanationOptions(integration_batch_size=16)
    ).explain_inputs(**inputs())
    assert [row["attribution"] for row in first.features] == pytest.approx(
        [row["attribution"] for row in second.features], abs=3e-7
    )


def test_inadequate_quadrature_is_reported():
    result = FlairdExplainer(
        tiny_model(),
        ExplanationOptions(
            integration_steps=1,
            max_integration_steps=1,
            completeness_atol=0,
            completeness_rtol=0,
        ),
    ).explain_inputs(**inputs())
    assert not result.diagnostics["completeness_passed"]
    assert any("tolerance exceeded" in w for w in result.warnings)


def test_invalid_inputs_and_execution_modes():
    model = tiny_model()
    explainer = FlairdExplainer(model)
    batch = inputs()
    model.train()
    with pytest.raises(ValueError, match="eval"):
        explainer.explain_inputs(**batch)
    model.eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        explainer.explain_inputs(**batch)
    for bad in [torch.zeros(34), torch.full((35,), float("nan"))]:
        with pytest.raises(ValueError, match="finite vector"):
            explainer.explain_inputs(**{**batch, "forensic_features": bad})
        with pytest.raises(ValueError, match="finite vector"):
            explainer.explain_inputs(**batch, baseline_features=bad)
    for mask in [torch.zeros(1, 5), torch.full((1, 5), 0.5), torch.ones(1, 2)]:
        with pytest.raises(ValueError, match="attention mask"):
            explainer.explain_inputs(**{**batch, "attention_mask": mask})
    model.config.feature_group_indices[0] = (0,)
    with pytest.raises(ValueError, match="partition"):
        FlairdExplainer(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integration_steps": 0},
        {"integration_batch_size": True},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"completeness_atol": -1},
        {"completeness_rtol": float("inf")},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        ExplanationOptions(**kwargs)


class RecordingTokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        ids = [1] + [4] * len(text.split()) + [2]
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids}


@pytest.mark.parametrize("mode", ["training", "tira", "none"])
def test_text_preprocessing_and_truncation_are_explicit(mode):
    model = tiny_model()
    tokenizer = RecordingTokenizer()
    text = "Email research@example.com about the changing climate and the news report."
    with patch.object(
        model, "extract_forensic_features", return_value=inputs()["forensic_features"]
    ) as extract:
        result = FlairdExplainer(model).explain_text(
            text, tokenizer, max_length=5, preprocessing=mode
        )
    semantic_text = (
        text if mode == "none" else text.replace("research@example.com", "[EMAIL]")
    )
    assert tokenizer.texts == [semantic_text, semantic_text]
    assert extract.call_args.args[0] == [semantic_text if mode == "training" else text]
    assert result.metadata["semantic_truncated"]
    assert result.metadata["tokens_used"] == 5
    assert result.metadata["forensic_scope"] == "full_feature_text"
    assert any("truncated" in w for w in result.warnings)


def test_feature_extractor_on_all_demo_examples():
    # Verify actual extraction, including whitespace, zero-width and short input.
    import csv

    from flaird.modeling.features import ForensicFeatureExtractor

    path = Path(__file__).parents[1] / "demo" / "examples.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 20
    extractor = ForensicFeatureExtractor()
    for text in [row["text"] for row in rows] + [
        "",
        "a",
        "!",
        "A short English sentence.",
    ]:
        values = extractor(text)
        assert len(values) == 35
        assert torch.isfinite(torch.tensor(values)).all()


def test_experiment_configs_are_the_full_factorial_design():
    paths = (Path(__file__).parents[1] / "configs").glob(
        "flaird-modernbert-large-*.json"
    )
    variants = {
        tuple(
            json.loads(p.read_text())[key]
            for key in ("fusion_type", "use_generator_classifier", "freeze_encoder")
        )
        for p in paths
    }
    assert variants == set(
        itertools.product(["attention", "concatenation"], [True, False], [True, False])
    )
    assert sorted(
        i for indices in DEFAULT_FEATURE_GROUPS.values() for i in indices
    ) == list(range(35))


def test_cli_exports_local_checkpoint(tmp_path):
    from click.testing import CliRunner
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from flaird.scripts.explain import main

    checkpoint = tmp_path / "checkpoint"
    tiny_model(fusion="concatenation").save_pretrained(checkpoint)
    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(checkpoint)
    text_file = tmp_path / "input.txt"
    text_file.write_text(
        "A research team published its analysis of the new policy today."
    )
    output = tmp_path / "result.json"
    response = CliRunner().invoke(
        main,
        [
            "--model",
            str(checkpoint),
            "--text-file",
            str(text_file),
            "--output",
            str(output),
            "--max-integration-steps",
            "1024",
        ],
    )
    assert response.exit_code == 0, response.output
    report = json.loads(output.read_text())
    assert len(report["features"]) == 35
    assert report["metadata"]["requested_model"] == str(checkpoint)
    assert report["diagnostics"]["completeness_passed"]
