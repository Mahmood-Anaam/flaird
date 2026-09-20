"""Local, model-grounded explanations for FLAIRD predictions.

The module deliberately reports evidence rather than a causal verdict: feature and
word scores are gradient × input sensitivities for the current prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from flaird.modeling.configuration_flaird import DEFAULT_FEATURE_GROUPS
from flaird.modeling.features import FEATURE_NAMES


@dataclass
class Explanation:
    """Serializable evidence associated with one binary prediction."""

    machine_probability: float
    predicted_label: str
    confidence: float
    generator_probabilities: dict[str, float]
    fusion_gate: float | None
    feature_attention: dict[str, float] | None
    feature_contributions: dict[str, float]
    token_contributions: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1))
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(f"Expected one or two binary logits, got {tuple(logits.shape)}.")


def _label(mapping: Any, index: int) -> str:
    return str(mapping.get(index, mapping.get(str(index), f"class_{index}")))


class FlairdExplainer:
    """Compute token and forensic evidence from a loaded FLAIRD model.

    A single backward pass obtains gradient × input scores for all feature groups
    and input tokens, keeping interactive explanations practical on a Space.
    """

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def explain(self, text: str, *, preprocess=None, top_k: int = 12) -> Explanation:
        if not text or not text.strip():
            raise ValueError("Enter some text before requesting an analysis.")
        prepared_text = preprocess(text) if preprocess else text.strip()
        device = self.model.device
        encoded = self.tokenizer(
            prepared_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        features = (
            self.model.extract_forensic_features([prepared_text])
            .detach()
            .requires_grad_(True)
        )

        embedding_layer = self.model.encoder.get_input_embeddings()
        captured: list[torch.Tensor] = []

        def capture_embeddings(_module, _inputs, output):
            output.retain_grad()
            captured.append(output)

        handle = embedding_layer.register_forward_hook(capture_embeddings)
        try:
            self.model.zero_grad(set_to_none=True)
            output = self.model(
                **encoded,
                forensic_features=features,
                output_fusion_states=True,
            )
            binary_logit = (
                output.logits.reshape(-1)[0]
                if output.logits.shape[-1] == 1
                else output.logits[0, 1]
            )
            binary_logit.backward()
        finally:
            handle.remove()

        probability = float(_probability(output.logits).item())
        feature_scores = (features.grad[0] * features.detach()[0]).detach().cpu()
        feature_contributions = {
            name: float(feature_scores[list(indices)].sum())
            for name, indices in DEFAULT_FEATURE_GROUPS.items()
        }

        token_contributions: list[dict[str, float | str]] = []
        if captured and captured[0].grad is not None:
            scores = (captured[0].grad[0] * captured[0].detach()[0]).sum(dim=-1).cpu()
            token_ids = encoded["input_ids"][0].detach().cpu().tolist()
            mask = (
                encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0]
                .bool()
                .cpu()
            )
            special = set(self.tokenizer.all_special_ids)
            token_contributions = [
                {
                    "token": self.tokenizer.convert_ids_to_tokens(token_id),
                    "score": float(score),
                }
                for token_id, score, valid in zip(
                    token_ids, scores.tolist(), mask.tolist(), strict=True
                )
                if valid and token_id not in special
            ]
            token_contributions.sort(
                key=lambda item: abs(float(item["score"])), reverse=True
            )
            token_contributions = token_contributions[:top_k]

        fusion = output.fusion_states.fusion_outputs if output.fusion_states else None
        attention = None
        if fusion is not None and fusion.feature_attention is not None:
            attention = {
                name: float(value)
                for name, value in zip(
                    self.model.config.feature_group_names,
                    fusion.feature_attention[0].detach().cpu(),
                    strict=True,
                )
            }
        gate = (
            float(fusion.fusion_gate[0])
            if fusion is not None and fusion.fusion_gate is not None
            else None
        )

        generator_probabilities: dict[str, float] = {}
        if output.generator_logits is not None:
            for index, value in enumerate(
                torch.softmax(output.generator_logits[0], dim=-1)
                .detach()
                .cpu()
                .tolist()
            ):
                generator_probabilities[
                    _label(self.model.config.id2generator_label, index)
                ] = float(value)

        return Explanation(
            machine_probability=probability,
            predicted_label=(
                "machine-generated" if probability >= 0.5 else "human-written"
            ),
            confidence=max(probability, 1.0 - probability),
            generator_probabilities=generator_probabilities,
            fusion_gate=gate,
            feature_attention=attention,
            feature_contributions=feature_contributions,
            token_contributions=token_contributions,
        )


def feature_values(text: str, model: Any) -> dict[str, float]:
    """Return the named raw forensic indicators for a text."""
    values = model.extract_forensic_features_dict([text])[0]
    return {name: float(values[name]) for name in FEATURE_NAMES}
