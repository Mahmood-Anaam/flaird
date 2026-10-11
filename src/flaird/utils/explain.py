"""Local, model-aware explanations for FLAIRD sequence predictions."""

from __future__ import annotations

from typing import Any

import torch

from flaird.modeling.features import FEATURE_NAMES


def _label(mapping: dict[Any, str], index: int, fallback: str) -> str:
    return str(mapping.get(index, mapping.get(str(index), fallback)))


def explain_text(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int = 512,
    top_k: int = 10,
) -> dict[str, Any]:
    """Predict and explain one text using the model's exposed fusion states.

    Feature attributions are signed gradient-times-input values for the machine
    logit. Attention scores describe fusion allocation, not causal importance.
    """
    if not text or not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    device = model.device
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in encoded.items()
    }
    forensic_features = model.extract_forensic_features([text], return_tensors=True)
    forensic_features = forensic_features.to(device).detach().requires_grad_(True)
    inputs["forensic_features"] = forensic_features

    was_training = model.training
    model.eval()
    try:
        outputs = model(**inputs, output_fusion_states=True)
        logits = outputs.logits[0].reshape(-1)
        if logits.numel() not in (1, 2):
            raise ValueError(
                f"Expected one or two classification logits, got {logits.numel()}"
            )

        id2label = getattr(model.config, "id2label", {})
        if logits.numel() == 1:
            machine_logit = logits[0]
            machine_score = torch.sigmoid(machine_logit)
        else:
            class_probabilities = torch.softmax(logits, dim=-1)
            machine_index = next(
                (
                    index
                    for index in range(2)
                    if _label(id2label, index, "").lower() == "machine"
                ),
                1,
            )
            machine_logit = logits[machine_index]
            machine_score = class_probabilities[machine_index]

        gradients = torch.autograd.grad(
            machine_logit,
            forensic_features,
            allow_unused=True,
        )[0]
        if gradients is None:
            gradients = torch.zeros_like(forensic_features)
        attributions = (gradients * forensic_features)[0].detach().cpu().tolist()
        raw_features = forensic_features[0].detach().cpu().tolist()

        ranked_features = [
            {
                "feature": name,
                "value": float(value),
                "attribution": float(attribution),
                "importance": abs(float(attribution)),
                "direction": "machine"
                if attribution > 0
                else "human"
                if attribution < 0
                else "neutral",
            }
            for name, value, attribution in zip(
                FEATURE_NAMES, raw_features, attributions, strict=True
            )
        ]
        ranked_features.sort(key=lambda item: item["importance"], reverse=True)

        fusion_states = outputs.fusion_states
        fusion_outputs = (
            None
            if fusion_states is None
            else getattr(fusion_states, "fusion_outputs", None)
        )
        feature_attention = (
            None
            if fusion_outputs is None
            else getattr(fusion_outputs, "feature_attention", None)
        )
        group_attention = None
        if feature_attention is not None:
            scores = feature_attention[0].detach().cpu().tolist()
            names = model.config.feature_group_names
            group_attention = [
                {"group": name, "attention": float(score)}
                for name, score in zip(names, scores, strict=True)
            ]
            group_attention.sort(key=lambda item: item["attention"], reverse=True)

        fusion_gate = None
        if fusion_outputs is not None and fusion_outputs.fusion_gate is not None:
            fusion_gate = float(fusion_outputs.fusion_gate[0].detach().cpu().item())

        generator_probabilities = None
        if outputs.generator_logits is not None:
            probabilities = (
                torch.softmax(outputs.generator_logits[0], dim=-1).detach().cpu()
            )
            id2generator_label = getattr(model.config, "id2generator_label", {})
            generator_probabilities = [
                {
                    "generator": _label(id2generator_label, index, str(index)),
                    "probability": float(probability),
                }
                for index, probability in enumerate(probabilities.tolist())
            ]
            generator_probabilities.sort(
                key=lambda item: item["probability"], reverse=True
            )

        score = float(machine_score.detach().cpu().item())
        return {
            "machine_probability": score,
            "human_probability": 1.0 - score,
            "predicted_label": "machine" if score >= 0.5 else "human",
            "generator_probabilities": generator_probabilities,
            "feature_attributions": ranked_features[:top_k],
            "all_feature_attributions": ranked_features,
            "feature_group_attention": group_attention,
            "fusion_gate": fusion_gate,
            "fusion_type": model.config.fusion_type,
            "token_count": int(inputs["attention_mask"].sum().item())
            if "attention_mask" in inputs
            else int(inputs["input_ids"].shape[-1]),
            "possibly_truncated": int(inputs["input_ids"].shape[-1]) >= max_length,
        }
    finally:
        model.train(was_training)
