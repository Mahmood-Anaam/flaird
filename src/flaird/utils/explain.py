"""Checkpoint-compatible, local explanations of FLAIRD's binary machine logit.

Attention and gate values are diagnostics. Integrated gradients and group
replacement measure feature effects conditional on an unchanged semantic input.
No encoder gradients, checkpoint changes, or retraining are required.
"""

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

from flaird.data.data_collator import preprocess_text
from flaird.modeling.features import FEATURE_NAMES


@dataclass(frozen=True)
class ExplanationOptions:
    """Bounded numerical settings; tolerances apply to machine-logit units."""

    integration_steps: int = 64
    integration_batch_size: int = 8
    max_integration_steps: int = 1024
    completeness_atol: float = 0.001
    completeness_rtol: float = 0.01
    threshold: float = 0.5

    def __post_init__(self):
        for name in (
            "integration_steps",
            "integration_batch_size",
            "max_integration_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_integration_steps < self.integration_steps:
            raise ValueError("max_integration_steps must be at least integration_steps")
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must be finite and strictly between 0 and 1")
        for name in ("completeness_atol", "completeness_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class ExplanationResult:
    """JSON-safe chart/table data. Signed effects always target the machine logit."""

    prediction: dict
    features: list[dict]
    groups: list[dict]
    diagnostics: dict
    metadata: dict
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class FlairdExplainer:
    """Explain one example at a time using the loaded model's own modules.

    The caller owns device placement and must call ``model.eval()``. Existing
    parameter gradients and train/eval flags are never changed. ``no_grad`` is
    supported; ``inference_mode`` is rejected because attribution needs autograd.
    """

    def __init__(self, model, options: ExplanationOptions | None = None):
        self.model = model
        self.options = options or ExplanationOptions()
        config = model.config
        if (
            config.fusion_type not in ("attention", "concatenation")
            or config.num_labels != 1
        ):
            raise ValueError(
                "Explanations require a FLAIRD model with one binary logit"
            )
        self.group_names = tuple(config.feature_group_names)
        self.group_indices = tuple(tuple(g) for g in config.feature_group_indices)
        indices = [i for group in self.group_indices for i in group]
        if (
            config.forensic_feature_dim != len(FEATURE_NAMES)
            or len(self.group_names) != len(self.group_indices)
            or len(set(self.group_names)) != len(self.group_names)
            or any(not group for group in self.group_indices)
            or sorted(indices) != list(range(len(FEATURE_NAMES)))
        ):
            raise ValueError(
                "Feature groups must partition the 35 ordered FLAIRD features"
            )
        self.feature_groups = {
            index: name
            for name, group in zip(self.group_names, self.group_indices, strict=True)
            for index in group
        }

    def _vector(self, value, reference: torch.Tensor, name: str) -> torch.Tensor:
        vector = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if vector.shape == (len(FEATURE_NAMES),):
            vector = vector.unsqueeze(0)
        if vector.shape != (1, len(FEATURE_NAMES)) or not torch.isfinite(vector).all():
            raise ValueError(
                f"{name} must be a finite vector of 35 features for one example"
            )
        return vector.detach().clone()

    def _tail(self, semantic_states, attention_mask, features):
        """Reuse the exact trained feature, fusion, and classification modules."""
        batch_size = features.shape[0]
        fusion = self.model.model.fusion(
            semantic_states.expand(batch_size, -1, -1),
            self.model.feature_encoder(features),
            attention_mask.expand(batch_size, -1),
        )
        head = self.model.head(fusion.fused_state)
        logits = self.model.classifier(head).reshape(-1)
        generator = (
            self.model.generator_classifier(head)
            if self.model.generator_classifier is not None
            else None
        )
        return logits, generator, fusion

    def explain_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        forensic_features: torch.Tensor,
        *,
        baseline_features=None,
    ) -> ExplanationResult:
        """Explain the exact supplied tensors; no preprocessing is performed.

        A zero raw-feature baseline is the default numerical reference, not a
        typical human document. A caller may supply an empirical training-only
        reference vector. The semantic representation stays fixed throughout.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Call the explainer outside torch.inference_mode(); no_grad is supported"
            )
        if any(module.training for module in self.model.modules()):
            raise ValueError(
                "Call model.eval() before generating deterministic explanations"
            )
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] == 0
            or attention_mask.shape != input_ids.shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
            or not attention_mask.any()
        ):
            raise ValueError(
                "Expected one nonempty token sequence with a matching binary attention mask"
            )
        reference = next(self.model.feature_encoder.parameters())
        input_ids = input_ids.to(reference.device)
        attention_mask = attention_mask.to(reference.device)
        features = self._vector(forensic_features, reference, "forensic_features")
        baseline = (
            torch.zeros_like(features)
            if baseline_features is None
            else self._vector(baseline_features, reference, "baseline_features")
        )
        with torch.no_grad():
            # ModernBERT is evaluated once, including when its parameters are frozen.
            semantic = self.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.detach()
            logits, generator, fusion = self._tail(semantic, attention_mask, features)
            baseline_logits, _, _ = self._tail(semantic, attention_mask, baseline)
            replacements = features.expand(len(self.group_names), -1).clone()
            for row, indices in enumerate(self.group_indices):
                replacements[row, list(indices)] = baseline[0, list(indices)]
            replacement_logits, _, _ = self._tail(
                semantic, attention_mask, replacements
            )

        logit = float(logits[0])
        baseline_logit = float(baseline_logits[0])
        score = float(logits[0].float().sigmoid())
        tolerance = (
            self.options.completeness_atol
            + self.options.completeness_rtol * abs(logit - baseline_logit)
        )
        delta = features - baseline
        steps = self.options.integration_steps
        convergence = []
        while True:
            # Interior nodes avoid endpoint derivatives of sign(x)*log1p(abs(x)).
            nodes, weights = np.polynomial.legendre.leggauss(steps)
            nodes = torch.tensor(
                (nodes + 1) / 2, device=reference.device, dtype=reference.dtype
            )
            weights = torch.tensor(
                weights / 2, device=reference.device, dtype=torch.float64
            )
            integral = torch.zeros_like(features, dtype=torch.float64)
            with torch.enable_grad():
                for start in range(0, len(nodes), self.options.integration_batch_size):
                    stop = start + self.options.integration_batch_size
                    path = (
                        (baseline + nodes[start:stop, None] * delta)
                        .detach()
                        .requires_grad_(True)
                    )
                    path_logits, _, _ = self._tail(semantic, attention_mask, path)
                    gradient = torch.autograd.grad(
                        path_logits.sum(), path, only_inputs=True
                    )[0]
                    integral += (gradient.double() * weights[start:stop, None]).sum(
                        0, keepdim=True
                    )
            attributions = (delta.double() * integral)[0].detach()
            if not all(
                torch.isfinite(tensor).all()
                for tensor in (
                    logits,
                    baseline_logits,
                    replacement_logits,
                    attributions,
                )
            ):
                raise RuntimeError(
                    "Nonfinite model outputs or attributions; check checkpoint, inputs, and precision"
                )
            residual = logit - baseline_logit - float(attributions.sum())
            convergence.append({"steps": steps, "residual": residual})
            if (
                abs(residual) <= tolerance
                or steps >= self.options.max_integration_steps
            ):
                break
            steps = min(steps * 2, self.options.max_integration_steps)
        warnings = [
            "Feature effects are conditional on a fixed semantic representation and the selected baseline.",
            "A machine score does not establish factual falsity, intent, or a coordinated disinformation campaign.",
        ]
        if baseline_features is None:
            warnings.append(
                "The zero-feature baseline is a numerical reference and may be outside the data distribution."
            )
        if abs(residual) > tolerance:
            warnings.append(
                "Integrated-gradients completeness tolerance exceeded; increase max_integration_steps and inspect baseline/precision."
            )
        if reference.dtype != torch.float32 and reference.dtype != torch.float64:
            warnings.append(
                "Reduced precision can degrade attribution accuracy; prefer float32 for analysis."
            )

        attention = fusion.feature_attention
        gate = fusion.fusion_gate
        if attention is not None and not torch.isfinite(attention).all():
            raise RuntimeError("Nonfinite fusion attention")
        if gate is not None and not torch.isfinite(gate).all():
            raise RuntimeError("Nonfinite fusion gate")
        feature_rows = [
            {
                "index": i,
                "name": name,
                "group": self.feature_groups[i],
                "value": float(features[0, i]),
                "baseline": float(baseline[0, i]),
                "attribution": float(attributions[i]),
            }
            for i, name in enumerate(FEATURE_NAMES)
        ]
        # Rank magnitudes without discarding the signed contribution or original index.
        for rank, row in enumerate(
            sorted(feature_rows, key=lambda row: -abs(row["attribution"])), 1
        ):
            row["rank"] = rank
        group_rows = [
            {
                "name": name,
                "feature_indices": list(indices),
                "attribution": float(attributions[list(indices)].sum()),
                "attention": float(attention[0, i]) if attention is not None else None,
                "replacement_logit": float(replacement_logits[i]),
                "logit_delta": logit - float(replacement_logits[i]),
                "probability_delta": score
                - float(replacement_logits[i].float().sigmoid()),
            }
            for i, (name, indices) in enumerate(
                zip(self.group_names, self.group_indices, strict=True)
            )
        ]
        generator_scores = None
        if generator is not None:
            if not torch.isfinite(generator).all():
                raise RuntimeError("Nonfinite generator logits")
            probs = generator[0].float().softmax(-1).tolist()
            labels = self.model.config.id2generator_label
            generator_scores = [
                {
                    "index": i,
                    "label": labels.get(i, labels.get(str(i), str(i))),
                    "probability": prob,
                }
                for i, prob in enumerate(probs)
            ]
            warnings.append(
                "Generator scores are an independent auxiliary distribution, including human; they are not calibrated identity evidence."
            )
        if attention is not None:
            warnings.append(
                "Group attention and the mean semantic gate are internal diagnostics, not causal importance or branch contribution percentages."
            )
        return ExplanationResult(
            prediction={
                "machine_logit": logit,
                "machine_probability": score,
                "human_probability": 1 - score,
                "threshold": self.options.threshold,
                "label": "machine" if score > self.options.threshold else "human",
                "generator_probabilities": generator_scores,
            },
            features=feature_rows,
            groups=group_rows,
            diagnostics={
                "baseline_logit": baseline_logit,
                "baseline_machine_probability": float(
                    baseline_logits[0].float().sigmoid()
                ),
                "attribution_sum": float(attributions.sum()),
                "completeness_residual": residual,
                "completeness_tolerance": tolerance,
                "completeness_passed": abs(residual) <= tolerance,
                "integration_steps_used": steps,
                "convergence": convergence,
                "mean_semantic_gate": float(gate[0]) if gate is not None else None,
            },
            metadata={
                "schema_version": 1,
                "target": "machine_logit",
                "method": "integrated_gradients",
                "quadrature": "gauss_legendre",
                "baseline_kind": "zero" if baseline_features is None else "provided",
                "fusion_type": self.model.config.fusion_type,
                "model_id": getattr(self.model.config, "_name_or_path", ""),
                "model_revision": getattr(self.model.config, "_commit_hash", None),
                "dtype": str(reference.dtype),
                "device": str(reference.device),
                "options": asdict(self.options),
                "tokens_used": int(attention_mask.sum()),
            },
            warnings=warnings,
        )

    def explain_text(
        self,
        text: str,
        tokenizer,
        *,
        max_length: int = 512,
        preprocessing: Literal["training", "tira", "none"] = "training",
        baseline_features=None,
    ) -> ExplanationResult:
        """Prepare both branches explicitly and record semantic truncation.

        training: normalize both branches as in the dataset notebook/demo.
        tira: normalize tokens but extract raw features as in TIRA's collator.
        none: raw tokens/features, e.g. already-normalized RAID feature text.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide nonempty English text")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 2
        ):
            raise ValueError("max_length must be an integer of at least 2")
        if preprocessing not in ("training", "tira", "none"):
            raise ValueError("preprocessing must be training, tira, or none")
        semantic_text = preprocess_text(text) if preprocessing != "none" else text
        feature_text = semantic_text if preprocessing == "training" else text
        encoded = tokenizer(
            semantic_text, return_tensors="pt", truncation=True, max_length=max_length
        )
        total_tokens = len(tokenizer(semantic_text, truncation=False)["input_ids"])
        features = self.model.extract_forensic_features(
            [feature_text], return_tensors=True
        )
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
        result = self.explain_inputs(
            encoded["input_ids"], mask, features, baseline_features=baseline_features
        )
        result.metadata.update(
            {
                "preprocessing": preprocessing,
                "max_length": max_length,
                "tokens_before_truncation": total_tokens,
                "semantic_truncated": total_tokens > encoded["input_ids"].shape[1],
                "forensic_scope": "full_feature_text",
                "semantic_characters": len(semantic_text),
                "forensic_characters": len(feature_text),
            }
        )
        if result.metadata["semantic_truncated"]:
            result.warnings.append(
                "Semantic tokens were truncated; forensic features still describe the full feature text."
            )
        return result
