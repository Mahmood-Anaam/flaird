"""Forensic explanation utilities for FLAIRD models."""

from typing import Any

import numpy as np
import torch

from flaird.data.data_collator import DataCollator
from flaird.modeling.features import FEATURE_NAMES, ForensicFeatureExtractor


class FlairdExplainer:
    """Utility to generate detailed forensic interpretations and explanations for FLAIRD predictions."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = ForensicFeatureExtractor()
        self.data_collator = DataCollator(tokenizer, include_labels=False)

    def explain(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        """Generate a complete diagnostic explanation for a single input text."""

        # 1. Extract raw features
        raw_features_dict = self.feature_extractor.extract_dict(text)
        features_vector = [raw_features_dict[name] for name in FEATURE_NAMES]

        # 2. Tokenize and run model inference with output_fusion_states=True
        batch = self.data_collator([{"text": text}])
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**batch, output_fusion_states=True)

        logits = outputs.logits.cpu().numpy().reshape(-1)
        prob = float(1.0 / (1.0 + np.exp(-logits[0])))
        verdict = "machine" if prob >= threshold else "human"

        # 3. Handle Generator Classification (Multi-task)
        generator_info = None
        if outputs.generator_logits is not None:
            gen_logits = outputs.generator_logits.cpu().numpy().reshape(-1)
            gen_exp = np.exp(gen_logits - np.max(gen_logits))
            gen_probs = gen_exp / np.sum(gen_exp)

            id2gen = getattr(self.model.config, "id2generator_label", None) or {
                0: "GPT",
                1: "Meta-LLaMA",
                2: "MPT",
                3: "Cohere",
                4: "Mistral",
                5: "Gemini",
                6: "DeepSeek",
                7: "Falcon",
                8: "Bison",
                9: "Qwen",
                10: "human",
            }
            # Convert keys if integer strings or ints
            id2gen = {int(k): str(v) for k, v in id2gen.items()}

            gen_prob_dict = {
                id2gen.get(i, f"Generator_{i}"): float(gen_probs[i])
                for i in range(len(gen_probs))
            }

            top_gen_idx = int(np.argmax(gen_probs))
            top_gen_label = id2gen.get(top_gen_idx, f"Generator_{top_gen_idx}")
            top_gen_prob = float(gen_probs[top_gen_idx])

            generator_info = {
                "top_generator": top_gen_label,
                "top_generator_probability": top_gen_prob,
                "generator_probabilities": gen_prob_dict,
            }

        # 4. Feature Group Importance & Attention Analysis
        feature_group_names = getattr(
            self.model.config,
            "feature_group_names",
            [
                "lexical_diversity",
                "function_word_style",
                "surface_composition",
                "structural_organization",
                "punctuation_and_rhetoric",
                "information_entropy",
                "repetition_and_burstiness",
            ],
        )

        feature_group_indices = getattr(
            self.model.config,
            "feature_group_indices",
            [
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                (10, 11, 12, 13, 14, 15, 16),
                (17, 21, 22, 23, 24),
                (18, 27, 28, 34),
                (19, 20, 31, 32, 33),
                (25, 26),
                (29, 30),
            ],
        )

        group_attention_dict = {}
        fusion_states = outputs.fusion_states
        has_attn = False

        if (
            fusion_states is not None
            and hasattr(fusion_states, "fusion_outputs")
            and fusion_states.fusion_outputs is not None
            and fusion_states.fusion_outputs.feature_attention is not None
        ):
            has_attn = True
            attn_weights = (
                fusion_states.fusion_outputs.feature_attention.cpu().numpy().reshape(-1)
            )
            # Normalize attention weights if needed
            if np.sum(attn_weights) > 0:
                attn_weights = attn_weights / np.sum(attn_weights)
            for name, weight in zip(feature_group_names, attn_weights):
                group_attention_dict[name] = float(weight)
        else:
            # Fallback for concatenation fusion or missing attention states:
            # Calculate group norms from feature_states if available
            if (
                fusion_states is not None
                and hasattr(fusion_states, "feature_states")
                and fusion_states.feature_states is not None
            ):
                f_states = fusion_states.feature_states.cpu().numpy()[0]  # [num_groups, dim]
                norms = np.linalg.norm(f_states, axis=-1)
                total_norm = np.sum(norms) if np.sum(norms) > 0 else 1.0
                normalized_norms = norms / total_norm
                for name, norm in zip(feature_group_names, normalized_norms):
                    group_attention_dict[name] = float(norm)
            else:
                # Equal weights fallback
                for name in feature_group_names:
                    group_attention_dict[name] = float(1.0 / len(feature_group_names))

        # 5. Build Group Detailed Breakdown
        groups_detail = {}
        for group_name, group_idx_tuple in zip(
            feature_group_names, feature_group_indices, strict=False
        ):
            group_features = {
                FEATURE_NAMES[idx]: raw_features_dict[FEATURE_NAMES[idx]]
                for idx in group_idx_tuple
                if idx < len(FEATURE_NAMES)
            }
            groups_detail[group_name] = {
                "importance_weight": group_attention_dict.get(group_name, 0.0),
                "features": group_features,
            }

        # 6. Generate Human-Readable Narrative Summary
        sorted_groups = sorted(
            group_attention_dict.items(), key=lambda x: x[1], reverse=True
        )
        top_groups = [g[0].replace("_", " ").title() for g in sorted_groups[:3]]

        if verdict == "machine":
            summary = (
                f"The text exhibits characteristic machine-generated stylistic patterns with a probability of {prob:.2%}. "
                f"The classification is predominantly influenced by key forensic domains: {', '.join(top_groups)}. "
            )
            if generator_info:
                summary += f"The multi-task generator detector identifies the primary candidate model as '{generator_info['top_generator']}' ({generator_info['top_generator_probability']:.2%} likelihood)."
        else:
            summary = (
                f"The text demonstrates natural human authorship patterns with a machine probability of only {prob:.2%}. "
                f"Forensic emphasis was observed across: {', '.join(top_groups)}."
            )

        return {
            "verdict": verdict,
            "machine_probability": prob,
            "human_probability": 1.0 - prob,
            "top_feature_groups": sorted_groups,
            "feature_group_importance": group_attention_dict,
            "feature_group_details": groups_detail,
            "raw_features": raw_features_dict,
            "generator_analysis": generator_info,
            "narrative_summary": summary,
            "has_attention_weights": has_attn,
        }


def explain_text(model, tokenizer, text: str) -> dict[str, Any]:
    """Functional shortcut for quick explanation generation."""
    explainer = FlairdExplainer(model, tokenizer)
    return explainer.explain(text)
