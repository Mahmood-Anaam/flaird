import re
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from flaird.modeling.features import ForensicFeatureExtractor


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    text_column: str = "text"
    feature_column: str | None = None
    feature_extractor: ForensicFeatureExtractor | None = None
    use_forensic_features: bool = True
    apply_text_preprocessing: bool = True
    include_labels: bool = True

    def __post_init__(self):
        self.feature_extractor = self.feature_extractor or ForensicFeatureExtractor()

    def __call__(self, examples):

        texts = [
            preprocess_text(str(example[self.text_column]))
            if self.apply_text_preprocessing
            else str(example[self.text_column])
            for example in examples
        ]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = None
        if self.use_forensic_features:
            features = []
            for example in examples:
                stored = example.get(self.feature_column) if self.feature_column else None
                features.append(
                    stored
                    if stored is not None
                    else self.feature_extractor(str(example[self.text_column]))
                )
        if features is not None:
            batch["forensic_features"] = torch.tensor(features, dtype=torch.float32)
        if self.include_labels:
            binary_labels = [int(example["label"]) for example in examples]
            generator_labels = [int(example["generator_label"]) for example in examples]
            batch["labels"] = torch.tensor(binary_labels, dtype=torch.long)
            batch["generator_labels"] = torch.tensor(generator_labels, dtype=torch.long)
        return batch


def preprocess_text(text: str) -> str:
    text = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"@[A-Za-z0-9_-]+", "[USER]", text)
    return re.sub(r"(?<!\w)(?:\+?\d[\d(). *-]{5,}\d)(?!\w)", " [PHONE]", text)
