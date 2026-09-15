"""Configuration-driven FLAIRD training entry point."""

import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, set_seed

from flaird.data.data_collator import DataCollator
from flaird.data.dataset import load_flaird_dataset
from flaird.modeling import FlairdForSequenceClassification, register_auto_classes
from flaird.trainer import FlairdTrainer, compute_metrics
from flaird.utils.arguments import parse_args


def _build_model(model_args):
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if getattr(config, "model_type", None) == "flaird":
        return FlairdForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    return FlairdForSequenceClassification.from_pretrained_encoder(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        fusion_type=model_args.fusion_type,
        feature_token_dim=model_args.feature_token_dim,
        fusion_dim=model_args.fusion_dim,
        num_attention_heads=model_args.num_attention_heads,
        dropout=model_args.dropout,
        use_forensic_features=model_args.use_forensic_features,
        use_generator_head=model_args.use_generator_head,
        num_generator_labels=model_args.num_generator_labels,
        generator_loss_weight=model_args.generator_loss_weight,
        binary_class_weights=model_args.binary_class_weights,
    )


def main():
    model_args, data_args, training_args = parse_args()
    set_seed(training_args.seed)
    register_auto_classes()
    training_args.remove_unused_columns = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = _build_model(model_args)
    if model_args.freeze_encoder:
        model.freeze_encoder()
    train_dataset, eval_dataset = load_flaird_dataset(data_args)
    collator = DataCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_seq_length,
        text_column=data_args.text_column,
        feature_column=data_args.feature_column,
        use_forensic_features=model_args.use_forensic_features,
    )
    trainer = FlairdTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "parameters": total,
                "trainable_parameters": trainable,
            },
            indent=2,
        )
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    Path(training_args.output_dir, "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message="FLAIRD experiment complete")


if __name__ == "__main__":
    main()
