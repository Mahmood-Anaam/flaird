"""Configuration-driven FLAIRD training entry point."""

import json
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from flaird.data.data_collator import DataCollator
from flaird.data.dataset import load_flaird_dataset
from flaird.modeling import FlairdForSequenceClassification
from flaird.trainer import FlairdTrainer, compute_metrics
from flaird.utils.arguments import parse_args


def _print_summary(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | fusion_type={model.config.fusion_type}")
    print(f"  total params:     {total:,}")
    print(f"  trainable params: {trainable:,} ({100 * trainable / max(total, 1):.1f}%)")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"    - {name}: {n:,} params")


def main():
    model_args, data_args, training_args = parse_args()
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.model_name_or_path is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )

    else:
        model = FlairdForSequenceClassification.from_pretrained_encoder(
            model_args.encoder_name_or_path,
            fusion_type=model_args.fusion_type,
            use_generator_classifier=model_args.use_generator_classifier,
            num_generator_labels=model_args.num_generator_labels,
            generator_loss_weight=model_args.generator_loss_weight,
            pos_weight=model_args.pos_weight,
            generator_class_weights=model_args.generator_class_weights,
            freeze_encoder=model_args.freeze_encoder,
        )

    if model_args.freeze_encoder:
        model.freeze_encoder()
    train_dataset, eval_dataset = load_flaird_dataset(data_args)
    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_seq_length,
        text_column=data_args.text_column,
        feature_column=data_args.feature_column,
        apply_text_preprocessing=False,
    )

    print("train_dataset\n", train_dataset)
    print("eval_dataset\n", eval_dataset)
    _print_summary(model)
    print("Training arguments:\n", training_args)

    trainer = FlairdTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("Evaluating the model on the validation dataset...")
    metrics = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    Path(training_args.output_dir, "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if training_args.push_to_hub:
        print("Pushing the model to the Hugging Face Hub...")
        trainer.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()
