"""Regenerate/check documentation tables against committed research artifacts.

Run from any directory: python docs/tools/research_tables.py [--check]
No network access, model loading, training, or benchmark submission is performed.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AXES = ("domain", "model", "decoding", "repetition_penalty", "attack")
PAN26 = "pan26-generative-ai-detection-20260507-test"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def collect():
    records = []
    feature_source = ROOT / "src/flaird/modeling/features.py"

    def feature_names(source):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FEATURE_NAMES"
                for t in node.targets
            ):
                return list(ast.literal_eval(node.value))
        raise AssertionError("Missing canonical feature order")

    names = feature_names(feature_source.read_text())
    notebook = json.loads(
        (ROOT / "notebooks/flaird_dataset_preparation.ipynb").read_text()
    )
    feature_cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "FEATURE_NAMES =" in "".join(c["source"])
    )
    assert names == feature_names(feature_cell) and len(names) == 35
    architecture = (ROOT / "docs/MODEL_ARCHITECTURE.md").read_text()
    assert all(f"| {i} | `{name}`" in architecture for i, name in enumerate(names)), (
        "Documented feature order differs"
    )
    inputs = sorted((ROOT / "configs").glob("flaird-modernbert-large-*.json"))
    inputs += sorted((ROOT / "docs/tira-evaluation").glob("*.json"))
    raid_path = (
        ROOT
        / "docs/raid-evaluation/flaird-modernbert-large-attention-multitask/results.json"
    )
    inputs += [
        ROOT / "src/flaird/modeling/features.py",
        ROOT / "src/flaird/modeling/configuration_flaird.py",
        ROOT / "src/flaird/data/data_collator.py",
        ROOT / "src/flaird/scripts/evaluator.py",
        raid_path,
        raid_path.with_name("metadata.json"),
        ROOT / "notebooks/flaird_dataset_preparation.ipynb",
        ROOT / "notebooks/flaird_training.ipynb",
    ]
    for path in sorted((ROOT / "configs").glob("flaird-modernbert-large-*.json")):
        config = json.loads(path.read_text())
        results = json.loads((ROOT / "docs/tira-evaluation" / path.name).read_text())
        assert len(results) == 4
        pan = next(r for r in results if r["Dataset"] == PAN26)
        records.append(
            {"name": path.stem, "config": config, "evaluations": results, "pan26": pan}
        )
    assert len(records) == 8
    # Read saved notebook outputs, preserving the model's family-label order.
    outputs = [
        "".join(o.get("data", {}).get("text/plain", []))
        for cell in notebook["cells"]
        for o in cell.get("outputs", [])
    ]
    frequency_output = next(
        o for o in outputs if o.startswith("model_family\n") and "1450356" in o
    )
    family_rows = {
        name: int(count)
        for name, count in re.findall(
            r"^([\w-]+)\s+(\d+)$", frequency_output, flags=re.MULTILINE
        )
    }
    sizes_output = next(
        o for o in outputs if "num_rows: 5077945" in o and "num_rows: 565171" in o
    )
    training_rows, validation_rows = map(
        int, re.findall(r"num_rows: (\d+)", sizes_output)
    )
    family_rows["human"] = training_rows - sum(family_rows.values())
    families = [
        "GPT",
        "Meta-LLaMA",
        "MPT",
        "Cohere",
        "Mistral",
        "Gemini",
        "DeepSeek",
        "Falcon",
        "Bison",
        "Qwen",
        "human",
    ]
    family_counts = [family_rows[name] for name in families]
    assert sum(family_counts) == training_rows == 5077945
    assert validation_rows == 565171
    inverse_roots = [n**-0.5 for n in family_counts]
    reconstructed = [w / (sum(inverse_roots) / 11) for w in inverse_roots]
    for record in records:
        config = record["config"]
        assert config["pos_weight"] == family_counts[-1] / sum(family_counts[:-1])
        if config["use_generator_classifier"]:
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(reconstructed, config["generator_class_weights"])
                )
                < 6e-8
            )
    raid = json.loads(raid_path.read_text())
    slices = {
        axis: [
            r
            for r in raid["scores"]
            if r[axis] != "all"
            and all(r[other] == "all" for other in AXES if other != axis)
        ]
        for axis in AXES
    }
    assert len(raid["scores"]) == 10881
    for r in [*raid["score_agg"].values(), *raid["scores"]]:
        for counts in r["accuracy"].values():
            assert (
                abs(counts["accuracy"] - counts["tp"] / (counts["tp"] + counts["fn"]))
                < 1e-12
            )
    sections = {}

    def label(r):
        c = r["config"]
        return (
            ("Attention" if c["fusion_type"] == "attention" else "Concatenation")
            + " / "
            + ("multi-task" if c["use_generator_classifier"] else "single-task")
            + " / "
            + ("frozen" if c["freeze_encoder"] else "trainable")
        )

    metrics = ["Roc-Auc", "Brier", "C@1", "F1", "F05U", "Mean"]
    sections["pan"] = table(
        ["Variant", *metrics],
        [[label(r), *[f"{r['pan26'][m]:.3f}" for m in metrics]] for r in records],
    )
    sections["models"] = table(
        ["Variant", "PAN26 Mean", "Hugging Face checkpoint"],
        [
            [
                label(r),
                f"{r['pan26']['Mean']:.3f}",
                f"[{r['name']}](https://huggingface.co/{r['config']['hub_model_id']})",
            ]
            for r in records
        ],
    )
    sections["experiments"] = table(
        [
            "Configuration",
            "Fusion",
            "Auxiliary head",
            "Encoder",
            "Train/eval batch per device",
            "Best-model reload explicitly set",
        ],
        [
            [
                f"[{r['name']}](../configs/{r['name']}.json)",
                r["config"]["fusion_type"],
                "11-class" if r["config"]["use_generator_classifier"] else "none",
                "frozen" if r["config"]["freeze_encoder"] else "trainable",
                r["config"]["per_device_train_batch_size"],
                str(r["config"].get("load_best_model_at_end", "omitted")).lower(),
            ]
            for r in records
        ],
    )
    datasets = [
        "pan25-generative-ai-detection-20260508-test",
        PAN26,
        "eloquent-20260617-test",
        "pan26-generative-ai-detection-smoke-test-20260330-training",
    ]
    sections["cross"] = table(
        ["Variant", "PAN25", "PAN26", "Eloquent", "Smoke"],
        [
            [
                label(r),
                *[
                    f"{next(v for v in r['evaluations'] if v['Dataset'] == d)['Mean']:.3f}"
                    for d in datasets
                ],
            ]
            for r in records
        ],
    )
    sections["runs"] = table(
        ["Configuration", "PAN26 TIRA run"],
        [[r["name"], r["pan26"]["Run"]] for r in records],
    )

    def raid_table(rows, axis):
        return table(
            [axis, "Recall @ 5% FPR", "Recall @ 1% FPR", "AUROC", "TP / FN @ 1%"],
            [
                [
                    r[axis],
                    *[
                        f"{100 * r['accuracy'][p]['accuracy']:.4f}%"
                        for p in ("0.05", "0.01")
                    ],
                    f"{r['auroc']:.6f}",
                    f"{r['accuracy']['0.01']['tp']:,} / {r['accuracy']['0.01']['fn']:,}",
                ]
                for r in rows
            ],
        )

    sections["raid-summary"] = raid_table(
        [raid["score_agg"]["all"], raid["score_agg"]["no_adversarial"]], "attack"
    )
    # The aggregate no-adversarial object stores attack='none'; labels are explicit below.
    for axis, rows in slices.items():
        sections["raid-" + axis] = raid_table(rows, axis)
    sections["thresholds"] = table(
        ["Domain", "Threshold @ 5%", "Threshold @ 1%", "Achieved FPR @ 5% / 1%"],
        [
            [
                d,
                f"{raid['thresholds']['0.05'][d]:.9f}",
                f"{raid['thresholds']['0.01'][d]:.9f}",
                f"{raid['fpr']['0.05'][d]:.2f} / {raid['fpr']['0.01'][d]:.2f}",
            ]
            for d in raid["thresholds"]["0.05"]
        ],
    )
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    ledger = {
        "schema_version": 1,
        "scope": "Committed artifacts; benchmark checkpoint revisions and training logs are not supplied.",
        "source_sha256": hashes,
        "canonical_features": names,
        "prepared_split_sizes": {"train": training_rows, "validation": validation_rows},
        "training_family_counts_in_label_order": family_counts,
        "auxiliary_weight_reconstruction_max_error": max(
            abs(a - b)
            for a, b in zip(
                reconstructed, records[0]["config"]["generator_class_weights"]
            )
        ),
        "experiments": records,
        "raid": {
            "date_released": raid["date_released"],
            "score_records": len(raid["scores"]),
            "score_agg": raid["score_agg"],
            "single_axis_slices": slices,
            "thresholds": raid["thresholds"],
            "fpr": raid["fpr"],
        },
    }
    return sections, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sections, ledger = collect()
    updates = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        original = path.read_text()

        def replace(match):
            key = match.group(1)
            assert key in sections, f"Unknown section: {key}"
            return f"<!-- research:{key}:start -->\n{sections[key]}\n<!-- research:{key}:end -->"

        generated = re.sub(
            r"<!-- research:([\w-]+):start -->.*?<!-- research:\1:end -->",
            replace,
            original,
            flags=re.DOTALL,
        )
        if args.check:
            assert generated == original, f"Stale research table: {path}"
        elif generated != original:
            path.write_text(generated)
            updates += 1
    ledger_path = ROOT / "docs/validation/research-evidence.json"
    serialized = json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        assert ledger_path.read_text() == serialized, "Stale research evidence ledger"
    else:
        ledger_path.write_text(serialized)
    print(
        f"Verified 8 configurations, 32 TIRA records, 10,881 RAID slices; {'checked' if args.check else 'updated'} {updates} documents."
    )


if __name__ == "__main__":
    main()
