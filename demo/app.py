"""FLAIRD Gradio 6 workbench. Run: python demo/app.py."""

import json
import logging
import os
import tempfile
from html import escape
from pathlib import Path

import gradio as gr
import torch

if __package__:
    from .presentation import charts, diagnostics, empty_plot, provenance, summary
    from .runtime import (
        DEFAULT_MODEL,
        MAX_CHARACTERS,
        MODELS,
        POLICIES,
        load_examples,
        runtime,
    )
else:
    from presentation import charts, diagnostics, empty_plot, provenance, summary
    from runtime import (
        DEFAULT_MODEL,
        MAX_CHARACTERS,
        MODELS,
        POLICIES,
        load_examples,
        runtime,
    )

EXAMPLES = load_examples()
IDLE = '<div class="empty-state"><span class="eyebrow">READY WHEN YOU ARE</span><h2>Read the signal behind the score.</h2><p>Paste English text or explore a sample, then analyze to see the decision and its forensic explanation.</p></div>'


def remove_export(path):
    if path:
        Path(path).unlink(missing_ok=True)


def export_report(report, old_path):
    if report is None:
        raise gr.Error("Analyze a text before exporting")
    remove_export(old_path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="flaird-report-",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return gr.update(value=file.name, interactive=True), file.name


def choose_example(index):
    if index is None:
        return gr.skip(), ""
    index = int(index)
    if not 0 <= index < len(EXAMPLES):
        raise gr.Error("Choose an example from the list")
    return EXAMPLES[index]["text"], provenance(EXAMPLES[index], index)


def reset_views():
    return [
        IDLE,
        empty_plot("Your origin scores will appear here"),
        None,
        None,
        None,
        None,
        None,
        [],
        "Analyze a text to inspect numerical quality and model provenance.",
        None,
        None,
        gr.update(interactive=False),
        gr.update(value=None, interactive=False),
    ]


def on_text_change(value, index):
    source = ""
    if (
        index is not None
        and 0 <= int(index) < len(EXAMPLES)
        and value == EXAMPLES[int(index)]["text"]
    ):
        source = provenance(EXAMPLES[int(index)], int(index))
    return source, *reset_views()


def analyze_text(value, model_id, cutoff, normalization):
    """Disable controls throughout a streamed request; always restore after errors."""
    yield [gr.update(interactive=False) for _ in range(7)] + reset_views()
    try:
        report = runtime.explain(value, model_id, cutoff, normalization)
        figures = charts(report)
        rows = [
            [
                r["name"],
                r["group"],
                r["value"],
                r["baseline"],
                r["attribution"],
                r["rank"],
            ]
            for r in report["features"]
        ]
        output = [
            summary(report),
            *figures,
            rows,
            diagnostics(report),
            report,
            report,
            gr.update(interactive=True),
            gr.update(value=None, interactive=False),
        ]
    except ValueError as error:
        output = reset_views()
        output[0] = (
            '<div class="error-state" role="alert">' + escape(str(error)) + "</div>"
        )
    except Exception:
        logging.getLogger(__name__).exception("FLAIRD analysis failed")
        output = reset_views()
        output[0] = (
            '<div class="error-state" role="alert">Analysis could not finish. Check model availability and host memory, then try again.</div>'
        )
    yield [gr.update(interactive=True) for _ in range(7)] + output


def build_app():
    if DEFAULT_MODEL not in MODELS.values():
        raise ValueError("MODEL_ID must be one of the eight published FLAIRD models")
    with gr.Blocks(
        title="FLAIRD · Detection & explanation",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        report_state = gr.State(None)
        export_path = gr.State(None, time_to_live=3600, delete_callback=remove_export)
        gr.HTML(
            """<header class="hero"><div><span class="eyebrow">FORENSIC LINGUISTICS × AI</span><h1>FLAIRD<span class="brand-dot">.</span></h1><p>Detect machine-generated text.<br>Explore the linguistic signals behind the decision.</p></div><div class="hero-tag">DETECTION & EXPLANATION<br><span>Research workbench · English text</span></div></header>"""
        )
        with gr.Row(equal_height=False, elem_classes="workbench-row"):
            with gr.Column(scale=4, min_width=310, elem_classes="input-panel"):
                gr.Markdown("### 01 / Your text")
                example = gr.Dropdown(
                    choices=[
                        (
                            f"{i + 1:02d} · {r['model']} · {r['attack'] or 'original / unspecified'}",
                            i,
                        )
                        for i, r in enumerate(EXAMPLES)
                    ],
                    label="Explore 20 examples",
                    value=None,
                )
                source = gr.HTML("")
                text = gr.Textbox(
                    label="English text",
                    lines=12,
                    max_lines=18,
                    placeholder="Paste a passage to analyze…",
                    max_length=MAX_CHARACTERS,
                    buttons=["copy"],
                    elem_id="analysis-text",
                )
                gr.Markdown(
                    "Up to 30,000 characters. The semantic encoder reads the first 512 tokens; forensic features cover the full text.",
                    elem_classes="helper",
                )
                with gr.Accordion("Model & analysis settings", open=False):
                    model = gr.Dropdown(
                        choices=[(name, value) for name, value in MODELS.items()],
                        value=DEFAULT_MODEL,
                        label="Trained checkpoint",
                    )
                    threshold = gr.Slider(
                        0.05,
                        0.95,
                        value=0.5,
                        step=0.01,
                        label="Machine decision threshold",
                        info="A machine decision requires a score strictly above this threshold.",
                    )
                    policy = gr.Dropdown(
                        choices=[(name, value) for name, value in POLICIES.items()],
                        value="training",
                        label="Text normalization",
                        info="Training matches the dataset workflow. TIRA keeps raw forensic text; Raw keeps both branches unchanged.",
                    )
                with gr.Row():
                    analyze = gr.Button("Analyze & explain", variant="primary", scale=3)
                    clear = gr.Button("Clear", scale=1)
                gr.Markdown(
                    "The score concerns text origin. It does not establish factual falsity, author intent, or a coordinated campaign.",
                    elem_classes="helper",
                )
            with gr.Column(scale=7, min_width=340):
                gr.Markdown("### 02 / Decision & evidence")
                result = gr.HTML(IDLE)
                probability = gr.Plot(
                    label="Origin scores",
                    value=empty_plot("Your origin scores will appear here"),
                )
                with gr.Tabs():
                    with gr.Tab("Feature effects"):
                        gr.Markdown(
                            "**Coral → toward machine · Teal → toward human.** Signed effects on the machine logit, conditional on fixed semantics and a zero-feature reference. Largest 15 of 35 features."
                        )
                        features = gr.Plot(label="Forensic feature effects")
                        table = gr.Dataframe(
                            headers=[
                                "Feature",
                                "Group",
                                "Raw value",
                                "Reference",
                                "Signed effect",
                                "Rank",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "number",
                                "number",
                                "number",
                                "number",
                            ],
                            interactive=False,
                            label="All 35 features",
                            wrap=True,
                            column_widths=[150, 150, 110, 85, 110, 60],
                        )
                    with gr.Tab("Feature groups"):
                        gr.Markdown(
                            "Group effects sum their feature attributions. Replacing a group with its zero reference gives a separate sensitivity test; replacement effects are not additive."
                        )
                        group_effect = gr.Plot(label="Group attribution")
                        replacement = gr.Plot(label="Group replacement sensitivity")
                        with gr.Accordion(
                            "Internal attention & gate diagnostics", open=False
                        ):
                            gr.Markdown(
                                "Attention is over **seven feature groups**, not words. Attention and the semantic gate are internal diagnostics, not causal importance or branch contribution percentages. Gate values appear under Quality & report."
                            )
                            attention = gr.Plot(label="Group attention")
                    with gr.Tab("Generator family"):
                        gr.Markdown(
                            "Independent auxiliary distribution, including the human class. These scores do not verify generator identity and are not conditional on a machine decision. Available only for multitask checkpoints."
                        )
                        generator = gr.Plot(label="Auxiliary generator distribution")
                    with gr.Tab("Quality & report"):
                        quality = gr.Markdown(
                            "Analyze a text to inspect numerical quality and model provenance."
                        )
                        with gr.Accordion("Full structured report", open=False):
                            structured = gr.JSON(label="Explanation report")
                        prepare = gr.Button("Prepare JSON export", interactive=False)
                        download = gr.DownloadButton(
                            "Download report.json", interactive=False
                        )
                        gr.Markdown(
                            "Exports contain model outputs, forensic values and settings. The original text and example annotations are excluded.",
                            elem_classes="helper",
                        )
        gr.HTML(
            '<footer class="footer"><b>FLAIRD</b> · Forensic Linguistics and AI for text origin analysis <a href="https://github.com/Mahmood-Anaam/flaird" target="_blank" rel="noopener">Project & methodology ↗</a></footer>'
        )
        views = [
            result,
            probability,
            features,
            group_effect,
            replacement,
            attention,
            generator,
            table,
            quality,
            structured,
            report_state,
            prepare,
            download,
        ]
        controls = [text, model, threshold, policy, example, analyze, clear]
        analyze.click(
            analyze_text,
            inputs=[text, model, threshold, policy],
            outputs=controls + views,
            concurrency_limit=1,
            concurrency_id="model-inference",
            api_name="analyze",
        )
        example.change(
            choose_example,
            inputs=example,
            outputs=[text, source],
            queue=False,
            api_visibility="private",
        ).then(reset_views, outputs=views, queue=False, api_visibility="private")
        text.change(
            on_text_change,
            inputs=[text, example],
            outputs=[source, *views],
            queue=False,
            api_visibility="private",
        )
        for component in [model, threshold, policy]:
            component.change(
                reset_views, outputs=views, queue=False, api_visibility="private"
            )
        clear.click(
            lambda: ("", None, "", *reset_views()),
            outputs=[text, example, source, *views],
            queue=False,
            api_visibility="private",
        )
        prepare.click(
            export_report,
            inputs=[report_state, export_path],
            outputs=[download, export_path],
            api_visibility="private",
        )
    return app.queue(max_size=8, default_concurrency_limit=1)


def main():
    threads = int(os.getenv("FLAIRD_CPU_THREADS", "4"))
    if threads < 1:
        raise ValueError("FLAIRD_CPU_THREADS must be positive")
    torch.set_num_threads(threads)
    build_app().launch(
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=["Arial", "sans-serif"],
        ),
        css=Path(__file__).with_name("style.css").read_text(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=False,
        share=False,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
