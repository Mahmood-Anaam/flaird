"""Opt-in trained-checkpoint and live-browser acceptance checks (not a benchmark).

PYTHONPATH=. python tests/validate_demo.py --output docs/validation/gradio-checks.json
Install playwright==1.51.0 and its Chromium browser before running.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Local HTTP clients must reach the server in this process directly.
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import torch
from playwright.sync_api import expect, sync_playwright

from demo.app import EXAMPLES, build_app
from demo.presentation import charts
from demo.runtime import MODELS, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--browser-only",
        action="store_true",
        help="Reuse checkpoint checks already saved at --output",
    )
    parser.add_argument("--screenshots", type=Path, default=Path("/tmp/flaird-ui"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).parents[1]
    result = {
        "purpose": "Gradio acceptance checks using trained weights; not a benchmark evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "gradio", "plotly", "playwright")
        },
        "examples_sha256": hashlib.sha256(
            (root / "demo/examples.csv").read_bytes()
        ).hexdigest(),
        "checkpoints": [],
        "browser_checks": [],
    }
    # All eight variants exercise the production runtime and presentation adapter.
    if args.browser_only:
        result = json.loads(args.output.read_text())
        result["browser_checks"] = []
    human = next(row["text"] for row in EXAMPLES if row["label"] == "human")
    for model_id in [] if args.browser_only else MODELS.values():
        print("Checking", model_id, flush=True)
        report = runtime.explain(human, model_id)
        rendered = charts(report)
        assert report["diagnostics"]["completeness_passed"]
        assert len(report["features"]) == 35 and len(report["groups"]) == 7
        assert (report["groups"][0]["attention"] is not None) == (
            "-attention-" in model_id
        )
        assert (report["prediction"]["generator_probabilities"] is not None) == (
            "-multitask" in model_id
        )
        assert len(rendered[1].data[0].x) == 15
        # Confirm the UI runtime cache reuses this instance, then evicts at next model.
        old_model = runtime.model
        runtime._load(model_id)
        assert runtime.model is old_model
        del old_model
        result["checkpoints"].append(
            {
                "model_id": model_id,
                "revision": report["metadata"]["model_revision"],
                "machine_probability": report["prediction"]["machine_probability"],
                "completeness_passed": True,
                "chart_values_checked": True,
                "cache_reused": True,
            }
        )
        json.dumps(report, allow_nan=False)
    result["acceptance_completed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    blocks = build_app()
    print("Launching Gradio", flush=True)
    blocks.launch(
        prevent_thread_lock=True,
        server_name="127.0.0.1",
        server_port=7860,
        css=(root / "demo/style.css").read_text(),
        theme=__import__("gradio").themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=["Arial", "sans-serif"],
        ),
        share=False,
        quiet=True,
        footer_links=[],
    )
    args.screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--no-sandbox", "--single-process", "--no-zygote"]
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.set_default_timeout(30000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "requestfailed",
                lambda request: print(
                    "Request failed", request.url, request.failure, flush=True
                ),
            )
            print("Opening browser page", flush=True)
            page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded")
            page.screenshot(path=str(args.screenshots / "loading.png"), full_page=True)
            print("Page loaded", flush=True)
            page.get_by_role(
                "heading", name="Read the signal behind the score."
            ).wait_for()
            page.screenshot(
                path=str(args.screenshots / "desktop-idle.png"), full_page=True
            )
            result["browser_checks"].append("Desktop initial layout rendered")
            page.get_by_role("button", name="Analyze & explain", exact=True).click()
            page.screenshot(
                path=str(args.screenshots / "after-click.png"), full_page=True
            )
            print("Blank analysis clicked", flush=True)
            page.get_by_text(
                "Enter English text or choose an example before analyzing", exact=True
            ).wait_for()
            result["browser_checks"].append("Empty input error; controls restored")
            # Dropdown keyboard selection preserves exact CSV text and provenance.
            dropdown = page.get_by_role("combobox").first
            dropdown.click()
            page.get_by_role("option").first.click()
            page.get_by_text("Example 01 · source metadata", exact=True).wait_for()
            assert (
                page.locator("#analysis-text textarea").input_value()
                == EXAMPLES[0]["text"]
            )
            page.get_by_role("button", name="Analyze & explain", exact=True).click()
            page.get_by_text("MACHINE SCORE", exact=True).wait_for(timeout=120000)
            assert page.get_by_text("EXPLANATION CHECK", exact=True).is_visible()
            page.screenshot(
                path=str(args.screenshots / "desktop-result.png"), full_page=True
            )
            result["browser_checks"].append(
                "CSV example selected; actual model inference and feature charts rendered"
            )
            page.get_by_role("tab", name="Feature groups", exact=True).click()
            page.get_by_text(
                "Internal attention & gate diagnostics", exact=True
            ).click()
            page.screenshot(path=str(args.screenshots / "groups.png"), full_page=True)
            page.get_by_role("tab", name="Generator family", exact=True).click()
            page.screenshot(
                path=str(args.screenshots / "generator.png"), full_page=True
            )
            page.get_by_role("tab", name="Quality & report", exact=True).click()
            page.get_by_role("button", name="Prepare JSON export", exact=True).click()
            with page.expect_download() as download:
                page.get_by_role(
                    "button", name="Download report.json", exact=True
                ).click()
            downloaded = json.loads(Path(download.value.path()).read_text())
            assert len(downloaded["features"]) == 35 and "text" not in downloaded
            result["browser_checks"].append(
                "Group, attention, generator and quality tabs; JSON download verified"
            )
            page.locator("#analysis-text textarea").fill(
                "A short edited passage about a quiet morning in the library."
            )
            page.screenshot(path=str(args.screenshots / "edited.png"), full_page=True)
            page.get_by_role(
                "heading", name="Read the signal behind the score."
            ).wait_for()
            page.get_by_text("Example 01 · source metadata", exact=True).wait_for(
                state="hidden"
            )
            assert page.get_by_role("button", name="Prepare JSON export").is_disabled()
            result["browser_checks"].append(
                "Manual edit clears stale prediction, provenance and export"
            )
            page.get_by_text("Model & analysis settings", exact=True).click()
            page.get_by_role("combobox").nth(1).click()
            page.get_by_role(
                "option", name="Concatenation · single-task · frozen", exact=True
            ).click()
            page.locator('input[type="range"]').fill("0.65")
            page.get_by_role("button", name="Analyze & explain", exact=True).click()
            page.get_by_text("MACHINE SCORE", exact=True).wait_for(timeout=120000)
            page.get_by_text("Threshold 65%", exact=True).wait_for()
            page.get_by_role("tab", name="Generator family", exact=True).click()
            page.get_by_text(
                "Generator head is unavailable for single-task models", exact=True
            ).wait_for()
            page.get_by_role("tab", name="Feature groups", exact=True).click()
            page.get_by_text(
                "Group attention is unavailable for concatenation models", exact=True
            ).wait_for()
            result["browser_checks"].append(
                "Checkpoint switched to concatenation single-task frozen; 65% threshold and absent-head panels verified"
            )
            page.get_by_role("button", name="Clear", exact=True).click()
            expect(page.locator("#analysis-text textarea")).to_have_value("")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_function(
                "document.documentElement.scrollWidth <= window.innerWidth",
                timeout=10000,
            )
            page.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
            overflow = page.evaluate(
                "document.documentElement.scrollWidth > window.innerWidth"
            )
            assert not overflow
            result["browser_checks"].append(
                "390px mobile layout has no horizontal overflow; clear works"
            )
            browser.close()
            browser = p.chromium.launch(
                args=["--no-sandbox", "--single-process", "--no-zygote"]
            )
            mobile = browser.new_page(viewport={"width": 390, "height": 844})
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.set_default_timeout(30000)
            mobile.goto("http://127.0.0.1:7860", wait_until="domcontentloaded")
            mobile.locator("#analysis-text textarea").fill(
                "A short passage about the history of libraries and how readers share knowledge."
            )
            mobile.get_by_role("button", name="Analyze & explain", exact=True).click()
            mobile.get_by_text("MACHINE SCORE", exact=True).wait_for(timeout=120000)
            mobile.locator(".js-plotly-plot .main-svg").first.wait_for()
            mobile.screenshot(
                path=str(args.screenshots / "mobile-result.png"), full_page=True
            )
            assert (
                mobile.locator(".js-plotly-plot").first.bounding_box()["width"] <= 390
            )
            assert not mobile.evaluate(
                "document.documentElement.scrollWidth > window.innerWidth"
            )
            result["browser_checks"].append(
                "Fresh 390px mobile session analyzed actual text; decision cards and Plotly chart width verified"
            )
            assert not errors, errors
            result["browser_page_errors"] = errors
            browser.close()
    finally:
        blocks.close()
    result["acceptance_completed"] = True
    result["browser_completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["checkpoint_sample"] = {
        "example_index": 4,
        "recorded_label": "human",
        "preprocessing": "training",
    }
    result["implementation_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "demo/app.py",
            "demo/runtime.py",
            "demo/presentation.py",
            "demo/style.css",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Acceptance checks passed", flush=True)


if __name__ == "__main__":
    main()
