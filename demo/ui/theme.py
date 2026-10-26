"""Design tokens and CSS of the FLAIRD dashboard.

The look follows a professional blue operations dashboard: a solid blue header, translucent
KPI tiles on a blue strip, white rounded cards on a light grey page, teal and navy data colors.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

TOKENS: dict[str, str] = {
    "primary": "#1F5498",
    "primary_dark": "#264772",
    "accent": "#0CB9C0",
    "page_bg": "#F4F5F7",
    "card_bg": "#FFFFFF",
    "panel_bg": "#F0FBFF",
    "text": "#1E2A3B",
    "text_muted": "#6B7A90",
    "muted_series": "#A9B5C1",
    "track": "#E6ECF3",
    "good": "#1E9E5A",
    "fair": "#F0BA3C",
    "poor": "#C62828",
    "machine": "#E5533D",
    "human": "#0CB9C0",
}

FONT_STACK = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

# Loaded in <head>: Inter from Google Fonts (falls back to the system font offline) and a
# script that pins the light color scheme so the dashboard looks the same everywhere.
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
  (function () {
    var url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url);
    }
  })();
</script>
"""


def build_css() -> str:
    """Return the dashboard CSS with the design tokens injected as CSS variables."""
    variables = "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in TOKENS.items())
    body = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    return f":root {{\n{variables}\n  --font: {FONT_STACK};\n}}\n{body}"


def build_theme() -> gr.themes.Base:
    """Gradio base theme aligned with the tokens (component-level details live in the CSS)."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#EAF1FB",
            c100="#D5E3F6",
            c200="#ABC7ED",
            c300="#7EA7DD",
            c400="#4F83C8",
            c500="#2F66AD",
            c600="#1F5498",
            c700="#1A4580",
            c800="#153866",
            c900="#102B4F",
            c950="#0B1E38",
        ),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill=TOKENS["page_bg"],
        body_text_color=TOKENS["text"],
        button_primary_background_fill=TOKENS["primary"],
        button_primary_background_fill_hover=TOKENS["primary_dark"],
        button_primary_text_color="#FFFFFF",
        block_radius="12px",
        input_radius="10px",
    )
