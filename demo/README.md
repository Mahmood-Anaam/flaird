---
title: FLAIRD Forensic Disinformation & Machine Text Detector
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns
---

# FLAIRD Hugging Face Space Demo

This Space hosts the interactive user interface for **FLAIRD** (*Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns*).

## Features
- **Multi-Model Support**: Choose between 8 distinct trained FLAIRD architectures (Attention vs Concatenation, Single-task vs Multi-task, Frozen vs Unfrozen encoders).
- **Forensic Visualization**: View interactive Plotly bar charts of feature group attention weights and 35 extracted stylometric metrics.
- **Generator Attribution**: Multi-task models identify candidate generative LLM families (e.g., GPT, LLaMA, Mistral, Gemini, etc.).
- **Structured JSON Output**: Retrieve full structured diagnostic reports suitable for downstream pipelines.
