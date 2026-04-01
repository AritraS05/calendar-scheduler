---
title: Meta Calendar Scheduler
emoji: 📅
colorFrom: blue
colorTo: indigo
sdk: streamlit # Or "gradio" / "docker" depending on your app.py
app_file: app.py
pinned: false
---


A streamlined tool for managing and scheduling calendar events, optimized for performance and ease of use.

* Automated event scheduling logic.
* Inference-ready backend for intelligent time-slot suggestions.
* Lightweight architecture using `uv` for fast dependency management.


This project uses **uv** for extremely fast Python package management.

1. Install uv if you haven't: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Sync dependencies:
   ```bash
   uv sync