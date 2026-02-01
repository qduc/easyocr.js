# Python Reference

This directory contains the original EasyOCR implementation to generate reference data for validation.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Generate reference

```bash
uv run python generate_reference.py path/to/image.jpg
```
