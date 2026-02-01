# Python Reference

This directory contains the original EasyOCR implementation to generate reference data for validation.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Generate reference JSON (one image)

```bash
uv run python generate_reference.py path/to/image.jpg
```

Outputs JSON into `python_reference/out/` by default.

## Generate reference JSON (batch)

Scan directories recursively and preserve relative paths under `--out-dir`:

```bash
uv run python generate_reference.py ./my_images --relative-to ./my_images --out-dir ./out
```

## Optional: HTTP reference server

This is convenient for driving the Python reference from Node-based integration tests later.

```bash
uv run uvicorn server:app --reload --port 8008
```

Then `POST /readtext` with a `multipart/form-data` file field named `image`.
