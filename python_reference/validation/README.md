# Validation Set (Add Images Here)

Put your validation images in `python_reference/validation/images/` and generate expected outputs into
`python_reference/validation/expected/`.

Example (CPU-only, stable-ish reference):

```bash
cd python_reference
uv run python generate_reference.py ./validation/images \
  --relative-to ./validation/images \
  --out-dir ./validation/expected \
  --force
```

This produces one JSON per image (plus `manifest.json`) that can later be consumed by JS integration tests.

