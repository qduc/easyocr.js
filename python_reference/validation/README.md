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

## JS-vs-Python Drift Tracing (Step-by-step)

When the JS OCR output differs from Python EasyOCR, use the tracing scripts to find the **first step that drifts**
(image decode → resize/pad → normalize → detector outputs → box decode → crops → recognizer input).

### What you get

Both JS and Python emit a **trace directory** with a shared structure:

- `trace.json`: ordered step index (names + kinds + per-step folder)
- `steps/<NNN>_<step_name>/meta.json`: summary + `sha256_raw`
- `steps/<NNN>_<step_name>/image.png`: visual artifact (when the step is image-like)
- `steps/<NNN>_<step_name>/raw.bin` / `tensor.bin` / `boxes.bin`: canonical raw bytes for stable hashing/diffing
- `steps/<NNN>_<step_name>/*.meta.json`: dtype/layout/shape metadata for the raw artifact

### 1) Build JS packages (so tracing exports exist)

From repo root:

```bash
bun run build
```

### 2) Generate the JS trace

From repo root:

```bash
node examples/node-ocr.mjs ./python_reference/validation/images/Screenshot_20260201_193653.png \
  --trace-dir /tmp/trace_js
```

Notes:
- `examples/node-ocr.mjs` also loads a grayscale image for recognition (to match Python EasyOCR).
- Tracing only runs if the built Node bundle exports `createFsTraceWriter` (so run `bun run build` first).

### 3) Generate the Python trace (baseline = installed EasyOCR)

From repo root:

```bash
python3 python_reference/trace_easyocr.py ./python_reference/validation/images/Screenshot_20260201_193653.png \
  --trace-dir /tmp/trace_py
```

Optional: also store the final Python `readtext()` results inside the trace:

```bash
python3 python_reference/trace_easyocr.py ./python_reference/validation/images/Screenshot_20260201_193653.png \
  --trace-dir /tmp/trace_py --run-readtext
```

Prereqs:
- Your Python environment must have `easyocr` installed (and its deps like `opencv-python`, `torch`, etc.).
- This script uses `easyocr==1.7.2` APIs (but should remain mostly stable across patch versions).

### 4) Diff the traces to find the first drift

From repo root:

```bash
python3 python_reference/validation/diff_traces.py \
  --js /tmp/trace_js \
  --py /tmp/trace_py \
  --out /tmp/trace_report
```

To see all drifts instead of stopping at the first:

```bash
python3 python_reference/validation/diff_traces.py \
  --js /tmp/trace_js \
  --py /tmp/trace_py \
  --out /tmp/trace_report \
  --continue
```

The report directory can include `*_diff.png` images for quick visual inspection (when Pillow is installed).

### 5) How to interpret results

Common “first drift” patterns:
- Drift at `load_image`: image decoder mismatch (color space / alpha removal / orientation).
- Drift at `resize_aspect_ratio` or `pad_to_stride`: resize interpolation or 32-stride padding mismatch.
- Drift at `normalize_mean_variance`: mean/std constants or formula mismatch.
- Drift at `detector_raw_output_*`: runtime/model/provider mismatch (or preprocessing still differs).
- Drift at `threshold_and_box_decode`: postprocess mismatch (connected components / dilation / min-area-rect / scaling).
- Drift at crop/recognizer steps: perspective warp direction/interpolation, resize/pad policy, or recognizer input height.

### Extending the trace (adding more checkpoints)

To add a new checkpoint:
1. Add a `traceStep(...)` call in `packages/core/src/pipeline.ts` with a stable step name.
2. Add the matching step emission in `python_reference/trace_easyocr.py` with the **same step name**.
3. Rebuild JS (`bun run build`), re-run both traces, and re-run the diff.

Tip: keep step names stable and append new steps to the end to avoid breaking existing comparisons.
