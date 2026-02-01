# Pipeline Contract

This document defines the *observable* interface (inputs/outputs) and the required
transformations at each stage of the EasyOCR-style pipeline implemented by this repo.

The goal is: for a fixed Python reference (see `python_reference/`), the JS pipeline should
produce equivalent boxes/text/confidence within tolerances.

## Scope + reference

### Reference entrypoint

The Python reference used by this repo currently calls:

- `easyocr.Reader(langs, gpu=...)`
- `reader.readtext(image)` with **no extra arguments** (defaults)

See:
- `python_reference/generate_reference.py`
- `python_reference/server.py`

Because EasyOCR defaults can change across versions, always capture the exact signature/config
of `readtext(...)` for the reference environment and record it in the JSON manifest.

Helper:
- Run `uv run python python_reference/dump_readtext_signature.py` in the Python venv.

## Conventions

### Coordinate system

- Image coordinates are in pixels in the decoded image raster.
- Origin is top-left.
- X increases to the right; Y increases downward.
- A point is `[x, y]` as `number` (float).

### Box (quadrilateral)

EasyOCR emits a 4-point polygon for each item. We treat it as:

- `box`: `[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]`
- Point order is considered *opaque*: preserve the order as emitted by the Python reference.
- Boxes are in the *original image coordinate space*.

JS should preserve this ordering when possible; if internal ops reorder points, they must be
normalized back to the reference convention before returning results.

### Image decode

Contractually, the pipeline accepts:

- Node: file path, `Buffer`, `Uint8Array`
- Web: `Blob`, `File`, `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, `OffscreenCanvas`

The runtime must decode into a consistent internal representation:

- `uint8` pixels
- 3 channels (RGB) or 4 channels (RGBA), but the *detector/recognizer preprocessors* must
  operate on a well-defined channel order.

Note: the Python reference server decodes via OpenCV (`cv2.imdecode(..., IMREAD_COLOR)`), which
produces BGR; EasyOCR handles this internally. JS runtimes should pick a single canonical order
(recommendation: RGB) and ensure preprocessing matches the reference outputs empirically.

## Revised contract (EasyOCR-aligned)

This is the staged contract we port against (mirrors EasyOCR’s public API more closely).

### 1) Input

**Python reference accepts**
- image: path | numpy array | bytes

**JS runtime accepts**
- Node/Web inputs as described in “Image decode” above, then normalizes to a decoded raster.

### 2) Detector preprocess

**Input**
- decoded image raster

**Output**
- detector input tensor (layout depends on exported model)
- mapping to original coordinates

**Required behavior**
- Resize bounded by `canvas_size`, scaled by `mag_ratio` (preserve aspect ratio).
- Apply the detector’s required normalization.

### 3) Detector (CRAFT)

**Internal (model output)**
- region/affinity heatmaps

**Public (post-detector) output**
- `horizontal_list`, `free_list`

Interpretation:
- `horizontal_list`: boxes suitable for “horizontal” text handling (axis-aligned / near-horizontal)
- `free_list`: boxes requiring free-form handling (quadrilateral / polygon perspective warp)

### 4) Box post-process + merging

**Threshold parameters**
- `text_threshold`
- `low_text`
- `link_threshold`

**Merging / grouping heuristics**
- `slope_ths`
- `ycenter_ths`
- `height_ths`
- `width_ths`
- `add_margin`

**Paragraph merge (only when `paragraph=True`)**
- `x_ths`
- `y_ths`

**Output**
- an ordered set of boxes to recognize, preserving the EasyOCR reading order and point order.

### 5) Crop / rectify

**Input**
- original image + (`horizontal_list`, `free_list`)

**Output**
- rectified crops (and bookkeeping to map each crop back to its box)

**Required behavior**
- horizontal crops for `horizontal_list`
- perspective warp for `free_list`
- optional: `rotation_info` search (try multiple rotations and pick best)

### 6) Recognition preprocess

**Required behavior**
- optional contrast retry: if below `contrast_ths`, run `adjust_contrast` and retry recognition.

### 7) Recognizer (CRNN-style)

**Model family**
- feature extractor (ResNet/VGG) + LSTM + CTC

**Output**
- per-timestep character probabilities / logits + decoder charset

### 8) Decoding

Decoder must match the Python reference:

- `greedy`
- `beamsearch`
- `wordbeamsearch`

### 9) Output assembly

**Output**
- list of `(box, text, confidence)`
- optionally paragraph-combined output when `paragraph=True`

## Output format (JS + Python reference)

The reference JSON payload emitted by `python_reference/*` is:

```json
{
  "formatVersion": 1,
  "image": "filename-or-relative-path.jpg",
  "results": [
    {
      "box": [[0, 0], [10, 0], [10, 5], [0, 5]],
      "text": "hello",
      "confidence": 0.98
    }
  ]
}
```

The JS pipeline output should be equivalent:

- preserve `box` point order and coordinate space
- `text` is UTF-8 string (no forced ASCII)
- `confidence` is float in `[0,1]`

## Contract checklist (for porting)

When implementing the JS pipeline, for each stage above, ensure the following are explicitly
captured somewhere stable (tests or JSON):

1) Exact EasyOCR version used by the Python reference.
2) `readtext(...)` signature and all parameter defaults used by the reference run.
3) Detector preprocess parameters: `canvas_size`, `mag_ratio`, normalization, channel order.
4) Detector postprocess thresholds: `text_threshold`, `low_text`, `link_threshold`.
5) Merge/group heuristics: `slope_ths`, `ycenter_ths`, `height_ths`, `width_ths`, `add_margin`.
6) Paragraph merge knobs (if used): `paragraph`, `x_ths`, `y_ths`.
7) Crop/rectify: `free_list` warp, `rotation_info` (if used).
8) Recognition extras: `contrast_ths`, `adjust_contrast` behavior (if used).
9) Decoder: `greedy`/`beamsearch`/`wordbeamsearch` + confidence definition.
