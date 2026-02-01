# Task: Match ONNX Output to Python EasyOCR Reference (New Session)

## Goal
Make the JS/ONNX pipeline output match the Python EasyOCR reference for `python_reference/validation/expected/Screenshot_20260201_193653.json`.

## Current State / Findings
- **Python reference used**: EasyOCR 1.7.2 (see `python_reference/validation/expected/manifest.json`).
- **Input image**: `examples/Screenshot_20260201_193653.png` (identical to `python_reference/validation/images/Screenshot_20260201_193653.png`).
- **Python expected output**: includes multiple lines, e.g. `V13.2`, `30 May 2021`, `Version 1,.3.2`, etc.
- **Current ONNX output (JS)**: still wrong; only short gibberish fragments from top line, missing the rest of the text.

## Changes Already Made (Do NOT lose)
These changes are already in the repo; keep them unless intentionally reverting:
1) **Detector scaling fix** (heatmap scale instead of input scale):
   - `packages/core/src/pipeline.ts`
   - Uses `textMap.width/height` vs original image to scale boxes.
2) **Channel-order aware normalization**:
   - `packages/core/src/utils.ts` `toFloatImage` now respects `channelOrder` (RGB vs BGR).
3) **Resize interpolation**:
   - `packages/core/src/utils.ts` `resizeImage` switched from nearest neighbor to bilinear.
4) **Recognizer preprocess tweaks**:
   - `packages/core/src/recognizer.ts` now builds planar in grayscale, then normalizes after padding.
5) **CTC decode mapping attempt**:
   - `packages/core/src/recognizer.ts` added `indexToChar` to shift indices when blank is 0.
6) **Default OCR options updated to match Python defaults**:
   - `packages/core/src/types.ts`: `magRatio=1.0`, `minSize=20`, detector mean/std set to ImageNet values.
7) **Detector postprocess**:
   - Added link-only suppression and dilation/box adjustment; still not matching Python.

## Key Reference Behavior (From EasyOCR Source)
Location: `/ub/home/qduc/src/easyocr.js/.venv/lib/python3.13/site-packages/easyocr`

**Recognizer pipeline** (from `recognition.py` + `utils.py`):
- Convert crop to grayscale (`'L'`).
- Resize to height 32 with **BICUBIC**, width = `ceil(imgH * w/h)` capped at 100.
- **NormalizePAD**: `ToTensor()` (0..1) then `(x - 0.5) / 0.5`. Pad **right** to width 100 using **last column replication** (not zeros).
- Run model, then `softmax` over classes.
- Zero out `ignore_idx` (includes blank index 0 + separators), **renormalize**.
- Greedy decode using **CTCLabelConverter**: indices are 1-based for chars, index 0 is `[blank]`.
- Confidence = `custom_mean` = `prod(max_probs)^(2/sqrt(n))`.

**Detector behavior** (from EasyOCR CRAFT):
- OpenCV `cv2` decode (BGR).
- Connected components on text/link maps, link-only suppression, dilation iteration based on component size; then generate boxes.
- Postprocessing uses original image scale + add_margin.

## Current ONNX Model Metadata
- Detector ONNX outputs two tensors:
  - `detector_output_0` shape `[1, 96, 368, 2]`
  - `detector_output_1` shape `[1, 32, 96, 368]`
- Recognizer ONNX:
  - Inputs: `input` (float32, shape `[1,1,32,100]`), `text` (int64, shape `[1,1]`)
  - Output: `[1, 24, 97]` (steps=24, classes=97)
- Charset length: 96 (`models/english_g2.charset.txt`).
  => blank index is 0, characters are indices 1..96.

## Known Gaps vs EasyOCR
- We still **do not match** EasyOCR recognizer preprocess:
  - BICUBIC resize not guaranteed (current is bilinear).
  - NormalizePAD logic not implemented (right pad with last column).
  - The softmax + ignore + renorm + confidence formula is missing.
- Decoder should match CTCLabelConverter logic (1-based char mapping, ignore_idx filtering).

## Immediate Next Steps (Recommended)
1) Implement recognizer preprocessing to **match EasyOCR exactly**:
   - Grayscale conversion (use formula 0.299/0.587/0.114).
   - Resize with **BICUBIC** to height=32, width=ceil(32*w/h), cap=100.
   - NormalizePAD: pad to width 100 with last-column replication.
   - Normalize `(x - 0.5) / 0.5`.
2) Implement **softmax + ignore + renormalize** and **CTCLabelConverter greedy decode**.
3) Implement **custom_mean** confidence (geometric-ish mean as in EasyOCR).
4) Re-run `node examples/node-ocr.mjs` and compare to `python_reference/validation/expected/Screenshot_20260201_193653.json`.

## Files to Edit Next
- `packages/core/src/recognizer.ts`
- `packages/core/src/utils.ts`
- Possibly `packages/core/src/pipeline.ts` if decoder output handling needs adjustment.

## Commands Used Previously
- `bun run -F @easyocrjs/core build`
- `bun run -F @easyocrjs/node build`
- `node examples/node-ocr.mjs ./python_reference/validation/images/Screenshot_20260201_193653.png`

## Success Criteria
The ONNX pipeline output JSON matches the Python reference output JSON for the test image, including all boundaries and texts.
