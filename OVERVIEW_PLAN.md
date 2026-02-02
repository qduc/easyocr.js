### High-level plan to build a JS EasyOCR variant

1. **Scope + repo** (DONE: 2026-02-01)

   * Decide targets: **Node**, **Browser**, or both. Answer: both.
   * Create a monorepo: `core` (shared pipeline), `node` runtime, `web` runtime, `cli`, `examples`, `benchmarks`.

2. **Freeze a Python reference (DONE: 2026-02-01)**

   * Stand up a tiny Python service (or scripts) running EasyOCR.
   * Collect a validation set of images + expected outputs (boxes + text) to compare against while porting.

3. **Extract the pipeline contract** (DONE: 2026-02-01)

   * Precisely document: preprocessing, detector output format, cropping/deskew, recognizer inputs, decoding method (CTC/attention), and postprocessing/merging rules.

4. **Convert models** (DONE: 2026-02-01)

   * Added `models/export_onnx.py` to export detector + recognizer to **ONNX** (uses local `.pth` weights).
   * Run `python models/export_onnx.py --validate` after installing Python deps to validate with ONNX Runtime.
   * If conversion breaks, either rewrite the offending ops or split the model into exportable parts.

5. **Build the JS core pipeline** (DONE: 2026-02-01)

   * Implement: image normalization → detector inference → box/polygon postprocess → crop/warp → recognizer inference → decode → final results.
   * Keep model inference abstract so Node and Browser can share most logic.
   * Plan (breakdown):
     1. Define core public API in `@qduc/easyocr-core`
        - Types: `Point`, `Box`, `OcrResult`, `OcrOptions` (mirror Python defaults captured in `PIPELINE_CONTRACT.md`)
        - Runtime-agnostic image type: decoded raster + channel order metadata
        - Runtime-agnostic inference interface: `run(feeds) -> outputs` for detector + recognizer
     2. Implement detector preprocessing
        - Resize rule (`canvas_size`, `mag_ratio`) + coordinate mapping back to original pixels
        - Normalization + layout packing exactly as required by the exported ONNX detector
     3. Implement detector postprocess (CRAFT-style)
        - Threshold heatmaps (`text_threshold`, `low_text`, `link_threshold`)
        - Connected-components + polygon/quad extraction
        - Emit `horizontal_list` + `free_list` in the contract format and order
     4. Implement grouping / merging heuristics
        - Reading-order sort + line grouping (`slope_ths`, `ycenter_ths`, `height_ths`, `width_ths`, `add_margin`)
        - Optional paragraph merge (`paragraph`, `x_ths`, `y_ths`) as a separate, testable stage
     5. Implement crop / rectify
        - Horizontal crop path for `horizontal_list`
        - Perspective warp path for `free_list` (preserve point ordering expectations)
        - Optional `rotation_info` search (try rotations, pick best by recognition confidence)
     6. Implement recognizer preprocessing
        - Resize/pad to recognizer input height/width; grayscale vs RGB decision documented + tested
        - Optional contrast retry (`contrast_ths`, `adjust_contrast`) wired as a loop around recognition
     7. Implement decoding + confidence
        - CTC greedy decoder first (match Python reference)
        - Add beam search and/or word beam search once greedy matches reference
        - Define confidence computation to match reference (store in contract + tests)
     8. Implement orchestrator + batching
        - Single `recognize(image, options)` pipeline that composes all stages
        - Batch detector (1 image) + batch recognizer (N crops) for performance
        - Deterministic output ordering and stable float handling (no hidden nondeterminism)
     9. Add stage-level tests in `packages/core/test/`
        - Golden tests for: resize/mapping, postprocess box extraction, perspective warp, CTC decode
        - Contract tests that assert shape/layout of every intermediate artifact
     10. Add end-to-end parity harness hooks (without binding to Node/Web)
        - Consume a reference JSON sample and compare output within tolerances (wired up in step 7)

6. **Implement runtimes** (DONE: 2026-02-02)

   * **Node**: `onnxruntime-node` + `sharp` (and OpenCV only if needed).
   * **Web**: `onnxruntime-web` (WebGPU/WASM) + Canvas / OpenCV.js; run inference in **WebWorkers**.

7. **Match behavior with tests**

   * Unit test preprocessing + decoding + postprocessing.
   * Integration tests that compare JS results to the Python reference on the validation set (tolerances + text-level metrics).
