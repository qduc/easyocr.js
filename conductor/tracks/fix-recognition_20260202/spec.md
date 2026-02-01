# Specification - Fix Recognition Discrepancy (Screenshot_20260201_193653)

## Overview
The goal of this track is to identify and fix a bug in the `node-ocr` recognition pipeline where the extracted text for `Screenshot_20260201_193653.png` does not match the results from the Python reference implementation.

## Functional Requirements
- **Recognition Accuracy:** The text extracted by `@easyocrjs/node` for `Screenshot_20260201_193653.png` must match the text content in the reference validation data (`python_reference/validation/expected/Screenshot_20260201_193653.json`).
- **Pipeline Integrity:** Verify that the image crops passed from the detector to the recognizer are correctly pre-processed (resized, normalized, etc.) according to the EasyOCR recognition model requirements.
- **Functional Parity:** Ensure that the CRNN recognizer logic (including character mapping/decoding) is functionally equivalent to the Python reference.

## Non-Functional Requirements
- **Maintainability:** Ensure debugging tools used (if any) are documented or cleaned up.
- **Performance:** Fixes should not introduce significant latency to the OCR pipeline.

## Acceptance Criteria
- Running the validation against `Screenshot_20260201_193653.png` yields the correct text strings as defined in the Python reference output.
- Existing tests in `packages/core` and `packages/node` pass.

## Out of Scope
- Improving detection bounding box accuracy (unless it is found to be the root cause of recognition failure).
- Achieving exact numerical parity for confidence scores (floating point bit-perfection).
