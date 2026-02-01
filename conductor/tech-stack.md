# Tech Stack - easyocr.js

## Core Technologies
- **Language:** TypeScript
- **Runtime Environments:** Node.js (v18+), Modern Web Browsers
- **Build System:** Bun (Workspaces)

## AI & ML
- **Inference Engine:** ONNX Runtime
  - `onnxruntime-node` for server-side execution.
  - `onnxruntime-web` for client-side execution.
- **Models:** CRAFT (Detection), Recognition models (CRNN) exported to ONNX format.

## Image Processing
- **Node.js:** `sharp` for efficient image manipulation and pre-processing.
- **Web:** Browser `Canvas` API for client-side image operations.

## Testing & Quality
- **Framework:** `vitest` for unit and integration testing.
- **Linting/Formatting:** Integrated into the development workflow.
