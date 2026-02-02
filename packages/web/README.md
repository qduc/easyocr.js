# @qduc/easyocr-web

Browser runtime for [easyocr.js](https://github.com/qduc/easyocr.js), a JavaScript port of EasyOCR.

This package provides browser-compatible implementations for image loading (via Canvas) and inference (via `onnxruntime-web`).

## Installation

```bash
npm install @qduc/easyocr-web @qduc/easyocr-core
```

## Quick Start (React/Vue/Svelte)

```typescript
import * as ort from 'onnxruntime-web';
import { loadImage, recognize } from '@qduc/easyocr-web';

// Configure WASM path
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

// ... see main README for full browser integration details
```

See the [main repository README](https://github.com/qduc/easyocr.js#readme) for more details.
