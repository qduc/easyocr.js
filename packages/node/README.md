# @qduc/easyocr-node

Node.js runtime for [easyocr.js](https://github.com/qduc/easyocr.js), a JavaScript port of EasyOCR.

This package provides Node.js implementations for image loading (via `sharp`) and inference (via `onnxruntime-node`).

## Installation

```bash
npm install @qduc/easyocr-node @qduc/easyocr-core
```

## Quick Start

```typescript
import { createOCR } from '@qduc/easyocr-node';

const ocr = await createOCR({
  modelDir: './models',
  lang: 'en',
});

const results = await ocr.read('path/to/image.png');
console.log(results);
```

See the [main repository README](https://github.com/qduc/easyocr.js#readme) for more details.
