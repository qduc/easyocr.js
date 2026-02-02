# easyocr.js

A JavaScript port of [EasyOCR](https://github.com/JaidedAI/EasyOCR) for Node.js and the Browser.

`easyocr.js` provides an efficient, modular OCR pipeline that runs entirely in JavaScript environments using ONNX Runtime. It achieves feature parity with the Python reference implementation while offering superior integration with JavaScript/TypeScript ecosystems.

## Features

- **Multi-runtime Support**: Works in Node.js (via `sharp` and `onnxruntime-node`) and the Browser (via `onnxruntime-web` and Canvas).
- **TypeScript First**: Full type safety and high-level abstractions for OCR tasks.
- **Model Compatible**: Uses the same CRAFT detector and recognition models as the original Python EasyOCR.
- **High Performance**: Leverages hardware acceleration (CPU/GPU) through ONNX Runtime.
- **Multi-language**: Support for 8+ languages including English, Chinese, Japanese, Korean, and more.
- **Lightweight Core**: The core logic is decoupled from runtime-specific dependencies.

## Installation

### Understanding the Package Structure

- **`@qduc/easyocr-core`**: Shared types, pipeline logic, and image processing (required for all runtimes)
- **`@qduc/easyocr-node`**: Node.js runtime implementations using `sharp` for images and `onnxruntime-node` for inference
- **`@qduc/easyocr-web`**: Browser runtime implementations using Canvas APIs and `onnxruntime-web`

**When to use each:**
- Use **Node.js** for server-side OCR, CLI tools, or desktop applications with Node.js
- Use **Web** for in-browser OCR without server dependencies

### Node.js

```bash
npm install @qduc/easyocr-node @qduc/easyocr-core
# or
yarn add @qduc/easyocr-node @qduc/easyocr-core
# or
bun add @qduc/easyocr-node @qduc/easyocr-core
```

**Requirements:**
- Node.js 16+ (18+ recommended)
- `sharp` will be installed as a dependency (may require system libs on Linux)

### Browser

```bash
npm install @qduc/easyocr-web @qduc/easyocr-core
```

> **Note**: For browser usage, you need to host or point to the `onnxruntime-web` WASM files. See [Browser Integration](#browser-integration) for details.

## Quick Start (Node.js)

### Basic Usage (Simplified)

```typescript
import { createOCR } from '@qduc/easyocr-node';

async function run() {
  const ocr = await createOCR({
    modelDir: './models',
    lang: 'en', // or langs: ['en', 'ch_sim']
  });

  const results = await ocr.read('path/to/image.png');
  for (const item of results) {
    console.log(`Text: ${item.text}`);
    console.log(`Confidence: ${(item.confidence * 100).toFixed(1)}%`);
  }
}

run().catch(console.error);
```

> **Note**: `modelDir` must contain `onnx/` models and the matching `.charset.txt` files (see [Getting the Models](#getting-the-models)).

### Advanced Usage (Manual Setup)

```typescript
import {
  loadImage,
  loadDetectorModel,
  loadRecognizerModel,
  recognize,
  loadCharset
} from '@qduc/easyocr-node';

async function run() {
  // 1. Load your image (PNG, JPG, etc.)
  const image = await loadImage('path/to/image.png');

  // 2. Load the detector model
  // On first run, this auto-downloads from GitHub Releases (~200MB)
  // Subsequent runs use the cached copy from models/onnx/
  const detector = await loadDetectorModel('models/onnx/craft_mlt_25k.onnx');

  // 3. Load the recognizer model and charset
  // For English text:
  const charset = await loadCharset('models/english_g2.charset.txt');
  const recognizer = await loadRecognizerModel('models/onnx/english_g2.onnx', {
    charset,
    textInputName: 'text', // Required for g2 models
  });

  // 4. Run OCR
  const results = await recognize({
    image,
    detector,
    recognizer,
  });

  // 5. Process results
  for (const item of results) {
    console.log(`Text: ${item.text}`);
    console.log(`Confidence: ${(item.confidence * 100).toFixed(1)}%`);
    // item.box is a 4-point polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  }
}

run().catch(console.error);
```

### Using a Different Language

With `createOCR`, just pass the language(s) and the correct model is selected automatically:

```typescript
const ocr = await createOCR({ modelDir: './models', langs: ['en', 'ch_sim'] });
```

To recognize text in a different language manually, load the corresponding model and charset:

```typescript
// For Chinese (Simplified)
const charset = await loadCharset('models/zh_sim_g2.charset.txt');
const recognizer = await loadRecognizerModel('models/onnx/zh_sim_g2.onnx', {
  charset,
  textInputName: 'text',
});

// For Japanese
const charset = await loadCharset('models/japanese_g2.charset.txt');
const recognizer = await loadRecognizerModel('models/onnx/japanese_g2.onnx', {
  charset,
  textInputName: 'text',
});

// See Supported Models section for all available languages
```

### Configuration Options

The `recognize` function accepts an optional `options` object:

```typescript
const results = await recognize({
  image,
  detector,
  recognizer,
  options: {
    langList: ['en', 'ch_sim'], // Filter characters by language
    allowlist: '0123456789',    // Only recognize these characters
    blocklist: 'XYZ',           // Exclude these characters
    paragraph: true,            // Combine results into paragraphs
    canvasSize: 2560,          // Max canvas dimension (default: 2560)
  },
});
```

### Processing Multiple Images

```typescript
async function processMultiple(imagePaths: string[]) {
  // Load models once
  const detector = await loadDetectorModel('models/onnx/craft_mlt_25k.onnx');
  const charset = await loadCharset('models/english_g2.charset.txt');
  const recognizer = await loadRecognizerModel('models/onnx/english_g2.onnx', {
    charset,
    textInputName: 'text',
  });

  // Process images sequentially (for parallelism, use Promise.all with caution on memory)
  for (const path of imagePaths) {
    const image = await loadImage(path);
    const results = await recognize({ image, detector, recognizer });
    console.log(`${path}: ${results.map(r => r.text).join(' ')}`);
  }
}
```

### Error Handling

```typescript
import type { OcrResult } from '@qduc/easyocr-core';

try {
  const image = await loadImage('image.png');
  const results = await recognize({ image, detector, recognizer });

  // Filter high-confidence results
  const confident = results.filter(r => r.confidence > 0.5);
  console.log(`Found ${confident.length} confident detections`);
} catch (error) {
  if (error instanceof Error) {
    if (error.message.includes('ENOENT')) {
      console.error('Image file not found');
    } else if (error.message.includes('model')) {
      console.error('Failed to load model - check models/onnx/ directory');
    } else {
      console.error('OCR error:', error.message);
    }
  }
}
```

### Understanding the Output Format

Each `OcrResult` contains:
- **`text`**: Recognized text string
- **`confidence`**: Confidence score (0.0 to 1.0)
- **`box`**: 4-point polygon coordinates as `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` in pixel coordinates relative to the original image

## Browser Integration

### Setup (React/Vue/Svelte example)

When using `@qduc/easyocr-web`, you need to handle WASM paths and runtime differences.

```typescript
import * as ort from 'onnxruntime-web';
import { loadImage, recognize } from '@qduc/easyocr-web';
import type { RasterImage, OcrResult } from '@qduc/easyocr-core';

// Configure WASM path (choose one method):
// Method 1: Using CDN
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

// Method 2: Host locally or use your bundler's output
ort.env.wasm.wasmPaths = '/path/to/your/wasm/';

// Usage is similar to Node, but loadImage works with different input types:
// - File objects from <input type="file">
// - Blob objects
// - HTMLImageElement
// - Canvas/OffscreenCanvas

async function recognizeFromFileInput(fileInput: File): Promise<OcrResult[]> {
  const image = await loadImage(fileInput);
  return recognize({ image, detector, recognizer });
}
```

**Note:** Model loading in browser requires either:
1. Models hosted on a CORS-enabled server
2. Bundled models using your build tool
3. Pre-loaded `ArrayBuffer`s passed to model loaders

## Getting the Models

This package requires ONNX models and character sets to operate.

### Where Models Go

Models should be placed in the `models/onnx/` directory (relative to your project root or the working directory when running the code):

```
project/
├── models/
│   ├── onnx/
│   │   ├── craft_mlt_25k.onnx          (detector, ~200MB)
│   │   ├── english_g2.onnx             (recognizer)
│   │   ├── zh_sim_g2.onnx              (recognizer)
│   │   └── ...
│   ├── english_g2.charset.txt
│   ├── zh_sim_g2.charset.txt
│   └── ...
└── src/
    └── your-app.ts
```

### How to Get Models

**1. Automatic Download (Node.js only)**

`@qduc/easyocr-node` will automatically download missing models from GitHub Releases on first use:

```typescript
// This will download models/onnx/craft_mlt_25k.onnx if not found locally
const detector = await loadDetectorModel('models/onnx/craft_mlt_25k.onnx');
```

**2. Manual Download from Releases**

Download `.onnx` and `.charset.txt` files from the [GitHub Releases](https://github.com/qduc/easyocr.js/releases) page and place them in the `models/` directory.

**3. Manual Export from PyTorch**

If you have the original PyTorch weights from [EasyOCR](https://github.com/JaidedAI/EasyOCR):

```bash
# 1. Set up Python environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r python_reference/requirements.txt

# 3. Export and validate models
python models/export_onnx.py --detector --recognizer --validate
```

See [models/README.md](models/README.md) for more export options.

### Supported Models

| Language | Recognition Model | Charset File | Notes |
|----------|-------------------|--------------|-------|
| English | `english_g2.onnx` | `english_g2.charset.txt` | Default model |
| Latin | `latin_g2.onnx` | `latin_g2.charset.txt` | Covers European languages |
| Chinese (Simplified) | `zh_sim_g2.onnx` | `zh_sim_g2.charset.txt` | Mainland China |
| Japanese | `japanese_g2.onnx` | `japanese_g2.charset.txt` | Hiragana, Katakana, Kanji |
| Korean | `korean_g2.onnx` | `korean_g2.charset.txt` | Hangul |
| Cyrillic | `cyrillic_g2.onnx` | `cyrillic_g2.charset.txt` | Russian, Ukrainian, etc. |
| Telugu | `telugu_g2.onnx` | `telugu_g2.charset.txt` | South Indian language |
| Kannada | `kannada_g2.onnx` | `kannada_g2.charset.txt` | South Indian language |

**Detector:** All languages use the same detector `craft_mlt_25k.onnx` (multi-lingual text).

## Repository Structure

- [packages/core](packages/core): Runtime-agnostic types, pipeline logic, image processing, and post-processing. Use this for type definitions.
- [packages/node](packages/node): Node.js implementations using `sharp` for image loading and `onnxruntime-node` for inference.
- [packages/web](packages/web): Browser implementations using Canvas APIs and `onnxruntime-web`.
- [packages/cli](packages/cli): Command-line tool for running OCR from the terminal.
- [examples](examples): Sample code for Node.js and browser usage.
- [models](models): Model assets and export scripts (see [models/README.md](models/README.md)).
- [python_reference](python_reference): Original EasyOCR implementation and validation tools.

## TypeScript Types

If you need to work with types (e.g., for custom implementations):

```typescript
import type {
  RasterImage,     // Image data with width, height, channels
  OcrResult,       // Detection result with text, confidence, box
  DetectorModel,   // Loaded detector model
  RecognizerModel, // Loaded recognizer model
  OcrOptions,      // Recognition options
  Box,             // 4-point polygon coordinates
  Point,           // [x, y] coordinate
} from '@qduc/easyocr-core';
```

See [packages/core/src/types.ts](packages/core/src/types.ts) for full type definitions.

## Development

This is a monorepo using Bun workspaces.

### Getting Started

```bash
# Install all dependencies
bun install

# Build all packages (TypeScript → dist/)
bun run build

# Run tests
bun run test
```

### Working on a Single Package

```bash
# Build only @qduc/easyocr-node
bun run -F @qduc/easyocr-node build

# Test only @qduc/easyocr-core
bun run -F @qduc/easyocr-core test

# Run the CLI
bun run -F @qduc/easyocr-cli easyocr image.png
```

### Running Examples

```bash
# Node.js example (requires built packages)
node examples/node-ocr.mjs <image-path>

# Or with TypeScript directly
bun examples/node-ocr.ts <image-path>
```

### Debugging

**View debug traces:**

Both Python reference and JS implementation can emit detailed traces for comparison:

```bash
# Generate JS trace
EASYOCR_DEBUG=1 node examples/node-ocr.mjs image.png > debug_output/js/trace.json

# Generate Python trace
python python_reference/trace_easyocr.py image.png > debug_output/py/trace.json

# Compare traces
python python_reference/validation/diff_traces.py debug_output/py/trace.json debug_output/js/trace.json
```

See [python_reference/validation/README.md](python_reference/validation/README.md) for detailed validation instructions.

**Common Issues:**

| Problem | Solution |
|---------|----------|
| Models not found | Make sure `models/onnx/` directory exists and run from repo root |
| ONNX Runtime errors | Ensure correct Node/browser version; check [onnxruntime-node](https://github.com/microsoft/onnxruntime-javascript) compatibility |
| Image loading fails | Verify image format (PNG, JPG, WebP) and path; try absolute paths |
| Low accuracy | Check if using correct language model and charset; try disabling `langList` filter |

## FAQ

**Q: Why do I need both `@qduc/easyocr-core` and `@qduc/easyocr-node` (or `@qduc/easyocr-web`)?**

A: `@qduc/easyocr-core` contains the shared types and pipeline logic (runtime-agnostic). The node/web packages provide runtime-specific implementations for loading images and running inference.

**Q: Can I use this in production?**

A: Yes, but be aware of performance considerations. OCR is computationally intensive; on CPU, expect 1-5s per image depending on size. GPU acceleration via ONNX Runtime can improve performance significantly.

**Q: How accurate is this compared to the Python version?**

A: The JS port achieves numerical parity with the Python EasyOCR. See [python_reference/validation/README.md](python_reference/validation/README.md) for validation methodology.

**Q: Can I use custom models?**

A: Currently, only CRAFT (detector) and the provided g2 (recognizer) models are supported. Custom models require code changes.

**Q: Why is the first run slow?**

A: On the first run, models are downloaded from GitHub Releases (100-300MB total) and optionally quantized. Subsequent runs use cached models.

**Q: How do I handle multiple languages in one image?**

A: Use the `langList` option to specify multiple language codes. The recognizer will attempt to recognize characters from all specified languages. Note: You still need to load a single recognizer model; the language filter only affects which characters are accepted.

**Q: Can I run OCR in a Worker thread (Node.js or Browser)?**

A: Yes, but ONNX Runtime sessions may not be sharable across threads. Test thoroughly and consider creating separate model instances per worker.

**Q: What are the system requirements?**

A:
- **Node.js**: 16+ (18+ recommended)
- **Browser**: Modern browsers with Canvas and WebAssembly support (Chrome 74+, Firefox 79+, Safari 14+)
- **Memory**: 500MB+ recommended (more for large images or parallel processing)

## License

Apache-2.0 (Matches the original EasyOCR license).
