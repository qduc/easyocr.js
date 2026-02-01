# ONNX OCR Debugging - Quick Start Checklist

Use this checklist to quickly verify the current state and continue debugging.

## ✅ Verification Steps

### 1. Verify Fixes Are Applied
```bash
# Check if padding fix is in place
grep -A 5 "IMPORTANT: Python EasyOCR does NOT pad" packages/core/src/utils.ts

# Check if alpha removal is in place
grep "removeAlpha" packages/node/src/index.ts

# Both should show the fixed code
```

### 2. Rebuild Everything
```bash
rm -rf packages/*/dist
bun run build
```

### 3. Run Tensor Match Test
```bash
# Generate reference tensors
python3 debug_tensors.py python_reference/validation/images/Screenshot_20260201_193653.png --output-dir debug_output

# Generate JS tensors
node debug_tensors.mjs python_reference/validation/images/Screenshot_20260201_193653.png debug_output

# Compare
python3 test_tensor_match.py
```

**Expected:** ✓ Test PASSED: Tensors match within tolerance!

### 4. Run OCR Test
```bash
node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png
```

**Current Output:** 4 gibberish detections with very low confidence
**Expected Output:** 7 correct detections with high confidence

---

## 🔍 Debugging Next Steps

### Priority 1: Compare Detector Heatmaps

**Goal:** Determine if detector outputs match between Python and JS

```bash
# Run detector output inspector
node debug_detector_output.mjs

# Compare output with Python (run in Python):
python3 << 'EOF'
import numpy as np
import onnxruntime as ort

tensor = np.load('debug_output/detector_input_python.npy')
session = ort.InferenceSession('models/onnx/craft_mlt_25k.onnx', providers=['CPUExecutionProvider'])
outputs = session.run(None, {'input': tensor})

print(f"Output 0 shape: {outputs[0].shape}")
print(f"Output 0 range: [{outputs[0].min():.6f}, {outputs[0].max():.6f}]")
print(f"Output 0 mean: {outputs[0].mean():.6f}")
print(f"Text pixels > 0.7: {np.sum(outputs[0] > 0.7)}")
EOF
```

**Questions to answer:**
- [ ] Do heatmap shapes match?
- [ ] Do heatmap value ranges match?
- [ ] Do pixel counts above threshold match?

### Priority 2: Test Recognizer in Isolation

**Goal:** Verify recognizer preprocessing and inference work correctly

**Create test script:**
```javascript
// test_recognizer_isolated.mjs
import { loadImage } from './packages/node/dist/index.js';
import { recognizerPreprocess, resolveOcrOptions } from './packages/core/dist/index.js';

// Load a simple test image (single character if possible)
const image = await loadImage('path/to/simple_text.png');
const options = resolveOcrOptions();

// Preprocess
const { input } = recognizerPreprocess(image, options);

// Print tensor stats
console.log('Recognizer input:', {
  shape: input.shape,
  min: Math.min(...input.data),
  max: Math.max(...input.data),
  mean: input.data.reduce((a,b) => a+b, 0) / input.data.length,
});

// TODO: Compare with Python preprocessing of same image
```

**Questions to answer:**
- [ ] Does grayscale conversion match Python?
- [ ] Does resize match Python?
- [ ] Does normalization match Python?
- [ ] Do recognizer logits match Python?

### Priority 3: Binary Search the Pipeline

**Test each stage systematically:**

```javascript
// Stage 1: Detector preprocessing ✅ VERIFIED
const detectorInput = detectorPreprocess(image, options);
// Compare with Python - PASS ✅

// Stage 2: Detector inference
const detectorOutputs = await detector.session.run(...);
// Compare with Python - TODO ⚠️

// Stage 3: Heatmap extraction
const textMap = tensorToHeatmap(textTensor);
const linkMap = tensorToHeatmap(linkTensor);
// Compare with Python - TODO ⚠️

// Stage 4: Box extraction
const boxes = detectorPostprocess(textMap, linkMap, ...);
// Compare with Python - TODO ⚠️

// Stage 5: Image cropping
const crops = buildCrops(image, boxes, ...);
// Compare with Python - TODO ⚠️

// Stage 6: Recognizer preprocessing
const recognizerInput = recognizerPreprocess(crop, ...);
// Compare with Python - TODO ⚠️

// Stage 7: Recognizer inference
const logits = await recognizer.session.run(...);
// Compare with Python - TODO ⚠️

// Stage 8: CTC decoding
const decoded = ctcGreedyDecode(logits, ...);
// Compare with Python - TODO ⚠️
```

**Mark each stage as:**
- ✅ PASS: Matches Python
- ⚠️ TODO: Not yet tested
- ❌ FAIL: Doesn't match Python

---

## 🐛 Known Issues

### Issue #1: Recognition Produces Gibberish
- **Symptom:** Text is random characters, very low confidence
- **Status:** Under investigation
- **Suspects:**
  - Detector post-processing
  - Image cropping
  - Recognizer preprocessing
  - CTC decoding

### Issue #2: Only 4 Boxes Detected (Expected 7)
- **Symptom:** Missing text regions
- **Status:** Under investigation
- **Suspects:**
  - Detector heatmap processing
  - Threshold values
  - Box merging logic

---

## 📊 Expected vs Actual

### Test Image: Screenshot_20260201_193653.png

**Expected (Python):**
```
[0] "V13.2" (conf: 0.9277) at box [(6,18), (104,18), (104,50), (6,50)]
[1] "30 May 2021" (conf: 0.8519)
[2] "Version 1,.3.2" (conf: 0.5781)
[3] "Faster greedy decoder (thanks @samayala22)" (conf: 0.9538)
[4] "Fix bug when text box's aspect ratio is disproportional..." (conf: 0.8692)
[5] "report)" (conf: 0.6557)
[6] "bug" (conf: 1.0000)
```

**Actual (JS - Current):**
```
[0] "[1V/71,.3.2 " (conf: 0.0272) at box [(-1,11), (109,11), (109,55), (-1,55)]
[1] "J3Jimqjwd-+aoewrta32" (conf: 0.0005)
[2] "Imnipaitcvjpie jwut" (conf: 0.0010)
[3] "KaeziwmimaizfihvlLiwg" (conf: 0.0003)
```

**Observations:**
- Only 4 detections vs 7
- Boxes roughly in right areas but not exact
- Text completely wrong
- Confidence 100x-1000x lower

---

## 🛠️ Available Tools

### Diagnostic Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `debug_tensors.py` | Generate Python reference tensors | `python3 debug_tensors.py <image> --output-dir debug_output` |
| `debug_tensors.mjs` | Generate JS tensors | `node debug_tensors.mjs <image> debug_output` |
| `test_tensor_match.py` | Compare Python and JS tensors | `python3 test_tensor_match.py` |
| `test_onnx_python.py` | Validate ONNX models in Python | `python3 test_onnx_python.py` |
| `debug_detector_output.mjs` | Inspect detector heatmaps | `node debug_detector_output.mjs` |
| `debug_full_pipeline.mjs` | Full pipeline with comparison | `node debug_full_pipeline.mjs` |
| `compare_raw_pixels.mjs` | Compare raw pixel values | `node compare_raw_pixels.mjs` |

### Quick Commands

```bash
# Full rebuild
rm -rf packages/*/dist && bun run build

# Run all tests
python3 debug_tensors.py python_reference/validation/images/Screenshot_20260201_193653.png --output-dir debug_output && \
node debug_tensors.mjs python_reference/validation/images/Screenshot_20260201_193653.png debug_output && \
python3 test_tensor_match.py

# Test OCR
node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png

# Compare with Python
python3 << 'EOF'
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext('python_reference/validation/images/Screenshot_20260201_193653.png')
for bbox, text, conf in results[:5]:
    print(f'"{text}" ({conf:.4f})')
EOF
```

---

## 📝 Progress Tracking

### Completed ✅
- [x] Identified padding bug
- [x] Identified alpha channel bug
- [x] Fixed both preprocessing bugs
- [x] Created tensor comparison tests
- [x] Verified detector input tensors match Python
- [x] Validated ONNX models work in Python
- [x] Created diagnostic tools

### In Progress 🔄
- [ ] Identify where pipeline diverges from Python
- [ ] Fix detector post-processing (if needed)
- [ ] Fix recognizer preprocessing (if needed)
- [ ] Fix CTC decoding (if needed)

### Blocked ⛔
- None currently

---

## 💡 Tips

### When Adding Debug Logging

```javascript
// Add at critical points in the pipeline
console.log('='.repeat(60));
console.log('DEBUG: Stage Name');
console.log('='.repeat(60));
console.log('Input shape:', tensor.shape);
console.log('Input range:', [Math.min(...tensor.data), Math.max(...tensor.data)]);
console.log('Input mean:', tensor.data.reduce((a,b) => a+b, 0) / tensor.data.length);
```

### When Comparing Tensors

```python
# Python side
import numpy as np
np.save('debug_stage_X_python.npy', tensor)
print(f"Shape: {tensor.shape}, Range: [{tensor.min()}, {tensor.max()}]")
```

```javascript
// JS side
import { writeFile } from 'fs/promises';
await writeFile('debug_stage_X_js.bin', Buffer.from(tensor.data.buffer));
console.log(`Shape: [${tensor.shape}], Range: [${Math.min(...tensor.data)}, ${Math.max(...tensor.data)}]`);
```

### When Stuck

1. Go back to working state (detector input matches)
2. Move forward one stage at a time
3. Compare intermediate outputs at each stage
4. First divergence = bug location

---

## 📚 Key References

- **Full handoff:** `HANDOFF_ONNX_DEBUGGING.md`
- **Fixed files:** `packages/core/src/utils.ts`, `packages/node/src/index.ts`
- **Pipeline code:** `packages/core/src/pipeline.ts`
- **Python reference:** `.venv/lib/python3.11/site-packages/easyocr/`

---

**Last Updated:** 2026-02-01
