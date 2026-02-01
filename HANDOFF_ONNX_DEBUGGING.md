# ONNX OCR Debugging - Handoff Document

**Date:** 2026-02-01
**Author:** Claude (Sonnet 4.5)
**Status:** Partial Resolution - Preprocessing Fixed, Recognition Still Failing

---

## Executive Summary

Investigated why the ONNX-based JavaScript OCR implementation produces incorrect results compared to the Python EasyOCR reference. **Successfully fixed critical preprocessing bugs** that caused tensor mismatches, but **recognition still fails** with gibberish output and low confidence scores.

### What's Fixed ✅
- ✅ Detector input tensors now match Python exactly (max diff: 2.4e-7)
- ✅ Image loading correctly handles RGB without alpha channel corruption
- ✅ Removed unnecessary padding that caused shape mismatches
- ✅ Created comprehensive diagnostic tools for tensor comparison

### What's Still Broken ⚠️
- ⚠️ OCR produces gibberish text with very low confidence (0.02-0.0005 vs expected 0.85-0.97)
- ⚠️ Detecting only 4 boxes instead of expected 7
- ⚠️ Despite correct detector input, pipeline output is wrong

---

## Problem Statement

### Initial Symptoms

Running the ONNX OCR on a test image produced completely incorrect results:

**Actual Output (Bad):**
```json
{
  "text": "'0ri,.3 .2",
  "confidence": 0.052,
  "box": [[-3,13], [145,13], [145,66], [-3,66]]
}
```

**Expected Output (Good):**
```json
{
  "text": "V13.2",
  "confidence": 0.927,
  "box": [[6,18], [104,18], [104,50], [6,50]]
}
```

**Key observations:**
1. Text completely wrong (gibberish)
2. Confidence very low (0.052 vs 0.927)
3. Bounding boxes different
4. Both detector and recognizer failing

---

## Investigation Process

Followed systematic debugging approach recommended by the Strategist agent:

### Phase 1: Verify ONNX Models
- ✅ Confirmed ONNX models work correctly in Python with ONNX Runtime
- ✅ Python + ONNX produces correct results (7 detections, high confidence)
- ✅ Models were exported correctly with validation

### Phase 2: Compare Preprocessing Tensors
Created diagnostic scripts to dump and compare intermediate tensors:
1. `debug_tensors.py` - Saves Python reference tensors
2. `debug_tensors.mjs` - Saves JS tensors
3. `test_tensor_match.py` - Numerical comparison

**Initial findings:**
- Python tensor shape: `[1, 3, 182, 733]`
- JS tensor shape: `[1, 3, 192, 736]` ❌
- Shape mismatch identified as root cause

### Phase 3: Root Cause Analysis
Traced through preprocessing pipeline to find discrepancies.

---

## Bugs Found and Fixed

### Bug #1: Unnecessary Image Padding ⚠️ CRITICAL

**Location:** `packages/core/src/utils.ts` lines 171-199

**Problem:**
The `resizeLongSide` function was padding images to multiples of `align` (16) even when no resize was needed:

```typescript
// BEFORE (Wrong)
const paddedWidth = targetWidth % align === 0
  ? targetWidth
  : targetWidth + (align - (targetWidth % align));
const paddedHeight = targetHeight % align === 0
  ? targetHeight
  : targetHeight + (align - (targetHeight % align));
```

This caused:
- Input image: 733×182
- Padded to: 736×192
- Detector saw different dimensions than Python
- Wrong spatial features and box coordinates

**Root Cause:**
- ONNX CRAFT model was exported with **dynamic shapes**: `['batch', 3, 'height', 'width']`
- Model accepts any input size - padding unnecessary and harmful
- Python EasyOCR doesn't pad to alignment boundaries

**Fix:**
```typescript
// AFTER (Correct)
export const resizeLongSide = (image: RasterImage, maxSide: number, align: number): ResizeResult => {
  const { width, height } = image;
  const maxDim = Math.max(width, height);
  const scale = maxSide / maxDim;
  const targetWidth = Math.max(1, Math.floor(width * scale));
  const targetHeight = Math.max(1, Math.floor(height * scale));
  const resized = resizeImage(image, targetWidth, targetHeight);

  // IMPORTANT: Python EasyOCR does NOT pad to alignment boundaries.
  // The ONNX CRAFT model was exported with dynamic shapes and accepts any input size.
  // Padding was causing incorrect detection results because the model sees different input dimensions.
  // Simply return the resized image without padding to match Python's behavior.
  return { image: resized, scale };
};
```

**Impact:** Tensor shapes now match Python exactly.

---

### Bug #2: Alpha Channel Corruption ⚠️ CRITICAL

**Location:** `packages/node/src/index.ts` lines 23-40

**Problem:**
Sharp image library was returning RGBA (4 channels) but code claimed RGB (3 channels):

```typescript
// BEFORE (Wrong)
const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
return {
  data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  width: info.width,
  height: info.height,
  channels: 3,  // ❌ LYING - data actually has 4 channels!
  channelOrder,
};
```

This caused:
- Data size: 533,624 bytes (733×182×4) but claimed 3 channels
- Expected: 400,218 bytes (733×182×3)
- All pixel values misaligned - reading alpha as RGB
- Tensor values completely wrong

**Visual example of corruption:**
```
Actual data:  [R1, G1, B1, A1, R2, G2, B2, A2, ...]
Read as RGB:  [R1, G1, B1,  R2, G2, B2,  A2, ...]
                             ↑ Wrong! Should be A1, but reading R2
```

**Root Cause:**
- PNG files often have alpha channel
- Sharp's `.raw()` preserves original format including alpha
- No explicit alpha removal

**Fix:**
```typescript
// AFTER (Correct)
const image =
  typeof input === 'string'
    ? sharp(input).toColourspace('srgb').removeAlpha().raw()  // ✅ Remove alpha!
    : sharp(Buffer.from(input instanceof Uint8Array ? input : new Uint8Array(input)))
        .toColourspace('srgb')
        .removeAlpha()  // ✅ Remove alpha!
        .raw();

const { data, info } = await image.toBuffer({ resolveWithObject: true });

// After removeAlpha() and toColourspace('srgb'), we should have 3 channels (RGB)
if (info.channels !== 3) {
  throw new Error(`Expected 3 channels after RGB conversion, got ${info.channels}`);
}

return {
  data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  width: info.width,
  height: info.height,
  channels: 3,  // ✅ Now truly 3 channels
  channelOrder,
};
```

**Impact:** Pixel values now match Python exactly.

---

## Test Suite Created

### Tensor Comparison Tests

**1. `debug_tensors.py`** - Python Reference Generator
```bash
python3 debug_tensors.py <image_path> --output-dir debug_output
```

Generates:
- `detector_input_python.npy` - NCHW float32 tensor
- `detector_input_info.json` - Statistics and metadata
- `image_rgb_uint8.npy` - Raw RGB image (uint8)

**2. `debug_tensors.mjs`** - JavaScript Tensor Generator
```bash
node debug_tensors.mjs <image_path> debug_output
```

Generates:
- `detector_input_js.bin` - Raw float32 tensor data
- `detector_input_js.json` - Statistics and metadata
- `image_rgb_uint8.json` - Image metadata

**3. `test_tensor_match.py`** - Numerical Comparison
```bash
python3 test_tensor_match.py
```

Compares Python and JS tensors:
- Shape validation
- Element-wise difference
- Statistical analysis
- Reports mismatches

**Sample Output (After Fixes):**
```
✓ Test PASSED: Tensors match within tolerance!
  All 400218 elements are within 1e-05 of each other.

  Max absolute difference: 0.00000024
  Mean absolute difference: 0.00000000
```

### Additional Diagnostic Tools

**4. `test_onnx_python.py`** - Validate ONNX Models
```bash
python3 test_onnx_python.py
```

Tests ONNX models in Python to verify correct export.

**5. `debug_detector_output.mjs`** - Inspect Detector Outputs
```bash
node debug_detector_output.mjs
```

Shows detector heatmap statistics and thresholds.

**6. `debug_full_pipeline.mjs`** - End-to-End Pipeline Debug
```bash
node debug_full_pipeline.mjs
```

Runs full OCR pipeline with detailed logging and comparison to expected results.

**7. `compare_raw_pixels.mjs`** - Raw Pixel Comparison
```bash
node compare_raw_pixels.mjs
```

Compares raw pixel values before preprocessing (detects alpha channel issues).

---

## Current Status

### ✅ What's Working

1. **Image Loading**
   - Sharp correctly loads RGB without alpha channel
   - Data size matches expected (width × height × 3)
   - Pixel values match Python exactly

2. **Detector Preprocessing**
   - No unnecessary padding
   - Tensor shape: `[1, 3, 182, 733]` matches Python ✓
   - Tensor values match Python (max diff: 2.4e-7) ✓
   - Normalization correct (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

3. **ONNX Models**
   - Detector ONNX model validates correctly in Python
   - Recognizer ONNX model validates correctly in Python
   - Both produce correct results when run in Python with ONNX Runtime

### ⚠️ What's Still Broken

1. **Text Recognition Failure**
   - Produces gibberish: "'0ri,.3 .2" instead of "V13.2"
   - Very low confidence: 0.052 instead of 0.927
   - Character sequences make no sense

2. **Detection Count Mismatch**
   - Detecting only 4 text regions
   - Expected 7 text regions (from Python)
   - Boxes are in approximately correct locations but not exact

3. **End-to-End Pipeline**
   - Despite correct detector input, final output is wrong
   - Issue must be in:
     - Detector post-processing (box extraction from heatmaps)
     - Image cropping for recognizer
     - Recognizer preprocessing
     - CTC decoding
     - Or some combination of the above

### Test Results

**Tensor Match Test:**
```bash
$ python3 test_tensor_match.py
============================================================
RESULT
============================================================
✓ Test PASSED: Tensors match within tolerance!
  All 400218 elements are within 1e-05 of each other.
```

**OCR Output Test:**
```bash
$ node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png

Actual:
[0] "[1V/71,.3.2 " (confidence: 0.0272)
[1] "J3Jimqjwd-+aoewrta32" (confidence: 0.0005)
[2] "Imnipaitcvjpie jwut" (confidence: 0.0010)
[3] "KaeziwmimaizfihvlLiwg" (confidence: 0.0003)

Expected:
[0] "V13.2" (confidence: 0.9277)
[1] "30 May 2021" (confidence: 0.8519)
[2] "Version 1,.3.2" (confidence: 0.5781)
[3] "Faster greedy decoder (thanks @samayala22)" (confidence: 0.9538)
```

---

## Remaining Issues - Deep Dive

### Issue Analysis

Given that:
1. ✅ Detector input tensor matches Python perfectly
2. ✅ ONNX models work correctly in Python
3. ⚠️ JS ONNX Runtime produces wrong results

The bug must be in one of these areas:

### Hypothesis 1: Detector Post-Processing

**Location:** `packages/core/src/detector.ts` lines 77-223

**Potential issues:**
- `detectorPostprocess()` extracts boxes from heatmaps
- Uses connected components, morphological operations, minimum area rectangles
- Complex algorithm with many edge cases
- May have implementation differences from Python

**How to test:**
1. Compare heatmap outputs between Python and JS ONNX Runtime
2. Verify threshold values (lowText, linkThreshold, textThreshold)
3. Check if connected components algorithm matches Python

**Diagnostic script:**
```javascript
// In debug_detector_output.mjs
const textMap = tensorToHeatmap(textTensor);
const linkMap = tensorToHeatmap(linkTensor);
// Save heatmaps and compare with Python
```

### Hypothesis 2: Image Cropping

**Location:** `packages/core/src/crop.ts`

**Potential issues:**
- Crops image patches from detected boxes
- Applies perspective transforms for non-rectangular boxes
- Rotation handling
- Margin calculations

**How to test:**
1. Save cropped image patches to disk
2. Compare with Python's cropped patches
3. Check if perspective transform is correct

### Hypothesis 3: Recognizer Preprocessing

**Location:** `packages/core/src/recognizer.ts` lines 16-90

**Potential issues:**
- Grayscale conversion: `0.299*R + 0.587*G + 0.114*B`
- Bicubic resize implementation
- Padding strategy (pad right with last column replication)
- Normalization: `(pixel - 0.5) / 0.5`

**Known differences:**
- JS uses custom bicubic interpolation
- Python uses OpenCV's resize
- Small differences in interpolation can compound

**How to test:**
1. Save recognizer input tensors
2. Compare with Python recognizer inputs
3. Check if grayscale conversion is exact

### Hypothesis 4: CTC Decoding

**Location:** `packages/core/src/recognizer.ts` lines 92-171

**Potential issues:**
- Greedy CTC decoder implementation
- Blank index handling (should be 0)
- Character indexing off-by-one errors
- Confidence calculation formula

**How to test:**
1. Save raw logits from recognizer
2. Manually decode and compare
3. Verify charset indexing

**Character indexing:**
```
Model output: 97 classes
- Class 0: blank (CTC blank symbol)
- Class 1: charset[0] = '0'
- Class 2: charset[1] = '1'
- ...
- Class 96: charset[95] = 'z'
```

### Hypothesis 5: Tensor Layout Issues

**Potential issues:**
- ONNX Runtime might return tensors in unexpected layout
- Channel ordering (RGB vs BGR) in intermediate steps
- Row-major vs column-major memory layout

**How to test:**
1. Inspect raw tensor data at each pipeline stage
2. Compare tensor shapes and strides
3. Verify data is being read in correct order

---

## Next Steps (Recommended)

### Immediate Actions

1. **Compare Detector Outputs** (30 min)
   ```bash
   # Add logging to debug_detector_output.mjs to save heatmaps
   # Compare text/link heatmaps between Python and JS
   ```

2. **Test Recognizer in Isolation** (1 hour)
   - Create a test that:
     - Loads a single cropped text patch
     - Runs recognizer preprocessing
     - Compares preprocessed tensor with Python
     - Runs recognizer inference
     - Compares logits with Python

3. **Verify Charset** (15 min)
   ```bash
   # Check if charset length matches model output
   wc -c models/english_g2.charset.txt  # Should be 96
   # Model outputs 97 classes (96 chars + 1 blank)
   ```

### Systematic Debugging Approach

**Step 1: Isolate the Failing Component**

Create tests for each pipeline stage:
```javascript
// Test 1: Detector heatmaps
const heatmaps = await detector.session.run(...)
// Compare with Python heatmaps

// Test 2: Box extraction
const boxes = detectorPostprocess(...)
// Compare with Python boxes

// Test 3: Image cropping
const crops = buildCrops(...)
// Save crops as images, compare visually

// Test 4: Recognizer preprocessing
const recognizerInput = recognizerPreprocess(...)
// Compare tensor with Python

// Test 5: Recognizer logits
const logits = await recognizer.session.run(...)
// Compare with Python logits

// Test 6: CTC decoding
const decoded = ctcGreedyDecode(...)
// Compare with Python decoded text
```

**Step 2: Binary Search the Pipeline**

Find the first point where JS diverges from Python:
1. If heatmaps match → bug is in box extraction
2. If boxes match → bug is in cropping
3. If crops match → bug is in recognizer preprocessing
4. If recognizer inputs match → bug is in ONNX Runtime or model
5. If logits match → bug is in CTC decoding

**Step 3: Fix and Verify**

For each bug found:
1. Implement fix
2. Add regression test
3. Verify tensors still match at that stage
4. Verify end-to-end results improve

### Tools to Build

1. **Heatmap Visualizer**
   - Save detector heatmaps as images
   - Overlay on original image
   - Compare side-by-side with Python

2. **Crop Extractor**
   - Save all cropped patches as PNG files
   - Name them with expected text for easy verification
   - Compare with Python crops

3. **Tensor Dumper**
   - Generic utility to save any tensor at any pipeline stage
   - Comparison utility to diff two tensor dumps

---

## Key Files Modified

### Fixed Files ✅

1. **packages/core/src/utils.ts**
   - Removed padding logic from `resizeLongSide()` (lines 171-199)

2. **packages/node/src/index.ts**
   - Added `.removeAlpha()` to Sharp pipeline (lines 23-40)
   - Added validation for channel count

### Diagnostic Files Created 📊

1. **debug_tensors.py** - Python tensor dump
2. **debug_tensors.mjs** - JS tensor dump
3. **test_tensor_match.py** - Tensor comparison
4. **test_onnx_python.py** - ONNX model validation
5. **debug_detector_output.mjs** - Detector output inspection
6. **debug_full_pipeline.mjs** - Full pipeline debugging
7. **compare_raw_pixels.mjs** - Raw pixel comparison

### Key Files to Investigate 🔍

1. **packages/core/src/detector.ts** - Post-processing logic
2. **packages/core/src/crop.ts** - Image cropping
3. **packages/core/src/recognizer.ts** - Preprocessing and CTC decode
4. **packages/core/src/utils.ts** - Resize and image utilities

---

## Technical Details

### Normalization Values

**Detector (RGB, 3 channels):**
- Mean: `[0.485, 0.456, 0.406]` (ImageNet normalization)
- Std: `[0.229, 0.224, 0.225]`
- Formula: `(pixel/255 - mean) / std`

**Recognizer (Grayscale, 1 channel):**
- Mean: `0.5`
- Std: `0.5`
- Grayscale conversion: `0.299*R + 0.587*G + 0.114*B`
- Formula: `(pixel/255 - 0.5) / 0.5`

### ONNX Model Details

**Detector (craft_mlt_25k.onnx):**
- Input: `['batch', 3, 'height', 'width']` (dynamic H/W)
- Output 0: `['batch', '(height//2)', '(width//2)', 2]` (text+link heatmap)
- Output 1: `['batch', 32, '(height//2)', '(width//2)']` (features)

**Recognizer (english_g2.onnx):**
- Input: `[1, 1, 32, 100]` (grayscale, fixed 32x100)
- Text input: `[1, 1]` (int64, for teacher forcing - set to [0])
- Output: `[1, 24, 97]` (24 time steps, 97 classes)
  - Class 0: CTC blank
  - Classes 1-96: charset characters

### Charset Format

File: `models/english_g2.charset.txt`
- 96 characters total
- Single line, no newlines
- Order: `0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €ABCD...xyz`
- First 10: digits 0-9
- Last 10: letters q-z

---

## Important Insights

### What We Learned

1. **ONNX Models Support Dynamic Shapes**
   - No need for padding/alignment
   - Models were correctly exported
   - Padding was harmful, not helpful

2. **Sharp Image Loading Pitfalls**
   - `.raw()` preserves alpha channel
   - Must explicitly call `.removeAlpha()`
   - `info.channels` reflects actual data

3. **Preprocessing is Critical**
   - Even small differences compound
   - Tensor shape mismatches cascade through pipeline
   - Validation at each step essential

4. **Python Reference is Golden**
   - EasyOCR's Python implementation is the source of truth
   - ONNX models work correctly in Python
   - Bug must be in JS implementation

### What Didn't Work

1. **Initial hypothesis: Model export issues** ❌
   - Models validate perfectly in Python
   - Python + ONNX produces correct results

2. **Initial hypothesis: Normalization differences** ❌
   - Normalization values are correct
   - Formula matches Python exactly

3. **Initial hypothesis: Channel ordering** ❌
   - RGB vs BGR handling is correct
   - Sharp loads as sRGB (RGB)

### What's Still Unknown

1. **Why is recognition still failing?**
   - Detector input is perfect
   - Models work in Python
   - But JS produces gibberish

2. **Where exactly does the pipeline diverge?**
   - Need to binary search the pipeline
   - Compare intermediate outputs at each stage

3. **Is it a single bug or multiple bugs?**
   - Could be multiple issues compounding
   - Each stage needs individual validation

---

## Environment

**System:**
- macOS (Darwin 25.2.0)
- Node.js v22.15.1
- Bun (package manager and build tool)

**Dependencies:**
- `sharp` - Image processing (loads PNG with sRGB color space)
- `onnxruntime-node` - ONNX inference
- Python 3.11.12 (for reference tests)
- EasyOCR 1.7.2 (for reference)

**Models:**
- Detector: `models/onnx/craft_mlt_25k.onnx` (17.4 MB)
- Recognizer: `models/onnx/english_g2.onnx` (11.8 MB)
- Charset: `models/english_g2.charset.txt` (96 chars)

---

## How to Use This Handoff

### For Immediate Debugging:

1. **Verify fixes are applied:**
   ```bash
   rm -rf packages/*/dist
   bun run build
   python3 test_tensor_match.py  # Should PASS
   ```

2. **Run diagnostics:**
   ```bash
   node debug_full_pipeline.mjs  # See current OCR output
   python3 test_onnx_python.py   # Verify models work in Python
   ```

3. **Start with detector output comparison:**
   ```bash
   node debug_detector_output.mjs  # Check heatmap statistics
   # Compare with Python heatmap statistics
   ```

### For Systematic Investigation:

Follow the "Next Steps" section above:
1. Compare detector outputs
2. Test recognizer in isolation
3. Binary search the pipeline
4. Fix and verify each issue

### For Adding New Tests:

Use the existing diagnostic scripts as templates:
- `debug_*.mjs` files show how to inspect intermediate values
- `test_*.py` files show how to compare with Python
- All scripts are well-commented

---

## References

### Code Locations

- Detector preprocessing: `packages/core/src/detector.ts:22-47`
- Detector postprocessing: `packages/core/src/detector.ts:77-223`
- Recognizer preprocessing: `packages/core/src/recognizer.ts:16-90`
- CTC decoding: `packages/core/src/recognizer.ts:92-171`
- Image loading: `packages/node/src/index.ts:23-40`
- Resize utilities: `packages/core/src/utils.ts:31-199`
- Full pipeline: `packages/core/src/pipeline.ts:35-96`

### Python Reference

- EasyOCR normalization: `.venv/lib/python3.11/site-packages/easyocr/imgproc.py:20`
- ONNX export script: `models/export_onnx.py`
- Reference generation: `python_reference/generate_reference.py`

### Documentation

- ONNX model export: `models/export_onnx.py` (lines 54-75 for patches)
- Pipeline contract: `PIPELINE_CONTRACT.md` (if exists)
- Task tracking: `TASK_MATCH_PYTHON_REFERENCE.md` (if exists)

---

## Conclusion

**Progress Made:**
- ✅ Fixed critical preprocessing bugs
- ✅ Achieved perfect tensor match with Python
- ✅ Created comprehensive diagnostic tools
- ✅ Validated ONNX models are correct

**Work Remaining:**
- ⚠️ Debug why recognition produces gibberish despite correct input
- ⚠️ Fix detector to find all 7 text regions (currently finds 4)
- ⚠️ Identify and fix remaining pipeline issues

**Confidence Level:**
- High confidence that preprocessing is now correct
- High confidence that models are correct (validated in Python)
- Medium confidence that issue is in post-processing or cropping
- Low confidence in specific root cause of recognition failure

**Recommended Next Owner:**
Someone with:
- Experience debugging ML pipelines
- Familiarity with ONNX Runtime
- Patience for systematic binary search debugging
- Python+JS skills for cross-language comparison

---

## Contact

For questions about this handoff:
- Review diagnostic scripts in project root
- Check git commit history for detailed change descriptions
- Run `git log --oneline --grep="ONNX"` to see related commits

**Last Updated:** 2026-02-01
**Claude Session ID:** (not tracked - used through CLI)
