# ONNX OCR Bug Fixes - Summary

**Date:** 2026-02-01
**Status:** Preprocessing Fixed ✅, Recognition Still Failing ⚠️

---

## Quick Summary

Fixed two critical bugs in preprocessing that caused tensor mismatches. Detector input now matches Python perfectly, but OCR still produces gibberish output. Issue is downstream in the pipeline.

---

## Bugs Fixed

### Bug #1: Unnecessary Padding ⚠️ CRITICAL

**File:** `packages/core/src/utils.ts` (lines 171-199)

**Before:**
```typescript
// Wrong: Padded image to multiples of 16
const paddedWidth = targetWidth % align === 0 ? targetWidth : targetWidth + (align - (targetWidth % align));
const paddedHeight = targetHeight % align === 0 ? targetHeight : targetHeight + (align - (targetHeight % align));
// Caused: 733×182 → 736×192 (wrong shape!)
```

**After:**
```typescript
// Correct: No padding - model supports dynamic shapes
return { image: resized, scale };
// Result: 733×182 → 733×182 (correct!)
```

**Impact:** Tensor shape now matches Python: `[1, 3, 182, 733]` ✅

---

### Bug #2: Alpha Channel Corruption ⚠️ CRITICAL

**File:** `packages/node/src/index.ts` (lines 23-40)

**Before:**
```typescript
// Wrong: Sharp returned RGBA but claimed RGB
const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
// Data had 533,624 bytes (RGBA) but claimed 3 channels
// All pixel values misaligned!
```

**After:**
```typescript
// Correct: Explicitly remove alpha channel
const image = sharp(input).toColourspace('srgb').removeAlpha().raw();
// Data now has 400,218 bytes (RGB) correctly
```

**Impact:** Pixel values now match Python exactly (max diff: 2.4e-7) ✅

---

## Test Results

### ✅ Detector Preprocessing Test
```bash
$ python3 test_tensor_match.py

============================================================
RESULT
============================================================
✓ Test PASSED: Tensors match within tolerance!
  All 400218 elements are within 1e-05 of each other.

Python tensor statistics:
  Min: -2.117904, Max: 2.640000
  Mean: 2.310644, Std: 0.616117

JS tensor statistics:
  Min: -2.117904, Max: 2.640000
  Mean: 2.310644, Std: 0.616117
```

### ⚠️ OCR Output Test
```bash
$ node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png

Expected:
[0] "V13.2" (confidence: 0.9277)
[1] "30 May 2021" (confidence: 0.8519)
[2] "Version 1,.3.2" (confidence: 0.5781)

Actual:
[0] "[1V/71,.3.2 " (confidence: 0.0272)  ❌
[1] "J3Jimqjwd-+aoewrta32" (confidence: 0.0005)  ❌
[2] "Imnipaitcvjpie jwut" (confidence: 0.0010)  ❌
```

**Conclusion:** Preprocessing fixed, but recognition still failing!

---

## What's Still Broken

1. **Text Recognition:** Produces gibberish instead of real text
2. **Confidence Scores:** 100x-1000x lower than expected
3. **Detection Count:** Only 4 boxes found instead of 7

**Root Cause:** Unknown - needs further investigation

**Suspects:**
- Detector post-processing (box extraction from heatmaps)
- Image cropping for recognizer
- Recognizer preprocessing
- CTC decoding

---

## How to Verify Fixes

```bash
# 1. Clean rebuild
rm -rf packages/*/dist && bun run build

# 2. Test tensor match (should PASS)
python3 debug_tensors.py python_reference/validation/images/Screenshot_20260201_193653.png --output-dir debug_output
node debug_tensors.mjs python_reference/validation/images/Screenshot_20260201_193653.png debug_output
python3 test_tensor_match.py

# 3. Test OCR (still FAILS but preprocessing is correct)
node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png
```

---

## Diagnostic Tools Created

All scripts are in the project root:

1. **`debug_tensors.py`** - Save Python reference tensors
2. **`debug_tensors.mjs`** - Save JS tensors
3. **`test_tensor_match.py`** - Compare tensors numerically
4. **`test_onnx_python.py`** - Validate ONNX models
5. **`debug_detector_output.mjs`** - Inspect detector outputs
6. **`debug_full_pipeline.mjs`** - Full pipeline debug with comparison
7. **`compare_raw_pixels.mjs`** - Raw pixel comparison

---

## Next Steps

1. **Compare detector heatmap outputs** between Python and JS
2. **Test recognizer in isolation** with a simple text image
3. **Binary search the pipeline** to find where JS diverges from Python
4. **Fix the remaining bugs** and verify end-to-end

See `DEBUGGING_CHECKLIST.md` for detailed next steps.

---

## Files Modified

### Core Changes
- ✅ `packages/core/src/utils.ts` - Removed padding
- ✅ `packages/node/src/index.ts` - Fixed alpha channel handling

### Documentation
- 📄 `HANDOFF_ONNX_DEBUGGING.md` - Full handoff document
- 📄 `DEBUGGING_CHECKLIST.md` - Quick start checklist
- 📄 `FIXES_SUMMARY.md` - This file

---

## Key Insights

1. **ONNX models are correct** - validated in Python
2. **Preprocessing is critical** - even small bugs cascade through pipeline
3. **Dynamic shapes work** - no padding needed for CRAFT model
4. **Sharp library gotcha** - must explicitly remove alpha channel
5. **Tensor comparison is essential** - caught bugs that were invisible otherwise

---

**For full details, see:** `HANDOFF_ONNX_DEBUGGING.md`
