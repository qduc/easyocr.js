# ONNX OCR Debugging - Documentation Index

This directory contains debugging work for the ONNX OCR implementation.

---

## 📚 Documentation

### Start Here

1. **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)** - Quick summary of what was fixed
   - Read this first for a quick overview
   - 2-minute read

2. **[DEBUGGING_CHECKLIST.md](DEBUGGING_CHECKLIST.md)** - Quick start checklist
   - Use this to verify fixes and continue debugging
   - Step-by-step commands and checks
   - 5-minute read

3. **[HANDOFF_ONNX_DEBUGGING.md](HANDOFF_ONNX_DEBUGGING.md)** - Complete handoff document
   - Full technical details
   - Bug analysis and investigation process
   - Recommended next steps
   - 20-minute read

---

## 🚀 Quick Start

### Verify Everything Works

```bash
# 1. Rebuild
rm -rf packages/*/dist && bun run build

# 2. Test preprocessing (should PASS ✅)
python3 debug_tensors.py python_reference/validation/images/Screenshot_20260201_193653.png --output-dir debug_output
node debug_tensors.mjs python_reference/validation/images/Screenshot_20260201_193653.png debug_output
python3 test_tensor_match.py

# 3. Test OCR (still FAILS ⚠️ but preprocessing is correct)
node examples/node-ocr.mjs python_reference/validation/images/Screenshot_20260201_193653.png
```

### Expected Results

✅ **Tensor test:** PASS - Tensors match within tolerance
⚠️ **OCR test:** FAIL - Still produces gibberish (needs further debugging)

---

## 🐛 Current Status

### Fixed ✅
- Preprocessing bugs (padding and alpha channel)
- Detector input tensors match Python perfectly

### Still Broken ⚠️
- OCR produces gibberish text
- Very low confidence scores
- Only 4 detections instead of 7

### Next Steps
See [DEBUGGING_CHECKLIST.md](DEBUGGING_CHECKLIST.md) for detailed next steps.

---

## 🛠️ Diagnostic Scripts

All diagnostic scripts are in the project root:

| Script | Purpose |
|--------|---------|
| `debug_tensors.py` | Generate Python reference tensors |
| `debug_tensors.mjs` | Generate JS tensors |
| `test_tensor_match.py` | Compare Python and JS tensors |
| `test_onnx_python.py` | Validate ONNX models in Python |
| `debug_detector_output.mjs` | Inspect detector heatmaps |
| `debug_full_pipeline.mjs` | Full pipeline debug with comparison |
| `compare_raw_pixels.mjs` | Compare raw pixel values |

---

## 📊 Test Results

### Detector Preprocessing ✅
```
✓ Test PASSED: Tensors match within tolerance!
  Shape: [1, 3, 182, 733]
  Max difference: 0.00000024
  All 400218 elements match
```

### OCR Output ⚠️
```
Expected: "V13.2" (conf: 0.9277)
Actual:   "[1V/71,.3.2 " (conf: 0.0272) ❌

Expected: "30 May 2021" (conf: 0.8519)
Actual:   "J3Jimqjwd-+aoewrta32" (conf: 0.0005) ❌
```

---

## 💡 Key Insights

1. **Preprocessing is now correct** - tensors match Python perfectly
2. **ONNX models are correct** - validated in Python
3. **Bug is downstream** - in detector post-processing, cropping, or recognition
4. **Need systematic investigation** - binary search through pipeline stages

---

## 📝 Files Modified

- `packages/core/src/utils.ts` - Removed unnecessary padding
- `packages/node/src/index.ts` - Fixed alpha channel handling

---

**Last Updated:** 2026-02-01

For questions or to continue debugging, start with [DEBUGGING_CHECKLIST.md](DEBUGGING_CHECKLIST.md).
