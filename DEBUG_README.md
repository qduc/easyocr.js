# ONNX OCR Debugging Hub

This document summarizes the debugging work for the ONNX OCR implementation.

## 🚀 Quick Start (Verification)

To verify the current state:

```bash
# 1. Build the project
bun run build

# 2. Run the full pipeline debug script
node debug_full_pipeline.mjs
```

**Status:** Resolved ✅ - JavaScript OCR now correctly recognizes text and matches the Python reference.

---

## 📊 Current Status

| Component | Status | Match % |
|-----------|---------|---------|
| Image Loading | ✅ Fixed | 100% |
| Detector Preprocessing | ✅ Fixed | 100% |
| Recognizer Preprocessing | ✅ Fixed | ~98% |
| Detector Grouping | ✅ Optimized | High |
| CTC Decoding | ✅ Fixed | Legible |

---

## 🐛 Resolved Issues

1. **Bug #1: Alignment Padding**: Removed unnecessary 16px alignment padding in `utils.ts` that was corrupting detector spatial features.
2. **Bug #2: Alpha Corruption**: Fixed Sharp pipeline to explicitly remove alpha channels, preventing pixel offset corruption.
3. **Bug #3: CTC Decoding**: Fixed loop logic in `ctcGreedyDecode` to correctly handle blanks and find max probability characters.
4. **Bug #4: Recognizer Padding**: Switched from edge padding to neutral mean padding in `padToWidth`.
5. **Optimization: Word Bundling**: Disabled paragraph merging when `paragraph: false` to prevent squashing in the fixed-width recognizer window.

---

## 🛠️ Diagnostic Toolkit

### Verification Scripts

| Script | Purpose |
|--------|---------|
| `test_tensor_match.py` | Compares JS tensors against Python master reference. |
| `debug_full_pipeline.mjs` | Runs end-to-end OCR and compares against expected text. |
| `compare_raw_pixels.mjs` | Verifies image loading consistency. |

---

## 📚 Documentation Index

- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)**: Technical breakdown of all applied patches.
- **[DEBUGGING_CHECKLIST.md](DEBUGGING_CHECKLIST.md)**: Step-by-step verification checklist.
- **[HANDOFF_ONNX_DEBUGGING.md](HANDOFF_ONNX_DEBUGGING.md)**: Full architectural post-mortem and final status.

---

**Last Updated:** 2026-02-02
