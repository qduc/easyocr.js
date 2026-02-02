# Exporting EasyOCR models to ONNX

This directory stores the PyTorch weights used by EasyOCR. Use the script below to
export detector + recognizer models to ONNX and optionally validate outputs with
ONNX Runtime.

## Prerequisites

Create a Python environment and install deps (same as the Python reference):

```bash
uv venv
source .venv/bin/activate
uv pip install -r python_reference/requirements.txt
uv pip install onnxruntime
# OR for GPU support:
uv pip install onnxruntime-gpu
```

## Export detector + recognizer

```bash
python models/export_onnx.py --detector --recognizer --validate
```

Outputs go to `models/onnx/` by default:
- `models/onnx/craft_mlt_25k.onnx`
- `models/onnx/english_g2.onnx`

## Options

```bash
python models/export_onnx.py --help
```

Highlights:
- `--output-dir` to change destination
- `--detector-shape` / `--recognizer-shape` to change dummy export shapes
- `--opset` to set ONNX opset
- `--validate` to compare ONNX Runtime outputs vs PyTorch

## Notes

- The exporter uses the `.pth` files in this folder and disables downloads.
- If EasyOCR changes internal attribute names, the script may need minor updates.
- The ONNX files are generated artifacts; commit them only if you explicitly want to version them.

## Quantize + compare

Quantize an existing ONNX model, run inference on dummy inputs, and compare
outputs with the full-precision version:

```bash
python models/quantize_onnx.py models/onnx/english_g2.onnx
```

You can override dynamic input shapes if needed:

```bash
python models/quantize_onnx.py models/onnx/craft_mlt_25k.onnx \
  --shape input=1,3,256,256
```

## Static QDQ quantization (recommended for Conv-heavy models)

Static QDQ quantization uses calibration images and avoids unsupported
`ConvInteger` kernels on CPU.

```bash
python models/quantize_onnx_static.py models/onnx/craft_mlt_25k.onnx \
  --mode detector \
  --calib-dir tmp/calib_images
```

```bash
python models/quantize_onnx_static.py models/onnx/english_g2.onnx \
  --mode recognizer \
  --calib-dir tmp/calib_images
```

### GPU Acceleration for Quantization

If you have `onnxruntime-gpu` (NVIDIA) or standard `onnxruntime` (macOS/CoreML) installed, you can use the `--gpu` flag to speed up the calibration forward passes. It will automatically detect `CUDAExecutionProvider` or `CoreMLExecutionProvider`.

```bash
python models/quantize_onnx_static.py ... --gpu
```
