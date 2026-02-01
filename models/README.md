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
