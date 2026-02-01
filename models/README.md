# Exporting EasyOCR models to ONNX

## Detector (CRAFT)
EasyOCR uses CRAFT. To export:
1. Load CRAFT model in PyTorch.
2. `torch.onnx.export`.

## Recognizer
Choose a model (e.g., `english_g2`).
Export to ONNX.

## Export Script
I will create a script `export_onnx.py` in this directory soon.
