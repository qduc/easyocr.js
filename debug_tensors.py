#!/usr/bin/env python3
"""
Debug script to dump intermediate tensors from Python EasyOCR for comparison with JS implementation.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Dump EasyOCR intermediate tensors for debugging')
    parser.add_argument('image_path', help='Path to the test image')
    parser.add_argument('--output-dir', default='./debug_output', help='Directory to save tensor dumps')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import after arg parsing for better error messages
    try:
        import easyocr
        from easyocr import imgproc
        import torch
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install easyocr torch")
        return 1

    # Load image using OpenCV (like EasyOCR does)
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Failed to load image: {image_path}")
        return 1

    print(f"Image shape (OpenCV BGR): {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image range: [{img.min()}, {img.max()}]")

    # Convert BGR to RGB (EasyOCR does this)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"\nAfter BGR->RGB conversion:")
    print(f"  Shape: {img_rgb.shape}")
    print(f"  First pixel (RGB): {img_rgb[0, 0, :]}")

    # Save the RGB image as reference
    np.save(output_dir / 'image_rgb_uint8.npy', img_rgb)
    with open(output_dir / 'image_info.json', 'w') as f:
        json.dump({
            'shape': list(img_rgb.shape),
            'dtype': str(img_rgb.dtype),
            'min': int(img_rgb.min()),
            'max': int(img_rgb.max()),
            'first_pixel': img_rgb[0, 0, :].tolist(),
        }, f, indent=2)

    # Apply detector preprocessing (normalizeMeanVariance)
    mean = (0.485, 0.456, 0.406)
    variance = (0.229, 0.224, 0.225)

    print(f"\nApplying normalization:")
    print(f"  mean: {mean}")
    print(f"  std: {variance}")

    # EasyOCR's normalizeMeanVariance implementation
    img_normalized = img_rgb.copy().astype(np.float32)
    img_normalized -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img_normalized /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

    print(f"\nAfter normalization:")
    print(f"  Shape: {img_normalized.shape}")
    print(f"  Range: [{img_normalized.min():.6f}, {img_normalized.max():.6f}]")
    print(f"  Mean: {img_normalized.mean():.6f}")
    print(f"  Std: {img_normalized.std():.6f}")
    print(f"  First pixel (normalized): {img_normalized[0, 0, :]}")
    print(f"  First 10 values (flattened): {img_normalized.flatten()[:10]}")

    # Convert to NCHW format (for ONNX/PyTorch)
    # Currently: HWC (height, width, channels)
    # Target: CHW (channels, height, width)
    img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    img_nchw = np.expand_dims(img_chw, axis=0)  # CHW -> NCHW (add batch dimension)

    print(f"\nAfter HWC->NCHW conversion:")
    print(f"  Shape: {img_nchw.shape}")
    print(f"  First 10 values (flattened): {img_nchw.flatten()[:10]}")

    # Save the detector input tensor
    np.save(output_dir / 'detector_input_python.npy', img_nchw)
    with open(output_dir / 'detector_input_info.json', 'w') as f:
        json.dump({
            'shape': list(img_nchw.shape),
            'dtype': str(img_nchw.dtype),
            'min': float(img_nchw.min()),
            'max': float(img_nchw.max()),
            'mean': float(img_nchw.mean()),
            'std': float(img_nchw.std()),
            'first_10_values': img_nchw.flatten()[:10].tolist(),
        }, f, indent=2)

    print(f"\nSaved tensors to {output_dir}/")
    print(f"  - image_rgb_uint8.npy (original RGB image)")
    print(f"  - detector_input_python.npy (normalized NCHW tensor)")
    print(f"  - *.json (metadata)")

    print("\nNote: EasyOCR full pipeline skipped. Tensor dumps are sufficient for debugging.")

    print(f"\n✓ All debug outputs saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
