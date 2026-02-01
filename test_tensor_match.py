#!/usr/bin/env python3
"""
Test to verify that JS preprocessing produces the same tensor as Python.
"""
import sys
from pathlib import Path
import numpy as np
import struct


def load_js_tensor(bin_path):
    """Load the float32 tensor from JS binary output."""
    with open(bin_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data


def main():
    python_npy = Path('debug_output/detector_input_python.npy')
    js_bin = Path('debug_output/detector_input_js.bin')

    if not python_npy.exists():
        print(f"Error: Python tensor not found: {python_npy}")
        print("Run: python3 debug_tensors.py <image> --output-dir debug_output")
        return 1

    if not js_bin.exists():
        print(f"Error: JS tensor not found: {js_bin}")
        print("Run: node debug_tensors.mjs <image> debug_output")
        return 1

    print('=' * 60)
    print('TENSOR MATCH TEST')
    print('=' * 60)

    print(f"\nLoading Python tensor: {python_npy}")
    py_tensor = np.load(python_npy)
    print(f"  Shape: {py_tensor.shape}")
    print(f"  Dtype: {py_tensor.dtype}")
    print(f"  Size: {py_tensor.size}")

    print(f"\nLoading JS tensor: {js_bin}")
    js_data = load_js_tensor(js_bin)
    print(f"  Size: {js_data.size}")

    # Reshape JS data to match Python shape (assuming NCHW format)
    if js_data.size == py_tensor.size:
        js_tensor = js_data.reshape(py_tensor.shape)
        print(f"  Reshaped to: {js_tensor.shape}")
    else:
        print(f"\n✗ ERROR: Size mismatch!")
        print(f"  Python: {py_tensor.size} elements")
        print(f"  JS:     {js_data.size} elements")
        return 1

    # Compare shapes
    print('\n' + '=' * 60)
    print('SHAPE COMPARISON')
    print('=' * 60)

    if py_tensor.shape == js_tensor.shape:
        print(f"✓ Shapes MATCH: {py_tensor.shape}")
    else:
        print(f"✗ Shapes DO NOT MATCH:")
        print(f"  Python: {py_tensor.shape}")
        print(f"  JS:     {js_tensor.shape}")
        return 1

    # Compare values
    print('\n' + '=' * 60)
    print('VALUE COMPARISON')
    print('=' * 60)

    diff = np.abs(py_tensor - js_tensor)
    max_diff = diff.max()
    mean_diff = diff.mean()
    tolerance = 1e-5

    print(f"Total elements: {py_tensor.size}")
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    mismatch_count = np.sum(diff > tolerance)
    mismatch_pct = (mismatch_count / py_tensor.size) * 100
    print(f"Elements exceeding tolerance ({tolerance}): {mismatch_count} ({mismatch_pct:.2f}%)")

    # Statistics
    print(f"\nPython tensor statistics:")
    print(f"  Min: {py_tensor.min():.6f}, Max: {py_tensor.max():.6f}")
    print(f"  Mean: {py_tensor.mean():.6f}, Std: {py_tensor.std():.6f}")

    print(f"\nJS tensor statistics:")
    print(f"  Min: {js_tensor.min():.6f}, Max: {js_tensor.max():.6f}")
    print(f"  Mean: {js_tensor.mean():.6f}, Std: {js_tensor.std():.6f}")

    # Show first 10 values
    py_flat = py_tensor.flatten()
    js_flat = js_tensor.flatten()
    diff_flat = diff.flatten()

    print('\nFirst 10 values comparison:')
    print('  Index | Python      | JS          | Diff       | Status')
    print('  ' + '-' * 60)
    for i in range(min(10, py_flat.size)):
        marker = '✗' if diff_flat[i] > tolerance else '✓'
        print(f"  {i:5d} | {py_flat[i]:11.6f} | {js_flat[i]:11.6f} | {diff_flat[i]:10.8f} | {marker}")

    # Overall result
    print('\n' + '=' * 60)
    print('RESULT')
    print('=' * 60)

    if max_diff < tolerance:
        print('✓ Test PASSED: Tensors match within tolerance!')
        print(f"  All {py_tensor.size} elements are within {tolerance} of each other.")
        return 0
    else:
        print(f'✗ Test FAILED: Tensors differ by up to {max_diff:.8f}')
        print(f"  {mismatch_count} out of {py_tensor.size} elements exceed tolerance.")

        # Show worst mismatches
        worst_indices = np.argsort(diff_flat)[-5:][::-1]
        print(f"\nTop 5 worst mismatches:")
        print('  Index | Python      | JS          | Diff')
        print('  ' + '-' * 50)
        for idx in worst_indices:
            print(f"  {idx:5d} | {py_flat[idx]:11.6f} | {js_flat[idx]:11.6f} | {diff_flat[idx]:10.8f}")

        return 1


if __name__ == '__main__':
    sys.exit(main())
