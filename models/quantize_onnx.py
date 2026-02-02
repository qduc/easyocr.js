#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_shape_override(value: str) -> tuple[str, list[int]]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('Shape override must be in name=1,3,256,256 form.')
    name, shape_str = value.split('=', 1)
    parts = [int(part) for part in shape_str.split(',') if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('Shape override must include at least one dimension.')
    return name, parts


def ort_type_to_dtype(type_str: str) -> np.dtype:
    mapping: dict[str, Any] = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(int16)': np.int16,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(bool)': np.bool_,
    }
    if type_str not in mapping:
        raise ValueError(f'Unsupported ONNX input type: {type_str}')
    return mapping[type_str]


def default_shape_for_input(input_def: ort.NodeArg, default_hw: int) -> list[int]:
    shape = []
    for dim in input_def.shape:
        if isinstance(dim, int):
            shape.append(dim)
        else:
            shape.append(None)

    if len(shape) == 4:
        batch = shape[0] if shape[0] is not None else 1
        channels = shape[1] if shape[1] is not None else 3
        height = shape[2] if shape[2] is not None else default_hw
        width = shape[3] if shape[3] is not None else default_hw
        return [batch, channels, height, width]

    return [dim if dim is not None else 1 for dim in shape]


def build_inputs(
    session: ort.InferenceSession,
    *,
    seed: int,
    default_hw: int,
    shape_overrides: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for input_def in session.get_inputs():
        name = input_def.name
        shape = shape_overrides.get(name, default_shape_for_input(input_def, default_hw))
        dtype = ort_type_to_dtype(input_def.type)
        if np.issubdtype(dtype, np.floating):
            data = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 10, size=shape, dtype=dtype)
        elif dtype == np.bool_:
            data = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            raise ValueError(f'Unsupported dtype for input {name}: {dtype}')
        feeds[name] = data
    return feeds


def compare_outputs(
    ref_outputs: list[np.ndarray],
    test_outputs: list[np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if len(ref_outputs) != len(test_outputs):
        raise ValueError(
            f'Output count mismatch: {len(ref_outputs)} (full) vs {len(test_outputs)} (quant)'
        )
    for idx, (ref, test) in enumerate(zip(ref_outputs, test_outputs)):
        diff = np.abs(ref - test)
        max_diff = float(diff.max()) if diff.size else 0.0
        mean_diff = float(diff.mean()) if diff.size else 0.0
        allclose = bool(np.allclose(ref, test, atol=atol, rtol=rtol))
        results.append(
            {
                'index': idx,
                'shape': list(ref.shape),
                'dtype_full': str(ref.dtype),
                'dtype_quant': str(test.dtype),
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'allclose': allclose,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Quantize an ONNX model and compare outputs.')
    parser.add_argument('model', help='Path to the full-precision ONNX model.')
    parser.add_argument(
        '--output',
        help='Path for quantized model. Defaults to <model>.quant.onnx',
    )
    parser.add_argument(
        '--default-hw',
        type=int,
        default=256,
        help='Default H/W for dynamic 4D inputs (when shape dims are None).',
    )
    parser.add_argument(
        '--shape',
        action='append',
        default=[],
        type=parse_shape_override,
        help='Override input shape (name=1,3,256,256). Can be repeated.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for dummy inputs.')
    parser.add_argument('--atol', type=float, default=1e-2, help='Abs tolerance for comparison.')
    parser.add_argument('--rtol', type=float, default=1e-2, help='Rel tolerance for comparison.')
    parser.add_argument(
        '--op-types',
        default='MatMul,Gemm',
        help='Comma-separated ONNX op types to quantize (use "all" for default behavior).',
    )
    parser.add_argument('--gpu', action='store_true', help='Use CUDAExecutionProvider for acceleration.')
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f'Model not found: {model_path}')

    output_path = Path(args.output) if args.output else model_path.with_suffix('.quant.onnx')

    op_types_to_quantize = None
    if args.op_types.strip().lower() != 'all':
        op_types_to_quantize = [op.strip() for op in args.op_types.split(',') if op.strip()]

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types_to_quantize,
    )

    available = ort.get_available_providers()
    gpu_providers = [p for p in ['CUDAExecutionProvider', 'CoreMLExecutionProvider'] if p in available]

    if args.gpu:
        providers = gpu_providers + ['CPUExecutionProvider']
    else:
        # Default to first available GPU provider if any, else CPU
        providers = [gpu_providers[0]] + ['CPUExecutionProvider'] if gpu_providers else ['CPUExecutionProvider']

    full_session = ort.InferenceSession(str(model_path), providers=providers)
    quant_session = ort.InferenceSession(str(output_path), providers=providers)

    shape_overrides = {name: shape for name, shape in args.shape}
    feeds = build_inputs(
        full_session,
        seed=args.seed,
        default_hw=args.default_hw,
        shape_overrides=shape_overrides,
    )

    full_outputs = full_session.run(None, feeds)
    quant_outputs = quant_session.run(None, feeds)

    results = compare_outputs(
        full_outputs,
        quant_outputs,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(f'Wrote quantized model to {output_path}')
    for result in results:
        print(
            'output[{index}] shape={shape} '
            'dtype_full={dtype_full} dtype_quant={dtype_quant} '
            'max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} allclose={allclose}'.format(**result)
        )


if __name__ == '__main__':
    main()
