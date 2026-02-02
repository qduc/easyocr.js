#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image, ImageEnhance


def list_images(root: Path) -> list[Path]:
    patterns = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(paths)


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.split(',') if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('Shape must be "N,C,H,W" (e.g. 1,3,768,768).')
    return tuple(parts)  # type: ignore[return-value]


def preprocess_detector(image_path: Path, shape: tuple[int, int, int, int]) -> np.ndarray:
    _, channels, height, width = shape
    if channels != 3:
        raise ValueError(f'Detector expects 3 channels, got {channels}')
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)
    return array


def preprocess_recognizer(image_path: Path, shape: tuple[int, int, int, int]) -> np.ndarray:
    _, channels, height, width = shape
    if channels != 1:
        raise ValueError(f'Recognizer expects 1 channel, got {channels}')
    image = Image.open(image_path).convert('L')
    image = image.resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    array = np.expand_dims(array, axis=0)
    return array


def identify_inputs(session: ort.InferenceSession) -> dict[str, str | None]:
    inputs = session.get_inputs()
    if len(inputs) == 1:
        return {'image': inputs[0].name, 'text': None}

    image_name = None
    text_name = None
    for input_def in inputs:
        if input_def.type == 'tensor(int64)':
            text_name = input_def.name
        elif input_def.type.startswith('tensor(float'):
            image_name = input_def.name

    if image_name is None:
        image_name = inputs[0].name
    if text_name is None and len(inputs) > 1:
        text_name = inputs[1].name

    return {'image': image_name, 'text': text_name}


def augment_image(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    if rng.random() < 0.7:
        scale = rng.uniform(0.75, 1.1)
        width, height = image.size
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        image = image.resize((new_width, new_height), Image.BILINEAR)

    if rng.random() < 0.7:
        width, height = image.size
        crop_scale = rng.uniform(0.7, 1.0)
        crop_width = max(1, int(width * crop_scale))
        crop_height = max(1, int(height * crop_scale))
        if crop_width < width or crop_height < height:
            left = int(rng.integers(0, max(1, width - crop_width + 1)))
            top = int(rng.integers(0, max(1, height - crop_height + 1)))
            image = image.crop((left, top, left + crop_width, top + crop_height))

    if rng.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return image


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        session: ort.InferenceSession,
        image_paths: list[tuple[Path, int]],
        *,
        mode: str,
        detector_shape: tuple[int, int, int, int],
        recognizer_shape: tuple[int, int, int, int],
        seed: int,
    ) -> None:
        self._session = session
        self._image_paths = image_paths
        self._mode = mode
        self._detector_shape = detector_shape
        self._recognizer_shape = recognizer_shape
        self._rng = np.random.default_rng(seed)
        self._iterator: Iterator[Path] | None = None
        self._input_names = identify_inputs(session)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = iter(self._image_paths)

        try:
            image_path, augment_index = next(self._iterator)
        except StopIteration:
            return None

        if self._mode == 'detector':
            image = Image.open(image_path).convert('RGB')
            if augment_index > 0:
                image = augment_image(image, self._rng)
            image = image.resize((self._detector_shape[3], self._detector_shape[2]), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
            array = np.expand_dims(array, axis=0)
            image = array
            return {self._input_names['image']: image}

        image = Image.open(image_path).convert('L')
        if augment_index > 0:
            image = augment_image(image, self._rng)
        image = image.resize((self._recognizer_shape[3], self._recognizer_shape[2]), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)
        array = np.expand_dims(array, axis=0)
        image = array
        feeds: dict[str, np.ndarray] = {self._input_names['image']: image}
        if self._input_names['text']:
            batch = self._recognizer_shape[0]
            feeds[self._input_names['text']] = np.zeros((batch, 1), dtype=np.int64)
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
    parser = argparse.ArgumentParser(description='Static QDQ quantization for ONNX models.')
    parser.add_argument('model', help='Path to the full-precision ONNX model.')
    parser.add_argument(
        '--output',
        help='Path for quantized model. Defaults to <model>.qdq.onnx',
    )
    parser.add_argument(
        '--mode',
        choices=['detector', 'recognizer'],
        required=True,
        help='Model type for preprocessing.',
    )
    parser.add_argument(
        '--calib-dir',
        required=True,
        help='Directory containing calibration images.',
    )
    parser.add_argument('--num-samples', type=int, default=20, help='Max calibration images.')
    parser.add_argument(
        '--detector-shape',
        type=parse_shape,
        default='1,3,768,768',
        help='Detector input shape.',
    )
    parser.add_argument(
        '--recognizer-shape',
        type=parse_shape,
        default='1,1,32,100',
        help='Recognizer input shape.',
    )
    parser.add_argument(
        '--augment-per-image',
        type=int,
        default=1,
        help='Number of augmented variants per image (1 disables augmentation).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for augmentation.')
    parser.add_argument(
        '--per-channel',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable or disable per-channel weight quantization.',
    )
    parser.add_argument(
        '--activation-type',
        choices=['qint8', 'quint8'],
        default='qint8',
        help='Activation quantization type.',
    )
    parser.add_argument(
        '--weight-type',
        choices=['qint8', 'quint8'],
        default='qint8',
        help='Weight quantization type.',
    )
    parser.add_argument(
        '--reduce-range',
        action='store_true',
        help='Enable reduce-range quantization (often more conservative).',
    )
    parser.add_argument('--atol', type=float, default=1e-2, help='Abs tolerance for comparison.')
    parser.add_argument('--rtol', type=float, default=1e-2, help='Rel tolerance for comparison.')
    parser.add_argument('--gpu', action='store_true', help='Use CUDAExecutionProvider for acceleration.')
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f'Model not found: {model_path}')

    calib_dir = Path(args.calib_dir).resolve()
    if not calib_dir.exists():
        raise SystemExit(f'Calibration dir not found: {calib_dir}')

    image_paths = list_images(calib_dir)
    if not image_paths:
        raise SystemExit(f'No images found in calibration dir: {calib_dir}')
    augmented: list[tuple[Path, int]] = []
    for path in image_paths:
        for augment_index in range(args.augment_per_image):
            augmented.append((path, augment_index))
    if len(augmented) > args.num_samples:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(augmented)
        augmented = augmented[: args.num_samples]

    output_path = Path(args.output) if args.output else model_path.with_suffix('.qdq.onnx')

    available = ort.get_available_providers()
    gpu_providers = [p for p in ['CUDAExecutionProvider', 'CoreMLExecutionProvider'] if p in available]

    if args.gpu:
        providers = gpu_providers + ['CPUExecutionProvider']
    else:
        # Default to first available GPU provider if any, else CPU
        providers = [gpu_providers[0]] + ['CPUExecutionProvider'] if gpu_providers else ['CPUExecutionProvider']

    session = ort.InferenceSession(str(model_path), providers=providers)
    reader = ImageCalibrationReader(
        session,
        augmented,
        mode=args.mode,
        detector_shape=args.detector_shape,
        recognizer_shape=args.recognizer_shape,
        seed=args.seed,
    )

    activation_type = QuantType.QInt8 if args.activation_type == 'qint8' else QuantType.QUInt8
    weight_type = QuantType.QInt8 if args.weight_type == 'qint8' else QuantType.QUInt8

    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        calibrate_method=CalibrationMethod.MinMax,
    )

    full_session = ort.InferenceSession(str(model_path), providers=providers)
    quant_session = ort.InferenceSession(str(output_path), providers=providers)

    first_path = augmented[0][0]
    if args.mode == 'detector':
        image = preprocess_detector(first_path, args.detector_shape)
        feeds = {identify_inputs(full_session)['image']: image}
    else:
        image = preprocess_recognizer(first_path, args.recognizer_shape)
        names = identify_inputs(full_session)
        feeds = {names['image']: image}
        if names['text']:
            batch = args.recognizer_shape[0]
            feeds[names['text']] = np.zeros((batch, 1), dtype=np.int64)

    full_outputs = full_session.run(None, feeds)
    quant_outputs = quant_session.run(None, feeds)

    results = compare_outputs(
        full_outputs,
        quant_outputs,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(f'Wrote quantized model to {output_path}')
    print(f'Calibration samples used: {len(augmented)}')
    for result in results:
        print(
            'output[{index}] shape={shape} '
            'dtype_full={dtype_full} dtype_quant={dtype_quant} '
            'max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} allclose={allclose}'.format(**result)
        )


if __name__ == '__main__':
    main()
