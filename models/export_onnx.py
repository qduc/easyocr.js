#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.split(',') if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('Shape must be "N,C,H,W" (e.g. 1,3,768,768).')
    return tuple(parts)  # type: ignore[return-value]


def require_easyocr():
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - message only
        raise SystemExit(
            'easyocr is not installed. Install python deps first, e.g.\n'
            '  uv pip install -r python_reference/requirements.txt\n'
            '  uv pip install onnxruntime\n'
        ) from exc
    return easyocr


def build_reader(models_dir: Path, lang: str, recog_network: str | None):
    easyocr = require_easyocr()
    base_kwargs = {
        'gpu': False,
        'quantize': False,
        'model_storage_directory': str(models_dir),
        'download_enabled': False,
    }
    attempts = []
    if recog_network:
        attempts.append({**base_kwargs, 'recog_network': recog_network})
    attempts.append(base_kwargs)
    attempts.append({'gpu': False, 'quantize': False})

    last_error = None
    for kwargs in attempts:
        try:
            return easyocr.Reader([lang], **kwargs)
        except TypeError as err:
            last_error = err
    raise SystemExit(f'Failed to initialize easyocr.Reader: {last_error}')


def patch_easyocr_for_onnx_export():
    """
    EasyOCR's BidirectionalLSTM calls `flatten_parameters()` inside forward(). During
    ONNX export (both dynamo and legacy), this can trigger ScriptObject-related
    tracing/export failures. For export purposes, flattening is unnecessary, so we
    patch the forward method to skip it.
    """
    try:
        import easyocr.model.modules as easyocr_modules
    except Exception:
        return

    if not hasattr(easyocr_modules, 'BidirectionalLSTM'):
        return

    def forward_no_flatten(self, input):  # noqa: ANN001
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

    easyocr_modules.BidirectionalLSTM.forward = forward_no_flatten


def pick_module(reader, candidates: list[str]):
    for name in candidates:
        value = getattr(reader, name, None)
        if isinstance(value, torch.nn.Module):
            return value
    return None


def normalize_outputs(outputs):
    if torch.is_tensor(outputs):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        return [output for output in outputs if torch.is_tensor(output)]
    if isinstance(outputs, dict):
        return [output for output in outputs.values() if torch.is_tensor(output)]
    raise RuntimeError('Unsupported model output type for ONNX export.')


def normalize_inputs(inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]):
    if torch.is_tensor(inputs):
        return [inputs]
    if isinstance(inputs, (list, tuple)) and all(torch.is_tensor(value) for value in inputs):
        return list(inputs)
    raise RuntimeError('Unsupported model input type for ONNX export.')


def export_model(
    name: str,
    model: torch.nn.Module,
    dummy_inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    output_path: Path,
    opset: int,
    *,
    input_names: list[str] | None = None,
    dynamic_spatial: bool = True,
    use_dynamic_axes: bool = True,
):
    model.eval()
    inputs = normalize_inputs(dummy_inputs)
    if input_names is None:
        input_names = ['input']
    if len(input_names) != len(inputs):
        raise ValueError(f'input_names length ({len(input_names)}) does not match inputs ({len(inputs)})')

    with torch.no_grad():
        torch_outputs = normalize_outputs(model(*inputs))

    output_names = [f'{name}_output_{idx}' for idx in range(len(torch_outputs))]
    dynamic_axes = None
    if use_dynamic_axes:
        dynamic_axes = {}
        for input_name, tensor in zip(input_names, inputs):
            dynamic_axes[input_name] = {0: 'batch'}
            if dynamic_spatial and tensor.dim() == 4:
                dynamic_axes[input_name].update({2: 'height', 3: 'width'})

        for output_name, output in zip(output_names, torch_outputs):
            if output.dim() >= 1:
                dynamic_axes[output_name] = {0: 'batch'}
                if output.dim() == 4:
                    dynamic_axes[output_name].update({2: 'height', 3: 'width'})

    torch.onnx.export(
        model,
        tuple(inputs) if len(inputs) > 1 else inputs[0],
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    return torch_outputs


def validate_onnx(
    name: str,
    onnx_path: Path,
    dummy_inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    torch_outputs: list[torch.Tensor],
    atol: float,
    rtol: float,
):
    try:
        import onnxruntime as ort
    except Exception:  # pragma: no cover - message only
        print('onnxruntime not installed; skipping validation.')
        return False

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    inputs = normalize_inputs(dummy_inputs)
    session_inputs = session.get_inputs()
    if len(session_inputs) != len(inputs):
        print(f'{name}: input count mismatch ({len(inputs)} vs {len(session_inputs)})')
        return False
    feeds = {session_input.name: tensor.cpu().numpy() for session_input, tensor in zip(session_inputs, inputs)}
    ort_outputs = session.run(None, feeds)

    if len(ort_outputs) != len(torch_outputs):
        print(f'{name}: output count mismatch ({len(torch_outputs)} vs {len(ort_outputs)})')
        return False

    all_ok = True
    for idx, (torch_output, ort_output) in enumerate(zip(torch_outputs, ort_outputs)):
        torch_np = torch_output.detach().cpu().numpy()
        diff = np.abs(torch_np - ort_output)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        ok = np.allclose(torch_np, ort_output, atol=atol, rtol=rtol)
        all_ok = all_ok and ok
        print(
            f'{name}[{idx}] max_diff={max_diff:.6f} '
            f'mean_diff={mean_diff:.6f} allclose={ok}'
        )
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Export EasyOCR models to ONNX.')
    parser.add_argument('--detector', action='store_true', help='Export CRAFT detector.')
    parser.add_argument('--recognizer', action='store_true', help='Export recognizer.')
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).resolve().parent / 'onnx'),
        help='Directory for exported ONNX files.',
    )
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version.')
    parser.add_argument('--lang', default='en', help='Language for EasyOCR reader.')
    parser.add_argument(
        '--recog-network',
        default='english_g2',
        help='Recognizer network name (matches *.pth filename).',
    )
    parser.add_argument(
        '--detector-shape',
        type=parse_shape,
        default='1,3,2560,2560',
        help='Dummy input shape for detector export.',
    )
    parser.add_argument(
        '--recognizer-shape',
        type=parse_shape,
        default='1,1,32,100',
        help='Dummy input shape for recognizer export.',
    )
    parser.add_argument('--validate', action='store_true', help='Validate with ONNX Runtime.')
    parser.add_argument('--atol', type=float, default=1e-3, help='Abs tolerance for validation.')
    parser.add_argument('--rtol', type=float, default=1e-3, help='Rel tolerance for validation.')
    args = parser.parse_args()

    if not args.detector and not args.recognizer:
        args.detector = True
        args.recognizer = True

    models_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = build_reader(models_dir, args.lang, args.recog_network)
    patch_easyocr_for_onnx_export()

    if args.detector:
        detector = pick_module(reader, ['detector', 'detector_model', 'craft_model'])
        if detector is None:
            raise SystemExit('Failed to locate detector model on EasyOCR reader.')
        detector_input = torch.randn(*args.detector_shape, dtype=torch.float32)
        detector_path = output_dir / 'craft_mlt_25k.onnx'
        detector_outputs = export_model(
            'detector',
            detector,
            detector_input,
            detector_path,
            args.opset,
        )
        print(f'Wrote detector ONNX to {detector_path}')
        if args.validate:
            validate_onnx(
                'detector',
                detector_path,
                detector_input,
                detector_outputs,
                args.atol,
                args.rtol,
            )

    if args.recognizer:
        recognizer = pick_module(reader, ['recognizer', 'recog_network', 'recognizer_model'])
        if recognizer is None:
            raise SystemExit('Failed to locate recognizer model on EasyOCR reader.')
        recognizer_input = torch.randn(*args.recognizer_shape, dtype=torch.float32)
        recognizer_text = torch.zeros((args.recognizer_shape[0], 1), dtype=torch.long)
        recognizer_name = (args.recog_network or 'recognizer').replace('.pth', '')
        recognizer_path = output_dir / f'{recognizer_name}.onnx'
        recognizer_outputs = export_model(
            'recognizer',
            recognizer,
            (recognizer_input, recognizer_text),
            recognizer_path,
            args.opset,
            input_names=['input', 'text'],
            dynamic_spatial=False,
            use_dynamic_axes=False,
        )
        print(f'Wrote recognizer ONNX to {recognizer_path}')
        if args.validate:
            validate_onnx(
                'recognizer',
                recognizer_path,
                (recognizer_input, recognizer_text),
                recognizer_outputs,
                args.atol,
                args.rtol,
            )


if __name__ == '__main__':
    main()
