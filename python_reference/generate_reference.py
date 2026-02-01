import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import easyocr

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _iter_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for p in inputs:
        if p.is_dir():
            for child in sorted(p.rglob('*')):
                if child.is_file() and child.suffix.lower() in _IMAGE_EXTS:
                    images.append(child)
        else:
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                images.append(p)
    return images


def _format_results(results):
    # EasyOCR emits (bbox, text, confidence) where bbox is a 4-point polygon.
    return [
        {
            'box': [[float(pt[0]), float(pt[1])] for pt in bbox],
            'text': text,
            'confidence': float(prob),
        }
        for (bbox, text, prob) in results
    ]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description='Generate EasyOCR reference JSON for one or more images (file(s) and/or directories).'
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Image file(s) and/or directory(ies) to scan recursively.',
    )
    parser.add_argument(
        '--out-dir',
        default=str(Path(__file__).parent / 'out'),
        help='Output directory for JSON (default: python_reference/out).',
    )
    parser.add_argument(
        '--lang',
        action='append',
        default=['en'],
        help='Language(s) to load in EasyOCR (repeatable). Default: --lang en',
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU (default: CPU-only for more stable reference).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing JSON files.',
    )
    parser.add_argument(
        '--relative-to',
        default=None,
        help='If set, store image paths relative to this directory in the JSON.',
    )
    args = parser.parse_args(argv)

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    out_dir = Path(args.out_dir).expanduser().resolve()
    relative_root = Path(args.relative_to).expanduser().resolve() if args.relative_to else None

    images = _iter_images(input_paths)
    if not images:
        print('No images found.', file=sys.stderr)
        return 2

    # Build once; EasyOCR model init is expensive.
    reader = easyocr.Reader(args.lang, gpu=bool(args.gpu))

    manifest = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'easyocrVersion': getattr(easyocr, '__version__', None),
        'langs': args.lang,
        'gpu': bool(args.gpu),
        'inputs': [str(p) for p in input_paths],
        'outDir': str(out_dir),
        'numImages': len(images),
        'images': [],
    }

    for image_path in images:
        if relative_root:
            try:
                rel_image = str(image_path.relative_to(relative_root))
            except ValueError:
                rel_image = str(image_path)
        else:
            # Prefer a stable relative-ish label if inputs are directories.
            rel_image = str(image_path)

        # Mirror folder structure under out-dir when possible.
        if relative_root:
            rel_no_ext = Path(rel_image).with_suffix('')
            out_json = out_dir / rel_no_ext.with_suffix('.json')
        else:
            out_json = out_dir / f'{image_path.stem}.json'

        if out_json.exists() and not args.force:
            manifest['images'].append({'image': rel_image, 'json': str(out_json), 'skipped': True})
            continue

        results = reader.readtext(str(image_path))
        payload = {
            'formatVersion': 1,
            'image': rel_image,
            'results': _format_results(results),
        }
        _write_json(out_json, payload)
        manifest['images'].append({'image': rel_image, 'json': str(out_json), 'skipped': False})

    _write_json(out_dir / 'manifest.json', manifest)
    print(f'Wrote {len(manifest["images"])} reference file(s) into {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
