import argparse
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _safe_signature(obj) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description='Dump EasyOCR Reader signatures/config for pinning a reproducible reference.'
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Optional output path. If omitted, prints JSON to stdout.',
    )
    args = parser.parse_args(argv)

    try:
        import easyocr  # type: ignore
    except ModuleNotFoundError:
        print(
            'easyocr is not installed. Activate the Python venv and install python_reference/requirements.txt.',
            file=sys.stderr,
        )
        return 2

    payload = {
        'formatVersion': 1,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'easyocrVersion': getattr(easyocr, '__version__', None),
        'easyocrFile': getattr(easyocr, '__file__', None),
        'signatures': {
            'Reader.__init__': _safe_signature(getattr(easyocr, 'Reader', None).__init__),
            'Reader.readtext': _safe_signature(getattr(easyocr, 'Reader', None).readtext),
        },
    }

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
