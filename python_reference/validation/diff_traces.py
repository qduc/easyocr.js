#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


Kind = Literal["image", "tensor", "boxes", "params"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(trace_dir: Path) -> dict[str, Any]:
    trace_json = trace_dir / "trace.json"
    if not trace_json.exists():
        raise FileNotFoundError(f"Missing trace.json: {trace_json}")
    data = _read_json(trace_json)
    if data.get("formatVersion") != 1:
        raise ValueError(f"Unsupported trace formatVersion: {data.get('formatVersion')}")
    return data


def _step_dir(trace_dir: Path, step: dict[str, Any]) -> Path:
    return trace_dir / step["dir"]


def _dtype_from_str(s: str) -> np.dtype:
    if s in ("float32", "int32", "uint8"):
        return np.dtype(s)
    # numpy may stringify like 'float32' already; try directly
    return np.dtype(s)


def _load_tensor(step_dir: Path) -> Tuple[np.ndarray, dict[str, Any]]:
    meta = _read_json(step_dir / "tensor.meta.json")
    dtype = _dtype_from_str(meta["dtype"])
    shape = tuple(int(x) for x in meta["shape"])
    raw = (step_dir / "tensor.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr, meta


def _load_image(step_dir: Path) -> Tuple[np.ndarray, dict[str, Any]]:
    meta = _read_json(step_dir / "raw.meta.json")
    dtype = _dtype_from_str(meta["dtype"])
    shape = tuple(int(x) for x in meta["shape"])
    raw = (step_dir / "raw.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr, meta


def _load_boxes(step_dir: Path) -> Tuple[np.ndarray, dict[str, Any]]:
    meta = _read_json(step_dir / "boxes.meta.json")
    raw = (step_dir / "boxes.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32).reshape((int(meta["shape"][0]), 4, 2))
    return arr, meta


def _summarize_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    absd = np.abs(diff)
    return {
        "mae": float(np.mean(absd)) if absd.size else 0.0,
        "max_abs": float(np.max(absd)) if absd.size else 0.0,
        "p99_abs": float(np.percentile(absd, 99)) if absd.size else 0.0,
    }


def _write_image_diff(out_path: Path, a: np.ndarray, b: np.ndarray) -> None:
    if Image is None:
        return
    if a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
        return
    aa = a.astype(np.int16)
    bb = b.astype(np.int16)
    d = np.abs(aa - bb).astype(np.uint8)
    # boost visibility
    boosted = np.clip(d.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
    Image.fromarray(boosted, mode="RGB").save(out_path, format="PNG")


def _sort_boxes(boxes: np.ndarray) -> np.ndarray:
    # boxes: [N,4,2]
    if boxes.size == 0:
        return boxes
    mins = np.min(boxes, axis=1)  # [N,2]
    maxs = np.max(boxes, axis=1)  # [N,2]
    keys = np.stack([mins[:, 1], mins[:, 0], maxs[:, 1], maxs[:, 0]], axis=1)
    order = np.lexsort((keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
    return boxes[order]


def _print_step_header(i: int, name: str, kind: str) -> None:
    print(f"\n[{i:03d}] {name} ({kind})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Diff two EasyOCR.js trace directories (JS vs Python).")
    parser.add_argument("--js", required=True, help="JS trace dir")
    parser.add_argument("--py", required=True, help="Python trace dir")
    parser.add_argument("--out", default=None, help="Optional report output dir")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue after first drift")
    args = parser.parse_args(argv)

    js_dir = Path(args.js).expanduser().resolve()
    py_dir = Path(args.py).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    js_trace = _load_trace(js_dir)
    py_trace = _load_trace(py_dir)

    js_steps = js_trace.get("steps", [])
    py_steps = py_trace.get("steps", [])

    js_names = [s.get("name") for s in js_steps]
    py_names = [s.get("name") for s in py_steps]
    if js_names != py_names:
        print("Step list mismatch (by name/order).", file=sys.stderr)
        print(f"  JS steps: {js_names}", file=sys.stderr)
        print(f"  PY steps: {py_names}", file=sys.stderr)
        print("Proceeding with index-based comparison of shared prefix.\n", file=sys.stderr)

    n = min(len(js_steps), len(py_steps))
    drift_found = False

    for i in range(n):
        js_step = js_steps[i]
        py_step = py_steps[i]
        name = js_step.get("name")
        kind = js_step.get("kind")
        py_kind = py_step.get("kind")

        _print_step_header(i, str(name), str(kind))
        if name != py_step.get("name") or kind != py_kind:
            print(f"  ✗ step identity mismatch: JS=({name},{kind}) PY=({py_step.get('name')},{py_kind})")
            drift_found = True
            if not args.cont:
                break
            continue

        js_step_dir = _step_dir(js_dir, js_step)
        py_step_dir = _step_dir(py_dir, py_step)

        js_meta = _read_json(js_step_dir / "meta.json")
        py_meta = _read_json(py_step_dir / "meta.json")

        js_sha = js_meta.get("sha256_raw")
        py_sha = py_meta.get("sha256_raw")
        if js_sha and py_sha and js_sha == py_sha:
            print("  ✓ sha256_raw match")
            continue

        print("  ✗ sha256_raw mismatch")

        if kind == "image":
            a, a_meta = _load_image(js_step_dir)
            b, b_meta = _load_image(py_step_dir)
            if a.shape != b.shape or str(a.dtype) != str(b.dtype):
                print(f"  Shape/dtype mismatch: JS={a.shape}/{a.dtype} PY={b.shape}/{b.dtype}")
                drift_found = True
            else:
                d = _summarize_diff(a, b)
                print(f"  Diff: mae={d['mae']:.6f} p99_abs={d['p99_abs']:.6f} max_abs={d['max_abs']:.6f}")
                if d["max_abs"] != 0.0:
                    drift_found = True
                if out_dir and Image is not None:
                    out_path = out_dir / f"{i:03d}_{name}_diff.png"
                    _write_image_diff(out_path, a, b)
                    print(f"  Wrote {out_path}")

        elif kind == "tensor":
            a, a_meta = _load_tensor(js_step_dir)
            b, b_meta = _load_tensor(py_step_dir)
            if a.shape != b.shape or str(a.dtype) != str(b.dtype):
                print(f"  Shape/dtype mismatch: JS={a.shape}/{a.dtype} PY={b.shape}/{b.dtype}")
                print(f"  Layout: JS={a_meta.get('layout')} PY={b_meta.get('layout')}")
                drift_found = True
            else:
                d = _summarize_diff(a, b)
                print(f"  Diff: mae={d['mae']:.6f} p99_abs={d['p99_abs']:.6f} max_abs={d['max_abs']:.6f}")
                print(f"  Layout: {a_meta.get('layout')}")
                if d["max_abs"] != 0.0:
                    drift_found = True

        elif kind == "boxes":
            a, _ = _load_boxes(js_step_dir)
            b, _ = _load_boxes(py_step_dir)
            a2 = _sort_boxes(a)
            b2 = _sort_boxes(b)
            m = min(a2.shape[0], b2.shape[0])
            if a2.shape[0] != b2.shape[0]:
                print(f"  Box count mismatch: JS={a2.shape[0]} PY={b2.shape[0]} (comparing first {m})")
                drift_found = True
            if m > 0:
                d = _summarize_diff(a2[:m], b2[:m])
                print(f"  Coord diff (sorted): mae={d['mae']:.6f} p99_abs={d['p99_abs']:.6f} max_abs={d['max_abs']:.6f}")
                if d["max_abs"] != 0.0:
                    drift_found = True
            else:
                print("  No boxes to compare.")

        elif kind == "params":
            print("  Params differ (see params.json / meta.json).")
            drift_found = True
        else:
            print("  Unknown kind; skipping.")
            drift_found = True

        if not args.cont:
            break

    if not drift_found:
        print("\n✓ No drift detected in shared steps.")
        return 0

    print("\n✗ Drift detected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
