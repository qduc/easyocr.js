from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


TraceKind = Literal["image", "tensor", "boxes", "params"]


def _sanitize_step_name(name: str) -> str:
    out = []
    prev_us = False
    for ch in name.strip().lower():
        ok = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if ok:
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s = "".join(out).strip("_")
    return s or "step"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    a = arr.astype(np.float64, copy=False)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _stable_json(obj: Any) -> str:
    def normalize(x: Any) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, list):
            return [normalize(v) for v in x]
        if isinstance(x, dict):
            return {k: normalize(x[k]) for k in sorted(x.keys())}
        return str(x)

    return json.dumps(normalize(obj), separators=(",", ":"), ensure_ascii=False)


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HWC image with >=3 channels, got shape={image.shape}")
    img = image[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _write_png(path: Path, image_rgb_uint8: np.ndarray) -> None:
    if Image is None:
        return
    im = Image.fromarray(image_rgb_uint8, mode="RGB")
    im.save(path, format="PNG")


@dataclass
class TraceIndexStep:
    index: int
    name: str
    kind: str
    dir: str


class TraceWriter:
    def __init__(self, trace_dir: Path, run_meta: Optional[dict[str, Any]] = None) -> None:
        self.trace_dir = trace_dir
        self.steps_dir = trace_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self._step_index = 0
        self._index: dict[str, Any] = {
            "formatVersion": 1,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "runMeta": run_meta or {},
            "steps": [],
        }
        self._flush_index()

    def _flush_index(self) -> None:
        out = self.trace_dir / "trace.json"
        out.write_text(json.dumps(self._index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _new_step_dir(self, name: str) -> tuple[int, Path, str]:
        idx = self._step_index
        dir_name = f"{idx:03d}_{_sanitize_step_name(name)}"
        step_dir = self.steps_dir / dir_name
        step_dir.mkdir(parents=True, exist_ok=True)
        rel = os.path.join("steps", dir_name).replace(os.sep, "/")
        self._step_index += 1
        return idx, step_dir, rel

    def add_image(self, name: str, image_rgb_uint8: np.ndarray, meta: Optional[dict[str, Any]] = None) -> None:
        idx, step_dir, rel = self._new_step_dir(name)
        img = _ensure_uint8_rgb(image_rgb_uint8)
        raw = img.tobytes(order="C")
        h, w, _ = img.shape
        sha = _sha256(raw)

        (step_dir / "raw.bin").write_bytes(raw)
        (step_dir / "raw.meta.json").write_text(
            json.dumps(
                {
                    "dtype": "uint8",
                    "layout": "HWC",
                    "colorSpace": "RGB",
                    "shape": [int(h), int(w), 3],
                    "sha256_raw": sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_png(step_dir / "image.png", img)

        step_meta = {
            "name": name,
            "kind": "image",
            "dtype": "uint8",
            "layout": "HWC",
            "colorSpace": "RGB",
            "shape": [int(h), int(w), 3],
            "sha256_raw": sha,
            "stats": _stats(img),
        }
        if meta:
            step_meta.update(meta)
        (step_dir / "meta.json").write_text(json.dumps(step_meta, indent=2) + "\n", encoding="utf-8")

        self._index["steps"].append({"index": idx, "name": name, "kind": "image", "dir": rel})
        self._flush_index()

    def add_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        layout: Optional[str] = None,
        color_space: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        idx, step_dir, rel = self._new_step_dir(name)
        arr = np.asarray(tensor)
        raw = arr.tobytes(order="C")
        sha = _sha256(raw)

        (step_dir / "tensor.bin").write_bytes(raw)
        (step_dir / "tensor.meta.json").write_text(
            json.dumps(
                {
                    "dtype": str(arr.dtype),
                    "shape": [int(x) for x in arr.shape],
                    "layout": layout,
                    "colorSpace": color_space,
                    "sha256_raw": sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        step_meta = {
            "name": name,
            "kind": "tensor",
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "layout": layout,
            "colorSpace": color_space,
            "sha256_raw": sha,
            "stats": _stats(arr),
        }
        if meta:
            step_meta.update(meta)
        (step_dir / "meta.json").write_text(json.dumps(step_meta, indent=2) + "\n", encoding="utf-8")

        # Optional preview for simple float images / heatmaps.
        try:
            if Image is not None and arr.dtype in (np.float32, np.float64) and layout in ("HW", "HWC"):
                if layout == "HW" and arr.ndim == 2:
                    a = arr.astype(np.float32, copy=False)
                    mn = float(np.min(a))
                    mx = float(np.max(a))
                    denom = max(1e-6, mx - mn)
                    img = np.clip(((a - mn) / denom) * 255.0, 0, 255).astype(np.uint8)
                    _write_png(step_dir / "preview.png", np.stack([img, img, img], axis=-1))
                if layout == "HWC" and arr.ndim == 3 and arr.shape[2] in (1, 3):
                    a = arr.astype(np.float32, copy=False)
                    mn = float(np.min(a))
                    mx = float(np.max(a))
                    denom = max(1e-6, mx - mn)
                    img = np.clip(((a - mn) / denom) * 255.0, 0, 255).astype(np.uint8)
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    _write_png(step_dir / "preview.png", img)
        except Exception:
            pass

        self._index["steps"].append({"index": idx, "name": name, "kind": "tensor", "dir": rel})
        self._flush_index()

    def add_boxes(self, name: str, boxes: list[list[list[float]]], meta: Optional[dict[str, Any]] = None) -> None:
        idx, step_dir, rel = self._new_step_dir(name)
        flat = np.array(boxes, dtype=np.float32).reshape((-1, 4, 2))
        raw = flat.tobytes(order="C")
        sha = _sha256(raw)

        (step_dir / "boxes.bin").write_bytes(raw)
        (step_dir / "boxes.meta.json").write_text(
            json.dumps({"dtype": "float32", "shape": [int(flat.shape[0]), 4, 2], "sha256_raw": sha}, indent=2) + "\n",
            encoding="utf-8",
        )
        (step_dir / "boxes.json").write_text(json.dumps(boxes, indent=2) + "\n", encoding="utf-8")

        step_meta = {
            "name": name,
            "kind": "boxes",
            "count": int(flat.shape[0]),
            "sha256_raw": sha,
            "stats": _stats(flat),
        }
        if meta:
            step_meta.update(meta)
        (step_dir / "meta.json").write_text(json.dumps(step_meta, indent=2) + "\n", encoding="utf-8")

        self._index["steps"].append({"index": idx, "name": name, "kind": "boxes", "dir": rel})
        self._flush_index()

    def add_params(self, name: str, params: Any, meta: Optional[dict[str, Any]] = None) -> None:
        idx, step_dir, rel = self._new_step_dir(name)
        raw = _stable_json(params).encode("utf-8")
        sha = _sha256(raw)
        (step_dir / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        step_meta = {"name": name, "kind": "params", "sha256_raw": sha}
        if meta:
            step_meta.update(meta)
        (step_dir / "meta.json").write_text(json.dumps(step_meta, indent=2) + "\n", encoding="utf-8")

        self._index["steps"].append({"index": idx, "name": name, "kind": "params", "dir": rel})
        self._flush_index()

