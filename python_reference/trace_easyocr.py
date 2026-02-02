#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _format_results(results: Any) -> list[dict[str, Any]]:
    # EasyOCR emits (bbox, text, confidence) where bbox is a 4-point polygon.
    out: list[dict[str, Any]] = []
    for bbox, text, prob in results:
        out.append(
            {
                "box": [[float(pt[0]), float(pt[1])] for pt in bbox],
                "text": str(text),
                "confidence": float(prob),
            }
        )
    return out


def _horizontal_to_boxes(horizontal_list: list[list[float]]) -> list[list[list[float]]]:
    # horizontal_list entries are [x_min, x_max, y_min, y_max]
    boxes: list[list[list[float]]] = []
    for box in horizontal_list:
        x0, x1, y0, y1 = box[0], box[1], box[2], box[3]
        boxes.append([[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]])
    return boxes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Trace Python EasyOCR steps for JS-vs-Python drift debugging.")
    parser.add_argument("image", help="Path to the input image (same as JS run).")
    parser.add_argument("--trace-dir", required=True, help="Output trace directory.")
    parser.add_argument("--lang", action="append", default=["en"], help="Repeatable language(s). Default: en")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (default: CPU).")
    parser.add_argument("--run-readtext", action="store_true", help="Also run readtext() and store final results.")
    args = parser.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    try:
        import easyocr  # type: ignore
        import cv2  # type: ignore
        import torch  # type: ignore
        from easyocr import craft_utils, detection, imgproc, utils  # type: ignore
    except Exception as e:
        print(f"Failed to import dependencies: {e}", file=sys.stderr)
        return 2

    # Allow running from repo root without packaging python_reference/.
    sys.path.insert(0, str(Path(__file__).parent))
    from trace_writer import TraceWriter  # type: ignore

    trace_dir = Path(args.trace_dir).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "impl": "py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "easyocrVersion": getattr(easyocr, "__version__", None),
        "easyocrFile": getattr(easyocr, "__file__", None),
        "imagePath": str(image_path),
        "gpu": bool(args.gpu),
        "langs": args.lang,
    }
    tw = TraceWriter(trace_dir, run_meta=run_meta)

    # Keep this aligned with JS DEFAULT_OCR_OPTIONS (packages/core/src/types.ts).
    tw.add_params(
        "ocr_options",
        {
            "canvasSize": 2560,
            "magRatio": 1,
            "align": 32,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "textThreshold": 0.7,
            "lowText": 0.4,
            "linkThreshold": 0.4,
            "minSize": 20,
            "slopeThs": 0.1,
            "ycenterThs": 0.5,
            "heightThs": 0.5,
            "widthThs": 0.5,
            "addMargin": 0.1,
            "paragraph": False,
            "xThs": 1,
            "yThs": 0.5,
            "rotationInfo": [],
            "contrastThs": 0.1,
            "adjustContrast": 0.5,
            "decoder": "greedy",
            "recognizer": {"inputHeight": 64, "inputWidth": 100, "inputChannels": 1, "mean": 0.5, "std": 0.5},
        },
    )

    # Load exactly like EasyOCR does (reformat_input uses imgproc.loadImage -> skimage.io.imread, RGB order).
    img_rgb, img_grey = utils.reformat_input(str(image_path))
    if img_rgb is None:
        raise RuntimeError("EasyOCR failed to load image.")
    tw.add_image(
        "load_image",
        np.asarray(img_rgb),
        meta={"width": int(img_rgb.shape[1]), "height": int(img_rgb.shape[0]), "channels": int(img_rgb.shape[2])},
    )

    # Patch detection.test_net so we can trace *exactly* what EasyOCR feeds into the detector.
    original_test_net = detection.test_net

    def traced_test_net(
        canvas_size: int,
        mag_ratio: float,
        net: Any,
        image: Any,
        text_threshold: float,
        link_threshold: float,
        low_text: float,
        poly: bool,
        device: Any,
        estimate_num_chars: bool = False,
    ):
        if isinstance(image, np.ndarray) and len(image.shape) == 4:
            image_arrs = image
        else:
            image_arrs = [image]

        img_resized_list = []
        ratio_info = None

        for img in image_arrs:
            height, width, _ = img.shape
            target_size = mag_ratio * max(height, width)
            if target_size > canvas_size:
                target_size = canvas_size
            ratio = target_size / max(height, width)
            target_h, target_w = int(height * ratio), int(width * ratio)

            proc = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            tw.add_image(
                "resize_aspect_ratio",
                proc,
                meta={
                    "canvasSize": int(canvas_size),
                    "magRatio": float(mag_ratio),
                    "targetRatio": float(ratio),
                    "targetWidth": int(target_w),
                    "targetHeight": int(target_h),
                },
            )

            resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
                img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
            )
            img_resized_list.append(resized)
            if ratio_info is None:
                pad_h = int(resized.shape[0]) - int(target_h)
                pad_w = int(resized.shape[1]) - int(target_w)
                tw.add_image(
                    "pad_to_stride",
                    resized.astype(np.uint8, copy=False),
                    meta={
                        "stride": 32,
                        "padded": True,
                        "padBottom": int(pad_h),
                        "padRight": int(pad_w),
                        "targetRatio": float(target_ratio),
                        "sizeHeatmap": [int(size_heatmap[1]), int(size_heatmap[0])],
                    },
                )
                ratio_info = (target_ratio, size_heatmap)

        ratio_h = ratio_w = 1 / ratio_info[0] if ratio_info else 1.0

        # preprocessing (normalizeMeanVariance expects RGB order, float32 in 0..255 range)
        norm_list = [imgproc.normalizeMeanVariance(n_img) for n_img in img_resized_list]
        if norm_list:
            tw.add_tensor(
                "normalize_mean_variance",
                norm_list[0],
                layout="HWC",
                color_space="RGB",
                meta={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            )

        x = [np.transpose(n_img, (2, 0, 1)) for n_img in norm_list]
        x_np = np.array(x, dtype=np.float32)
        tw.add_tensor("to_tensor_layout", x_np, layout="NCHW", color_space="RGB")
        tw.add_tensor("detector_input_final", x_np, layout="NCHW", color_space="RGB")

        x_t = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            y, _feature = net(x_t)

        boxes_list, polys_list = [], []
        for out in y:
            score_text = out[:, :, 0].cpu().data.numpy()
            score_link = out[:, :, 1].cpu().data.numpy()
            tw.add_tensor("detector_raw_output_text", score_text.astype(np.float32, copy=False), layout="HW")
            tw.add_tensor("detector_raw_output_link", score_link.astype(np.float32, copy=False), layout="HW")
            tw.add_tensor("heatmap_text", score_text.astype(np.float32, copy=False), layout="HW")
            tw.add_tensor("heatmap_link", score_link.astype(np.float32, copy=False), layout="HW")

            boxes, polys, mapper = craft_utils.getDetBoxes(
                score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars
            )

            # Before coordinate adjustment (scoremap-space / net-space).
            try:
                tw.add_boxes(
                    "threshold_and_box_decode",
                    [[[float(x), float(y)] for x, y in box] for box in boxes],
                    meta={"coordSpace": "scoremap"},
                )
            except Exception:
                pass

            boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

            if estimate_num_chars:
                boxes = list(boxes)
                polys = list(polys)
            for k in range(len(polys)):
                if estimate_num_chars:
                    boxes[k] = (boxes[k], mapper[k])
                if polys[k] is None:
                    polys[k] = boxes[k]

            try:
                tw.add_boxes(
                    "adjust_coordinates_to_original",
                    [[[float(x), float(y)] for x, y in box] for box in boxes],
                    meta={"coordSpace": "image", "ratioW": float(ratio_w), "ratioH": float(ratio_h)},
                )
            except Exception:
                pass

            boxes_list.append(boxes)
            polys_list.append(polys)

        return boxes_list, polys_list

    detection.test_net = traced_test_net  # type: ignore

    try:
        reader = easyocr.Reader(args.lang, gpu=bool(args.gpu), verbose=False)
        horizontal_list_agg, free_list_agg = reader.detect(
            img_rgb,
            reformat=False,
        )
    finally:
        detection.test_net = original_test_net  # type: ignore

    horizontal_list = horizontal_list_agg[0] if horizontal_list_agg else []
    free_list = free_list_agg[0] if free_list_agg else []

    tw.add_boxes("detector_boxes_horizontal", _horizontal_to_boxes(horizontal_list))
    tw.add_boxes("detector_boxes_free", [[[float(x), float(y)] for x, y in box] for box in free_list])
    ordered = _horizontal_to_boxes(horizontal_list) + [[[float(x), float(y)] for x, y in box] for box in free_list]
    tw.add_boxes("detector_boxes_ordered", ordered)

    if args.run_readtext:
        results = reader.readtext(str(image_path))
        tw.add_params("final_results", {"results": _format_results(results)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
