import type { Box, OcrOptions, RasterImage, Tensor } from './types.js';
import { clamp, resizeLongSide, toFloatImage } from './utils.js';

export interface DetectorPreprocessResult {
  input: Tensor;
  resized: RasterImage;
  padded: RasterImage;
  padRight: number;
  padBottom: number;
  scaleX: number;
  scaleY: number;
}

export interface Heatmap {
  data: Float32Array;
  width: number;
  height: number;
}

export interface DetectorPostprocessResult {
  horizontalList: Box[];
  freeList: Box[];
}

export interface DetectorPostprocessDebugResult extends DetectorPostprocessResult {
  rawBoxesHeatmap: Box[];
  rawBoxesAdjusted: Box[];
}

export const detectorPreprocess = (image: RasterImage, options: OcrOptions): DetectorPreprocessResult => {
  const targetSize = Math.min(options.canvasSize, Math.max(image.width, image.height) * options.magRatio);
  const { image: resized, scale } = resizeLongSide(image, targetSize, options.align);
  const stride = options.align || 32;
  const padBottom = resized.height % stride === 0 ? 0 : stride - (resized.height % stride);
  const padRight = resized.width % stride === 0 ? 0 : stride - (resized.width % stride);
  const paddedWidth = resized.width + padRight;
  const paddedHeight = resized.height + padBottom;
  const paddedData = new Uint8Array(paddedWidth * paddedHeight * resized.channels);
  for (let y = 0; y < resized.height; y += 1) {
    const srcRow = y * resized.width * resized.channels;
    const dstRow = y * paddedWidth * resized.channels;
    paddedData.set(
      resized.data.subarray(srcRow, srcRow + resized.width * resized.channels),
      dstRow,
    );
  }
  const padded: RasterImage = {
    data: paddedData,
    width: paddedWidth,
    height: paddedHeight,
    channels: resized.channels,
    channelOrder: resized.channelOrder,
  };

  const floatImage = toFloatImage(padded, options.mean, options.std);
  const { width, height } = floatImage;
  const data = new Float32Array(1 * 3 * height * width);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const srcIndex = (y * width + x) * 3;
      const dstIndex = y * width + x;
      data[dstIndex] = floatImage.data[srcIndex];
      data[height * width + dstIndex] = floatImage.data[srcIndex + 1];
      data[2 * height * width + dstIndex] = floatImage.data[srcIndex + 2];
    }
  }
  return {
    input: {
      data,
      shape: [1, 3, height, width],
      type: 'float32',
    },
    resized,
    padded,
    padRight,
    padBottom,
    scaleX: resized.width / image.width,
    scaleY: resized.height / image.height,
  };
};

export const tensorToHeatmap = (tensor: Tensor): Heatmap => {
  if (tensor.type !== 'float32') {
    throw new Error('Expected float32 tensor for heatmap.');
  }
  const shape = tensor.shape;
  const width = shape[shape.length - 1];
  const height = shape[shape.length - 2];
  const channelStride = width * height;
  const data = tensor.data as Float32Array;
  const slice = data.length === channelStride ? data : data.subarray(0, channelStride);
  const out = slice.length === channelStride ? slice : new Float32Array(slice);
  return { data: out, width, height };
};

const extractBoundingBox = (
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  scaleX: number,
  scaleY: number,
): Box => [
  [minX / scaleX, minY / scaleY],
  [maxX / scaleX, minY / scaleY],
  [maxX / scaleX, maxY / scaleY],
  [minX / scaleX, maxY / scaleY],
];

export const detectorPostprocess = (
  textMap: Heatmap,
  linkMap: Heatmap,
  options: OcrOptions,
  scaleX: number,
  scaleY: number,
): DetectorPostprocessResult => {
  const { horizontalList, freeList } = detectorPostprocessDebug(textMap, linkMap, options, scaleX, scaleY);
  return { horizontalList, freeList };
};

export const detectorPostprocessDebug = (
  textMap: Heatmap,
  linkMap: Heatmap,
  options: OcrOptions,
  scaleX: number,
  scaleY: number,
): DetectorPostprocessDebugResult => {
  if (textMap.width !== linkMap.width || textMap.height !== linkMap.height) {
    throw new Error('Detector output heatmaps must share the same shape.');
  }
  const text = textMap;
  const link = linkMap;
  const width = text.width;
  const height = text.height;
  const size = width * height;

  // Match EasyOCR craft_utils.getDetBoxes_core (poly=False, estimate_num_chars=False).
  const textScore = new Uint8Array(size);
  const linkScore = new Uint8Array(size);
  const combined = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    const isText = text.data[i] > options.lowText ? 1 : 0;
    const isLink = link.data[i] > options.linkThreshold ? 1 : 0;
    textScore[i] = isText;
    linkScore[i] = isLink;
    combined[i] = isText || isLink ? 1 : 0;
  }

  const visited = new Uint8Array(size);
  const queue = new Int32Array(size);
  const rawBoxesHeatmap: Box[] = [];
  const rawBoxesAdjusted: Box[] = [];

  const dilateRectInPlace = (
    segmap: Uint8Array,
    sx: number,
    ex: number,
    sy: number,
    ey: number,
    kSize: number,
  ): void => {
    const roiW = ex - sx;
    const roiH = ey - sy;
    if (roiW <= 0 || roiH <= 0) return;
    const src = new Uint8Array(roiW * roiH);
    for (let y = 0; y < roiH; y += 1) {
      const base = (sy + y) * width + sx;
      src.set(segmap.subarray(base, base + roiW), y * roiW);
    }
    const dst = new Uint8Array(roiW * roiH);
    const anchor = Math.floor(kSize / 2);
    for (let y = 0; y < roiH; y += 1) {
      for (let x = 0; x < roiW; x += 1) {
        let hit = 0;
        for (let ky = 0; ky < kSize && !hit; ky += 1) {
          const yy = y + ky - anchor;
          if (yy < 0 || yy >= roiH) continue;
          const row = yy * roiW;
          for (let kx = 0; kx < kSize; kx += 1) {
            const xx = x + kx - anchor;
            if (xx < 0 || xx >= roiW) continue;
            if (src[row + xx]) {
              hit = 255;
              break;
            }
          }
        }
        dst[y * roiW + x] = hit;
      }
    }
    for (let y = 0; y < roiH; y += 1) {
      const base = (sy + y) * width + sx;
      segmap.set(dst.subarray(y * roiW, y * roiW + roiW), base);
    }
  };

  const orderClockwiseStartTopLeft = (box: Point2D[]): Point2D[] => {
    if (box.length !== 4) return box;
    let start = 0;
    let minSum = Infinity;
    for (let i = 0; i < 4; i += 1) {
      const sum = box[i][0] + box[i][1];
      if (sum < minSum) {
        minSum = sum;
        start = i;
      }
    }
    const rolled: Point2D[] = [
      box[(start + 0) % 4],
      box[(start + 1) % 4],
      box[(start + 2) % 4],
      box[(start + 3) % 4],
    ];
    return rolled;
  };

  for (let i = 0; i < size; i += 1) {
    if (!combined[i] || visited[i]) continue;
    let qHead = 0;
    let qTail = 0;
    queue[qTail++] = i;
    visited[i] = 1;

    let minX = width - 1;
    let maxX = 0;
    let minY = height - 1;
    let maxY = 0;
    let maxText = -Infinity;
    let area = 0;
    const component: number[] = [];

    while (qHead < qTail) {
      const idx = queue[qHead++];
      component.push(idx);
      area += 1;
      const x = idx % width;
      const y = Math.floor(idx / width);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (text.data[idx] > maxText) maxText = text.data[idx];

      if (x > 0) {
        const n = idx - 1;
        if (!visited[n] && combined[n]) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (x < width - 1) {
        const n = idx + 1;
        if (!visited[n] && combined[n]) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (y > 0) {
        const n = idx - width;
        if (!visited[n] && combined[n]) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
      if (y < height - 1) {
        const n = idx + width;
        if (!visited[n] && combined[n]) {
          visited[n] = 1;
          queue[qTail++] = n;
        }
      }
    }

    if (area < 10) continue;
    if (maxText < options.textThreshold) continue;

    const segmap = new Uint8Array(size);
    for (const idx of component) {
      segmap[idx] = 255;
    }
    for (const idx of component) {
      if (linkScore[idx] && !textScore[idx]) {
        segmap[idx] = 0;
      }
    }

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    const niter = Math.trunc(Math.sqrt((area * Math.min(boxW, boxH)) / (boxW * boxH)) * 2);
    const sx = clamp(minX - niter, 0, width - 1);
    const sy = clamp(minY - niter, 0, height - 1);
    const ex = clamp(minX + boxW + niter + 1, 0, width);
    const ey = clamp(minY + boxH + niter + 1, 0, height);

    if (niter > 0) {
      dilateRectInPlace(segmap, sx, ex, sy, ey, 1 + niter);
    }

    const points: Point2D[] = [];
    let segMinX = Infinity;
    let segMaxX = -Infinity;
    let segMinY = Infinity;
    let segMaxY = -Infinity;
    for (let y = sy; y < ey; y += 1) {
      for (let x = sx; x < ex; x += 1) {
        if (!segmap[y * width + x]) continue;
        points.push([x, y]);
        if (x < segMinX) segMinX = x;
        if (x > segMaxX) segMaxX = x;
        if (y < segMinY) segMinY = y;
        if (y > segMaxY) segMaxY = y;
      }
    }
    if (!points.length) continue;

    let boxPoints = minAreaRect(points);
    if (!boxPoints.length) continue;

    const w = Math.hypot(boxPoints[0][0] - boxPoints[1][0], boxPoints[0][1] - boxPoints[1][1]);
    const h = Math.hypot(boxPoints[1][0] - boxPoints[2][0], boxPoints[1][1] - boxPoints[2][1]);
    const ratio = Math.max(w, h) / (Math.min(w, h) + 1e-5);
    if (Math.abs(1 - ratio) <= 0.1) {
      boxPoints = [
        [segMinX, segMinY],
        [segMaxX, segMinY],
        [segMaxX, segMaxY],
        [segMinX, segMaxY],
      ];
    }

    boxPoints = orderClockwiseStartTopLeft(boxPoints);
    const heatmapBox: Box = [
      [boxPoints[0][0], boxPoints[0][1]],
      [boxPoints[1][0], boxPoints[1][1]],
      [boxPoints[2][0], boxPoints[2][1]],
      [boxPoints[3][0], boxPoints[3][1]],
    ];
    rawBoxesHeatmap.push(heatmapBox);

    rawBoxesAdjusted.push([
      [heatmapBox[0][0] / scaleX, heatmapBox[0][1] / scaleY],
      [heatmapBox[1][0] / scaleX, heatmapBox[1][1] / scaleY],
      [heatmapBox[2][0] / scaleX, heatmapBox[2][1] / scaleY],
      [heatmapBox[3][0] / scaleX, heatmapBox[3][1] / scaleY],
    ]);
  }

  const grouped = groupTextBoxes(rawBoxesAdjusted, options);
  return { ...grouped, rawBoxesHeatmap, rawBoxesAdjusted };
};

const groupTextBoxes = (boxes: Box[], options: OcrOptions): DetectorPostprocessResult => {
  if (!boxes.length) {
    return { horizontalList: [], freeList: [] };
  }
  const polys = boxes.map((box) => [
    Math.trunc(box[0][0]), Math.trunc(box[0][1]),
    Math.trunc(box[1][0]), Math.trunc(box[1][1]),
    Math.trunc(box[2][0]), Math.trunc(box[2][1]),
    Math.trunc(box[3][0]), Math.trunc(box[3][1]),
  ]);

  const horizontalList: Array<[number, number, number, number, number, number]> = [];
  const freeList: Box[] = [];

  for (const poly of polys) {
    const slopeUp = (poly[3] - poly[1]) / Math.max(10, poly[2] - poly[0]);
    const slopeDown = (poly[5] - poly[7]) / Math.max(10, poly[4] - poly[6]);
    if (Math.max(Math.abs(slopeUp), Math.abs(slopeDown)) < options.slopeThs) {
      const xMax = Math.max(poly[0], poly[2], poly[4], poly[6]);
      const xMin = Math.min(poly[0], poly[2], poly[4], poly[6]);
      const yMax = Math.max(poly[1], poly[3], poly[5], poly[7]);
      const yMin = Math.min(poly[1], poly[3], poly[5], poly[7]);
      horizontalList.push([xMin, xMax, yMin, yMax, 0.5 * (yMin + yMax), yMax - yMin]);
    } else {
      const height = Math.hypot(poly[6] - poly[0], poly[7] - poly[1]);
      const width = Math.hypot(poly[2] - poly[0], poly[3] - poly[1]);
      const margin = Math.trunc(1.44 * options.addMargin * Math.min(width, height));
      const theta13 = Math.abs(Math.atan((poly[1] - poly[5]) / Math.max(10, poly[0] - poly[4])));
      const theta24 = Math.abs(Math.atan((poly[3] - poly[7]) / Math.max(10, poly[2] - poly[6])));
      const x1 = poly[0] - Math.cos(theta13) * margin;
      const y1 = poly[1] - Math.sin(theta13) * margin;
      const x2 = poly[2] + Math.cos(theta24) * margin;
      const y2 = poly[3] - Math.sin(theta24) * margin;
      const x3 = poly[4] + Math.cos(theta13) * margin;
      const y3 = poly[5] + Math.sin(theta13) * margin;
      const x4 = poly[6] - Math.cos(theta24) * margin;
      const y4 = poly[7] + Math.sin(theta24) * margin;
      freeList.push([
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4],
      ]);
    }
  }

  horizontalList.sort((a, b) => a[4] - b[4]);

  const combinedList: Array<Array<[number, number, number, number, number, number]>> = [];
  let newBox: Array<[number, number, number, number, number, number]> = [];
  let bHeight: number[] = [];
  let bYcenter: number[] = [];

  for (const poly of horizontalList) {
    if (!newBox.length) {
      bHeight = [poly[5]];
      bYcenter = [poly[4]];
      newBox = [poly];
      continue;
    }
    const meanY = bYcenter.reduce((a, b) => a + b, 0) / bYcenter.length;
    const meanH = bHeight.reduce((a, b) => a + b, 0) / bHeight.length;
    if (Math.abs(meanY - poly[4]) < options.ycenterThs * meanH) {
      bHeight.push(poly[5]);
      bYcenter.push(poly[4]);
      newBox.push(poly);
    } else {
      combinedList.push(newBox);
      bHeight = [poly[5]];
      bYcenter = [poly[4]];
      newBox = [poly];
    }
  }
  if (newBox.length) {
    combinedList.push(newBox);
  }

  const mergedList: Array<[number, number, number, number]> = [];
  for (const boxesInLine of combinedList) {
    if (boxesInLine.length === 1) {
      const box = boxesInLine[0];
      const margin = Math.trunc(options.addMargin * Math.min(box[1] - box[0], box[5]));
      mergedList.push([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]);
      continue;
    }
    boxesInLine.sort((a, b) => a[0] - b[0]);
    const mergedBox: Array<Array<[number, number, number, number, number, number]>> = [];
    newBox = [];
    bHeight = [];
    let xMax = 0;
    for (const box of boxesInLine) {
      if (!newBox.length) {
        bHeight = [box[5]];
        xMax = box[1];
        newBox = [box];
        continue;
      }
      const meanH = bHeight.reduce((a, b) => a + b, 0) / bHeight.length;
      if (Math.abs(meanH - box[5]) < options.heightThs * meanH && (box[0] - xMax) < options.widthThs * (box[3] - box[2])) {
        bHeight.push(box[5]);
        xMax = box[1];
        newBox.push(box);
      } else {
        mergedBox.push(newBox);
        bHeight = [box[5]];
        xMax = box[1];
        newBox = [box];
      }
    }
    if (newBox.length) {
      mergedBox.push(newBox);
    }

    for (const mbox of mergedBox) {
      if (mbox.length !== 1) {
        const xMin = Math.min(...mbox.map((b) => b[0]));
        const xMax2 = Math.max(...mbox.map((b) => b[1]));
        const yMin = Math.min(...mbox.map((b) => b[2]));
        const yMax = Math.max(...mbox.map((b) => b[3]));
        const boxWidth = xMax2 - xMin;
        const boxHeight = yMax - yMin;
        const margin = Math.trunc(options.addMargin * Math.min(boxWidth, boxHeight));
        mergedList.push([xMin - margin, xMax2 + margin, yMin - margin, yMax + margin]);
      } else {
        const box = mbox[0];
        const boxWidth = box[1] - box[0];
        const boxHeight = box[3] - box[2];
        const margin = Math.trunc(options.addMargin * Math.min(boxWidth, boxHeight));
        mergedList.push([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]);
      }
    }
  }

  let horizontalBoxes: Box[] = mergedList.map((box) => [
    [box[0], box[2]],
    [box[1], box[2]],
    [box[1], box[3]],
    [box[0], box[3]],
  ]);

  if (options.minSize) {
    horizontalBoxes = horizontalBoxes.filter((box) => {
      const width = Math.abs(box[1][0] - box[0][0]);
      const height = Math.abs(box[2][1] - box[1][1]);
      return Math.max(width, height) > options.minSize;
    });
  }

  let filteredFree = freeList;
  if (options.minSize) {
    filteredFree = freeList.filter((box) => {
      const xs = box.map((point) => point[0]);
      const ys = box.map((point) => point[1]);
      const width = Math.max(...xs) - Math.min(...xs);
      const height = Math.max(...ys) - Math.min(...ys);
      return Math.max(width, height) > options.minSize;
    });
  }
  return { horizontalList: horizontalBoxes, freeList: filteredFree };
};

interface LineGroup {
  boxes: Box[];
  centerY: number;
  height: number;
}

type Point2D = [number, number];

const cross = (o: Point2D, a: Point2D, b: Point2D): number =>
  (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

const convexHull = (points: Point2D[]): Point2D[] => {
  if (points.length <= 1) return points.slice();
  const sorted = points.slice().sort((p1, p2) =>
    p1[0] === p2[0] ? p1[1] - p2[1] : p1[0] - p2[0],
  );
  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
};

const rotatePoint = (point: Point2D, cos: number, sin: number): Point2D => [
  point[0] * cos - point[1] * sin,
  point[0] * sin + point[1] * cos,
];

const minAreaRect = (points: Point2D[]): Point2D[] => {
  if (points.length === 0) return [];
  if (points.length === 1) return [points[0], points[0], points[0], points[0]];
  const hull = convexHull(points);
  if (hull.length === 2) {
    const [p0, p1] = hull;
    return [p0, p1, p1, p0];
  }

  let bestArea = Infinity;
  let best: { angle: number; minX: number; maxX: number; minY: number; maxY: number } | null = null;

  for (let i = 0; i < hull.length; i += 1) {
    const p0 = hull[i];
    const p1 = hull[(i + 1) % hull.length];
    const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
    const cos = Math.cos(-angle);
    const sin = Math.sin(-angle);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of hull) {
      const [rx, ry] = rotatePoint(p, cos, sin);
      if (rx < minX) minX = rx;
      if (rx > maxX) maxX = rx;
      if (ry < minY) minY = ry;
      if (ry > maxY) maxY = ry;
    }
    const area = (maxX - minX) * (maxY - minY);
    if (area < bestArea) {
      bestArea = area;
      best = { angle, minX, maxX, minY, maxY };
    }
  }

  if (!best) return [];
  const cosA = Math.cos(best.angle);
  const sinA = Math.sin(best.angle);
  const rect = [
    [best.minX, best.minY],
    [best.maxX, best.minY],
    [best.maxX, best.maxY],
    [best.minX, best.maxY],
  ] as Point2D[];
  const box = rect.map((p) => rotatePoint(p, cosA, sinA)) as Point2D[];
  let startIdx = 0;
  let minSum = Infinity;
  for (let i = 0; i < box.length; i += 1) {
    const sum = box[i][0] + box[i][1];
    if (sum < minSum) {
      minSum = sum;
      startIdx = i;
    }
  }
  return [
    box[startIdx],
    box[(startIdx + 1) % 4],
    box[(startIdx + 2) % 4],
    box[(startIdx + 3) % 4],
  ];
};

const boxMetrics = (box: Box) => {
  const xs = box.map((point) => point[0]);
  const ys = box.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    height: maxY - minY,
  };
};

export const groupBoxesByLine = (boxes: Box[], options: OcrOptions): LineGroup[] => {
  const sorted = boxes
    .map((box) => ({ box, ...boxMetrics(box) }))
    .sort((a, b) => a.centerY - b.centerY);
  const lines: LineGroup[] = [];
  for (const item of sorted) {
    const target = lines.find((line) => {
      const heightRatio = Math.max(item.height, line.height) / Math.max(1, Math.min(item.height, line.height));
      return (
        Math.abs(item.centerY - line.centerY) <= Math.max(item.height, line.height) * options.ycenterThs &&
        heightRatio <= 1 + options.heightThs
      );
    });
    if (target) {
      target.boxes.push(item.box);
      target.centerY = (target.centerY * (target.boxes.length - 1) + item.centerY) / target.boxes.length;
      target.height = Math.max(target.height, item.height);
    } else {
      lines.push({ boxes: [item.box], centerY: item.centerY, height: item.height });
    }
  }
  return lines;
};

export const orderBoxes = (boxes: Box[], options: OcrOptions): Box[] => {
  // Match Python EasyOCR ordering used in utils.get_image_list(sort_output=True):
  // sort by the vertical position of the first point.
  return boxes
    .map((box, index) => ({ box, index, y: box[0][1] }))
    .sort((a, b) => (a.y === b.y ? a.index - b.index : a.y - b.y))
    .map((item) => item.box);
};
