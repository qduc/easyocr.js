import type { Box, OcrOptions, RasterImage, Tensor } from './types.js';
import { clamp, resizeLongSide, toFloatImage } from './utils.js';

export interface DetectorPreprocessResult {
  input: Tensor;
  resized: RasterImage;
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

export const detectorPreprocess = (image: RasterImage, options: OcrOptions): DetectorPreprocessResult => {
  const target = Math.min(options.canvasSize, Math.max(image.width, image.height) * options.magRatio);
  const { image: resized } = resizeLongSide(image, target, options.align);
  const floatImage = toFloatImage(resized, options.mean, options.std);
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
  if (textMap.width !== linkMap.width || textMap.height !== linkMap.height) {
    throw new Error('Detector output heatmaps must share the same shape.');
  }
  const width = textMap.width;
  const height = textMap.height;
  const size = width * height;
  const combined = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    combined[i] =
      textMap.data[i] > options.lowText || linkMap.data[i] > options.linkThreshold ? 1 : 0;
  }
  const visited = new Uint8Array(size);
  const horizontalList: Box[] = [];
  const freeList: Box[] = [];
  const queue = new Int32Array(size);

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
    let maxText = 0;
    while (qHead < qTail) {
      const idx = queue[qHead++];
      const x = idx % width;
      const y = Math.floor(idx / width);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (textMap.data[idx] > maxText) maxText = textMap.data[idx];
      const neighbors = [
        idx - 1,
        idx + 1,
        idx - width,
        idx + width,
      ];
      for (const nIdx of neighbors) {
        if (nIdx < 0 || nIdx >= size || visited[nIdx] || !combined[nIdx]) continue;
        visited[nIdx] = 1;
        queue[qTail++] = nIdx;
      }
    }
    if (maxText < options.textThreshold) continue;
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    if (Math.min(boxWidth, boxHeight) < options.minSize) continue;
    const margin = options.addMargin * Math.min(boxWidth, boxHeight);
    const paddedMinX = clamp(Math.floor(minX - margin), 0, width - 1);
    const paddedMaxX = clamp(Math.ceil(maxX + margin), 0, width - 1);
    const paddedMinY = clamp(Math.floor(minY - margin), 0, height - 1);
    const paddedMaxY = clamp(Math.ceil(maxY + margin), 0, height - 1);
    const box = extractBoundingBox(paddedMinX, paddedMinY, paddedMaxX, paddedMaxY, scaleX, scaleY);
    if (boxWidth >= boxHeight) {
      horizontalList.push(box);
    } else {
      freeList.push(box);
    }
  }
  return { horizontalList, freeList };
};

interface LineGroup {
  boxes: Box[];
  centerY: number;
  height: number;
}

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
  const lines = groupBoxesByLine(boxes, options);
  const ordered: Box[] = [];
  for (const line of lines) {
    line.boxes.sort((a, b) => boxMetrics(a).centerX - boxMetrics(b).centerX);
    ordered.push(...line.boxes);
  }
  return ordered;
};
