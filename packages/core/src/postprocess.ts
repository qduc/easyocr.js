import type { Box, OcrOptions, OcrResult } from './types.js';

interface ResultMetrics {
  result: OcrResult;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  height: number;
  yCenter: number;
  angleDeg: number;
}

const buildMetrics = (result: OcrResult): ResultMetrics => {
  const xs = result.box.map((point) => point[0]);
  const ys = result.box.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const height = maxY - minY;
  const yCenter = (minY + maxY) / 2;
  const [p0, p1] = result.box;
  const angleRaw = Math.abs((Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180) / Math.PI);
  const angleDeg = angleRaw > 90 ? 180 - angleRaw : angleRaw;
  return { result, minX, maxX, minY, maxY, height, yCenter, angleDeg };
};

const boxUnion = (items: ResultMetrics[]): Box => {
  const minX = Math.min(...items.map((item) => item.minX));
  const maxX = Math.max(...items.map((item) => item.maxX));
  const minY = Math.min(...items.map((item) => item.minY));
  const maxY = Math.max(...items.map((item) => item.maxY));
  return [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ];
};

const shouldGroup = (item: ResultMetrics, line: LineGroup, options: OcrOptions): boolean => {
  const overlap = Math.max(0, Math.min(item.maxY, line.maxY) - Math.max(item.minY, line.minY));
  const minHeight = Math.max(1, Math.min(item.height, line.height));
  const overlapRatio = overlap / minHeight;
  const centerDistanceOk =
    Math.abs(item.yCenter - line.centerY) <= Math.max(item.height, line.height) * options.yThs;
  const heightRatio = Math.max(item.height, line.height) / Math.max(1, Math.min(item.height, line.height));
  const heightSimilar = heightRatio <= 1 + options.heightThs;
  return (overlapRatio >= options.yThs || centerDistanceOk) && heightSimilar;
};

interface LineGroup {
  items: ResultMetrics[];
  centerY: number;
  height: number;
  minY: number;
  maxY: number;
}

const groupByLine = (items: ResultMetrics[], options: OcrOptions): LineGroup[] => {
  const sorted = items.slice().sort((a, b) => a.yCenter - b.yCenter);
  const lines: LineGroup[] = [];
  for (const item of sorted) {
    const target = lines.find((line) => shouldGroup(item, line, options));
    if (target) {
      target.items.push(item);
      target.centerY = (target.centerY * (target.items.length - 1) + item.yCenter) / target.items.length;
      target.height = Math.max(target.height, item.height);
      target.minY = Math.min(target.minY, item.minY);
      target.maxY = Math.max(target.maxY, item.maxY);
    } else {
      lines.push({
        items: [item],
        centerY: item.yCenter,
        height: item.height,
        minY: item.minY,
        maxY: item.maxY,
      });
    }
  }
  return lines;
};

const mergeLineItems = (line: LineGroup, options: OcrOptions): ResultMetrics[] => {
  const ordered = line.items.slice().sort((a, b) => (a.minX === b.minX ? a.minY - b.minY : a.minX - b.minX));
  const merged: ResultMetrics[] = [];
  let current: ResultMetrics[] = [];
  for (const item of ordered) {
    if (current.length === 0) {
      current.push(item);
      continue;
    }
    const prev = current[current.length - 1];
    const gap = item.minX - prev.maxX;
    const lineHeight = Math.max(line.height, item.height, 1);
    if (gap <= options.xThs * lineHeight) {
      current.push(item);
      continue;
    }
    merged.push(buildMergedResult(current));
    current = [item];
  }
  if (current.length > 0) {
    merged.push(buildMergedResult(current));
  }
  return merged;
};

const buildMergedResult = (items: ResultMetrics[]): ResultMetrics => {
  const textParts = items.map((item) => item.result.text).filter((text) => text.length > 0);
  const mergedText = textParts.join(' ');
  const confidence = Math.min(...items.map((item) => item.result.confidence));
  const box = boxUnion(items);
  const merged: OcrResult = { box, text: mergedText, confidence };
  return buildMetrics(merged);
};

export const mergeOcrResultsByLine = (results: OcrResult[], options: OcrOptions): OcrResult[] => {
  if (!options.mergeLines || results.length === 0) {
    return results.slice();
  }

  const metrics = results.map(buildMetrics);
  const horizontal = metrics.filter((item) => item.angleDeg <= options.maxAngleDeg);
  const passthrough = metrics.filter((item) => item.angleDeg > options.maxAngleDeg);

  const lines = groupByLine(horizontal, options);
  const mergedLines = lines.flatMap((line) => mergeLineItems(line, options));

  const combined = [...mergedLines, ...passthrough];
  return combined
    .sort((a, b) => (a.minY === b.minY ? a.minX - b.minX : a.minY - b.minY))
    .map((item) => item.result);
};
