import type { OcrOptions, RasterImage, Tensor } from './types.js';
import { padToWidth, resizeImage, toFloatImage } from './utils.js';

export interface RecognizerPreprocessResult {
  input: Tensor;
  scale: number;
  width: number;
  height: number;
}

export interface DecodeResult {
  text: string;
  confidence: number;
}

export const recognizerPreprocess = (image: RasterImage, options: OcrOptions): RecognizerPreprocessResult => {
  const targetHeight = options.recognizer.inputHeight;
  const scale = targetHeight / image.height;
  const targetWidth = Math.min(
    options.recognizer.inputWidth,
    Math.max(1, Math.round(image.width * scale)),
  );
  const resized = resizeImage(image, targetWidth, targetHeight);
  const floatImage = toFloatImage(resized, [options.recognizer.mean, options.recognizer.mean, options.recognizer.mean], [
    options.recognizer.std,
    options.recognizer.std,
    options.recognizer.std,
  ]);
  const width = floatImage.width;
  const height = floatImage.height;
  const channels = options.recognizer.inputChannels;
  const imageChannels = floatImage.channels;
  const planar = new Float32Array(channels * width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const srcIndex = (y * width + x) * imageChannels;
      const gray = (floatImage.data[srcIndex] + floatImage.data[srcIndex + 1] + floatImage.data[srcIndex + 2]) / 3;
      const dstIndex = y * width + x;
      if (channels === 1) {
        planar[dstIndex] = gray;
      } else {
        planar[dstIndex] = floatImage.data[srcIndex];
        planar[height * width + dstIndex] = floatImage.data[srcIndex + 1];
        planar[2 * height * width + dstIndex] = floatImage.data[srcIndex + 2];
      }
    }
  }
  const padded = padToWidth(planar, width, height, channels, options.recognizer.inputWidth);
  return {
    input: {
      data: padded,
      shape: [1, channels, height, options.recognizer.inputWidth],
      type: 'float32',
    },
    scale,
    width,
    height,
  };
};

export const ctcGreedyDecode = (
  logits: Float32Array,
  steps: number,
  classes: number,
  charset: string,
  blankIndex = 0,
): DecodeResult => {
  let text = '';
  let confidenceTotal = 0;
  let confidenceCount = 0;
  let prevIndex = -1;
  for (let t = 0; t < steps; t += 1) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < classes; c += 1) {
      const score = logits[t * classes + c];
      if (score > bestScore) {
        bestScore = score;
        bestIndex = c;
      }
    }
    if (bestIndex !== blankIndex && bestIndex !== prevIndex) {
      text += charset[bestIndex] ?? '';
      confidenceTotal += softmaxMax(logits, t, classes);
      confidenceCount += 1;
    }
    prevIndex = bestIndex;
  }
  const confidence = confidenceCount ? confidenceTotal / confidenceCount : 0;
  return { text, confidence };
};

const softmaxMax = (logits: Float32Array, step: number, classes: number): number => {
  let max = -Infinity;
  for (let c = 0; c < classes; c += 1) {
    const score = logits[step * classes + c];
    if (score > max) {
      max = score;
    }
  }
  let sum = 0;
  for (let c = 0; c < classes; c += 1) {
    sum += Math.exp(logits[step * classes + c] - max);
  }
  return Math.exp(max - max) / sum;
};
