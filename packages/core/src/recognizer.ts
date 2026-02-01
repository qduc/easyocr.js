import type { OcrOptions, RasterImage, Tensor } from './types.js';
import { padToWidth, resizeGrayscaleBicubic, resizeImageBicubic } from './utils.js';

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
  const channels = options.recognizer.inputChannels;
  const mean = options.recognizer.mean;
  const std = options.recognizer.std;
  const targetWidth = Math.max(1, Math.ceil(image.width * scale));

  let width = targetWidth;
  let height = targetHeight;
  let planar: Float32Array;


  if (channels === 1) {
    const isBgr = image.channelOrder === 'bgr' || image.channelOrder === 'bgra';
    const gray = new Float32Array(image.width * image.height);
    for (let y = 0; y < image.height; y += 1) {
      for (let x = 0; x < image.width; x += 1) {
        const srcIndex = (y * image.width + x) * image.channels;
        const c0 = image.data[srcIndex];
        const c1 = image.data[srcIndex + 1];
        const c2 = image.data[srcIndex + 2];
        const r = isBgr ? c2 : c0;
        const g = c1;
        const b = isBgr ? c0 : c2;
        gray[y * image.width + x] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      }
    }
    const resized = resizeGrayscaleBicubic(gray, image.width, image.height, targetWidth, targetHeight);
    planar = new Float32Array(width * height);
    for (let i = 0; i < resized.length; i += 1) {
      planar[i] = resized[i] / 255;
    }
  } else {
    const resized = resizeImageBicubic(image, targetWidth, targetHeight);
    width = resized.width;
    height = resized.height;
    const isBgr = resized.channelOrder === 'bgr' || resized.channelOrder === 'bgra';
    planar = new Float32Array(channels * width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const srcIndex = (y * width + x) * resized.channels;
        const c0 = resized.data[srcIndex] / 255;
        const c1 = resized.data[srcIndex + 1] / 255;
        const c2 = resized.data[srcIndex + 2] / 255;
        const r = isBgr ? c2 : c0;
        const g = c1;
        const b = isBgr ? c0 : c2;
        const dstIndex = y * width + x;
        planar[dstIndex] = r;
        planar[height * width + dstIndex] = g;
        planar[2 * height * width + dstIndex] = b;
      }
    }
  }

  const paddingWidth = Math.ceil(Math.max(options.recognizer.inputWidth, targetWidth) / 4) * 4;

  const padded = padToWidth(planar, width, height, channels, paddingWidth, mean);
  const invStd = 1 / std;
  for (let i = 0; i < padded.length; i += 1) {
    padded[i] = (padded[i] - mean) * invStd;
  }
  return {
    input: {
      data: padded,
      shape: [1, channels, height, paddingWidth],
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
  ignoreIndices: number[] = [],
): DecodeResult => {
  const indexToChar = (index: number): string => {
    if (index === blankIndex) return '';
    if (blankIndex === 0) {
      return charset[index - 1] ?? '';
    }
    const mapped = index > blankIndex ? index - 1 : index;
    return charset[mapped] ?? '';
  };
  const ignoreIdx = new Set([blankIndex, ...ignoreIndices]);
  const maxProbs: number[] = [];
  let text = '';
  let prevIndex = -1;
  for (let t = 0; t < steps; t += 1) {
    let maxLogit = -Infinity;
    let bestIndex = 0;
    for (let c = 0; c < classes; c += 1) {
      const score = logits[t * classes + c];
      if (score > maxLogit) {
        maxLogit = score;
        bestIndex = c;
      }
    }

    // Calculate confidence for this step
    let sum = 0;
    for (let c = 0; c < classes; c += 1) {
      sum += Math.exp(logits[t * classes + c] - maxLogit);
    }
    const prob = 1 / sum; // probability of the bestIndex
    if (bestIndex !== blankIndex) {
      maxProbs.push(prob);
    }

    if (bestIndex !== blankIndex && bestIndex !== prevIndex && !new Set(ignoreIndices).has(bestIndex)) {
      text += indexToChar(bestIndex);
    }
    prevIndex = bestIndex;
  }

  const charCount = text.length;
  let confidence = 0;
  if (charCount > 0) {
    let logSum = 0;
    for (let i = 0; i < maxProbs.length; i += 1) {
      const p = Math.max(maxProbs[i], 1e-8);
      logSum += Math.log(p);
    }
    confidence = Math.exp(logSum / maxProbs.length);
  }
  return { text, confidence };
};
