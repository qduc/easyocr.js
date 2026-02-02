import type { OcrOptions, RasterImage, Tensor } from './types.js';
import {
  padToWidth,
  padToWidthReplicateLast,
  resizeGrayscaleBicubic,
  resizeGrayscaleLinear,
  resizeImageBicubic,
} from './utils.js';

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
  const targetWidth = Math.max(1, Math.ceil(image.width * scale));
  const channels = options.recognizer.inputChannels;
  const mean = options.recognizer.mean;
  const std = options.recognizer.std;

  let width = targetWidth;
  let height = targetHeight;
  let planar: Float32Array;


  if (channels === 1) {
    // Match Python EasyOCR recognition preprocessing:
    // - grayscale input
    // - LANCZOS resize to model height preserving aspect ratio (floor)
    // - BICUBIC resize to ceil(width) (AlignCollate)
    // - normalize (x/255 - 0.5)/0.5
    // - pad to max_width (ceil(ratio)*imgH) by repeating last column
    const gray = new Float32Array(image.width * image.height);
    if (image.channels === 1 || image.channelOrder === 'gray') {
      for (let i = 0; i < gray.length; i += 1) {
        gray[i] = image.data[i] ?? 0;
      }
    } else {
      const isBgr = image.channelOrder === 'bgr' || image.channelOrder === 'bgra';
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
    }

    const ratioRaw = image.width / Math.max(1, image.height);
    const ratio = ratioRaw < 1 ? 1 / ratioRaw : ratioRaw;
    const maxWidth = Math.max(1, Math.ceil(ratio) * targetHeight);

    // Stage 1: compute_ratio_and_resize (cv2.resize with interpolation=Image.Resampling.LANCZOS, which is 1 => INTER_LINEAR).
    // Uses floor for the aspect-preserved side.
    let stage1Width: number;
    let stage1Height: number;
    if (ratioRaw < 1) {
      stage1Width = targetHeight;
      stage1Height = Math.max(1, Math.trunc(targetHeight * ratio));
    } else {
      stage1Width = Math.max(1, Math.trunc(targetHeight * ratioRaw));
      stage1Height = targetHeight;
    }
    const stage1 = resizeGrayscaleLinear(gray, image.width, image.height, stage1Width, stage1Height);

    // Stage 2: AlignCollate (BICUBIC) to resized_w x imgH where resized_w = min(maxWidth, ceil(imgH * (w/h))).
    const stage1Ratio = stage1Width / Math.max(1, stage1Height);
    const resizedW = Math.min(maxWidth, Math.max(1, Math.ceil(targetHeight * stage1Ratio)));
    const stage2 = resizeGrayscaleBicubic(stage1, stage1Width, stage1Height, resizedW, targetHeight);

    width = resizedW;
    height = targetHeight;
    planar = new Float32Array(width * height);
    for (let i = 0; i < stage2.length; i += 1) {
      planar[i] = stage2[i] / 255;
    }

    // Normalize and pad (replicate last column) to maxWidth.
    const invStd = 1 / std;
    for (let i = 0; i < planar.length; i += 1) {
      planar[i] = (planar[i] - mean) * invStd;
    }
    const padded = padToWidthReplicateLast(planar, width, height, 1, maxWidth);
    return {
      input: {
        data: padded,
        shape: [1, 1, height, maxWidth],
        type: 'float32',
      },
      scale: targetHeight / image.height,
      width,
      height,
    };
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
    scale: targetHeight / image.height,
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
  const ignoreIdx = new Set(ignoreIndices);
  const maxProbs: number[] = [];
  let text = '';
  let prevIndex = -1;
  for (let t = 0; t < steps; t += 1) {
    let maxLogit = -Infinity;
    let bestIndex = 0;
    for (let c = 0; c < classes; c += 1) {
      if (ignoreIdx.has(c)) continue;
      const score = logits[t * classes + c];
      if (score > maxLogit) {
        maxLogit = score;
        bestIndex = c;
      }
    }

    // Calculate confidence for this step
    let sum = 0;
    for (let c = 0; c < classes; c += 1) {
      if (ignoreIdx.has(c)) continue;
      sum += Math.exp(logits[t * classes + c] - maxLogit);
    }
    const prob = 1 / sum; // probability of the bestIndex after ignore-index renormalization
    if (bestIndex !== blankIndex && !ignoreIdx.has(bestIndex)) {
      maxProbs.push(prob);
    }

    if (bestIndex !== blankIndex && bestIndex !== prevIndex && !ignoreIdx.has(bestIndex)) {
      text += indexToChar(bestIndex);
    }
    prevIndex = bestIndex;
  }

  let confidence = 0;
  if (maxProbs.length > 0) {
    // Match Python EasyOCR: confidence = custom_mean(max_probs)
    // where custom_mean(x) = x.prod() ** (2.0 / sqrt(len(x))).
    let logProd = 0;
    for (const p of maxProbs) {
      if (p <= 0) {
        logProd = -Infinity;
        break;
      }
      logProd += Math.log(p);
    }
    confidence = logProd === -Infinity ? 0 : Math.exp(logProd * (2.0 / Math.sqrt(maxProbs.length)));
  }
  return { text, confidence };
};
