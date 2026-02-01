import type { Box, Point, RasterImage } from './types.js';

export interface FloatImage {
  data: Float32Array;
  width: number;
  height: number;
  channels: number;
}

export interface ResizeResult {
  image: RasterImage;
  scale: number;
}

export const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

export const rotateBox = (box: Box, angleDeg: number, width: number, height: number): Box => {
  const angle = (angleDeg * Math.PI) / 180;
  const cx = width / 2;
  const cy = height / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return box.map(([x, y]) => {
    const dx = x - cx;
    const dy = y - cy;
    return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos] as Point;
  }) as Box;
};

export const resizeImage = (image: RasterImage, targetWidth: number, targetHeight: number): RasterImage => {
  const { data, width, height, channels, channelOrder } = image;
  if (targetWidth === width && targetHeight === height) {
    return image;
  }
  const resized = new Uint8Array(targetWidth * targetHeight * channels);
  const xScale = width / targetWidth;
  const yScale = height / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const srcY = Math.min(height - 1, Math.floor(y * yScale));
    for (let x = 0; x < targetWidth; x += 1) {
      const srcX = Math.min(width - 1, Math.floor(x * xScale));
      const srcIndex = (srcY * width + srcX) * channels;
      const dstIndex = (y * targetWidth + x) * channels;
      for (let c = 0; c < channels; c += 1) {
        resized[dstIndex + c] = data[srcIndex + c];
      }
    }
  }
  return {
    data: resized,
    width: targetWidth,
    height: targetHeight,
    channels,
    channelOrder,
  };
};

export const resizeLongSide = (image: RasterImage, maxSide: number, align: number): ResizeResult => {
  const { width, height } = image;
  const maxDim = Math.max(width, height);
  const scale = maxSide / maxDim;
  const targetWidth = Math.max(align, Math.floor((width * scale + align - 1) / align) * align);
  const targetHeight = Math.max(align, Math.floor((height * scale + align - 1) / align) * align);
  return { image: resizeImage(image, targetWidth, targetHeight), scale };
};

export const extractChannel = (image: RasterImage, channel: number): Float32Array => {
  const { data, width, height, channels } = image;
  const out = new Float32Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    out[i] = data[i * channels + channel] ?? 0;
  }
  return out;
};

export const toFloatImage = (
  image: RasterImage,
  mean: [number, number, number],
  std: [number, number, number],
): FloatImage => {
  const { data, width, height, channels } = image;
  const outChannels = 3;
  const out = new Float32Array(width * height * outChannels);
  const invStd = [1 / std[0], 1 / std[1], 1 / std[2]];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const srcIndex = (y * width + x) * channels;
      const dstIndex = (y * width + x) * outChannels;
      const r = data[srcIndex] / 255;
      const g = data[srcIndex + 1] / 255;
      const b = data[srcIndex + 2] / 255;
      out[dstIndex] = (r - mean[0]) * invStd[0];
      out[dstIndex + 1] = (g - mean[1]) * invStd[1];
      out[dstIndex + 2] = (b - mean[2]) * invStd[2];
    }
  }
  return { data: out, width, height, channels: outChannels };
};

export const rotateImage = (image: RasterImage, angleDeg: number): RasterImage => {
  const normalized = ((angleDeg % 360) + 360) % 360;
  if (normalized === 0) {
    return image;
  }
  const { width, height, channels, channelOrder, data } = image;
  if (normalized === 90 || normalized === 270) {
    const out = new Uint8Array(width * height * channels);
    const outWidth = height;
    const outHeight = width;
    for (let y = 0; y < outHeight; y += 1) {
      for (let x = 0; x < outWidth; x += 1) {
        const srcX = normalized === 90 ? y : outHeight - 1 - y;
        const srcY = normalized === 90 ? width - 1 - x : x;
        const srcIndex = (srcY * width + srcX) * channels;
        const dstIndex = (y * outWidth + x) * channels;
        for (let c = 0; c < channels; c += 1) {
          out[dstIndex + c] = data[srcIndex + c];
        }
      }
    }
    return { data: out, width: outWidth, height: outHeight, channels, channelOrder };
  }
  if (normalized === 180) {
    const out = new Uint8Array(width * height * channels);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const srcIndex = ((height - 1 - y) * width + (width - 1 - x)) * channels;
        const dstIndex = (y * width + x) * channels;
        for (let c = 0; c < channels; c += 1) {
          out[dstIndex + c] = data[srcIndex + c];
        }
      }
    }
    return { data: out, width, height, channels, channelOrder };
  }
  return image;
};

export const cropBox = (image: RasterImage, box: Box): RasterImage => {
  const xs = box.map((point) => point[0]);
  const ys = box.map((point) => point[1]);
  const minX = clamp(Math.floor(Math.min(...xs)), 0, image.width - 1);
  const maxX = clamp(Math.ceil(Math.max(...xs)), 0, image.width - 1);
  const minY = clamp(Math.floor(Math.min(...ys)), 0, image.height - 1);
  const maxY = clamp(Math.ceil(Math.max(...ys)), 0, image.height - 1);
  const width = Math.max(1, maxX - minX + 1);
  const height = Math.max(1, maxY - minY + 1);
  const out = new Uint8Array(width * height * image.channels);
  for (let y = 0; y < height; y += 1) {
    const srcY = minY + y;
    for (let x = 0; x < width; x += 1) {
      const srcX = minX + x;
      const srcIndex = (srcY * image.width + srcX) * image.channels;
      const dstIndex = (y * width + x) * image.channels;
      for (let c = 0; c < image.channels; c += 1) {
        out[dstIndex + c] = image.data[srcIndex + c];
      }
    }
  }
  return {
    data: out,
    width,
    height,
    channels: image.channels,
    channelOrder: image.channelOrder,
  };
};

export interface WarpResult {
  image: RasterImage;
  transform: number[];
}

const solveHomography = (src: Point[], dst: Point[]): number[] => {
  const matrix = Array.from({ length: 8 }, () => new Array(9).fill(0));
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    const rowA = i * 2;
    const rowB = rowA + 1;
    matrix[rowA][0] = x;
    matrix[rowA][1] = y;
    matrix[rowA][2] = 1;
    matrix[rowA][6] = -u * x;
    matrix[rowA][7] = -u * y;
    matrix[rowA][8] = u;
    matrix[rowB][3] = x;
    matrix[rowB][4] = y;
    matrix[rowB][5] = 1;
    matrix[rowB][6] = -v * x;
    matrix[rowB][7] = -v * y;
    matrix[rowB][8] = v;
  }
  for (let i = 0; i < 8; i += 1) {
    let maxRow = i;
    for (let r = i + 1; r < 8; r += 1) {
      if (Math.abs(matrix[r][i]) > Math.abs(matrix[maxRow][i])) {
        maxRow = r;
      }
    }
    const temp = matrix[i];
    matrix[i] = matrix[maxRow];
    matrix[maxRow] = temp;
    const pivot = matrix[i][i] || 1e-12;
    for (let c = i; c < 9; c += 1) {
      matrix[i][c] /= pivot;
    }
    for (let r = 0; r < 8; r += 1) {
      if (r === i) continue;
      const factor = matrix[r][i];
      for (let c = i; c < 9; c += 1) {
        matrix[r][c] -= factor * matrix[i][c];
      }
    }
  }
  const h = matrix.map((row) => row[8]);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
};

export const warpPerspective = (image: RasterImage, box: Box, outputWidth: number, outputHeight: number): WarpResult => {
  const dst: Point[] = [
    [0, 0],
    [outputWidth - 1, 0],
    [outputWidth - 1, outputHeight - 1],
    [0, outputHeight - 1],
  ];
  const transform = solveHomography(box, dst);
  const out = new Uint8Array(outputWidth * outputHeight * image.channels);
  for (let y = 0; y < outputHeight; y += 1) {
    for (let x = 0; x < outputWidth; x += 1) {
      const denom = transform[6] * x + transform[7] * y + transform[8];
      const srcX = (transform[0] * x + transform[1] * y + transform[2]) / denom;
      const srcY = (transform[3] * x + transform[4] * y + transform[5]) / denom;
      const sx = clamp(Math.round(srcX), 0, image.width - 1);
      const sy = clamp(Math.round(srcY), 0, image.height - 1);
      const srcIndex = (sy * image.width + sx) * image.channels;
      const dstIndex = (y * outputWidth + x) * image.channels;
      for (let c = 0; c < image.channels; c += 1) {
        out[dstIndex + c] = image.data[srcIndex + c];
      }
    }
  }
  return {
    image: {
      data: out,
      width: outputWidth,
      height: outputHeight,
      channels: image.channels,
      channelOrder: image.channelOrder,
    },
    transform,
  };
};

export const padToWidth = (
  data: Float32Array,
  width: number,
  height: number,
  channels: number,
  targetWidth: number,
): Float32Array => {
  if (width >= targetWidth) {
    return data;
  }
  const out = new Float32Array(targetWidth * height * channels);
  for (let y = 0; y < height; y += 1) {
    const srcOffset = y * width * channels;
    const dstOffset = y * targetWidth * channels;
    out.set(data.subarray(srcOffset, srcOffset + width * channels), dstOffset);
  }
  return out;
};
