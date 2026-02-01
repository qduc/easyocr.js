import { describe, expect, it } from 'vitest';
import { ctcGreedyDecode, detectorPostprocess, detectorPreprocess, DEFAULT_OCR_OPTIONS } from '../src/index';
import { warpPerspective } from '../src/utils';
import type { RasterImage } from '../src/types';

const makeImage = (width: number, height: number): RasterImage => {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i += 1) {
    data[i * 3] = (i % 255) & 0xff;
    data[i * 3 + 1] = ((i * 2) % 255) & 0xff;
    data[i * 3 + 2] = ((i * 3) % 255) & 0xff;
  }
  return { data, width, height, channels: 3, channelOrder: 'rgb' };
};

describe('core pipeline stages', () => {
  it('preprocesses detector input with resizing and scaling', () => {
    const image = makeImage(200, 100);
    const prep = detectorPreprocess(image, DEFAULT_OCR_OPTIONS);
    expect(prep.input.shape[1]).toBe(3);
    expect(prep.scaleX).toBeGreaterThan(0);
    expect(prep.scaleY).toBeGreaterThan(0);
  });

  it('postprocesses detector heatmaps into boxes', () => {
    const width = 8;
    const height = 8;
    const size = width * height;
    const text = new Float32Array(size).fill(0);
    const link = new Float32Array(size).fill(0);
    for (let y = 2; y <= 4; y += 1) {
      for (let x = 2; x <= 5; x += 1) {
        text[y * width + x] = 0.9;
      }
    }
    const { horizontalList, freeList } = detectorPostprocess(
      { data: text, width, height },
      { data: link, width, height },
      { ...DEFAULT_OCR_OPTIONS, minSize: 1 },
      1,
      1,
    );
    expect(horizontalList.length + freeList.length).toBe(1);
  });

  it('warps a perspective crop to a rectangular image', () => {
    const image = makeImage(10, 10);
    const box: [number, number][] = [
      [1, 1],
      [8, 1],
      [8, 8],
      [1, 8],
    ];
    const warped = warpPerspective(image, box as any, 6, 6);
    expect(warped.image.width).toBe(6);
    expect(warped.image.height).toBe(6);
  });

  it('decodes CTC logits with greedy decoder', () => {
    const charset = '_abc';
    const steps = 3;
    const classes = 4;
    const logits = new Float32Array([
      0, 0, 5, 0,
      0, 0, 5, 0,
      0, 0, 0, 5,
    ]);
    const decoded = ctcGreedyDecode(logits, steps, classes, charset, 0);
    expect(decoded.text).toBe('ab');
    expect(decoded.confidence).toBeGreaterThan(0);
  });
});
