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

const makeImageWithChannels = (
  width: number,
  height: number,
  channels: 1 | 3 | 4,
  channelOrder: RasterImage['channelOrder'],
): RasterImage => {
  const data = new Uint8Array(width * height * channels);
  for (let i = 0; i < width * height * channels; i += 1) {
    data[i] = (i * 17) & 0xff;
  }
  return { data, width, height, channels, channelOrder };
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

  it('postprocesses detector heatmaps with edge-case inputs', () => {
    const options = { ...DEFAULT_OCR_OPTIONS, minSize: 0 };

    const allZero = detectorPostprocess(
      { data: new Float32Array(16).fill(0), width: 4, height: 4 },
      { data: new Float32Array(16).fill(0), width: 4, height: 4 },
      options,
      1,
      1,
    );
    expect(allZero.horizontalList).toHaveLength(0);
    expect(allZero.freeList).toHaveLength(0);

    const allOnes = detectorPostprocess(
      { data: new Float32Array(16).fill(1), width: 4, height: 4 },
      { data: new Float32Array(16).fill(1), width: 4, height: 4 },
      options,
      1,
      1,
    );
    expect(allOnes.horizontalList.length + allOnes.freeList.length).toBe(1);

    const tinySizes: Array<[number, number]> = [
      [1, 1],
      [2, 2],
    ];
    for (const [width, height] of tinySizes) {
      const size = width * height;
      const result = detectorPostprocess(
        { data: new Float32Array(size).fill(1), width, height },
        { data: new Float32Array(size).fill(0), width, height },
        options,
        1,
        1,
      );
      expect(result.horizontalList.length + result.freeList.length).toBeLessThanOrEqual(1);
    }
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

  it('decodes CTC logits with ignoreIndices', () => {
    const charset = 'abc'; // blankIndex is 0, so 1='a', 2='b', 3='c'
    const steps = 3;
    const classes = 4;
    const logits = new Float32Array([
      0, 5, 0, 0, // 'a' (1)
      0, 0, 5, 0, // 'b' (2)
      0, 0, 5, 0  // 'b' (2)
    ]);
    // Ignore 'b' (index 2)
    const decoded = ctcGreedyDecode(logits, steps, classes, charset, 0, [2]);
    // If 'b' is ignored, the next best in the second step might be index 0 or 1 or 3.
    // In our logits, 0 is the max after ignoring 2.
    // So it should be 'a' + '' + '' -> 'a'
    expect(decoded.text).toBe('a');
  });

  it('decodes CTC logits with all blanks', () => {
    const charset = 'ab';
    const steps = 4;
    const classes = 3;
    const logits = new Float32Array([
      5, 0, 0,
      5, 0, 0,
      5, 0, 0,
      5, 0, 0,
    ]);
    const decoded = ctcGreedyDecode(logits, steps, classes, charset, 0);
    expect(decoded.text).toBe('');
    expect(decoded.confidence).toBe(0);
  });

  it('decodes repeated characters with blanks collapsing correctly', () => {
    const charset = 'ab';
    const steps = 4;
    const classes = 3;
    const logits = new Float32Array([
      0, 5, 0,
      0, 5, 0,
      5, 0, 0,
      0, 5, 0,
    ]);
    const decoded = ctcGreedyDecode(logits, steps, classes, charset, 0);
    expect(decoded.text).toBe('aa');
    expect(decoded.confidence).toBeGreaterThan(0);
  });

  it('decodes with an empty charset', () => {
    const charset = '';
    const steps = 2;
    const classes = 1;
    const logits = new Float32Array([
      5,
      5,
    ]);
    const decoded = ctcGreedyDecode(logits, steps, classes, charset, 0);
    expect(decoded.text).toBe('');
  });

  it('preprocesses detector inputs with non-RGB channel orders', () => {
    const bgr = makeImageWithChannels(10, 8, 3, 'bgr');
    const bgra = makeImageWithChannels(10, 8, 4, 'bgra');
    const gray = makeImageWithChannels(10, 8, 1, 'gray');

    const bgrPrep = detectorPreprocess(bgr, DEFAULT_OCR_OPTIONS);
    expect(bgrPrep.input.shape[1]).toBe(3);

    const bgraPrep = detectorPreprocess(bgra, DEFAULT_OCR_OPTIONS);
    expect(bgraPrep.input.shape[1]).toBe(3);

    const grayPrep = detectorPreprocess(gray, DEFAULT_OCR_OPTIONS);
    expect(grayPrep.input.shape[1]).toBe(3);
  });
});
