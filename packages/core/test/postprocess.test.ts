import { describe, expect, it } from 'vitest';
import { DEFAULT_OCR_OPTIONS, mergeOcrResultsByLine } from '../src/index';
import type { OcrResult } from '../src/types';

const box = (minX: number, minY: number, maxX: number, maxY: number): OcrResult['box'] => [
  [minX, minY],
  [maxX, minY],
  [maxX, maxY],
  [minX, maxY],
];

const rotateRect = (minX: number, minY: number, maxX: number, maxY: number, angleDeg: number): OcrResult['box'] => {
  const angle = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const points: OcrResult['box'] = [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ];
  return points.map(([x, y]) => {
    const dx = x - cx;
    const dy = y - cy;
    return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos] as [number, number];
  }) as OcrResult['box'];
};

describe('mergeOcrResultsByLine', () => {
  it('groups results into lines and merges adjacent boxes', () => {
    const results: OcrResult[] = [
      { box: box(0, 0, 10, 10), text: 'foo', confidence: 0.9 },
      { box: box(15, 0, 25, 10), text: 'bar', confidence: 0.7 },
      { box: box(40, 0, 50, 10), text: 'baz', confidence: 0.8 },
      { box: box(0, 30, 10, 40), text: 'line', confidence: 0.95 },
      {
        box: [
          [0, 60],
          [10, 65],
          [10, 75],
          [0, 70],
        ],
        text: 'tilt',
        confidence: 0.6,
      },
    ];

    const merged = mergeOcrResultsByLine(results, {
      ...DEFAULT_OCR_OPTIONS,
      mergeLines: true,
      xThs: 1.0,
      yThs: 0.5,
      heightThs: 0.5,
      maxAngleDeg: 10,
    });

    expect(merged).toHaveLength(4);
    expect(merged[0].text).toBe('foo bar');
    expect(merged[0].confidence).toBe(0.7);
    expect(merged[0].box).toEqual(box(0, 0, 25, 10));
    expect(merged[1].text).toBe('baz');
    expect(merged[2].text).toBe('line');
    expect(merged[3].text).toBe('tilt');
  });

  it('merges only boxes under the maxAngleDeg boundary', () => {
    const results: OcrResult[] = [
      { box: rotateRect(0, 0, 20, 10, 9.9), text: 'foo', confidence: 0.9 },
      { box: rotateRect(22, 0, 42, 10, 9.9), text: 'bar', confidence: 0.8 },
    ];

    const merged = mergeOcrResultsByLine(results, {
      ...DEFAULT_OCR_OPTIONS,
      mergeLines: true,
      maxAngleDeg: 10,
      xThs: 1.0,
      yThs: 0.5,
      heightThs: 0.5,
    });

    expect(merged).toHaveLength(1);
    expect(merged[0].text).toBe('foo bar');
  });

  it('does not merge boxes across the maxAngleDeg boundary', () => {
    const results: OcrResult[] = [
      { box: rotateRect(0, 0, 20, 10, 9.9), text: 'foo', confidence: 0.9 },
      { box: rotateRect(22, 0, 42, 10, 10.1), text: 'bar', confidence: 0.8 },
    ];

    const merged = mergeOcrResultsByLine(results, {
      ...DEFAULT_OCR_OPTIONS,
      mergeLines: true,
      maxAngleDeg: 10,
      xThs: 1.0,
      yThs: 0.5,
      heightThs: 0.5,
    });

    expect(merged).toHaveLength(2);
    expect(merged.map((item) => item.text).sort()).toEqual(['bar', 'foo']);
  });
});
