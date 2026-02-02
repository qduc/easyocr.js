import type { Box, RasterImage, Tensor, TraceStepInput, TraceWriter } from '@qduc/easyocr-core';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

const sanitizeStepName = (name: string): string =>
  name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'step';

const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const sha256Text = (text: string): string => createHash('sha256').update(text).digest('hex');

const statsUint8 = (data: Uint8Array) => {
  let min = 255;
  let max = 0;
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = data.length ? sum / data.length : 0;
  let varSum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const d = data[i] - mean;
    varSum += d * d;
  }
  const std = data.length ? Math.sqrt(varSum / data.length) : 0;
  return { min, max, mean, std };
};

const statsFloat32 = (data: Float32Array) => {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = data.length ? sum / data.length : 0;
  let varSum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const d = data[i] - mean;
    varSum += d * d;
  }
  const std = data.length ? Math.sqrt(varSum / data.length) : 0;
  return { min, max, mean, std };
};

const stableStringify = (value: unknown): string => {
  const normalize = (input: unknown): JsonValue => {
    if (
      input === null ||
      typeof input === 'boolean' ||
      typeof input === 'number' ||
      typeof input === 'string'
    ) {
      return input;
    }
    if (Array.isArray(input)) {
      return input.map((v) => normalize(v));
    }
    if (typeof input === 'object') {
      const record = input as Record<string, unknown>;
      const out: Record<string, JsonValue> = {};
      for (const key of Object.keys(record).sort()) {
        out[key] = normalize(record[key]);
      }
      return out;
    }
    return String(input);
  };
  return JSON.stringify(normalize(value));
};

const toRgbImage = (image: RasterImage): { data: Uint8Array; width: number; height: number; channels: 3 } => {
  const { width, height } = image;
  const src = image.data;
  const out = new Uint8Array(width * height * 3);
  const isBgr = image.channelOrder === 'bgr' || image.channelOrder === 'bgra';
  const step = image.channels;
  for (let i = 0, o = 0; i < src.length; i += step, o += 3) {
    const c0 = src[i] ?? 0;
    const c1 = src[i + 1] ?? 0;
    const c2 = src[i + 2] ?? 0;
    const r = isBgr ? c2 : c0;
    const g = c1;
    const b = isBgr ? c0 : c2;
    out[o] = r;
    out[o + 1] = g;
    out[o + 2] = b;
  }
  return { data: out, width, height, channels: 3 };
};

const boxesToFloat32 = (boxes: Box[]): Float32Array => {
  const out = new Float32Array(boxes.length * 8);
  let o = 0;
  for (const box of boxes) {
    for (const [x, y] of box) {
      out[o++] = x;
      out[o++] = y;
    }
  }
  return out;
};

export interface FsTraceWriterOptions {
  traceDir: string;
  runMeta?: Record<string, unknown>;
}

export class FsTraceWriter implements TraceWriter {
  private traceDir: string;
  private stepsDir: string;
  private stepIndex = 0;
  private index: {
    formatVersion: number;
    createdAt: string;
    runMeta?: Record<string, unknown>;
    steps: Array<{ index: number; name: string; kind: string; dir: string }>;
  };

  constructor(options: FsTraceWriterOptions) {
    this.traceDir = options.traceDir;
    this.stepsDir = path.join(this.traceDir, 'steps');
    this.index = {
      formatVersion: 1,
      createdAt: new Date().toISOString(),
      runMeta: options.runMeta,
      steps: [],
    };
  }

  private async ensureBaseDirs(): Promise<void> {
    await mkdir(this.stepsDir, { recursive: true });
  }

  private async flushIndex(): Promise<void> {
    const outPath = path.join(this.traceDir, 'trace.json');
    await writeFile(outPath, JSON.stringify(this.index, null, 2) + '\n', 'utf8');
  }

  async addStep(step: TraceStepInput): Promise<void> {
    await this.ensureBaseDirs();

    const stepNo = this.stepIndex;
    const dirName = `${String(stepNo).padStart(3, '0')}_${sanitizeStepName(step.name)}`;
    const stepDir = path.join(this.stepsDir, dirName);
    await mkdir(stepDir, { recursive: true });

    const baseMeta: Record<string, unknown> = {
      name: step.name,
      kind: step.kind,
      ...((step.meta ?? {}) as Record<string, unknown>),
    };

    if (step.kind === 'image') {
      const rgb = toRgbImage(step.image);
      const raw = rgb.data;
      const rawBytes = raw;
      const hash = sha256(rawBytes);
      const stats = statsUint8(raw);

      await writeFile(path.join(stepDir, 'raw.bin'), rawBytes);
      await writeFile(
        path.join(stepDir, 'raw.meta.json'),
        JSON.stringify(
          {
            dtype: 'uint8',
            layout: 'HWC',
            colorSpace: 'RGB',
            shape: [rgb.height, rgb.width, 3],
            sha256_raw: hash,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );
      await sharp(Buffer.from(rawBytes), {
        raw: { width: rgb.width, height: rgb.height, channels: 3 },
      })
        .png()
        .toFile(path.join(stepDir, 'image.png'));

      await writeFile(
        path.join(stepDir, 'meta.json'),
        JSON.stringify(
          {
            ...baseMeta,
            dtype: 'uint8',
            layout: 'HWC',
            colorSpace: 'RGB',
            shape: [rgb.height, rgb.width, 3],
            sha256_raw: hash,
            stats,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );
    } else if (step.kind === 'tensor') {
      const tensor = step.tensor;
      const rawBytes = new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
      const hash = sha256(rawBytes);
      const stats =
        tensor.type === 'float32'
          ? statsFloat32(tensor.data as Float32Array)
          : tensor.type === 'uint8'
            ? statsUint8(tensor.data as Uint8Array)
            : undefined;

      await writeFile(path.join(stepDir, 'tensor.bin'), rawBytes);
      await writeFile(
        path.join(stepDir, 'tensor.meta.json'),
        JSON.stringify(
          {
            dtype: tensor.type,
            shape: tensor.shape,
            layout: step.layout,
            colorSpace: step.colorSpace,
            sha256_raw: hash,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );

      // Optional visualization for simple cases.
      if (tensor.type === 'float32' && step.layout && (step.layout === 'HW' || step.layout === 'HWC')) {
        try {
          const shape = tensor.shape;
          const width = step.layout === 'HW' ? shape[1] : shape[1];
          const height = step.layout === 'HW' ? shape[0] : shape[0];
          const channels = step.layout === 'HW' ? 1 : shape[2];
          if (width > 0 && height > 0 && (channels === 1 || channels === 3)) {
            const data = tensor.data as Float32Array;
            const { min, max } = statsFloat32(data);
            const denom = Math.max(1e-6, max - min);
            const preview = new Uint8Array(width * height * channels);
            for (let i = 0; i < preview.length; i += 1) {
              const v = (data[i] - min) / denom;
              preview[i] = Math.max(0, Math.min(255, Math.round(v * 255)));
            }
            await sharp(Buffer.from(preview), {
              raw: { width, height, channels },
            })
              .png()
              .toFile(path.join(stepDir, 'preview.png'));
          }
        } catch {
          // Ignore preview failures.
        }
      }

      await writeFile(
        path.join(stepDir, 'meta.json'),
        JSON.stringify(
          {
            ...baseMeta,
            dtype: tensor.type,
            shape: tensor.shape,
            layout: step.layout,
            colorSpace: step.colorSpace,
            sha256_raw: hash,
            stats,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );
    } else if (step.kind === 'boxes') {
      const flat = boxesToFloat32(step.boxes);
      const rawBytes = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
      const hash = sha256(rawBytes);
      const stats = statsFloat32(flat);

      await writeFile(path.join(stepDir, 'boxes.bin'), rawBytes);
      await writeFile(
        path.join(stepDir, 'boxes.meta.json'),
        JSON.stringify(
          {
            dtype: 'float32',
            shape: [step.boxes.length, 4, 2],
            sha256_raw: hash,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );
      await writeFile(path.join(stepDir, 'boxes.json'), JSON.stringify(step.boxes, null, 2) + '\n', 'utf8');

      await writeFile(
        path.join(stepDir, 'meta.json'),
        JSON.stringify(
          {
            ...baseMeta,
            count: step.boxes.length,
            sha256_raw: hash,
            stats,
          },
          null,
          2,
        ) + '\n',
        'utf8',
      );
    } else if (step.kind === 'params') {
      const json = stableStringify(step.params);
      const hash = sha256Text(json);
      await writeFile(path.join(stepDir, 'params.json'), JSON.stringify(step.params, null, 2) + '\n', 'utf8');
      await writeFile(
        path.join(stepDir, 'meta.json'),
        JSON.stringify({ ...baseMeta, sha256_raw: hash }, null, 2) + '\n',
        'utf8',
      );
    }

    this.index.steps.push({ index: stepNo, name: step.name, kind: step.kind, dir: path.posix.join('steps', dirName) });
    this.stepIndex += 1;
    await this.flushIndex();
  }
}

export const createFsTraceWriter = (options: FsTraceWriterOptions): FsTraceWriter => new FsTraceWriter(options);
