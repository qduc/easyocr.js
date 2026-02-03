import { afterEach, describe, expect, it, vi } from 'vitest';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp } from 'node:fs/promises';
import type { DetectorModel, InferenceRunner, RecognizerModel, Tensor } from '@qduc/easyocr-core';

const makeEmptyDetector = (): DetectorModel => {
  const runner: InferenceRunner = {
    async run() {
      const data = new Float32Array([0]);
      const tensor: Tensor = { data, shape: [1, 1], type: 'float32' };
      return { text: tensor, link: tensor };
    },
  };
  return {
    session: runner,
    inputName: 'input',
    textOutputName: 'text',
    linkOutputName: 'link',
  };
};

const makeNoopRecognizer = (): RecognizerModel => {
  const runner: InferenceRunner = {
    async run() {
      throw new Error('Recognizer should not run when no detections are found.');
    },
  };
  return {
    session: runner,
    inputName: 'input',
    outputName: 'output',
    charset: 'abc',
  };
};

const makeTestImage = () => ({
  data: new Uint8Array(3 * 2 * 2),
  width: 2,
  height: 2,
  channels: 3,
  channelOrder: 'rgb' as const,
});

afterEach(() => {
  vi.unmock('node:fs/promises');
  vi.unmock('onnxruntime-node');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.resetModules();
});

describe('model load failures', () => {
  it('surfaces missing model file with path context', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    }));

    const { loadSession } = await import('../src/index');
    const missingPath = path.join(os.tmpdir(), `missing-model-${Date.now()}.onnx`);

    await expect(loadSession(missingPath)).rejects.toThrow(missingPath);
    await expect(loadSession(missingPath)).rejects.toThrow(/Failed to download model/i);
  });

  it('surfaces permission errors with model path context', async () => {
    vi.doMock('node:fs/promises', async () => {
      const actual = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
      return {
        ...actual,
        access: vi
          .fn()
          .mockRejectedValue(Object.assign(new Error('EACCES: permission denied'), { code: 'EACCES' })),
      };
    });

    const { loadSession } = await import('../src/index');
    const blockedPath = '/restricted/model.onnx';

    await expect(loadSession(blockedPath)).rejects.toThrow(blockedPath);
    await expect(loadSession(blockedPath)).rejects.toThrow(/Cannot access model/i);
  });

  it('adds context for invalid ONNX bytes', async () => {
    vi.doMock('onnxruntime-node', () => ({
      InferenceSession: {
        create: vi.fn(() => {
          throw new Error('Invalid ONNX payload');
        }),
      },
      Tensor: class {},
    }));

    const { loadSession } = await import('../src/index');
    const payload = new Uint8Array([1, 2, 3]);

    await expect(loadSession(payload)).rejects.toThrow(/in-memory model bytes/i);
    await expect(loadSession(payload)).rejects.toThrow(/Invalid ONNX payload/i);
  });
});

describe('recognize with empty detections', () => {
  it('returns an empty result list without throwing', async () => {
    const { recognize } = await import('../src/index');
    const results = await recognize({
      image: makeTestImage(),
      detector: makeEmptyDetector(),
      recognizer: makeNoopRecognizer(),
      options: { mergeLines: false },
    });

    expect(results).toEqual([]);
  });
});

describe('createOCR validation', () => {
  it('rejects unsupported language codes with a clear error', async () => {
    const { createOCR } = await import('../src/index');

    await expect(
      createOCR({
        modelDir: 'models',
        lang: 'xx',
      }),
    ).rejects.toThrow(/Unsupported language/i);
  });

  it('throws clear errors for invalid modelDir contents', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    }));

    const { createOCR } = await import('../src/index');
    const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'easyocr-models-'));

    await expect(
      createOCR({
        modelDir: tmpDir,
        lang: 'en',
      }),
    ).rejects.toThrow(new RegExp(tmpDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    await expect(
      createOCR({
        modelDir: tmpDir,
        lang: 'en',
      }),
    ).rejects.toThrow(/craft_mlt_25k\.onnx/);
  });
});

describe('image load failures', () => {
  it('loadImage rejects corrupt buffers with a clear error', async () => {
    const { loadImage } = await import('../src/index');
    const badBuffer = Buffer.from('not-an-image');

    await expect(loadImage(badBuffer)).rejects.toThrow(/unsupported|input/i);
  });

  it('loadGrayscaleImage rejects corrupt buffers with a clear error', async () => {
    const { loadGrayscaleImage } = await import('../src/index');
    const badBuffer = Buffer.from('not-an-image');

    await expect(loadGrayscaleImage(badBuffer)).rejects.toThrow(/unsupported|input/i);
  });
});
