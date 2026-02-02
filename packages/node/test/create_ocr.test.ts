import { describe, expect, it } from 'vitest';
import { createOCR } from '../src/index';
import type { DetectorModel, InferenceRunner, RecognizerModel, Tensor } from '@qduc/easyocr-core';

const makeDetector = (): DetectorModel => {
  const runner: InferenceRunner = {
    async run() {
      const data = new Float32Array(4);
      const tensor: Tensor = { data, shape: [1, 2, 2], type: 'float32' };
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

const makeRecognizer = (): RecognizerModel => {
  const runner: InferenceRunner = {
    async run() {
      return {};
    },
  };
  return {
    session: runner,
    inputName: 'input',
    outputName: 'output',
    charset: 'abc',
  };
};

describe('createOCR', () => {
  it('requires a modelDir', async () => {
    await expect(createOCR({ modelDir: '' })).rejects.toThrow('modelDir');
  });

  it('returns an instance that can read images', async () => {
    const detector = makeDetector();
    const recognizer = makeRecognizer();
    const ocr = await createOCR({
      modelDir: 'models',
      lang: 'en',
      detector,
      recognizer,
    });
    const image = {
      data: new Uint8Array(12),
      width: 2,
      height: 2,
      channels: 3,
      channelOrder: 'rgb' as const,
    };
    const results = await ocr.read(image);
    expect(results).toEqual([]);
  });
});
