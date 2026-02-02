import { describe, it, expect } from 'vitest';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadCharset, loadRecognizerModel } from '../src/index';

describe('loadRecognizerModel textInputName', () => {
  it('ignores configured textInputName when the session has no such input', async () => {
    const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
    const modelPath = path.join(repoRoot, 'models/onnx/english_g2.onnx');
    const charsetPath = path.join(repoRoot, 'models/english_g2.charset.txt');

    const charset = await loadCharset(charsetPath);
    const recognizer = await loadRecognizerModel(modelPath, {
      charset,
      textInputName: 'text',
    });

    expect(recognizer.inputName).toBe('input');
    expect(recognizer.textInputName).toBeUndefined();
  });
});
