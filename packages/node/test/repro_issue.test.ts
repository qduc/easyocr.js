import { describe, it, expect } from 'vitest';
import {
  loadImage,
  loadDetectorModel,
  loadRecognizerModel,
  loadCharset,
  recognize,
} from '../src/index';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

describe('Recognition Discrepancy Repro', () => {
  it('should match Python reference for Screenshot_20260201_193653.png', async () => {
    const rootDir = path.resolve(__dirname, '../../..');
    const imagePath = path.join(rootDir, 'python_reference/validation/images/Screenshot_20260201_193653.png');
    const expectedPath = path.join(rootDir, 'python_reference/validation/expected/Screenshot_20260201_193653.json');
    
    const detectorPath = path.join(rootDir, 'models/onnx/craft_mlt_25k.onnx');
    const recognizerPath = path.join(rootDir, 'models/onnx/english_g2.onnx');
    const charsetPath = path.join(rootDir, 'models/english_g2.charset.txt');

    const image = await loadImage(imagePath);
    const detector = await loadDetectorModel(detectorPath);
    const charset = await loadCharset(charsetPath);
    const recognizer = await loadRecognizerModel(recognizerPath, { charset });

    const results = await recognize({ image, detector, recognizer });
    const expected = JSON.parse(await readFile(expectedPath, 'utf8'));

    console.log('Detected results:', results.map(r => r.text));
    console.log('Expected results:', expected.results.map((r: any) => r.text));

    expect(results.length).toBe(expected.results.length);
    
    for (let i = 0; i < expected.results.length; i++) {
      expect(results[i].text).toBe(expected.results[i].text);
    }
  }, 30000); // 30s timeout for model loading and inference
});
