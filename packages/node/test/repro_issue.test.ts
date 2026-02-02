import { describe, it, expect } from 'vitest';
import {
  loadImage,
  loadGrayscaleImage,
  loadDetectorModel,
  loadRecognizerModel,
  loadCharset,
  recognize,
} from '../src/index';
import { 
  buildCrops, 
  ctcGreedyDecode, 
  recognizerPreprocess,
  DEFAULT_OCR_OPTIONS
} from '@qduc/easyocr-core';
import { readFile } from 'node:fs/promises';
import fs from 'node:fs';
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
    const recognitionImage = await loadGrayscaleImage(imagePath);
    const detector = await loadDetectorModel(detectorPath);
    const charset = await loadCharset(charsetPath);
    const recognizer = await loadRecognizerModel(recognizerPath, { charset });

    const results = await recognize({ image, recognitionImage, detector, recognizer, options: { mergeLines: false } });
    const expected = JSON.parse(await readFile(expectedPath, 'utf8'));

    expect(results.length).toBe(expected.results.length);
    for (let i = 0; i < expected.results.length; i += 1) {
      expect(results[i].text).toBe(expected.results[i].text);
      for (let p = 0; p < 4; p += 1) {
        expect(results[i].box[p][0]).toBeCloseTo(expected.results[i].box[p][0], 6);
        expect(results[i].box[p][1]).toBeCloseTo(expected.results[i].box[p][1], 6);
      }
    }
  }, 30000);

  it('should match recognition for expected boxes (bypassing detector)', async () => {
    const rootDir = path.resolve(__dirname, '../../..');
    const imagePath = path.join(rootDir, 'python_reference/validation/images/Screenshot_20260201_193653.png');
    const expectedPath = path.join(rootDir, 'python_reference/validation/expected/Screenshot_20260201_193653.json');
    
    const recognizerPath = path.join(rootDir, 'models/onnx/english_g2.onnx');
    const charsetPath = path.join(rootDir, 'models/english_g2.charset.txt');

    const image = await loadImage(imagePath);
    const recognitionImage = await loadGrayscaleImage(imagePath);
    const charset = await loadCharset(charsetPath);
    const recognizer = await loadRecognizerModel(recognizerPath, { charset });

    const expected = JSON.parse(await readFile(expectedPath, 'utf8'));
    
    const boxes = expected.results.map((r: any) => r.box);
    const crops = buildCrops(recognitionImage, boxes, [], DEFAULT_OCR_OPTIONS);
    
    for (let i = 0; i < crops.length; i++) {
      const crop = crops[i];
      const prep = recognizerPreprocess(crop.image, DEFAULT_OCR_OPTIONS);
      const feeds = { [recognizer.inputName]: prep.input };
      const outputs = await recognizer.session.run(feeds);
      const logitsTensor = outputs[recognizer.outputName];
      
      const shape = logitsTensor.shape;
      const steps = shape[shape.length - 2];
      const classes = shape[shape.length - 1];
      const decoded = ctcGreedyDecode(
        logitsTensor.data as Float32Array,
        steps,
        classes,
        recognizer.charset,
        recognizer.blankIndex ?? 0,
        [],
      );

      expect(decoded.text).toBe(expected.results[i].text);
    }
  }, 30000);
});
