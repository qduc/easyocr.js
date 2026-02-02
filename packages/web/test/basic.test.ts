import { describe, it, expect } from 'vitest';
import { version, loadImage, loadDetectorModel, loadRecognizerModel, loadCharset } from '../src/index';

describe('easyocr-js web', () => {
  it('should export version from core', () => {
    expect(version).toBe('0.0.1');
  });

  it('should export web runtime helpers', () => {
    expect(loadImage).toBeTypeOf('function');
    expect(loadDetectorModel).toBeTypeOf('function');
    expect(loadRecognizerModel).toBeTypeOf('function');
    expect(loadCharset).toBeTypeOf('function');
  });
});
