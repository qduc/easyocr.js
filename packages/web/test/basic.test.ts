import { describe, it, expect } from 'vitest';
import { version } from '../src/index';

describe('easyocr-js web', () => {
  it('should export version from core', () => {
    expect(version).toBe('0.0.1');
  });
});
