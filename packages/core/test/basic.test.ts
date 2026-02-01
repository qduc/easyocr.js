import { describe, it, expect } from 'vitest';
import { version } from '../src/index';

describe('easyocr-js core', () => {
  it('should have a version', () => {
    expect(version).toBe('0.0.1');
  });
});
