import { describe, it, expect } from 'vitest';
import { version } from '../src/index';
import { version as coreVersion } from '@qduc/easyocr-core';

describe('easyocr-js node', () => {
  it('should export version from core', () => {
    expect(version).toBe(coreVersion);
  });
});
